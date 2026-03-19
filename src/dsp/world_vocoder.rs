//! WORLDアルゴリズムベースの声質変換DSPモジュール
//!
//! 50%オーバーラップ + Hann窓によるoverlap-add方式を採用。
//! 2つのHann窓の50%オーバーラップは定数1.0に合計されるため、
//! ブロック境界での音途切れが原理的に発生しない。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use ndarray::Array2;
use parking_lot::Mutex;

/// フレーム周期 (ms)
const FRAME_PERIOD: f64 = 5.0;

/// F0推定アルゴリズム
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum F0Method {
    Yin,
    Dio,
    Harvest,
}

/// ピッチ操作モード
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PitchMode {
    /// 相対シフト（半音単位）
    Shift(f64),
    /// 固定周波数（Hz）。0 = 強制無声音
    Fixed(f64),
}

/// ワーカースレッドとの共有状態
struct SharedState {
    /// ワーカーへの入力キュー（(分析窓, 出力サンプル数)）
    input_queue: VecDeque<(Vec<f64>, usize)>,
    /// ワーカーからの出力キュー（Hann窓適用済みブロック）
    output_queue: VecDeque<Vec<f32>>,
    /// パラメータ
    pitch_mode: PitchMode,
    formant_shift_semitones: f64,
    /// F0推定アルゴリズム
    f0_method: F0Method,
}

/// WORLDボコーダー処理
pub struct WorldVocoder {
    shared: Arc<Mutex<SharedState>>,
    running: Arc<AtomicBool>,
    worker_handle: Option<thread::JoinHandle<()>>,
    /// 入力蓄積バッファ（f64）
    input_buffer: Vec<f64>,
    /// overlap-add用の出力蓄積バッファ
    ola_buffer: VecDeque<f32>,
    /// 確定済み出力キュー（process()で消費する）
    output_queue: VecDeque<f32>,
    /// パススルー用入力キュー
    dry_buffer: VecDeque<f32>,
    /// ブロックサイズ（分析窓サイズ）
    block_size: usize,
    /// ホップサイズ（= block_size / 2）
    hop_size: usize,
    sample_rate: f32,
    /// ワーカーに投入済みだが結果回収前のブロック数
    pending_blocks: usize,
}

impl WorldVocoder {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_block_ms(sample_rate, 200.0)
    }

    pub fn with_block_ms(sample_rate: f32, block_ms: f32) -> Self {
        let sr = sample_rate as i32;
        let fft_size = world_dsp::get_fft_size_for_cheaptrick(sr, 71.0);
        // block_sizeは偶数に揃える
        let block_size = (((sample_rate * block_ms / 1000.0) as usize).max(1024)) & !1;
        let hop_size = block_size / 2;

        let shared = Arc::new(Mutex::new(SharedState {
            input_queue: VecDeque::new(),
            output_queue: VecDeque::new(),
            pitch_mode: PitchMode::Shift(0.0),
            formant_shift_semitones: 0.0,
            f0_method: F0Method::Yin,
        }));

        let running = Arc::new(AtomicBool::new(true));

        let worker_shared = shared.clone();
        let worker_running = running.clone();
        let worker_handle = thread::spawn(move || {
            Self::worker_loop(worker_shared, worker_running, sr, fft_size);
        });

        Self {
            shared,
            running,
            worker_handle: Some(worker_handle),
            input_buffer: Vec::with_capacity(block_size * 2),
            ola_buffer: VecDeque::with_capacity(block_size * 2),
            output_queue: VecDeque::with_capacity(block_size * 4),
            dry_buffer: VecDeque::with_capacity(block_size * 4),
            block_size,
            hop_size,
            sample_rate,
            pending_blocks: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn set_block_size(&mut self, new_block_size: usize) {
        let new_block_size = (new_block_size.max(1024)) & !1;
        if new_block_size != self.block_size {
            self.block_size = new_block_size;
            self.hop_size = new_block_size / 2;
            self.input_buffer.clear();
            self.ola_buffer.clear();
            self.pending_blocks = 0;
        }
    }

    pub fn set_pitch_mode(&mut self, mode: PitchMode) {
        self.shared.lock().pitch_mode = mode;
    }

    pub fn set_formant_shift(&mut self, semitones: f64) {
        self.shared.lock().formant_shift_semitones = semitones;
    }

    pub fn set_f0_method(&mut self, method: F0Method) {
        self.shared.lock().f0_method = method;
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // 入力を蓄積
        self.input_buffer.extend(input.iter().map(|&s| s as f64));
        self.dry_buffer.extend(input.iter());

        // hop_sizeごとにブロックをワーカーへ投入
        // 最初のブロックはblock_size必要、以降はhop_sizeずつスライド
        while self.input_buffer.len() >= self.block_size {
            let block: Vec<f64> = self.input_buffer[..self.block_size].to_vec();
            self.input_buffer.drain(..self.hop_size);

            let target_len = self.block_size;
            self.shared
                .lock()
                .input_queue
                .push_back((block, target_len));
            self.pending_blocks += 1;
        }

        // ワーカーの処理結果を回収してoverlap-add
        {
            let results: Vec<Vec<f32>> = {
                let mut state = self.shared.lock();
                state.output_queue.drain(..).collect()
            };
            for windowed_block in &results {
                self.overlap_add(windowed_block);
                self.pending_blocks = self.pending_blocks.saturating_sub(1);
            }
        }

        // 出力
        for sample in output.iter_mut() {
            if let Some(s) = self.output_queue.pop_front() {
                *sample = s;
                // 処理済み分のdryバッファを消費
                self.dry_buffer.pop_front();
            } else if let Some(s) = self.dry_buffer.pop_front() {
                *sample = s;
            } else {
                *sample = 0.0;
            }
        }
    }

    /// Hann窓適用済みブロックをola_bufferにoverlap-add
    /// ola_bufferからhop_sizeサンプルをoutput_queueに確定出力
    fn overlap_add(&mut self, windowed_block: &[f32]) {
        let block_len = windowed_block.len();

        // ola_bufferを必要なサイズに拡張（0パディング）
        while self.ola_buffer.len() < block_len {
            self.ola_buffer.push_back(0.0);
        }

        // 加算
        for (i, &s) in windowed_block.iter().enumerate() {
            self.ola_buffer[i] += s;
        }

        // 先頭hop_sizeサンプルを確定出力に移動
        let out_len = self.hop_size.min(self.ola_buffer.len());
        for _ in 0..out_len {
            if let Some(s) = self.ola_buffer.pop_front() {
                self.output_queue.push_back(s);
            }
        }
    }

    /// ワーカースレッドのメインループ
    fn worker_loop(
        shared: Arc<Mutex<SharedState>>,
        running: Arc<AtomicBool>,
        sample_rate: i32,
        fft_size: usize,
    ) {
        let mut window_cache: Option<(usize, Vec<f32>)> = None;
        // 前ブロック合成結果の後半（位相アライメント参照用）
        let mut prev_block_tail: Vec<f32> = Vec::new();

        while running.load(Ordering::Relaxed) {
            let (item, pitch_mode, formant, f0_method) = {
                let mut state = shared.lock();
                if let Some(item) = state.input_queue.pop_front() {
                    (
                        Some(item),
                        state.pitch_mode,
                        state.formant_shift_semitones,
                        state.f0_method,
                    )
                } else {
                    (None, PitchMode::Shift(0.0), 0.0, F0Method::Yin)
                }
            };

            if let Some((analysis_block, target_len)) = item {
                let synthesized = Self::process_block(
                    &analysis_block,
                    target_len,
                    sample_rate,
                    fft_size,
                    pitch_mode,
                    formant,
                    f0_method,
                );

                // 位相アライメント: 前ブロック末尾との相関が最大になるオフセットを探索
                let aligned = if !prev_block_tail.is_empty() && synthesized.len() > 0 {
                    let best_offset =
                        find_best_phase_offset(&prev_block_tail, &synthesized, sample_rate);
                    apply_offset(&synthesized, best_offset)
                } else {
                    synthesized
                };

                // 次回の位相アライメント用に後半を保存
                let half = aligned.len() / 2;
                prev_block_tail = aligned[half..].to_vec();

                // Hann窓を適用
                let window = match &window_cache {
                    Some((size, w)) if *size == aligned.len() => w,
                    _ => {
                        let w = create_hann_window(aligned.len());
                        window_cache = Some((aligned.len(), w));
                        &window_cache.as_ref().unwrap().1
                    }
                };

                let windowed: Vec<f32> = aligned
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| s * w)
                    .collect();

                shared.lock().output_queue.push_back(windowed);
            } else {
                thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// 1ブロック分のWORLD分析→変換→再合成
    fn process_block(
        analysis_block: &[f64],
        target_len: usize,
        sample_rate: i32,
        fft_size: usize,
        pitch_mode: PitchMode,
        formant_shift_semitones: f64,
        f0_method: F0Method,
    ) -> Vec<f32> {
        // F0推定（選択されたアルゴリズムを使用）
        let (temporal_positions, f0) = match f0_method {
            F0Method::Yin => {
                let estimator = world_dsp::Yin::new(sample_rate);
                world_dsp::F0Estimator::estimate(&estimator, analysis_block)
            }
            F0Method::Dio => {
                let estimator = world_dsp::Dio::new(sample_rate);
                world_dsp::F0Estimator::estimate(&estimator, analysis_block)
            }
            F0Method::Harvest => {
                let estimator = world_dsp::Harvest::new(sample_rate);
                world_dsp::F0Estimator::estimate(&estimator, analysis_block)
            }
        };

        if f0.is_empty() {
            return analysis_block.iter().map(|&s| s as f32).collect();
        }

        // スペクトル包絡推定
        let cheaptrick = world_dsp::CheapTrick::new(sample_rate, fft_size);
        let spectrogram = cheaptrick.estimate(analysis_block, &temporal_positions, &f0);

        // 非周期性指標推定
        let d4c = world_dsp::D4C::new(sample_rate, fft_size);
        let aperiodicity = d4c.estimate(analysis_block, &temporal_positions, &f0);

        // ピッチ操作
        let modified_f0: Vec<f64> = match pitch_mode {
            PitchMode::Shift(semitones) => {
                let ratio = 2.0_f64.powf(semitones / 12.0);
                f0.iter()
                    .map(|&v| if v > 0.0 { v * ratio } else { 0.0 })
                    .collect()
            }
            PitchMode::Fixed(hz) => {
                // 全フレームを指定周波数に固定（0 = 無声音）
                vec![hz; f0.len()]
            }
        };

        // フォルマントシフト
        let formant_ratio = 2.0_f64.powf(formant_shift_semitones / 12.0);
        let mut modified_spectrogram = if (formant_ratio - 1.0).abs() > 1e-6 {
            shift_spectrogram(&spectrogram, formant_ratio)
        } else {
            spectrogram
        };

        // 再合成
        let synthesizer = world_dsp::Synthesizer::new(FRAME_PERIOD, sample_rate, fft_size);
        let full_result =
            synthesizer.synthesize(&modified_f0, &modified_spectrogram, &aperiodicity);
        let full: Vec<f64> = full_result.to_vec();

        let resampled = resample_linear(&full, target_len);
        resampled.iter().map(|&s| s as f32).collect()
    }
}

impl Drop for WorldVocoder {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Hann窓を生成
fn create_hann_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..size)
        .map(|i| {
            let t = i as f32 / size as f32;
            0.5 * (1.0 - (2.0 * PI * t).cos())
        })
        .collect()
}

/// 線形補間リサンプル
fn resample_linear(input: &[f64], target_len: usize) -> Vec<f64> {
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if target_len == 1 {
        return vec![input[0]];
    }
    let ratio = (input.len() - 1) as f64 / (target_len - 1) as f64;
    (0..target_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f64;
            if idx + 1 < input.len() {
                input[idx] * (1.0 - frac) + input[idx + 1] * frac
            } else {
                input[idx.min(input.len() - 1)]
            }
        })
        .collect()
}

/// スペクトル包絡を周波数方向にシフト（フォルマント操作）
/// ratio > 1.0: 高域へシフト（声が細くなる）
/// ratio < 1.0: 低域へシフト（声が太くなる）
fn shift_spectrogram(spectrogram: &Array2<f64>, ratio: f64) -> Array2<f64> {
    let (num_frames, freq_bins) = spectrogram.dim();
    let mut shifted = Array2::zeros((num_frames, freq_bins));
    let last_bin = freq_bins - 1;

    for frame in 0..num_frames {
        for bin in 0..freq_bins {
            let src_bin = bin as f64 / ratio;
            let src_idx = src_bin.floor() as usize;
            let frac = src_bin - src_idx as f64;

            if src_idx + 1 < freq_bins {
                // 範囲内: 線形補間
                shifted[[frame, bin]] = spectrogram[[frame, src_idx]] * (1.0 - frac)
                    + spectrogram[[frame, src_idx + 1]] * frac;
            } else {
                // 範囲外: 最終ビンの値で埋める（無音にしない）
                shifted[[frame, bin]] = spectrogram[[frame, last_bin]];
            }
        }
    }

    shifted
}

/// 位相アライメント: 前ブロック末尾と新ブロック先頭の相互相関が最大になるオフセットを探索
/// 戻り値: 最適オフセット（サンプル数、負=前にシフト、正=後にシフト）
fn find_best_phase_offset(prev_tail: &[f32], new_block: &[f32], sample_rate: i32) -> i32 {
    // 相関に使う長さ（前ブロック末尾の長さ、最大で新ブロックの1/4）
    let corr_len = prev_tail.len().min(new_block.len() / 4).min(2048);
    if corr_len < 16 {
        return 0;
    }

    // 探索範囲: ±(1周期分程度)。低音100Hzの1周期 = sample_rate/100 サンプル
    let max_search = (sample_rate as i32 / 100).min(new_block.len() as i32 / 4);

    let mut best_offset = 0i32;
    let mut best_corr = f32::MIN;

    // 粗い探索（4サンプル刻み）
    let coarse_step = 4i32.max(max_search / 64);
    for offset in (-max_search..=max_search).step_by(coarse_step as usize) {
        let corr = normalized_correlation(prev_tail, new_block, offset, corr_len);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }

    // 細かい探索（1サンプル刻み、粗い探索の周辺）
    let fine_range = coarse_step;
    let fine_best_base = best_offset;
    for offset in (fine_best_base - fine_range)..=(fine_best_base + fine_range) {
        if offset < -max_search || offset > max_search {
            continue;
        }
        let corr = normalized_correlation(prev_tail, new_block, offset, corr_len);
        if corr > best_corr {
            best_corr = corr;
            best_offset = offset;
        }
    }

    best_offset
}

/// 正規化相互相関を計算
/// prev_tail の末尾 corr_len サンプルと、new_block の (offset..) から corr_len サンプルを比較
fn normalized_correlation(
    prev_tail: &[f32],
    new_block: &[f32],
    offset: i32,
    corr_len: usize,
) -> f32 {
    let ref_start = prev_tail.len().saturating_sub(corr_len);
    let new_start = offset.max(0) as usize;

    if new_start + corr_len > new_block.len() {
        return f32::MIN;
    }

    let mut correlation = 0.0f32;
    let mut energy_ref = 0.0f32;
    let mut energy_new = 0.0f32;

    // サブサンプリングで高速化
    const STEP: usize = 4;
    for i in (0..corr_len).step_by(STEP) {
        let r = prev_tail[ref_start + i];
        let n = new_block[new_start + i];
        correlation += r * n;
        energy_ref += r * r;
        energy_new += n * n;
    }

    let denom = (energy_ref * energy_new).sqrt();
    if denom > 1e-10 {
        correlation / denom
    } else {
        0.0
    }
}

/// オフセットを適用してブロックをシフト（循環シフト）
fn apply_offset(block: &[f32], offset: i32) -> Vec<f32> {
    if offset == 0 || block.is_empty() {
        return block.to_vec();
    }
    let len = block.len();
    let shift = ((offset % len as i32) + len as i32) as usize % len;
    let mut result = vec![0.0f32; len];
    for i in 0..len {
        result[i] = block[(i + shift) % len];
    }
    result
}
