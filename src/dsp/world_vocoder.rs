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
/// デフォルトの倍音制御数
pub const DEFAULT_NUM_HARMONICS: usize = 8;

/// ワーカースレッドとの共有状態
struct SharedState {
    /// ワーカーへの入力キュー（(分析窓, 出力サンプル数)）
    input_queue: VecDeque<(Vec<f64>, usize)>,
    /// ワーカーからの出力キュー（Hann窓適用済みブロック）
    output_queue: VecDeque<Vec<f32>>,
    /// パラメータ
    pitch_shift_semitones: f64,
    formant_shift_semitones: f64,
    /// 倍音振幅係数（index 0 = 基本波, index 1 = 第2倍音, ...）
    /// 1.0 = 変化なし, 0.0 = 消音, 2.0 = 2倍
    harmonic_gains: Vec<f64>,
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
            pitch_shift_semitones: 0.0,
            formant_shift_semitones: 0.0,
            harmonic_gains: vec![1.0; DEFAULT_NUM_HARMONICS],
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

    pub fn set_pitch_shift(&mut self, semitones: f64) {
        self.shared.lock().pitch_shift_semitones = semitones;
    }

    pub fn set_formant_shift(&mut self, semitones: f64) {
        self.shared.lock().formant_shift_semitones = semitones;
    }

    pub fn set_harmonic_gains(&mut self, gains: &[f64]) {
        self.shared.lock().harmonic_gains = gains.to_vec();
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
        // Hann窓のキャッシュ（サイズごと）
        let mut window_cache: Option<(usize, Vec<f32>)> = None;

        while running.load(Ordering::Relaxed) {
            let (item, pitch, formant, harmonics) = {
                let mut state = shared.lock();
                if let Some(item) = state.input_queue.pop_front() {
                    (
                        Some(item),
                        state.pitch_shift_semitones,
                        state.formant_shift_semitones,
                        state.harmonic_gains.clone(),
                    )
                } else {
                    (None, 0.0, 0.0, Vec::new())
                }
            };

            if let Some((analysis_block, target_len)) = item {
                // WORLD処理
                let synthesized = Self::process_block(
                    &analysis_block,
                    target_len,
                    sample_rate,
                    fft_size,
                    pitch,
                    formant,
                    &harmonics,
                );

                // Hann窓を適用
                let window = match &window_cache {
                    Some((size, w)) if *size == synthesized.len() => w,
                    _ => {
                        let w = create_hann_window(synthesized.len());
                        window_cache = Some((synthesized.len(), w));
                        &window_cache.as_ref().unwrap().1
                    }
                };

                let windowed: Vec<f32> = synthesized
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
        pitch_shift_semitones: f64,
        formant_shift_semitones: f64,
        harmonic_gains: &[f64],
    ) -> Vec<f32> {
        // F0推定
        let yin = world_dsp::Yin::new(sample_rate);
        let (temporal_positions, f0) = world_dsp::F0Estimator::estimate(&yin, analysis_block);

        if f0.is_empty() {
            return analysis_block.iter().map(|&s| s as f32).collect();
        }

        // スペクトル包絡推定
        let cheaptrick = world_dsp::CheapTrick::new(sample_rate, fft_size);
        let spectrogram = cheaptrick.estimate(analysis_block, &temporal_positions, &f0);

        // 非周期性指標推定
        let d4c = world_dsp::D4C::new(sample_rate, fft_size);
        let aperiodicity = d4c.estimate(analysis_block, &temporal_positions, &f0);

        // ピッチシフト
        let pitch_ratio = 2.0_f64.powf(pitch_shift_semitones / 12.0);
        let modified_f0: Vec<f64> = f0
            .iter()
            .map(|&v| if v > 0.0 { v * pitch_ratio } else { 0.0 })
            .collect();

        // フォルマントシフト
        let formant_ratio = 2.0_f64.powf(formant_shift_semitones / 12.0);
        let mut modified_spectrogram = if (formant_ratio - 1.0).abs() > 1e-6 {
            shift_spectrogram(&spectrogram, formant_ratio)
        } else {
            spectrogram
        };

        // 倍音振幅制御
        if !harmonic_gains.is_empty() {
            apply_harmonic_gains(
                &mut modified_spectrogram,
                &f0,
                harmonic_gains,
                sample_rate,
                fft_size,
            );
        }

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
fn shift_spectrogram(spectrogram: &Array2<f64>, ratio: f64) -> Array2<f64> {
    let (num_frames, freq_bins) = spectrogram.dim();
    let mut shifted = Array2::zeros((num_frames, freq_bins));

    for frame in 0..num_frames {
        for bin in 0..freq_bins {
            let src_bin = bin as f64 / ratio;
            let src_idx = src_bin.floor() as usize;
            let frac = src_bin - src_idx as f64;

            if src_idx + 1 < freq_bins {
                shifted[[frame, bin]] = spectrogram[[frame, src_idx]] * (1.0 - frac)
                    + spectrogram[[frame, src_idx + 1]] * frac;
            } else if src_idx < freq_bins {
                shifted[[frame, bin]] = spectrogram[[frame, src_idx]];
            }
        }
    }

    shifted
}

/// 倍音振幅制御: 各フレームのF0に基づいてn次倍音のスペクトル包絡にゲインを適用
///
/// CheapTrickの出力はパワースペクトル密度(PSD)なので、振幅ゲインgに対して
/// g^2をPSDに乗算する必要がある（振幅 = sqrt(PSD)のため）。
/// 各倍音の中心ビンにガウシアン重みでゲインを適用し、隣接倍音への干渉を防ぐ。
fn apply_harmonic_gains(
    spectrogram: &mut Array2<f64>,
    f0: &[f64],
    harmonic_gains: &[f64],
    sample_rate: i32,
    fft_size: usize,
) {
    let (num_frames, freq_bins) = spectrogram.dim();
    let bin_freq = sample_rate as f64 / fft_size as f64;

    for frame in 0..num_frames {
        let frame_f0 = if frame < f0.len() { f0[frame] } else { 0.0 };
        if frame_f0 <= 0.0 {
            continue;
        }

        // 各ビンに適用するPSDゲインを計算
        let mut bin_gains = vec![1.0_f64; freq_bins];

        for (n, &gain) in harmonic_gains.iter().enumerate() {
            if (gain - 1.0).abs() < 1e-6 {
                continue;
            }

            let harmonic_freq = frame_f0 * (n + 1) as f64;
            let center_bin = harmonic_freq / bin_freq;

            if center_bin >= freq_bins as f64 {
                break;
            }

            // PSDドメインのゲイン（振幅ゲインの2乗）
            let psd_gain = gain * gain;

            // σ = 倍音間隔の半分程度（F0/bin_freqが1倍音分のビン幅）
            let sigma = (frame_f0 / bin_freq) * 0.35;
            let sigma_sq = sigma * sigma;
            let range = (sigma * 3.0).ceil() as i64;
            let center = center_bin.round() as i64;

            for offset in -range..=range {
                let bin = center + offset;
                if bin >= 0 && (bin as usize) < freq_bins {
                    let dist = bin as f64 - center_bin;
                    let weight = (-0.5 * dist * dist / sigma_sq).exp();
                    // ガウシアン重みで1.0からpsd_gainへ補間
                    let effective_gain = 1.0 + weight * (psd_gain - 1.0);
                    bin_gains[bin as usize] *= effective_gain;
                }
            }
        }

        for bin in 0..freq_bins {
            spectrogram[[frame, bin]] *= bin_gains[bin];
        }
    }
}
