//! WORLDアルゴリズムベースの声質変換DSPモジュール
//!
//! 入力音声をブロック単位で蓄積し、ワーカースレッドでWORLD分析→パラメータ変換→再合成を行う。
//! 合成結果は入力と同じサンプル数にリサンプルし、入出力レートを厳密に一致させる。
//! 出力が間に合わない場合は入力信号をパススルーし、音途切れを防止する。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use ndarray::Array2;
use parking_lot::Mutex;

/// フレーム周期 (ms)
const FRAME_PERIOD: f64 = 5.0;
/// ブロック境界のクロスフェード長（サンプル数）
const CROSSFADE_LEN: usize = 128;
/// ピッチ推定・分析のオーバーラップ（前ブロック末尾から引き継ぐサンプル数）
const ANALYSIS_OVERLAP: usize = 2048;

/// ワーカースレッドとの共有状態
struct SharedState {
    /// ワーカーへの入力キュー（(分析窓全体, 出力サンプル数)）
    input_queue: VecDeque<(Vec<f64>, usize)>,
    /// ワーカーからの出力キュー（処理済み、target_lenにリサンプル済み）
    output_queue: VecDeque<Vec<f64>>,
    /// パラメータ
    pitch_shift_semitones: f64,
    formant_shift_semitones: f64,
}

/// WORLDボコーダー処理
pub struct WorldVocoder {
    shared: Arc<Mutex<SharedState>>,
    running: Arc<AtomicBool>,
    worker_handle: Option<thread::JoinHandle<()>>,
    /// 入力蓄積バッファ
    input_buffer: Vec<f64>,
    /// 処理済み出力キュー（ワーカーからの結果）
    output_buffer: VecDeque<f32>,
    /// パススルー用入力キュー（出力が間に合わない時のフォールバック）
    dry_buffer: VecDeque<f32>,
    /// 分析ブロックサイズ
    block_size: usize,
    sample_rate: f32,
    /// 前ブロック末尾（分析オーバーラップ用）
    prev_block_tail: Vec<f64>,
    /// 前ブロック末尾（クロスフェード用）
    prev_tail: [f64; CROSSFADE_LEN],
    has_prev_tail: bool,
}

impl WorldVocoder {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_block_ms(sample_rate, 80.0)
    }

    pub fn with_block_ms(sample_rate: f32, block_ms: f32) -> Self {
        let sr = sample_rate as i32;
        let fft_size = world_dsp::get_fft_size_for_cheaptrick(sr, 71.0);
        let block_size = ((sample_rate * block_ms / 1000.0) as usize).max(1024);

        let shared = Arc::new(Mutex::new(SharedState {
            input_queue: VecDeque::new(),
            output_queue: VecDeque::new(),
            pitch_shift_semitones: 0.0,
            formant_shift_semitones: 0.0,
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
            output_buffer: VecDeque::with_capacity(block_size * 4),
            dry_buffer: VecDeque::with_capacity(block_size * 4),
            block_size,
            sample_rate,
            prev_block_tail: Vec::new(),
            prev_tail: [0.0; CROSSFADE_LEN],
            has_prev_tail: false,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn set_block_size(&mut self, new_block_size: usize) {
        let new_block_size = new_block_size.max(1024);
        if new_block_size != self.block_size {
            self.block_size = new_block_size;
            self.input_buffer.clear();
            self.prev_block_tail.clear();
            self.has_prev_tail = false;
        }
    }

    pub fn set_pitch_shift(&mut self, semitones: f64) {
        self.shared.lock().pitch_shift_semitones = semitones;
    }

    pub fn set_formant_shift(&mut self, semitones: f64) {
        self.shared.lock().formant_shift_semitones = semitones;
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // 入力を蓄積
        self.input_buffer.extend(input.iter().map(|&s| s as f64));

        // パススルー用にも入力を保持
        self.dry_buffer.extend(input.iter());

        // ブロック単位でワーカーに投入
        while self.input_buffer.len() >= self.block_size {
            let block: Vec<f64> = self.input_buffer.drain(..self.block_size).collect();

            // 前ブロック末尾 + 現ブロックの連結データを作成
            let mut analysis_block =
                Vec::with_capacity(self.prev_block_tail.len() + block.len());
            analysis_block.extend_from_slice(&self.prev_block_tail);
            analysis_block.extend_from_slice(&block);

            // 現ブロックの末尾をオーバーラップとして保存
            let tail_start = block.len().saturating_sub(ANALYSIS_OVERLAP);
            self.prev_block_tail = block[tail_start..].to_vec();

            self.shared
                .lock()
                .input_queue
                .push_back((analysis_block, block.len()));
        }

        // ワーカーの処理結果を回収
        let results: Vec<Vec<f64>> = {
            let mut state = self.shared.lock();
            state.output_queue.drain(..).collect()
        };
        for block in &results {
            self.enqueue_block(block);
            // 処理済み分のdryバッファを消費
            let consume = block.len().min(self.dry_buffer.len());
            self.dry_buffer.drain(..consume);
        }

        // 出力: 処理済みがあればそれを使い、なければパススルー
        for sample in output.iter_mut() {
            if let Some(s) = self.output_buffer.pop_front() {
                *sample = s;
            } else if let Some(s) = self.dry_buffer.pop_front() {
                // 処理が追いつかない場合はパススルー
                *sample = s;
            } else {
                *sample = 0.0;
            }
        }
    }

    fn enqueue_block(&mut self, block: &[f64]) {
        if block.is_empty() {
            return;
        }

        if self.has_prev_tail && block.len() >= CROSSFADE_LEN {
            for i in 0..CROSSFADE_LEN {
                let t = (i + 1) as f64 / (CROSSFADE_LEN + 1) as f64;
                let mixed = self.prev_tail[i] * (1.0 - t) + block[i] * t;
                self.output_buffer.push_back(mixed as f32);
            }
            for &s in &block[CROSSFADE_LEN..] {
                self.output_buffer.push_back(s as f32);
            }
        } else {
            for &s in block {
                self.output_buffer.push_back(s as f32);
            }
        }

        if block.len() >= CROSSFADE_LEN {
            self.prev_tail
                .copy_from_slice(&block[block.len() - CROSSFADE_LEN..]);
            self.has_prev_tail = true;
        }
    }

    fn worker_loop(
        shared: Arc<Mutex<SharedState>>,
        running: Arc<AtomicBool>,
        sample_rate: i32,
        fft_size: usize,
    ) {
        while running.load(Ordering::Relaxed) {
            let (item, pitch, formant) = {
                let mut state = shared.lock();
                if let Some(item) = state.input_queue.pop_front() {
                    (
                        Some(item),
                        state.pitch_shift_semitones,
                        state.formant_shift_semitones,
                    )
                } else {
                    (None, 0.0, 0.0)
                }
            };

            if let Some((analysis_block, target_len)) = item {
                let synthesized = Self::process_block(
                    &analysis_block,
                    target_len,
                    sample_rate,
                    fft_size,
                    pitch,
                    formant,
                );
                shared.lock().output_queue.push_back(synthesized);
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
    ) -> Vec<f64> {
        // F0推定（YIN）
        let yin = world_dsp::Yin::new(sample_rate);
        let (temporal_positions, f0) = world_dsp::F0Estimator::estimate(&yin, analysis_block);

        if f0.is_empty() {
            // 推定失敗時は入力末尾をそのまま返す
            let start = analysis_block.len().saturating_sub(target_len);
            return analysis_block[start..].to_vec();
        }

        // スペクトル包絡推定（分析窓全体）
        let cheaptrick = world_dsp::CheapTrick::new(sample_rate, fft_size);
        let spectrogram = cheaptrick.estimate(analysis_block, &temporal_positions, &f0);

        // 非周期性指標推定（分析窓全体）
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
        let modified_spectrogram = if (formant_ratio - 1.0).abs() > 1e-6 {
            shift_spectrogram(&spectrogram, formant_ratio)
        } else {
            spectrogram
        };

        // 再合成（分析窓全体分）
        let synthesizer = world_dsp::Synthesizer::new(FRAME_PERIOD, sample_rate, fft_size);
        let full_result =
            synthesizer.synthesize(&modified_f0, &modified_spectrogram, &aperiodicity);
        let full = full_result.to_vec();

        // 末尾 target_len サンプルを取り出してリサンプル
        if full.len() >= target_len {
            let start = full.len() - target_len;
            resample_linear(&full[start..], target_len)
        } else {
            resample_linear(&full, target_len)
        }
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
