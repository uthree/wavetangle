//! DSP処理モジュール
//!
//! 各ノードのオーディオ処理アルゴリズムを実装

use std::f32::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::nodes::{FilterType, FFT_SIZE};

/// Biquadフィルター係数
#[derive(Clone, Debug)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadCoeffs {
    /// ローパスフィルター係数を計算
    pub fn lowpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// ハイパスフィルター係数を計算
    pub fn highpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// バンドパスフィルター係数を計算
    pub fn bandpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// フィルタータイプに応じた係数を計算
    pub fn from_filter_type(
        filter_type: FilterType,
        sample_rate: f32,
        cutoff: f32,
        q: f32,
    ) -> Self {
        match filter_type {
            FilterType::Low => Self::lowpass(sample_rate, cutoff, q),
            FilterType::High => Self::highpass(sample_rate, cutoff, q),
            FilterType::Band => Self::bandpass(sample_rate, cutoff, q),
        }
    }
}

/// Biquadフィルター状態
#[derive(Clone, Default)]
pub struct BiquadState {
    pub x1: f32,
    pub x2: f32,
    pub y1: f32,
    pub y2: f32,
}

impl BiquadState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 1サンプル処理
    pub fn process(&mut self, input: f32, coeffs: &BiquadCoeffs) -> f32 {
        let output = coeffs.b0 * input + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// リセット
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// コンプレッサーパラメータ
#[derive(Clone, Copy)]
pub struct CompressorParams {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_db: f32,
    pub sample_rate: f32,
}

/// コンプレッサー状態
#[derive(Clone)]
pub struct CompressorState {
    /// エンベロープ
    pub envelope: f32,
}

impl Default for CompressorState {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorState {
    pub fn new() -> Self {
        // 初期エンベロープを十分低い値（-120dB）に設定
        // 0.0（0dB）だと閾値より高くなり、最初からゲインリダクションが適用されてしまう
        Self { envelope: -120.0 }
    }

    /// 1サンプル処理
    pub fn process(&mut self, input: f32, params: &CompressorParams) -> f32 {
        let CompressorParams {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            sample_rate,
        } = *params;
        // 入力レベルをdBに変換
        let input_abs = input.abs();
        let input_db = if input_abs > 1e-6 {
            20.0 * input_abs.log10()
        } else {
            -120.0
        };

        // アタック/リリース係数
        let attack_coeff = (-1.0 / (attack_ms * sample_rate / 1000.0)).exp();
        let release_coeff = (-1.0 / (release_ms * sample_rate / 1000.0)).exp();

        // エンベロープフォロワー
        if input_db > self.envelope {
            self.envelope = attack_coeff * self.envelope + (1.0 - attack_coeff) * input_db;
        } else {
            self.envelope = release_coeff * self.envelope + (1.0 - release_coeff) * input_db;
        }

        // ゲインリダクション計算
        let gain_reduction_db = if self.envelope > threshold_db {
            (threshold_db - self.envelope) * (1.0 - 1.0 / ratio)
        } else {
            0.0
        };

        // 出力ゲイン（線形）
        let output_gain = 10.0_f32.powf((gain_reduction_db + makeup_db) / 20.0);

        input * output_gain
    }

    /// リセット
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.envelope = -120.0;
    }
}

/// スペクトラムアナライザー
pub struct SpectrumAnalyzer {
    /// FFTプランナー
    planner: FftPlanner<f32>,
    /// 入力バッファ
    input_buffer: Vec<f32>,
    /// 書き込み位置
    write_pos: usize,
    /// FFT入力（複素数）
    fft_input: Vec<Complex<f32>>,
    /// FFT出力（複素数）
    fft_output: Vec<Complex<f32>>,
    /// 窓関数
    window: Vec<f32>,
    /// スムージング済みスペクトラム
    smoothed: Vec<f32>,
    /// スムージング係数（0.0-1.0、高いほど遅い追従）
    smoothing: f32,
}

impl SpectrumAnalyzer {
    pub fn new() -> Self {
        let planner = FftPlanner::new();
        let half_size = FFT_SIZE / 2;

        // Hann窓を生成
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                let t = i as f32 / FFT_SIZE as f32;
                0.5 * (1.0 - (2.0 * PI * t).cos())
            })
            .collect();

        Self {
            planner,
            input_buffer: vec![0.0; FFT_SIZE],
            write_pos: 0,
            fft_input: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            fft_output: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            window,
            smoothed: vec![0.0; half_size],
            smoothing: 0.8, // 80%の前回値 + 20%の新値
        }
    }

    /// サンプルを追加
    pub fn push_sample(&mut self, sample: f32) {
        self.input_buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % FFT_SIZE;
    }

    /// スペクトラムを計算してマグニチュード（スムージング済み）を返す
    pub fn compute_spectrum(&mut self) -> Vec<f32> {
        // 窓関数を適用してFFT入力を準備
        for i in 0..FFT_SIZE {
            let idx = (self.write_pos + i) % FFT_SIZE;
            self.fft_input[i] = Complex::new(self.input_buffer[idx] * self.window[i], 0.0);
        }

        // FFTを実行
        let fft = self.planner.plan_fft_forward(FFT_SIZE);
        self.fft_output.copy_from_slice(&self.fft_input);
        fft.process(&mut self.fft_output);

        // マグニチュードを計算してスムージング適用（ナイキスト周波数まで）
        let half_size = FFT_SIZE / 2;
        for i in 0..half_size {
            let raw_mag = self.fft_output[i].norm() / (FFT_SIZE as f32).sqrt();
            // 指数移動平均でスムージング
            self.smoothed[i] = self.smoothing * self.smoothed[i] + (1.0 - self.smoothing) * raw_mag;
        }

        self.smoothed.clone()
    }

    /// リセット
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.smoothed.fill(0.0);
        self.write_pos = 0;
    }
}

impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// ピッチシフトのバッファサイズ
pub const PITCH_BUFFER_SIZE: usize = 16384;
/// グレインサイズ（サンプル数）
const GRAIN_SIZE: usize = 1024;
/// オーバーラップ数
const NUM_GRAINS: usize = 4;

/// グラニュラー合成ベースのピッチシフター
pub struct PitchShifter {
    /// 入力リングバッファ
    input_buffer: Vec<f32>,
    /// 入力書き込み位置
    write_pos: usize,
    /// 各グレインの読み取り位置（浮動小数点）
    grain_read_pos: [f64; NUM_GRAINS],
    /// 各グレインの位相（0.0〜1.0）
    grain_phase: [f64; NUM_GRAINS],
    /// ピッチシフト量（1.0 = 変化なし、2.0 = 1オクターブ上）
    pitch_ratio: f32,
    /// サンプルカウンター
    sample_count: usize,
    /// 窓関数
    window: Vec<f32>,
}

impl PitchShifter {
    pub fn new(_sample_rate: f32) -> Self {
        // ハン窓を生成
        let window: Vec<f32> = (0..GRAIN_SIZE)
            .map(|i| {
                let t = i as f32 / GRAIN_SIZE as f32;
                0.5 * (1.0 - (2.0 * PI * t).cos())
            })
            .collect();

        // グレインを均等に配置（位相）
        let mut grain_phase = [0.0; NUM_GRAINS];
        for (i, phase) in grain_phase.iter_mut().enumerate() {
            *phase = i as f64 / NUM_GRAINS as f64;
        }

        // 初期書き込み位置
        let write_pos = PITCH_BUFFER_SIZE / 2;

        // グレインの読み取り位置を書き込み位置の手前に設定
        // 各グレインは位相に応じてオフセットされる
        let grain_spacing = GRAIN_SIZE / NUM_GRAINS;
        let mut grain_read_pos = [0.0; NUM_GRAINS];
        for (i, read_pos) in grain_read_pos.iter_mut().enumerate() {
            // 書き込み位置の手前からスタートし、各グレインをずらす
            let offset = GRAIN_SIZE + (i * grain_spacing);
            *read_pos = (write_pos as f64 - offset as f64).rem_euclid(PITCH_BUFFER_SIZE as f64);
        }

        Self {
            input_buffer: vec![0.0; PITCH_BUFFER_SIZE],
            write_pos,
            grain_read_pos,
            grain_phase,
            pitch_ratio: 1.0,
            sample_count: 0,
            window,
        }
    }

    /// ピッチシフト量を設定
    #[allow(dead_code)]
    pub fn set_pitch_ratio(&mut self, ratio: f32) {
        self.pitch_ratio = ratio.clamp(0.5, 2.0);
    }

    /// 半音単位でピッチを設定（-12 = 1オクターブ下、+12 = 1オクターブ上）
    pub fn set_semitones(&mut self, semitones: f32) {
        self.pitch_ratio = 2.0_f32.powf(semitones / 12.0);
    }

    /// 線形補間でサンプルを取得
    fn interpolate(&self, pos: f64) -> f32 {
        let pos_mod = pos.rem_euclid(PITCH_BUFFER_SIZE as f64);
        let idx0 = pos_mod.floor() as usize;
        let idx1 = (idx0 + 1) % PITCH_BUFFER_SIZE;
        let frac = (pos_mod - idx0 as f64) as f32;

        self.input_buffer[idx0] * (1.0 - frac) + self.input_buffer[idx1] * frac
    }

    /// サンプルを処理
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let grain_size_f = GRAIN_SIZE as f64;
        let phase_increment = 1.0 / grain_size_f;

        for (i, out_sample) in output.iter_mut().enumerate() {
            // 入力をバッファに書き込み
            self.input_buffer[self.write_pos] = input[i];
            self.write_pos = (self.write_pos + 1) % PITCH_BUFFER_SIZE;

            // 各グレインからの出力を合成
            let mut sum = 0.0f32;

            for grain_idx in 0..NUM_GRAINS {
                let phase = self.grain_phase[grain_idx];

                // 窓関数の値を取得
                let window_pos = (phase * grain_size_f) as usize;
                let window_val = if window_pos < GRAIN_SIZE {
                    self.window[window_pos]
                } else {
                    0.0
                };

                // 補間してサンプルを取得
                let sample = self.interpolate(self.grain_read_pos[grain_idx]);
                sum += sample * window_val;

                // 読み取り位置を進める（ピッチ比率に応じて）
                self.grain_read_pos[grain_idx] += self.pitch_ratio as f64;
                if self.grain_read_pos[grain_idx] >= PITCH_BUFFER_SIZE as f64 {
                    self.grain_read_pos[grain_idx] -= PITCH_BUFFER_SIZE as f64;
                }

                // 位相を進める
                self.grain_phase[grain_idx] += phase_increment;

                // グレインが終了したら次の位置にリセット
                if self.grain_phase[grain_idx] >= 1.0 {
                    self.grain_phase[grain_idx] -= 1.0;
                    // 新しいグレインの開始位置を現在の書き込み位置の少し手前に設定
                    let offset = (GRAIN_SIZE * NUM_GRAINS / 2) as f64;
                    self.grain_read_pos[grain_idx] =
                        (self.write_pos as f64 - offset).rem_euclid(PITCH_BUFFER_SIZE as f64);
                }
            }

            // 正規化（グレイン数で割る）
            *out_sample = sum / (NUM_GRAINS as f32 / 2.0);
        }

        self.sample_count += input.len();
    }

    /// リセット
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.write_pos = PITCH_BUFFER_SIZE / 2;
        self.sample_count = 0;

        // グレインを均等に再配置
        for (i, phase) in self.grain_phase.iter_mut().enumerate() {
            *phase = i as f64 / NUM_GRAINS as f64;
        }

        // グレインの読み取り位置を書き込み位置の手前に設定
        let grain_spacing = GRAIN_SIZE / NUM_GRAINS;
        for (i, read_pos) in self.grain_read_pos.iter_mut().enumerate() {
            let offset = GRAIN_SIZE + (i * grain_spacing);
            *read_pos =
                (self.write_pos as f64 - offset as f64).rem_euclid(PITCH_BUFFER_SIZE as f64);
        }
    }
}

impl Default for PitchShifter {
    fn default() -> Self {
        Self::new(44100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass() {
        let coeffs = BiquadCoeffs::lowpass(44100.0, 1000.0, 0.707);
        let mut state = BiquadState::new();

        // インパルス応答をテスト
        let output = state.process(1.0, &coeffs);
        assert!(output > 0.0);

        // 後続サンプル
        for _ in 0..100 {
            let out = state.process(0.0, &coeffs);
            assert!(out.abs() < 1.0); // 減衰していくはず
        }
    }

    #[test]
    fn test_compressor() {
        let mut state = CompressorState::new();
        let params = CompressorParams {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_db: 0.0,
            sample_rate: 44100.0,
        };

        // 閾値以下の信号のテスト：初期状態ではエンベロープが低いのでゲインリダクションなし
        // まず小さい信号を複数回処理してエンベロープを安定させる
        for _ in 0..100 {
            state.process(0.01, &params); // -40dB、閾値より十分小さい
        }
        let output = state.process(0.01, &params);
        // 閾値以下では圧縮されない（makeup_gain=0なので入力≒出力）
        assert!(
            (output - 0.01).abs() < 0.005,
            "Expected ~0.01, got {}",
            output
        );

        // 閾値以上の信号（十分な時間エンベロープを追従させる）
        // 10msアタック @ 44100Hz = 441サンプルで63%到達
        // エンベロープが-120dBから0dBまで上昇するには約5-6時定数必要
        let mut state2 = CompressorState::new();
        for _ in 0..5000 {
            state2.process(1.0, &params);
        }
        let output = state2.process(1.0, &params);
        // エンベロープが0dB近くになり、閾値(-20dB)を超えているので圧縮される
        assert!(
            output < 0.95, // 4:1レシオで約6dBのゲインリダクションを期待
            "Expected compression, got {}",
            output
        );
    }

    #[test]
    fn test_spectrum_analyzer() {
        let mut analyzer = SpectrumAnalyzer::new();

        // サイン波を入力
        for i in 0..FFT_SIZE {
            let sample = (2.0 * PI * 440.0 * i as f32 / 44100.0).sin();
            analyzer.push_sample(sample);
        }

        let spectrum = analyzer.compute_spectrum();
        assert_eq!(spectrum.len(), FFT_SIZE / 2);

        // 440Hzにピークがあるはず
        let peak_idx = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // 440Hz / (44100Hz / FFT_SIZE) ≈ 10
        let expected_bin = (440.0 * FFT_SIZE as f32 / 44100.0) as usize;
        assert!((peak_idx as i32 - expected_bin as i32).abs() <= 2);
    }
}
