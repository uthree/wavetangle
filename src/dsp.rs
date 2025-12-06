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
        Self { envelope: 0.0 }
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
        self.envelope = 0.0;
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
}

impl SpectrumAnalyzer {
    pub fn new() -> Self {
        let planner = FftPlanner::new();

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
        }
    }

    /// サンプルを追加
    pub fn push_sample(&mut self, sample: f32) {
        self.input_buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % FFT_SIZE;
    }

    /// スペクトラムを計算してマグニチュードを返す
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

        // マグニチュードを計算（ナイキスト周波数まで）
        let half_size = FFT_SIZE / 2;
        let mut magnitudes = Vec::with_capacity(half_size);
        for i in 0..half_size {
            let mag = self.fft_output[i].norm() / (FFT_SIZE as f32).sqrt();
            magnitudes.push(mag);
        }

        magnitudes
    }

    /// リセット
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.write_pos = 0;
    }
}

impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
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

        // 閾値以下の信号
        let output = state.process(0.1, -20.0, 4.0, 10.0, 100.0, 0.0, 44100.0);
        assert!((output - 0.1).abs() < 0.01); // ほぼ変化なし

        // 閾値以上の信号
        let output = state.process(1.0, -20.0, 4.0, 10.0, 100.0, 0.0, 44100.0);
        assert!(output < 1.0); // 圧縮される
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
