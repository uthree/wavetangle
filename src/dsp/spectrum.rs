//! スペクトラムアナライザー

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::create_hann_window;
use crate::nodes::FFT_SIZE;

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

        let window = create_hann_window(FFT_SIZE);

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

    /// 複数サンプルを追加
    pub fn push_samples(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.push_sample(sample);
        }
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
}

impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

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
