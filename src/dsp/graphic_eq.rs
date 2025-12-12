//! FFTベースのグラフィックイコライザー

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::create_hann_window;

/// EQカーブ上のコントロールポイント
#[derive(Clone, Copy, Debug)]
pub struct EqPoint {
    /// 周波数 (Hz, 対数スケール用)
    pub freq: f32,
    /// ゲイン (dB)
    pub gain_db: f32,
}

impl EqPoint {
    pub fn new(freq: f32, gain_db: f32) -> Self {
        Self { freq, gain_db }
    }
}

/// グラフィックEQのFFTサイズ
pub const EQ_FFT_SIZE: usize = 2048;
/// ホップサイズ（75%オーバーラップ = FFT_SIZE / 4）
const EQ_HOP_SIZE: usize = EQ_FFT_SIZE / 4;

/// FFTベースのグラフィックイコライザー
/// シンプルなオーバーラップ加算方式を採用
pub struct GraphicEq {
    /// FFTプランナー
    fft: Arc<dyn rustfft::Fft<f32>>,
    /// IFFTプランナー
    ifft: Arc<dyn rustfft::Fft<f32>>,
    /// 入力FIFO（FFT_SIZE分保持）
    input_fifo: Vec<f32>,
    /// オーバーラップ加算バッファ（FFT_SIZE分）
    overlap_buffer: Vec<f32>,
    /// 出力FIFO（HOP_SIZE分保持）
    output_fifo: Vec<f32>,
    /// 出力FIFO読み取り位置
    output_fifo_pos: usize,
    /// FFT作業用バッファ
    fft_buffer: Vec<Complex<f32>>,
    /// 窓関数
    window: Vec<f32>,
    /// 周波数ごとのゲイン（線形スケール）
    freq_gains: Vec<f32>,
    /// サンプルレート
    sample_rate: f32,
    /// 入力スペクトラム（マグニチュード）
    input_spectrum: Vec<f32>,
    /// 初期化済みフラグ（最初のブロックを処理したか）
    initialized: bool,
}

impl GraphicEq {
    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(EQ_FFT_SIZE);
        let ifft = planner.plan_fft_inverse(EQ_FFT_SIZE);

        let window = create_hann_window(EQ_FFT_SIZE);

        // デフォルトは全帯域フラット（ゲイン1.0）
        let freq_gains = vec![1.0; EQ_FFT_SIZE / 2 + 1];

        Self {
            fft,
            ifft,
            input_fifo: vec![0.0; EQ_FFT_SIZE],
            overlap_buffer: vec![0.0; EQ_FFT_SIZE],
            output_fifo: vec![0.0; EQ_HOP_SIZE],
            output_fifo_pos: 0,
            fft_buffer: vec![Complex::new(0.0, 0.0); EQ_FFT_SIZE],
            window,
            freq_gains,
            sample_rate,
            input_spectrum: vec![0.0; EQ_FFT_SIZE / 2],
            initialized: false,
        }
    }

    /// EQカーブからゲインテーブルを更新
    pub fn update_curve(&mut self, points: &[EqPoint]) {
        if points.is_empty() {
            self.freq_gains.fill(1.0);
            return;
        }

        let nyquist = self.sample_rate / 2.0;
        let bin_count = EQ_FFT_SIZE / 2 + 1;

        for bin in 0..bin_count {
            let freq = (bin as f32 / bin_count as f32) * nyquist;
            let gain_db = Self::interpolate_gain(points, freq);
            // dBから線形ゲインに変換
            self.freq_gains[bin] = 10.0_f32.powf(gain_db / 20.0);
        }
    }

    /// ポイント間をCatmull-Romスプライン補間してゲインを取得
    fn interpolate_gain(points: &[EqPoint], freq: f32) -> f32 {
        if points.is_empty() {
            return 0.0;
        }

        if points.len() == 1 {
            return points[0].gain_db;
        }

        // 周波数でソートされていると仮定
        if freq <= points[0].freq {
            return points[0].gain_db;
        }
        if freq >= points[points.len() - 1].freq {
            return points[points.len() - 1].gain_db;
        }

        // 対数周波数スケールでの位置を計算
        let log_freq = freq.ln();

        // 補間区間を探す
        for i in 0..points.len() - 1 {
            if freq >= points[i].freq && freq <= points[i + 1].freq {
                let log_f0 = points[i].freq.ln();
                let log_f1 = points[i + 1].freq.ln();
                let t = (log_freq - log_f0) / (log_f1 - log_f0);

                // Catmull-Romスプライン用の4点を取得
                // 端点では隣接点を複製して使用
                let p0 = if i > 0 {
                    points[i - 1].gain_db
                } else {
                    // 端点: 傾きを維持するように外挿
                    2.0 * points[i].gain_db - points[i + 1].gain_db
                };
                let p1 = points[i].gain_db;
                let p2 = points[i + 1].gain_db;
                let p3 = if i + 2 < points.len() {
                    points[i + 2].gain_db
                } else {
                    // 端点: 傾きを維持するように外挿
                    2.0 * points[i + 1].gain_db - points[i].gain_db
                };

                return Self::catmull_rom(p0, p1, p2, p3, t);
            }
        }

        0.0
    }

    /// Catmull-Romスプライン補間
    /// p0, p1, p2, p3: 4つの制御点
    /// t: p1とp2の間のパラメータ (0.0 ~ 1.0)
    fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom係数 (tau = 0.5)
        // p(t) = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t² + (-p0 + 3*p1 - 3*p2 + p3)*t³)
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    /// オーディオを処理
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            // 出力FIFOから読み取り
            if self.initialized {
                output[i] = self.output_fifo[self.output_fifo_pos];
            } else {
                output[i] = 0.0;
            }

            // 入力FIFOをシフトして新しいサンプルを追加
            self.input_fifo.copy_within(1.., 0);
            self.input_fifo[EQ_FFT_SIZE - 1] = sample;

            self.output_fifo_pos += 1;

            // HOP_SIZE分溜まったらFFTブロック処理
            if self.output_fifo_pos >= EQ_HOP_SIZE {
                self.process_fft_block();
                self.output_fifo_pos = 0;
                self.initialized = true;
            }
        }
    }

    /// FFTブロック処理
    fn process_fft_block(&mut self) {
        // 入力を窓関数付きでFFTバッファにコピー
        for i in 0..EQ_FFT_SIZE {
            self.fft_buffer[i] = Complex::new(self.input_fifo[i] * self.window[i], 0.0);
        }

        // FFT
        self.fft.process(&mut self.fft_buffer);

        // 入力スペクトラムを保存（ゲイン適用前）
        for i in 0..EQ_FFT_SIZE / 2 {
            self.input_spectrum[i] = self.fft_buffer[i].norm();
        }

        // 周波数領域でゲイン適用
        let bin_count = EQ_FFT_SIZE / 2 + 1;
        for bin in 0..bin_count {
            let gain = self.freq_gains[bin];
            self.fft_buffer[bin] *= gain;
            // 対称成分（ナイキスト以下）
            if bin > 0 && bin < EQ_FFT_SIZE / 2 {
                self.fft_buffer[EQ_FFT_SIZE - bin] *= gain;
            }
        }

        // IFFT
        self.ifft.process(&mut self.fft_buffer);

        // IFFT正規化(1/N) × 窓補正(2/3 for 75% overlap Hann²)
        let norm = 2.0 / (3.0 * EQ_FFT_SIZE as f32);

        // オーバーラップ加算バッファに窓関数を適用して加算
        for i in 0..EQ_FFT_SIZE {
            self.overlap_buffer[i] += self.fft_buffer[i].re * norm * self.window[i];
        }

        // オーバーラップバッファの先頭HOP_SIZE分を出力FIFOにコピー
        self.output_fifo
            .copy_from_slice(&self.overlap_buffer[..EQ_HOP_SIZE]);

        // オーバーラップバッファをHOP_SIZE分シフト
        self.overlap_buffer.copy_within(EQ_HOP_SIZE.., 0);
        // シフトで空いた末尾をゼロクリア
        self.overlap_buffer[EQ_FFT_SIZE - EQ_HOP_SIZE..].fill(0.0);
    }

    /// 入力スペクトラムを取得
    pub fn get_input_spectrum(&self) -> &[f32] {
        &self.input_spectrum
    }
}

impl Default for GraphicEq {
    fn default() -> Self {
        Self::new(44100.0)
    }
}
