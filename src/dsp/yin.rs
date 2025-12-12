//! YINピッチ検出アルゴリズム
//!
//! De Cheveigné & Kawahara (2002) "YIN, a fundamental frequency estimator for speech and music"
//! に基づく実装

// 将来の拡張用に提供されているAPIについて警告を抑制
#![allow(dead_code)]

/// YINピッチ検出の結果
#[derive(Clone, Copy, Debug)]
pub struct PitchResult {
    /// 検出された周期（サンプル数）
    pub period: f32,
    /// 信頼度（0.0〜1.0、高いほど信頼性が高い）
    pub clarity: f32,
    /// 有声音として検出されたかどうか
    pub voiced: bool,
}

impl PitchResult {
    /// 周波数を取得（Hz）
    pub fn frequency(&self, sample_rate: f32) -> f32 {
        if self.period > 0.0 {
            sample_rate / self.period
        } else {
            0.0
        }
    }
}

/// YINピッチ検出器
pub struct YinPitchDetector {
    /// サンプルレート
    sample_rate: f32,
    /// 最小周期（サンプル数、対応する最高周波数）
    min_period: usize,
    /// 最大周期（サンプル数、対応する最低周波数）
    max_period: usize,
    /// YIN閾値（0.0〜1.0、低いほど厳格）
    threshold: f32,
    /// 差分関数バッファ
    diff_buffer: Vec<f32>,
    /// 累積平均正規化差分関数バッファ
    cmndf_buffer: Vec<f32>,
    /// 前回検出したピッチ結果（無声音時のフォールバック用）
    last_result: Option<PitchResult>,
}

impl YinPitchDetector {
    /// 新しいYINピッチ検出器を作成
    ///
    /// # 引数
    /// - `sample_rate`: サンプルレート (Hz)
    /// - `min_freq`: 検出する最低周波数 (Hz) - 例: 50Hz (男性の低い声)
    /// - `max_freq`: 検出する最高周波数 (Hz) - 例: 1000Hz (高いピッチ)
    pub fn new(sample_rate: f32, min_freq: f32, max_freq: f32) -> Self {
        // 周波数を周期（サンプル数）に変換
        let min_period = (sample_rate / max_freq).ceil() as usize;
        let max_period = (sample_rate / min_freq).ceil() as usize;

        Self {
            sample_rate,
            min_period: min_period.max(2),
            max_period,
            threshold: 0.15, // デフォルト閾値
            diff_buffer: vec![0.0; max_period + 1],
            cmndf_buffer: vec![0.0; max_period + 1],
            last_result: None,
        }
    }

    /// 閾値を設定
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.01, 0.5);
    }

    /// 閾値を取得
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// 最小必要サンプル数を取得
    /// YINでは分析窓の2倍のサンプルが必要
    pub fn required_samples(&self) -> usize {
        self.max_period * 2
    }

    /// ピッチを検出
    ///
    /// # 引数
    /// - `input`: 入力サンプル（最低でも `required_samples()` 以上必要）
    ///
    /// # 戻り値
    /// - `Some(PitchResult)`: 検出結果（無声音の場合でも前回の値を返す）
    /// - `None`: 十分なサンプルがない場合
    pub fn detect(&mut self, input: &[f32]) -> Option<PitchResult> {
        let required = self.required_samples();
        if input.len() < required {
            return self.last_result;
        }

        // 分析窓サイズ
        let window_size = self.max_period;

        // Step 1: 差分関数を計算
        self.compute_difference(input, window_size);

        // Step 2: 累積平均正規化差分関数を計算
        self.compute_cmndf();

        // Step 3: 閾値以下の最初の谷を探索
        let period_idx = self.find_first_valley();

        // Step 4: 結果を構築
        let result = if let Some(idx) = period_idx {
            // 放物線補間でサブサンプル精度の周期を取得
            let period = self.parabolic_interpolation(idx);
            let clarity = 1.0 - self.cmndf_buffer[idx];

            PitchResult {
                period,
                clarity: clarity.clamp(0.0, 1.0),
                voiced: true,
            }
        } else {
            // 無声音：前回の周期を継続（ただしvoicedはfalse）
            if let Some(last) = self.last_result {
                PitchResult {
                    period: last.period,
                    clarity: 0.0,
                    voiced: false,
                }
            } else {
                // 初めての検出で無声音の場合はデフォルト周期
                let default_period = (self.min_period + self.max_period) as f32 / 2.0;
                PitchResult {
                    period: default_period,
                    clarity: 0.0,
                    voiced: false,
                }
            }
        };

        self.last_result = Some(result);
        Some(result)
    }

    /// 差分関数を計算
    /// d(τ) = Σ (x[j] - x[j+τ])²
    fn compute_difference(&mut self, input: &[f32], window_size: usize) {
        self.diff_buffer[0] = 0.0;

        for tau in 1..=self.max_period {
            let mut sum = 0.0;
            for j in 0..window_size {
                let diff = input[j] - input[j + tau];
                sum += diff * diff;
            }
            self.diff_buffer[tau] = sum;
        }
    }

    /// 累積平均正規化差分関数を計算
    /// d'(0) = 1
    /// d'(τ) = d(τ) / ((1/τ) * Σ d(j))  for τ > 0
    fn compute_cmndf(&mut self) {
        self.cmndf_buffer[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..=self.max_period {
            running_sum += self.diff_buffer[tau];

            if running_sum > 0.0 {
                self.cmndf_buffer[tau] = self.diff_buffer[tau] * tau as f32 / running_sum;
            } else {
                self.cmndf_buffer[tau] = 1.0;
            }
        }
    }

    /// 閾値以下の最初の谷を探索
    fn find_first_valley(&self) -> Option<usize> {
        // 最小周期から探索開始
        let mut tau = self.min_period;

        // 閾値以下になる位置を探す
        while tau < self.max_period {
            if self.cmndf_buffer[tau] < self.threshold {
                // 谷（極小値）を探す
                while tau + 1 < self.max_period
                    && self.cmndf_buffer[tau + 1] < self.cmndf_buffer[tau]
                {
                    tau += 1;
                }
                return Some(tau);
            }
            tau += 1;
        }

        // 閾値以下が見つからない場合、全体の最小値を探す
        // ただし、その値が閾値の2倍以下であれば採用
        let mut min_tau = self.min_period;
        let mut min_val = self.cmndf_buffer[self.min_period];

        for tau in self.min_period..=self.max_period {
            if self.cmndf_buffer[tau] < min_val {
                min_val = self.cmndf_buffer[tau];
                min_tau = tau;
            }
        }

        if min_val < self.threshold * 2.0 {
            Some(min_tau)
        } else {
            None
        }
    }

    /// 放物線補間でサブサンプル精度の周期を取得
    fn parabolic_interpolation(&self, index: usize) -> f32 {
        if index <= self.min_period || index >= self.max_period {
            return index as f32;
        }

        let y0 = self.cmndf_buffer[index - 1];
        let y1 = self.cmndf_buffer[index];
        let y2 = self.cmndf_buffer[index + 1];

        // 放物線の頂点を計算
        let denominator = 2.0 * (2.0 * y1 - y0 - y2);
        if denominator.abs() < 1e-10 {
            return index as f32;
        }

        let delta = (y0 - y2) / denominator;
        index as f32 + delta
    }

    /// 現在の周波数を取得（Hz）
    pub fn current_frequency(&self) -> f32 {
        self.last_result
            .map(|r| r.frequency(self.sample_rate))
            .unwrap_or(0.0)
    }

    /// 前回の検出結果を取得
    pub fn last_result(&self) -> Option<PitchResult> {
        self.last_result
    }
}

impl Default for YinPitchDetector {
    fn default() -> Self {
        // 48kHzサンプルレート、50Hz〜1000Hzの範囲
        Self::new(48000.0, 50.0, 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// サイン波を生成
    fn generate_sine_wave(freq: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_yin_440hz() {
        let sample_rate = 48000.0;
        let freq = 440.0; // A4

        let mut detector = YinPitchDetector::new(sample_rate, 50.0, 1000.0);
        let samples = generate_sine_wave(freq, sample_rate, detector.required_samples());

        let result = detector.detect(&samples).unwrap();

        let detected_freq = result.frequency(sample_rate);
        let error_percent = ((detected_freq - freq) / freq).abs() * 100.0;

        assert!(result.voiced, "Should be detected as voiced");
        assert!(
            error_percent < 1.0,
            "Frequency error should be < 1%: detected {}Hz (expected {}Hz, error {:.2}%)",
            detected_freq,
            freq,
            error_percent
        );
        assert!(result.clarity > 0.8, "Clarity should be high for pure sine");
    }

    #[test]
    fn test_yin_low_frequency() {
        let sample_rate = 48000.0;
        let freq = 100.0; // 低い周波数

        let mut detector = YinPitchDetector::new(sample_rate, 50.0, 1000.0);
        let samples = generate_sine_wave(freq, sample_rate, detector.required_samples());

        let result = detector.detect(&samples).unwrap();
        let detected_freq = result.frequency(sample_rate);
        let error_percent = ((detected_freq - freq) / freq).abs() * 100.0;

        assert!(result.voiced);
        assert!(
            error_percent < 2.0,
            "Low frequency detection: {}Hz (expected {}Hz)",
            detected_freq,
            freq
        );
    }

    #[test]
    fn test_yin_high_frequency() {
        let sample_rate = 48000.0;
        let freq = 800.0; // 高い周波数

        let mut detector = YinPitchDetector::new(sample_rate, 50.0, 1000.0);
        let samples = generate_sine_wave(freq, sample_rate, detector.required_samples());

        let result = detector.detect(&samples).unwrap();
        let detected_freq = result.frequency(sample_rate);
        let error_percent = ((detected_freq - freq) / freq).abs() * 100.0;

        assert!(result.voiced);
        assert!(
            error_percent < 1.0,
            "High frequency detection: {}Hz (expected {}Hz)",
            detected_freq,
            freq
        );
    }

    #[test]
    fn test_yin_unvoiced_noise() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let sample_rate = 48000.0;
        let mut detector = YinPitchDetector::new(sample_rate, 50.0, 1000.0);

        // 擬似ランダムノイズ生成（再現可能）
        let noise: Vec<f32> = (0..detector.required_samples())
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                let hash = hasher.finish();
                (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let result = detector.detect(&noise).unwrap();

        // ノイズは無声音として検出されるべき
        assert!(
            !result.voiced || result.clarity < 0.5,
            "Noise should be unvoiced or low clarity"
        );
    }

    #[test]
    fn test_yin_pitch_continuity() {
        let sample_rate = 48000.0;
        let mut detector = YinPitchDetector::new(sample_rate, 50.0, 1000.0);

        // まず有声音を検出
        let samples = generate_sine_wave(440.0, sample_rate, detector.required_samples());
        let result1 = detector.detect(&samples).unwrap();
        assert!(result1.voiced);

        // その後無声音（ノイズ）を検出
        let noise: Vec<f32> = (0..detector.required_samples())
            .map(|i| {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                (i + 12345).hash(&mut hasher);
                let hash = hasher.finish();
                (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let result2 = detector.detect(&noise).unwrap();

        // 無声音でも前回の周期が維持される
        assert!(
            (result2.period - result1.period).abs() < 1.0,
            "Period should be maintained for unvoiced segments"
        );
    }

    #[test]
    fn test_yin_required_samples() {
        let detector = YinPitchDetector::new(48000.0, 50.0, 1000.0);

        // 48000Hz / 50Hz = 960サンプル = max_period
        // 必要サンプル数 = max_period * 2 = 1920
        assert!(detector.required_samples() >= 1920);
    }

    #[test]
    fn test_yin_threshold() {
        let mut detector = YinPitchDetector::new(48000.0, 50.0, 1000.0);

        detector.set_threshold(0.1);
        assert!((detector.threshold() - 0.1).abs() < 1e-6);

        // クランプされる
        detector.set_threshold(0.0);
        assert!(detector.threshold() >= 0.01);

        detector.set_threshold(1.0);
        assert!(detector.threshold() <= 0.5);
    }
}
