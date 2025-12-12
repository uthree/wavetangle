//! Biquadフィルター

use std::f32::consts::PI;

use crate::nodes::FilterType;

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
}
