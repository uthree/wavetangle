//! 補間アルゴリズム
//!
//! バッファからサンプルを補間して取得するためのtrait実装

// 将来の拡張用に提供されているAPIについて警告を抑制
#![allow(dead_code)]

/// 補間アルゴリズムを抽象化するtrait
pub trait Interpolator: Send + Sync {
    /// バッファから指定位置のサンプルを補間して取得
    ///
    /// - buffer: サンプルデータ（リングバッファ）
    /// - position: 浮動小数点位置（0.0からbuffer.len()の範囲）
    ///
    /// 戻り値: 補間されたサンプル値
    fn interpolate(&self, buffer: &[f32], position: f64) -> f32;

    /// 補間に必要な前後のサンプル数（マージン）
    /// リングバッファの端で正しく補間するために必要
    fn required_margin(&self) -> usize;

    /// 補間器の名前（デバッグ用）
    fn name(&self) -> &'static str;
}

/// 線形補間
/// 2点間を直線で結ぶ最もシンプルな補間方式
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearInterpolator;

impl LinearInterpolator {
    pub fn new() -> Self {
        Self
    }
}

impl Interpolator for LinearInterpolator {
    fn interpolate(&self, buffer: &[f32], position: f64) -> f32 {
        if buffer.is_empty() {
            return 0.0;
        }

        let len = buffer.len();
        let pos_mod = position.rem_euclid(len as f64);
        let idx0 = pos_mod.floor() as usize;
        let idx1 = (idx0 + 1) % len;
        let frac = (pos_mod - idx0 as f64) as f32;

        buffer[idx0] * (1.0 - frac) + buffer[idx1] * frac
    }

    fn required_margin(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "Linear"
    }
}

/// 3次（キュービック）補間
/// Catmull-Romスプラインを使用した滑らかな補間
#[derive(Clone, Copy, Debug, Default)]
pub struct CubicInterpolator;

impl CubicInterpolator {
    pub fn new() -> Self {
        Self
    }
}

impl Interpolator for CubicInterpolator {
    fn interpolate(&self, buffer: &[f32], position: f64) -> f32 {
        if buffer.is_empty() {
            return 0.0;
        }

        let len = buffer.len();
        let pos_mod = position.rem_euclid(len as f64);
        let idx1 = pos_mod.floor() as usize;

        // Catmull-Romには4点必要
        let idx0 = if idx1 == 0 { len - 1 } else { idx1 - 1 };
        let idx2 = (idx1 + 1) % len;
        let idx3 = (idx1 + 2) % len;

        let p0 = buffer[idx0];
        let p1 = buffer[idx1];
        let p2 = buffer[idx2];
        let p3 = buffer[idx3];

        let t = (pos_mod - idx1 as f64) as f32;
        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom係数 (tension = 0.5)
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    fn required_margin(&self) -> usize {
        2 // 前後2サンプル必要
    }

    fn name(&self) -> &'static str {
        "Cubic"
    }
}

/// デフォルトの補間器を作成
pub fn default_interpolator() -> Box<dyn Interpolator> {
    Box::new(LinearInterpolator::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation_exact() {
        let interp = LinearInterpolator::new();
        let buffer = vec![0.0, 1.0, 2.0, 3.0];

        // 整数位置では正確な値
        assert!((interp.interpolate(&buffer, 0.0) - 0.0).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 1.0) - 1.0).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 2.0) - 2.0).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 3.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_interpolation_between() {
        let interp = LinearInterpolator::new();
        let buffer = vec![0.0, 1.0, 2.0, 3.0];

        // 中間位置では線形補間
        assert!((interp.interpolate(&buffer, 0.5) - 0.5).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 1.5) - 1.5).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 2.5) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_linear_interpolation_wrap() {
        let interp = LinearInterpolator::new();
        let buffer = vec![0.0, 1.0, 2.0, 3.0];

        // ラップアラウンド
        assert!((interp.interpolate(&buffer, 3.5) - 1.5).abs() < 1e-6); // 3.0と0.0の中間
    }

    #[test]
    fn test_cubic_interpolation_exact() {
        let interp = CubicInterpolator::new();
        let buffer = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        // 整数位置では正確な値
        assert!((interp.interpolate(&buffer, 1.0) - 1.0).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 2.0) - 2.0).abs() < 1e-6);
        assert!((interp.interpolate(&buffer, 3.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_interpolation_smooth() {
        let interp = CubicInterpolator::new();
        // サイン波をサンプリング
        let buffer: Vec<f32> = (0..16)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 16.0).sin())
            .collect();

        // 中間点での補間が滑らか
        let val = interp.interpolate(&buffer, 0.5);
        let expected = (2.0 * std::f32::consts::PI * 0.5 / 16.0).sin();
        assert!((val - expected).abs() < 0.1); // 3次補間なのである程度の誤差は許容
    }

    #[test]
    fn test_interpolator_margin() {
        assert_eq!(LinearInterpolator::new().required_margin(), 1);
        assert_eq!(CubicInterpolator::new().required_margin(), 2);
    }

    #[test]
    fn test_interpolator_name() {
        assert_eq!(LinearInterpolator::new().name(), "Linear");
        assert_eq!(CubicInterpolator::new().name(), "Cubic");
    }
}
