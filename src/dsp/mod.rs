//! DSP処理モジュール
//!
//! 各ノードのオーディオ処理アルゴリズムを実装

use std::f32::consts::PI;

mod biquad;
mod compressor;
mod graphic_eq;
pub mod interpolation;
mod pitch_shifter;
mod spectrum;
mod yin;

// Re-exports
pub use biquad::{BiquadCoeffs, BiquadState};
pub use compressor::{CompressorParams, CompressorState};
pub use graphic_eq::{EqPoint, GraphicEq};
pub use pitch_shifter::{
    PhaseAlignmentParams, PitchShifter, DEFAULT_GRAIN_SIZE, DEFAULT_NUM_GRAINS,
    DEFAULT_PITCH_BUFFER_SIZE,
};
pub use spectrum::SpectrumAnalyzer;

/// Hann窓を生成する共通関数
pub fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let t = i as f32 / size as f32;
            0.5 * (1.0 - (2.0 * PI * t).cos())
        })
        .collect()
}
