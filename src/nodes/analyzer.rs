use std::sync::Arc;

use egui::Ui;
use egui_plot::{Bar, BarChart, Plot};
use parking_lot::Mutex;

use super::{
    hsv_to_rgb, impl_as_any, impl_input_port_nb, impl_single_output_port_nb, AudioInputPort,
    AudioOutputPort, ChannelBuffer, NodeBase, NodeBuffers, NodeType, NodeUI, NodeUIContext,
    PinType, FFT_SIZE,
};

// ============================================================================
// Spectrum Analyzer Node
// ============================================================================

/// スペクトラムアナライザーノード
pub struct SpectrumAnalyzerNode {
    /// スペクトラムデータ（マグニチュード）
    pub spectrum: Arc<Mutex<Vec<f32>>>,
    /// バッファ管理（1入力1出力）
    pub buffers: NodeBuffers,
    pub is_active: bool,
    /// FFTアナライザー（スレッドセーフ）
    pub analyzer: Arc<Mutex<crate::dsp::SpectrumAnalyzer>>,
}

impl Clone for SpectrumAnalyzerNode {
    fn clone(&self) -> Self {
        Self {
            spectrum: self.spectrum.clone(),
            buffers: self.buffers.clone(),
            is_active: self.is_active,
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }
}

impl SpectrumAnalyzerNode {
    pub fn new() -> Self {
        Self {
            spectrum: Arc::new(Mutex::new(vec![0.0; FFT_SIZE / 2])),
            buffers: NodeBuffers::single_io(),
            is_active: false,
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }
}

impl Default for SpectrumAnalyzerNode {
    fn default() -> Self {
        Self::new()
    }
}

// SpectrumAnalyzerNodeのトレイト実装（1入力1出力、NodeBuffers対応マクロを使用）
impl NodeBase for SpectrumAnalyzerNode {
    fn node_type(&self) -> NodeType {
        NodeType::SpectrumAnalyzer
    }

    fn title(&self) -> &str {
        "Spectrum"
    }

    impl_as_any!();
}

impl_input_port_nb!(SpectrumAnalyzerNode, ["In"]);
impl_single_output_port_nb!(SpectrumAnalyzerNode);

impl NodeUI for SpectrumAnalyzerNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            // スペクトラムデータを取得
            let spectrum = self.spectrum.lock();
            let bar_count = 48; // 表示するバーの数

            // バーデータを作成
            let bars: Vec<Bar> = (0..bar_count)
                .map(|i| {
                    // 対数スケールでインデックスをマッピング（低周波を細かく表示）
                    let freq_ratio = i as f32 / bar_count as f32;
                    let freq_idx = (freq_ratio.powf(2.0) * (FFT_SIZE / 2) as f32) as usize;
                    let freq_idx = freq_idx.min(spectrum.len().saturating_sub(1));

                    // マグニチュードを正規化（対数スケール、dB変換）
                    let magnitude = if freq_idx < spectrum.len() {
                        spectrum[freq_idx]
                    } else {
                        0.0
                    };
                    let db = if magnitude > 1e-6 {
                        20.0 * magnitude.log10()
                    } else {
                        -80.0
                    };
                    // -80dB〜0dBを0.0〜1.0にマッピング
                    let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0) as f64;

                    // 周波数に応じたグラデーションカラー（低周波=緑、高周波=シアン）
                    let hue = 120.0 + freq_ratio * 60.0; // 緑(120)→シアン(180)
                    let sat = 0.7 + normalized as f32 * 0.3;
                    let val = 0.3 + normalized as f32 * 0.7;
                    let color = hsv_to_rgb(hue, sat, val);

                    Bar::new(i as f64, normalized).fill(color).width(0.85)
                })
                .collect();

            drop(spectrum); // ロックを解放

            // egui_plotでバーチャートを表示
            Plot::new(format!("spectrum_{:?}", ctx.node_id))
                .height(100.0)
                .width(220.0)
                .show_axes([false, false])
                .show_grid([false, false])
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .include_y(0.0)
                .include_y(1.0)
                .show_background(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new("spectrum", bars));
                });
        });
    }
}
