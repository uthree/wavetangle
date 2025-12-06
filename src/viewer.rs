use egui::{Color32, Ui};
use egui_plot::{Bar, BarChart, Plot};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::nodes::{AudioNode, FilterType, NodeBehavior, PinType, FFT_SIZE};

/// ピンタイプに応じた色を取得
fn pin_color(pin_type: PinType) -> Color32 {
    match pin_type {
        PinType::Audio => Color32::from_rgb(100, 200, 100),
    }
}

/// HSVからRGBに変換
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color32 {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// オーディオグラフビューアー
pub struct AudioGraphViewer {
    pub input_devices: Vec<String>,
    pub output_devices: Vec<String>,
}

impl AudioGraphViewer {
    /// キャッシュされたデバイスリストから作成
    pub fn with_devices(input_devices: Vec<String>, output_devices: Vec<String>) -> Self {
        Self {
            input_devices,
            output_devices,
        }
    }
}

impl SnarlViewer<AudioNode> for AudioGraphViewer {
    fn title(&mut self, node: &AudioNode) -> String {
        node.title().to_string()
    }

    fn inputs(&mut self, node: &AudioNode) -> usize {
        node.input_count()
    }

    fn outputs(&mut self, node: &AudioNode) -> usize {
        node.output_count()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];

        // ピン名を表示
        if let Some(name) = node.input_pin_name(pin.id.input) {
            ui.label(name);
        }

        // ピンタイプに応じた色
        let color = node
            .input_pin_type(pin.id.input)
            .map(pin_color)
            .unwrap_or(Color32::GRAY);

        PinInfo::circle().with_fill(color)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];

        // ピン名を表示
        if let Some(name) = node.output_pin_name(pin.id.output) {
            ui.label(name);
        }

        // ピンタイプに応じた色
        let color = node
            .output_pin_type(pin.id.output)
            .map(pin_color)
            .unwrap_or(Color32::GRAY);

        PinInfo::circle().with_fill(color)
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) {
        let node = &mut snarl[node_id];

        match node {
            AudioNode::AudioInput(input_node) => {
                self.show_audio_input_body(node_id, input_node, ui);
            }
            AudioNode::AudioOutput(output_node) => {
                self.show_audio_output_body(node_id, output_node, ui);
            }
            AudioNode::Gain(gain_node) => {
                self.show_gain_body(node_id, gain_node, ui);
            }
            AudioNode::Add(add_node) => {
                self.show_add_body(node_id, add_node, ui);
            }
            AudioNode::Multiply(multiply_node) => {
                self.show_multiply_body(node_id, multiply_node, ui);
            }
            AudioNode::Filter(filter_node) => {
                self.show_filter_body(node_id, filter_node, ui);
            }
            AudioNode::SpectrumAnalyzer(spectrum_node) => {
                self.show_spectrum_body(node_id, spectrum_node, ui);
            }
            AudioNode::Compressor(compressor_node) => {
                self.show_compressor_body(node_id, compressor_node, ui);
            }
        }
    }

    fn has_body(&mut self, _node: &AudioNode) -> bool {
        true
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<AudioNode>) {
        let from_node = &snarl[from.id.node];
        let to_node = &snarl[to.id.node];

        let from_type = from_node.output_pin_type(from.id.output);
        let to_type = to_node.input_pin_type(to.id.input);

        // 同じピンタイプ同士のみ接続を許可
        if from_type.is_some() && from_type == to_type {
            // 既存の接続を削除（単一入力の場合）
            for &remote in &to.remotes {
                snarl.disconnect(remote, to.id);
            }
            snarl.connect(from.id, to.id);
        }
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<AudioNode>) {
        snarl.disconnect(from.id, to.id);
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<AudioNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut Ui, snarl: &mut Snarl<AudioNode>) {
        ui.label("Add Node");
        ui.separator();

        ui.menu_button("Input", |ui| {
            if ui.button("Audio Input").clicked() {
                let default_device = self
                    .input_devices
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "No device".to_string());
                snarl.insert_node(pos, AudioNode::new_audio_input(default_device));
                ui.close();
            }
        });

        ui.menu_button("Output", |ui| {
            if ui.button("Audio Output").clicked() {
                let default_device = self
                    .output_devices
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "No device".to_string());
                snarl.insert_node(pos, AudioNode::new_audio_output(default_device));
                ui.close();
            }
        });

        ui.menu_button("Effect", |ui| {
            if ui.button("Gain").clicked() {
                snarl.insert_node(pos, AudioNode::Gain(crate::nodes::GainNode::new()));
                ui.close();
            }
            if ui.button("Filter").clicked() {
                snarl.insert_node(pos, AudioNode::Filter(crate::nodes::FilterNode::new()));
                ui.close();
            }
            if ui.button("Compressor").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::Compressor(crate::nodes::CompressorNode::new()),
                );
                ui.close();
            }
        });

        ui.menu_button("Math", |ui| {
            if ui.button("Add").clicked() {
                snarl.insert_node(pos, AudioNode::Add(crate::nodes::AddNode::new()));
                ui.close();
            }
            if ui.button("Multiply").clicked() {
                snarl.insert_node(pos, AudioNode::Multiply(crate::nodes::MultiplyNode::new()));
                ui.close();
            }
        });

        ui.menu_button("Analysis", |ui| {
            if ui.button("Spectrum Analyzer").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::SpectrumAnalyzer(crate::nodes::SpectrumAnalyzerNode::new()),
                );
                ui.close();
            }
        });
    }

    fn has_node_menu(&mut self, _node: &AudioNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) {
        if ui.button("Delete").clicked() {
            snarl.remove_node(node_id);
            ui.close();
        }
    }
}

// ノードタイプ別のUI表示ヘルパーメソッド
impl AudioGraphViewer {
    fn show_audio_input_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::AudioInputNode,
        ui: &mut Ui,
    ) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut node.is_active, "Active");
            if node.is_active {
                ui.label(format!("{}ch", node.channels));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Device:");
            egui::ComboBox::from_id_salt(format!("input_device_{:?}", node_id))
                .selected_text(node.device_name.as_str())
                .show_ui(ui, |ui| {
                    for dev in &self.input_devices {
                        ui.selectable_value(&mut node.device_name, dev.clone(), dev);
                    }
                });
        });
    }

    fn show_audio_output_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::AudioOutputNode,
        ui: &mut Ui,
    ) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut node.is_active, "Active");
            if node.is_active {
                ui.label(format!("{}ch", node.channels));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Device:");
            egui::ComboBox::from_id_salt(format!("output_device_{:?}", node_id))
                .selected_text(node.device_name.as_str())
                .show_ui(ui, |ui| {
                    for dev in &self.output_devices {
                        ui.selectable_value(&mut node.device_name, dev.clone(), dev);
                    }
                });
        });
    }

    fn show_gain_body(&self, _node_id: NodeId, node: &mut crate::nodes::GainNode, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Gain:");
            ui.add(egui::Slider::new(&mut node.gain, 0.0..=2.0).suffix("x"));
        });

        // dB表示
        let db = 20.0 * node.gain.log10();
        if db.is_finite() {
            ui.label(format!("{:.1} dB", db));
        } else {
            ui.label("-∞ dB");
        }
    }

    fn show_add_body(&self, _node_id: NodeId, _node: &mut crate::nodes::AddNode, ui: &mut Ui) {
        ui.label("A + B");
    }

    fn show_multiply_body(
        &self,
        _node_id: NodeId,
        _node: &mut crate::nodes::MultiplyNode,
        ui: &mut Ui,
    ) {
        ui.label("A × B");
    }

    fn show_filter_body(&self, node_id: NodeId, node: &mut crate::nodes::FilterNode, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt(format!("filter_type_{:?}", node_id))
                .selected_text(match node.filter_type {
                    FilterType::Low => "Low Pass",
                    FilterType::High => "High Pass",
                    FilterType::Band => "Band Pass",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut node.filter_type, FilterType::Low, "Low Pass");
                    ui.selectable_value(&mut node.filter_type, FilterType::High, "High Pass");
                    ui.selectable_value(&mut node.filter_type, FilterType::Band, "Band Pass");
                });
        });

        ui.horizontal(|ui| {
            ui.label("Cutoff:");
            ui.add(
                egui::Slider::new(&mut node.cutoff, 20.0..=20000.0)
                    .logarithmic(true)
                    .suffix(" Hz"),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Q:");
            ui.add(egui::Slider::new(&mut node.resonance, 0.1..=10.0));
        });
    }

    fn show_spectrum_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::SpectrumAnalyzerNode,
        ui: &mut Ui,
    ) {
        // スペクトラムデータを取得
        let spectrum = node.spectrum.lock();
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
        Plot::new(format!("spectrum_{:?}", node_id))
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
    }

    fn show_compressor_body(
        &self,
        _node_id: NodeId,
        node: &mut crate::nodes::CompressorNode,
        ui: &mut Ui,
    ) {
        ui.horizontal(|ui| {
            ui.label("Threshold:");
            ui.add(egui::Slider::new(&mut node.threshold, -60.0..=0.0).suffix(" dB"));
        });

        ui.horizontal(|ui| {
            ui.label("Ratio:");
            ui.add(egui::Slider::new(&mut node.ratio, 1.0..=20.0).suffix(":1"));
        });

        ui.horizontal(|ui| {
            ui.label("Attack:");
            ui.add(egui::Slider::new(&mut node.attack, 0.1..=100.0).suffix(" ms"));
        });

        ui.horizontal(|ui| {
            ui.label("Release:");
            ui.add(egui::Slider::new(&mut node.release, 10.0..=1000.0).suffix(" ms"));
        });

        ui.horizontal(|ui| {
            ui.label("Makeup:");
            ui.add(egui::Slider::new(&mut node.makeup_gain, 0.0..=24.0).suffix(" dB"));
        });
    }
}
