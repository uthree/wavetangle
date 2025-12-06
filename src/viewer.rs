use egui::{Color32, Ui};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::nodes::{AudioNode, NodeBehavior, PinType};

/// ピンタイプに応じた色を取得
fn pin_color(pin_type: PinType) -> Color32 {
    match pin_type {
        PinType::Audio => Color32::from_rgb(100, 200, 100),
    }
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
}
