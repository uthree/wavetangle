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
        let ctx = crate::nodes::NodeUIContext {
            input_devices: &self.input_devices,
            output_devices: &self.output_devices,
            node_id,
        };
        let node = &mut snarl[node_id];
        node.show_body(ui, &ctx);
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
            if self.input_devices.is_empty() {
                ui.label("No input devices");
            } else {
                for device_name in &self.input_devices.clone() {
                    if ui.button(device_name).clicked() {
                        snarl.insert_node(pos, AudioNode::new_audio_input(device_name.clone()));
                        ui.close();
                    }
                }
            }
        });

        ui.menu_button("Output", |ui| {
            if self.output_devices.is_empty() {
                ui.label("No output devices");
            } else {
                for device_name in &self.output_devices.clone() {
                    if ui.button(device_name).clicked() {
                        snarl.insert_node(pos, AudioNode::new_audio_output(device_name.clone()));
                        ui.close();
                    }
                }
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
            if ui.button("Pitch Shift").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::PitchShift(crate::nodes::PitchShiftNode::new()),
                );
                ui.close();
            }
            if ui.button("Graphic EQ").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::GraphicEq(crate::nodes::GraphicEqNode::new()),
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
