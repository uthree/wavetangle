use egui::{Color32, Ui};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::nodes::{AudioNode, PinType};

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
        let pin_type = node.pin_type(true, pin.id.input);

        ui.label("In");

        match pin_type {
            PinType::AudioInput => PinInfo::circle().with_fill(Color32::from_rgb(100, 200, 100)),
            PinType::AudioOutput => PinInfo::circle().with_fill(Color32::from_rgb(200, 100, 100)),
        }
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let pin_type = node.pin_type(false, pin.id.output);

        ui.label("Out");

        match pin_type {
            PinType::AudioInput => PinInfo::circle().with_fill(Color32::from_rgb(100, 200, 100)),
            PinType::AudioOutput => PinInfo::circle().with_fill(Color32::from_rgb(200, 100, 100)),
        }
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
            AudioNode::AudioInput {
                device_name,
                is_active,
                ..
            } => {
                ui.horizontal(|ui| {
                    ui.label("Device:");
                    egui::ComboBox::from_id_salt(format!("input_device_{:?}", node_id))
                        .selected_text(device_name.as_str())
                        .show_ui(ui, |ui| {
                            for dev in &self.input_devices {
                                ui.selectable_value(device_name, dev.clone(), dev);
                            }
                        });
                });

                ui.horizontal(|ui| {
                    if ui
                        .button(if *is_active { "Stop" } else { "Start" })
                        .clicked()
                    {
                        *is_active = !*is_active;
                    }

                    if *is_active {
                        ui.label("Active");
                    }
                });
            }
            AudioNode::AudioOutput {
                device_name,
                is_active,
                ..
            } => {
                ui.horizontal(|ui| {
                    ui.label("Device:");
                    egui::ComboBox::from_id_salt(format!("output_device_{:?}", node_id))
                        .selected_text(device_name.as_str())
                        .show_ui(ui, |ui| {
                            for dev in &self.output_devices {
                                ui.selectable_value(device_name, dev.clone(), dev);
                            }
                        });
                });

                ui.horizontal(|ui| {
                    if ui
                        .button(if *is_active { "Stop" } else { "Start" })
                        .clicked()
                    {
                        *is_active = !*is_active;
                    }

                    if *is_active {
                        ui.label("Active");
                    }
                });
            }
        }
    }

    fn has_body(&mut self, _node: &AudioNode) -> bool {
        true
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<AudioNode>) {
        // 同じタイプのピン同士を接続
        let from_node = &snarl[from.id.node];
        let to_node = &snarl[to.id.node];

        let from_type = from_node.pin_type(false, from.id.output);
        let to_type = to_node.pin_type(true, to.id.input);

        // オーディオ出力からオーディオ入力への接続のみ許可
        if from_type == PinType::AudioOutput && to_type == PinType::AudioInput {
            // 既存の接続を削除
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

        if ui.button("Audio Input").clicked() {
            let default_device = self
                .input_devices
                .first()
                .cloned()
                .unwrap_or_else(|| "No device".to_string());
            snarl.insert_node(pos, AudioNode::new_audio_input(default_device));
            ui.close();
        }

        if ui.button("Audio Output").clicked() {
            let default_device = self
                .output_devices
                .first()
                .cloned()
                .unwrap_or_else(|| "No device".to_string());
            snarl.insert_node(pos, AudioNode::new_audio_output(default_device));
            ui.close();
        }
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
