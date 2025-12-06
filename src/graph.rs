use std::collections::HashMap;

use egui_snarl::{NodeId, Snarl};

use crate::audio::AudioSystem;
use crate::nodes::{AudioBuffer, AudioNode};

/// オーディオグラフの処理を管理
pub struct AudioGraphProcessor {
    /// アクティブなノードのバッファマッピング
    active_connections: HashMap<NodeId, AudioBuffer>,
}

impl AudioGraphProcessor {
    pub fn new() -> Self {
        Self {
            active_connections: HashMap::new(),
        }
    }

    /// アクティブなストリームがあるかチェック
    pub fn has_active_streams(&self) -> bool {
        !self.active_connections.is_empty()
    }

    /// グラフの接続を処理し、オーディオをルーティング
    pub fn process(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // 接続されたノードのペアを見つけて、バッファを共有
        let mut connections: Vec<(NodeId, NodeId)> = Vec::new();

        for (node_id, node) in snarl.node_ids() {
            if let AudioNode::AudioOutput { .. } = node {
                // 出力ノードの入力ピンに接続されているソースを探す
                let in_pin = snarl.in_pin(egui_snarl::InPinId {
                    node: node_id,
                    input: 0,
                });
                for remote in &in_pin.remotes {
                    connections.push((remote.node, node_id));
                }
            }
        }

        // 接続されたノード間でバッファを共有
        for (input_node_id, output_node_id) in connections {
            let input_buffer = {
                let node = &snarl[input_node_id];
                node.buffer().cloned()
            };

            if let Some(buffer) = input_buffer {
                // 出力ノードのバッファを更新
                let output_node = &mut snarl[output_node_id];
                if let AudioNode::AudioOutput {
                    buffer: out_buffer, ..
                } = output_node
                {
                    // 入力バッファからデータをコピー
                    let data = buffer.lock().clone();
                    *out_buffer.lock() = data;
                }
            }
        }

        // アクティブなノードのストリームを管理
        self.manage_streams(snarl, audio_system);
    }

    /// ストリームの開始/停止を管理
    fn manage_streams(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // まず、状態変更が必要なノードを収集
        let mut to_start_input: Vec<(NodeId, String, AudioBuffer)> = Vec::new();
        let mut to_stop_input: Vec<NodeId> = Vec::new();
        let mut to_start_output: Vec<(NodeId, String, AudioBuffer)> = Vec::new();
        let mut to_stop_output: Vec<NodeId> = Vec::new();

        for (node_id, node) in snarl.node_ids() {
            match node {
                AudioNode::AudioInput {
                    device_name,
                    buffer,
                    is_active,
                } => {
                    if *is_active && !self.active_connections.contains_key(&node_id) {
                        to_start_input.push((node_id, device_name.clone(), buffer.clone()));
                    } else if !*is_active && self.active_connections.contains_key(&node_id) {
                        to_stop_input.push(node_id);
                    }
                }
                AudioNode::AudioOutput {
                    device_name,
                    buffer,
                    is_active,
                } => {
                    if *is_active && !self.active_connections.contains_key(&node_id) {
                        to_start_output.push((node_id, device_name.clone(), buffer.clone()));
                    } else if !*is_active && self.active_connections.contains_key(&node_id) {
                        to_stop_output.push(node_id);
                    }
                }
            }
        }

        // ストリームを開始/停止
        for (node_id, device_name, buffer) in to_start_input {
            if let Err(e) = audio_system.start_input(&device_name, buffer.clone()) {
                eprintln!("Failed to start input: {}", e);
                // エラー時はノードを非アクティブに
                if let Some(node) = snarl.get_node_mut(node_id) {
                    node.set_active(false);
                }
            } else {
                self.active_connections.insert(node_id, buffer);
            }
        }

        for node_id in to_stop_input {
            audio_system.stop_input();
            self.active_connections.remove(&node_id);
        }

        for (node_id, device_name, buffer) in to_start_output {
            if let Err(e) = audio_system.start_output(&device_name, buffer.clone()) {
                eprintln!("Failed to start output: {}", e);
                if let Some(node) = snarl.get_node_mut(node_id) {
                    node.set_active(false);
                }
            } else {
                self.active_connections.insert(node_id, buffer);
            }
        }

        for node_id in to_stop_output {
            audio_system.stop_output();
            self.active_connections.remove(&node_id);
        }
    }
}

impl Default for AudioGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}
