use std::collections::HashMap;

use egui_snarl::{NodeId, Snarl};

use crate::audio::AudioSystem;
use crate::nodes::{AudioNode, ChannelBuffer, NodeBehavior};

/// オーディオグラフの処理を管理
pub struct AudioGraphProcessor {
    /// アクティブなノードのIDセット
    active_nodes: HashMap<NodeId, ()>,
}

impl AudioGraphProcessor {
    pub fn new() -> Self {
        Self {
            active_nodes: HashMap::new(),
        }
    }

    /// アクティブなストリームがあるかチェック
    pub fn has_active_streams(&self) -> bool {
        !self.active_nodes.is_empty()
    }

    /// グラフの接続を処理し、オーディオをルーティング
    pub fn process(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // 注: 実際のオーディオルーティングはcollect_output_buffersで行う
        // 出力ノード開始時に、接続された入力ノードのバッファを直接参照する

        // アクティブなノードのストリームを管理
        self.manage_streams(snarl, audio_system);
    }

    /// ストリームの開始/停止を管理
    fn manage_streams(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // まず、状態変更が必要なノードを収集
        let mut to_start_input: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_input: Vec<NodeId> = Vec::new();
        let mut to_start_output: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_output: Vec<NodeId> = Vec::new();

        for (node_id, node) in snarl.node_ids() {
            match node {
                AudioNode::AudioInput(input_node) => {
                    if input_node.is_active && !self.active_nodes.contains_key(&node_id) {
                        to_start_input.push((
                            node_id,
                            input_node.device_name.clone(),
                            input_node.channel_buffers.clone(),
                        ));
                    } else if !input_node.is_active && self.active_nodes.contains_key(&node_id) {
                        to_stop_input.push(node_id);
                    }
                }
                AudioNode::AudioOutput(output_node) => {
                    if output_node.is_active && !self.active_nodes.contains_key(&node_id) {
                        // 出力ノードの場合、接続されたソースのバッファを収集
                        let buffers = self.collect_output_buffers(snarl, node_id, output_node);
                        to_start_output.push((node_id, output_node.device_name.clone(), buffers));
                    } else if !output_node.is_active && self.active_nodes.contains_key(&node_id) {
                        to_stop_output.push(node_id);
                    }
                }
            }
        }

        // ストリームを開始/停止
        for (node_id, device_name, buffers) in to_start_input {
            match audio_system.start_input(&device_name, buffers) {
                Ok(channels) => {
                    // チャンネル数をノードに設定
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes.insert(node_id, ());
                }
                Err(e) => {
                    eprintln!("Failed to start input: {}", e);
                    // エラー時はノードを非アクティブに
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
        }

        for node_id in to_stop_input {
            audio_system.stop_input();
            self.active_nodes.remove(&node_id);
        }

        for (node_id, device_name, buffers) in to_start_output {
            match audio_system.start_output(&device_name, buffers) {
                Ok(channels) => {
                    // チャンネル数をノードに設定
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes.insert(node_id, ());
                }
                Err(e) => {
                    eprintln!("Failed to start output: {}", e);
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
        }

        for node_id in to_stop_output {
            audio_system.stop_output();
            self.active_nodes.remove(&node_id);
        }
    }

    /// 出力ノードに接続されたソースのバッファを収集
    fn collect_output_buffers(
        &self,
        snarl: &Snarl<AudioNode>,
        output_node_id: NodeId,
        output_node: &crate::nodes::AudioOutputNode,
    ) -> Vec<ChannelBuffer> {
        let channel_count = output_node.channels as usize;
        let mut buffers = Vec::with_capacity(channel_count);

        for ch in 0..channel_count {
            let in_pin = snarl.in_pin(egui_snarl::InPinId {
                node: output_node_id,
                input: ch,
            });

            // 接続されていればソースのバッファを使用、なければ出力ノード自身のバッファを使用
            let buffer = if let Some(&remote) = in_pin.remotes.first() {
                let source_node = &snarl[remote.node];
                source_node
                    .channel_buffer(remote.output)
                    .unwrap_or_else(|| output_node.channel_buffers[ch].clone())
            } else {
                // 接続がない場合は自身のバッファ（無音）
                output_node.channel_buffers[ch].clone()
            };

            buffers.push(buffer);
        }

        buffers
    }
}

impl Default for AudioGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}
