use std::collections::HashMap;

use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};

use crate::audio::AudioSystem;
use crate::dsp::{BiquadCoeffs, CompressorParams};
use crate::nodes::{AudioNode, ChannelBuffer, NodeBehavior};

/// オーディオグラフの処理を管理
pub struct AudioGraphProcessor {
    /// アクティブなノードのIDセット
    active_nodes: HashMap<NodeId, ()>,
    /// サンプルレート
    sample_rate: f32,
}

impl AudioGraphProcessor {
    pub fn new() -> Self {
        Self {
            active_nodes: HashMap::new(),
            sample_rate: 44100.0,
        }
    }

    /// アクティブなストリームがあるかチェック
    pub fn has_active_streams(&self) -> bool {
        !self.active_nodes.is_empty()
    }

    /// グラフの接続を処理し、オーディオをルーティング
    pub fn process(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // サンプルレートを更新
        self.sample_rate = audio_system.config().sample_rate as f32;

        // アクティブなノードのストリームを管理
        self.manage_streams(snarl, audio_system);

        // エフェクトノードの処理
        self.process_effects(snarl);
    }

    /// エフェクトノードの処理
    fn process_effects(&mut self, snarl: &mut Snarl<AudioNode>) {
        // 処理順序を決定するため、出力ノードから逆順にたどる
        let node_ids: Vec<NodeId> = snarl.node_ids().map(|(id, _)| id).collect();

        for node_id in node_ids {
            let node = &snarl[node_id];

            // エフェクトノードのみ処理
            match node {
                AudioNode::Gain(_)
                | AudioNode::Add(_)
                | AudioNode::Multiply(_)
                | AudioNode::Filter(_)
                | AudioNode::SpectrumAnalyzer(_)
                | AudioNode::Compressor(_) => {
                    self.process_effect_node(snarl, node_id);
                }
                _ => {}
            }
        }
    }

    /// 単一のエフェクトノードを処理
    fn process_effect_node(&mut self, snarl: &Snarl<AudioNode>, node_id: NodeId) {
        let block_size = 256;

        // 入力データを収集
        let input_data = self.collect_input_samples(snarl, node_id, 0, block_size);

        // ノードタイプに応じた処理
        let node = &snarl[node_id];
        let output_data: Vec<f32> = match node {
            AudioNode::Gain(gain_node) => input_data.iter().map(|&s| s * gain_node.gain).collect(),
            AudioNode::Add(_) => {
                let input_b = self.collect_input_samples(snarl, node_id, 1, block_size);
                input_data
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            AudioNode::Multiply(_) => {
                let input_b = self.collect_input_samples(snarl, node_id, 1, block_size);
                input_data
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a * b)
                    .collect()
            }
            AudioNode::Filter(filter_node) => {
                let coeffs = BiquadCoeffs::from_filter_type(
                    filter_node.filter_type,
                    self.sample_rate,
                    filter_node.cutoff,
                    filter_node.resonance,
                );
                let mut state = filter_node.biquad_state.lock();
                input_data
                    .iter()
                    .map(|&s| state.process(s, &coeffs))
                    .collect()
            }
            AudioNode::SpectrumAnalyzer(spectrum_node) => {
                // FFT用にサンプルを蓄積
                {
                    let mut analyzer = spectrum_node.analyzer.lock();
                    for &sample in &input_data {
                        analyzer.push_sample(sample);
                    }
                }
                // スペクトラムを更新
                spectrum_node.update_spectrum();
                // パススルー
                input_data.clone()
            }
            AudioNode::Compressor(comp_node) => {
                let params = CompressorParams {
                    threshold_db: comp_node.threshold,
                    ratio: comp_node.ratio,
                    attack_ms: comp_node.attack,
                    release_ms: comp_node.release,
                    makeup_db: comp_node.makeup_gain,
                    sample_rate: self.sample_rate,
                };
                let mut state = comp_node.compressor_state.lock();
                input_data
                    .iter()
                    .map(|&s| state.process(s, &params))
                    .collect()
            }
            _ => input_data.clone(),
        };

        // 出力バッファに書き込み
        if let Some(output_buffer) = snarl[node_id].channel_buffer(0) {
            let mut buffer = output_buffer.lock();
            buffer.write(&output_data);
        }
    }

    /// 指定した入力ピンからサンプルを収集
    fn collect_input_samples(
        &self,
        snarl: &Snarl<AudioNode>,
        node_id: NodeId,
        input_idx: usize,
        count: usize,
    ) -> Vec<f32> {
        let in_pin = snarl.in_pin(InPinId {
            node: node_id,
            input: input_idx,
        });

        let mut samples = vec![0.0; count];

        if let Some(&remote) = in_pin.remotes.first() {
            if let Some(buffer) = snarl[remote.node].channel_buffer(remote.output) {
                let mut buf = buffer.lock();
                buf.read(&mut samples);
            }
        }

        samples
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
                AudioNode::Gain(_)
                | AudioNode::Add(_)
                | AudioNode::Multiply(_)
                | AudioNode::Filter(_)
                | AudioNode::SpectrumAnalyzer(_)
                | AudioNode::Compressor(_) => {
                    // エフェクトノードはデバイスストリームを持たない
                    // process_effectsで処理される
                }
            }
        }

        // ストリームを開始/停止
        for (node_id, device_name, buffers) in to_start_input {
            match audio_system.start_input(&device_name, buffers) {
                Ok(channels) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes.insert(node_id, ());
                }
                Err(e) => {
                    eprintln!("Failed to start input: {}", e);
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
            let in_pin = snarl.in_pin(InPinId {
                node: output_node_id,
                input: ch,
            });

            // 接続チェーンをたどって最終的なバッファを取得
            let buffer = if let Some(&remote) = in_pin.remotes.first() {
                self.trace_buffer_chain(snarl, remote, &output_node.channel_buffers[ch])
            } else {
                output_node.channel_buffers[ch].clone()
            };

            buffers.push(buffer);
        }

        buffers
    }

    /// バッファチェーンをたどって最終的な出力バッファを取得
    fn trace_buffer_chain(
        &self,
        snarl: &Snarl<AudioNode>,
        out_pin_id: OutPinId,
        fallback: &ChannelBuffer,
    ) -> ChannelBuffer {
        let source_node = &snarl[out_pin_id.node];

        // エフェクトノードの場合、そのoutput_bufferを返す
        // これによりエフェクト処理後のデータが出力に送られる
        source_node
            .channel_buffer(out_pin_id.output)
            .unwrap_or_else(|| fallback.clone())
    }
}

impl Default for AudioGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}
