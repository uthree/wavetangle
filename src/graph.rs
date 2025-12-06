use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};

use crate::audio::AudioSystem;
use crate::effect_processor::{EffectNodeInfo, EffectNodeType, EffectProcessor};
use crate::nodes::{AudioNode, ChannelBuffer, NodeBehavior};

/// アクティブノードの状態
enum ActiveNodeState {
    /// 入力ノード
    Input,
    /// 出力ノード（接続されているバッファの参照を保持）
    Output(Vec<ChannelBuffer>),
}

/// オーディオグラフの処理を管理
pub struct AudioGraphProcessor {
    /// アクティブなノードの状態
    active_nodes: HashMap<NodeId, ActiveNodeState>,
    /// サンプルレート
    sample_rate: f32,
    /// エフェクトプロセッサー
    effect_processor: EffectProcessor,
}

impl AudioGraphProcessor {
    pub fn new() -> Self {
        Self {
            active_nodes: HashMap::new(),
            sample_rate: 44100.0,
            effect_processor: EffectProcessor::new(2), // 2ms間隔で高速処理
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
        self.effect_processor.set_sample_rate(self.sample_rate);

        // アクティブなノードのストリームを管理
        self.manage_streams(snarl, audio_system);

        // エフェクトノードの処理チェーンを構築・更新
        self.update_effect_chain(snarl);

        // スペクトラム解析を更新
        self.update_spectrum(snarl);
    }

    /// 入出力ノードとGraphicEQのスペクトラムを更新
    fn update_spectrum(&self, snarl: &mut Snarl<AudioNode>) {
        use crate::nodes::FFT_SIZE;

        for (node_id, node) in snarl.nodes_ids_mut() {
            match node {
                AudioNode::AudioInput(input_node) => {
                    if input_node.is_active {
                        // 最初のチャンネルからデータを取得してスペクトラム解析（常に更新）
                        if let Some(buffer) = input_node.channel_buffers.first() {
                            let mut samples = vec![0.0f32; FFT_SIZE];
                            buffer.lock().peek(&mut samples);

                            let mut analyzer = input_node.analyzer.lock();
                            analyzer.push_samples(&samples);
                            let spectrum_data = analyzer.compute_spectrum();

                            let mut spectrum = input_node.spectrum.lock();
                            spectrum.copy_from_slice(&spectrum_data);
                        }
                    }
                }
                AudioNode::AudioOutput(output_node) => {
                    if output_node.is_active {
                        // アクティブノードからソースバッファを取得してスペクトラム解析
                        if let Some(ActiveNodeState::Output(source_buffers)) =
                            self.active_nodes.get(&node_id)
                        {
                            if let Some(buffer) = source_buffers.first() {
                                let mut samples = vec![0.0f32; FFT_SIZE];
                                buffer.lock().peek(&mut samples);

                                let mut analyzer = output_node.analyzer.lock();
                                analyzer.push_samples(&samples);
                                let spectrum_data = analyzer.compute_spectrum();

                                let mut spectrum = output_node.spectrum.lock();
                                spectrum.copy_from_slice(&spectrum_data);
                            }
                        }
                    }
                }
                AudioNode::GraphicEq(eq_node) => {
                    if eq_node.show_spectrum {
                        // 入力バッファからスペクトラムを取得
                        // GraphicEqは内部でFFTを使用しているので、そのスペクトラムを再利用
                        let eq = eq_node.graphic_eq.lock();
                        let spectrum_data = eq.get_input_spectrum();
                        let mut spectrum = eq_node.spectrum.lock();
                        if spectrum.len() == spectrum_data.len() {
                            spectrum.copy_from_slice(spectrum_data);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// エフェクト処理チェーンを更新
    fn update_effect_chain(&mut self, snarl: &Snarl<AudioNode>) {
        // アクティブなオーディオがある場合のみ処理
        if self.active_nodes.is_empty() {
            if self.effect_processor.is_running() {
                self.effect_processor.stop();
                self.effect_processor.clear_nodes();
            }
            return;
        }

        // 処理順序を決定（トポロジカルソート）
        let sorted_nodes = self.topological_sort(snarl);

        // エフェクトノード情報を収集
        let mut effect_nodes = Vec::new();

        for node_id in sorted_nodes {
            let node = &snarl[node_id];

            if let Some(info) = self.build_effect_node_info(snarl, node_id, node) {
                effect_nodes.push(info);
            }
        }

        // プロセッサーを更新
        self.effect_processor.update_nodes(effect_nodes);

        // プロセッサーが停止していれば開始
        if !self.effect_processor.is_running() {
            self.effect_processor.start();
        }
    }

    /// トポロジカルソートでノードの処理順序を決定
    fn topological_sort(&self, snarl: &Snarl<AudioNode>) -> Vec<NodeId> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        // すべてのノードに対してDFS
        for (node_id, _) in snarl.node_ids() {
            if !visited.contains(&node_id) {
                Self::topological_visit(
                    snarl,
                    node_id,
                    &mut visited,
                    &mut temp_visited,
                    &mut sorted,
                );
            }
        }

        sorted
    }

    /// トポロジカルソートのDFS訪問
    fn topological_visit(
        snarl: &Snarl<AudioNode>,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        temp_visited: &mut HashSet<NodeId>,
        sorted: &mut Vec<NodeId>,
    ) {
        if temp_visited.contains(&node_id) || visited.contains(&node_id) {
            return;
        }

        temp_visited.insert(node_id);

        let node = &snarl[node_id];

        // このノードの入力に接続されているノードを先に訪問
        for input_idx in 0..node.input_count() {
            let in_pin = snarl.in_pin(InPinId {
                node: node_id,
                input: input_idx,
            });

            for &remote in &in_pin.remotes {
                Self::topological_visit(snarl, remote.node, visited, temp_visited, sorted);
            }
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        sorted.push(node_id);
    }

    /// エフェクトノード情報を構築
    fn build_effect_node_info(
        &self,
        snarl: &Snarl<AudioNode>,
        node_id: NodeId,
        node: &AudioNode,
    ) -> Option<EffectNodeInfo> {
        let (node_type, input_count) = match node {
            AudioNode::Gain(gain_node) => (
                EffectNodeType::Gain {
                    gain: gain_node.gain,
                },
                1,
            ),
            AudioNode::Add(_) => (EffectNodeType::Add, 2),
            AudioNode::Multiply(_) => (EffectNodeType::Multiply, 2),
            AudioNode::Filter(filter_node) => (
                EffectNodeType::Filter {
                    filter_type: filter_node.filter_type,
                    cutoff: filter_node.cutoff,
                    resonance: filter_node.resonance,
                    state: filter_node.biquad_state.clone(),
                },
                1,
            ),
            AudioNode::SpectrumAnalyzer(spectrum_node) => (
                EffectNodeType::SpectrumAnalyzer {
                    analyzer: spectrum_node.analyzer.clone(),
                    spectrum: spectrum_node.spectrum.clone(),
                },
                1,
            ),
            AudioNode::Compressor(comp_node) => (
                EffectNodeType::Compressor {
                    threshold: comp_node.threshold,
                    ratio: comp_node.ratio,
                    attack: comp_node.attack,
                    release: comp_node.release,
                    makeup_gain: comp_node.makeup_gain,
                    state: comp_node.compressor_state.clone(),
                },
                1,
            ),
            AudioNode::PitchShift(pitch_node) => (
                EffectNodeType::PitchShift {
                    semitones: pitch_node.semitones,
                    pitch_shifter: pitch_node.pitch_shifter.clone(),
                },
                1,
            ),
            AudioNode::GraphicEq(eq_node) => (
                EffectNodeType::GraphicEq {
                    graphic_eq: eq_node.graphic_eq.clone(),
                },
                1,
            ),
            // 入出力ノードはエフェクトノードではない
            AudioNode::AudioInput(_) | AudioNode::AudioOutput(_) => return None,
        };

        // ソースバッファを収集（接続されたノードの出力バッファ）
        let mut source_buffers = Vec::new();
        // ノード自身の入力バッファを収集
        let mut input_buffers = Vec::new();

        for input_idx in 0..input_count {
            let in_pin = snarl.in_pin(InPinId {
                node: node_id,
                input: input_idx,
            });

            // ソースバッファ（接続元ノードの出力）
            if let Some(&remote) = in_pin.remotes.first() {
                if let Some(buffer) = snarl[remote.node].channel_buffer(remote.output) {
                    source_buffers.push(buffer);
                }
            }

            // ノード自身の入力バッファ
            if let Some(buffer) = node.input_buffer(input_idx) {
                input_buffers.push(buffer);
            }
        }

        // 出力バッファを取得
        let output_buffer = node.channel_buffer(0)?;

        Some(EffectNodeInfo {
            node_type,
            source_buffers,
            input_buffers,
            output_buffer,
        })
    }

    /// バッファ参照が同じかチェック（Arc::ptr_eq で比較）
    fn buffers_equal(a: &[ChannelBuffer], b: &[ChannelBuffer]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(a, b)| Arc::ptr_eq(a, b))
    }

    /// ストリームの開始/停止を管理
    fn manage_streams(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // まず、状態変更が必要なノードを収集
        let mut to_start_input: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_input: Vec<NodeId> = Vec::new();
        let mut to_start_output: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_output: Vec<NodeId> = Vec::new();
        // 接続変更によりストリームを再起動するノード
        let mut to_restart_output: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();

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
                    } else if output_node.is_active && self.active_nodes.contains_key(&node_id) {
                        // アクティブな出力ノードの接続が変わったかチェック
                        let current_buffers =
                            self.collect_output_buffers(snarl, node_id, output_node);
                        if let Some(ActiveNodeState::Output(stored_buffers)) =
                            self.active_nodes.get(&node_id)
                        {
                            if !Self::buffers_equal(stored_buffers, &current_buffers) {
                                // バッファが変わった場合、ストリームを再起動
                                to_restart_output.push((
                                    node_id,
                                    output_node.device_name.clone(),
                                    current_buffers,
                                ));
                            }
                        }
                    }
                }
                AudioNode::Gain(_)
                | AudioNode::Add(_)
                | AudioNode::Multiply(_)
                | AudioNode::Filter(_)
                | AudioNode::SpectrumAnalyzer(_)
                | AudioNode::Compressor(_)
                | AudioNode::PitchShift(_)
                | AudioNode::GraphicEq(_) => {
                    // エフェクトノードはEffectProcessorで処理される
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
                    self.active_nodes.insert(node_id, ActiveNodeState::Input);
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
            match audio_system.start_output(&device_name, buffers.clone()) {
                Ok(channels) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes
                        .insert(node_id, ActiveNodeState::Output(buffers));
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

        // 接続変更により再起動が必要な出力ノード
        for (node_id, device_name, buffers) in to_restart_output {
            // 一旦停止してから再起動
            audio_system.stop_output();
            match audio_system.start_output(&device_name, buffers.clone()) {
                Ok(channels) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes
                        .insert(node_id, ActiveNodeState::Output(buffers));
                }
                Err(e) => {
                    eprintln!("Failed to restart output: {}", e);
                    self.active_nodes.remove(&node_id);
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
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
