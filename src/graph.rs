use std::collections::{HashMap, HashSet};

use egui_snarl::{InPinId, NodeId, Snarl};

use crate::audio::{AudioSystem, OutputStreamId};
use crate::effect_processor::{EffectNodeInfo, EffectNodeType, EffectProcessor};
use crate::nodes::{
    AudioInputNode, AudioInputPort, AudioNode, AudioOutputNode, AudioOutputPort, ChannelBuffer,
    CompressorNode, FilterNode, GainNode, GraphicEqNode, NodeType, PitchShiftNode,
    SpectrumAnalyzerNode,
};

/// アクティブノードの状態
enum ActiveNodeState {
    /// 入力ノード
    Input,
    /// 出力ノード（ストリームIDを保持）
    Output { stream_id: OutputStreamId },
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

    /// すべてのストリームを停止
    pub fn stop_all_streams(&mut self) {
        self.active_nodes.clear();
        self.effect_processor.clear_nodes();
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

        for (_node_id, node) in snarl.nodes_ids_mut() {
            match node.node_type() {
                NodeType::AudioInput => {
                    if let Some(input_node) = node.as_any_mut().downcast_mut::<AudioInputNode>() {
                        if input_node.is_active {
                            // 最初のチャンネルからデータを取得してスペクトラム解析（常に更新）
                            // read()は状態を変更しないので、データを消費せずに読み取れる
                            if let Some(buffer) = input_node.channel_buffers.first() {
                                let samples = buffer.lock().read(FFT_SIZE);

                                let mut analyzer = input_node.analyzer.lock();
                                analyzer.push_samples(&samples);
                                let spectrum_data = analyzer.compute_spectrum();

                                let mut spectrum = input_node.spectrum.lock();
                                spectrum.copy_from_slice(&spectrum_data);
                            }
                        }
                    }
                }
                NodeType::AudioOutput => {
                    if let Some(output_node) = node.as_any_mut().downcast_mut::<AudioOutputNode>() {
                        if output_node.is_active {
                            // 出力ノード自身のバッファからスペクトラム解析
                            // read()は状態を変更しないので、データを消費せずに読み取れる
                            if let Some(buffer) = output_node.channel_buffers.first() {
                                let samples = buffer.lock().read(FFT_SIZE);

                                let mut analyzer = output_node.analyzer.lock();
                                analyzer.push_samples(&samples);
                                let spectrum_data = analyzer.compute_spectrum();

                                let mut spectrum = output_node.spectrum.lock();
                                spectrum.copy_from_slice(&spectrum_data);
                            }
                        }
                    }
                }
                NodeType::GraphicEq => {
                    if let Some(eq_node) = node.as_any_mut().downcast_mut::<GraphicEqNode>() {
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

            // エフェクトノードの処理
            if let Some(info) = self.build_effect_node_info(snarl, node_id, node) {
                effect_nodes.push(info);
            }

            // 出力ノードへのデータコピー処理を追加
            if node.node_type() == NodeType::AudioOutput {
                if let Some(output_node) = node.as_any().downcast_ref::<AudioOutputNode>() {
                    if output_node.is_active {
                        // 各チャンネルに対してPassThroughエフェクトを作成
                        for ch in 0..output_node.channels as usize {
                            if let Some(source_buffer) = Self::get_source_buffer(snarl, node_id, ch)
                            {
                                // 出力ノードの入力バッファを使用
                                if let Some(input_buffer) = output_node.input_buffer(ch) {
                                    if let Some(output_buffer) = output_node.channel_buffer(ch) {
                                        effect_nodes.push(EffectNodeInfo {
                                            node_type: EffectNodeType::PassThrough,
                                            source_buffers: vec![source_buffer],
                                            input_buffers: vec![input_buffer],
                                            output_buffer,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
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
        let (node_type, input_count) = match node.node_type() {
            NodeType::Gain => {
                let gain_node = node.as_any().downcast_ref::<GainNode>()?;
                (
                    EffectNodeType::Gain {
                        gain: gain_node.gain,
                    },
                    1,
                )
            }
            NodeType::Add => (EffectNodeType::Add, 2),
            NodeType::Multiply => (EffectNodeType::Multiply, 2),
            NodeType::Filter => {
                let filter_node = node.as_any().downcast_ref::<FilterNode>()?;
                (
                    EffectNodeType::Filter {
                        filter_type: filter_node.filter_type,
                        cutoff: filter_node.cutoff,
                        resonance: filter_node.resonance,
                        state: filter_node.biquad_state.clone(),
                    },
                    1,
                )
            }
            NodeType::SpectrumAnalyzer => {
                let spectrum_node = node.as_any().downcast_ref::<SpectrumAnalyzerNode>()?;
                (
                    EffectNodeType::SpectrumAnalyzer {
                        analyzer: spectrum_node.analyzer.clone(),
                        spectrum: spectrum_node.spectrum.clone(),
                    },
                    1,
                )
            }
            NodeType::Compressor => {
                let comp_node = node.as_any().downcast_ref::<CompressorNode>()?;
                (
                    EffectNodeType::Compressor {
                        threshold: comp_node.threshold,
                        ratio: comp_node.ratio,
                        attack: comp_node.attack,
                        release: comp_node.release,
                        makeup_gain: comp_node.makeup_gain,
                        state: comp_node.compressor_state.clone(),
                    },
                    1,
                )
            }
            NodeType::PitchShift => {
                let pitch_node = node.as_any().downcast_ref::<PitchShiftNode>()?;
                (
                    EffectNodeType::PitchShift {
                        semitones: pitch_node.semitones,
                        phase_alignment_enabled: pitch_node.phase_alignment_enabled,
                        search_range_ratio: pitch_node.search_range_ratio,
                        correlation_length_ratio: pitch_node.correlation_length_ratio,
                        pitch_shifter: pitch_node.pitch_shifter.clone(),
                    },
                    1,
                )
            }
            NodeType::GraphicEq => {
                let eq_node = node.as_any().downcast_ref::<GraphicEqNode>()?;
                (
                    EffectNodeType::GraphicEq {
                        graphic_eq: eq_node.graphic_eq.clone(),
                    },
                    1,
                )
            }
            // 入出力ノードはエフェクトノードではない
            NodeType::AudioInput | NodeType::AudioOutput => return None,
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

    /// ストリームの開始/停止を管理
    fn manage_streams(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        // まず、状態変更が必要なノードを収集
        let mut to_start_input: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_input: Vec<NodeId> = Vec::new();
        let mut to_start_output: Vec<(NodeId, String, Vec<ChannelBuffer>)> = Vec::new();
        let mut to_stop_output: Vec<(NodeId, OutputStreamId)> = Vec::new();

        for (node_id, node) in snarl.node_ids() {
            match node.node_type() {
                NodeType::AudioInput => {
                    if let Some(input_node) = node.as_any().downcast_ref::<AudioInputNode>() {
                        if input_node.is_active && !self.active_nodes.contains_key(&node_id) {
                            to_start_input.push((
                                node_id,
                                input_node.device_name.clone(),
                                input_node.channel_buffers.clone(),
                            ));
                        } else if !input_node.is_active && self.active_nodes.contains_key(&node_id)
                        {
                            to_stop_input.push(node_id);
                        }
                    }
                }
                NodeType::AudioOutput => {
                    if let Some(output_node) = node.as_any().downcast_ref::<AudioOutputNode>() {
                        if output_node.is_active && !self.active_nodes.contains_key(&node_id) {
                            // 出力ノードは常に自身のバッファを使用
                            // データはeffect_processorによってソースからコピーされる
                            let buffers = output_node.channel_buffers.clone();
                            to_start_output.push((
                                node_id,
                                output_node.device_name.clone(),
                                buffers,
                            ));
                        } else if !output_node.is_active {
                            if let Some(ActiveNodeState::Output { stream_id }) =
                                self.active_nodes.get(&node_id)
                            {
                                to_stop_output.push((node_id, *stream_id));
                            }
                        }
                        // 接続変更はeffect_processorで自動的に処理されるため、再起動不要
                    }
                }
                // エフェクトノードはEffectProcessorで処理される
                NodeType::Gain
                | NodeType::Add
                | NodeType::Multiply
                | NodeType::Filter
                | NodeType::SpectrumAnalyzer
                | NodeType::Compressor
                | NodeType::PitchShift
                | NodeType::GraphicEq => {}
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
            match audio_system.start_output(&device_name, buffers) {
                Ok((channels, stream_id)) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes
                        .insert(node_id, ActiveNodeState::Output { stream_id });
                }
                Err(e) => {
                    eprintln!("Failed to start output: {}", e);
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
        }

        for (node_id, stream_id) in to_stop_output {
            audio_system.stop_output(stream_id);
            self.active_nodes.remove(&node_id);
        }
    }

    /// 接続されたソースノードの出力バッファを取得
    fn get_source_buffer(
        snarl: &Snarl<AudioNode>,
        node_id: NodeId,
        input_idx: usize,
    ) -> Option<ChannelBuffer> {
        let in_pin = snarl.in_pin(InPinId {
            node: node_id,
            input: input_idx,
        });

        if let Some(&remote) = in_pin.remotes.first() {
            snarl[remote.node].channel_buffer(remote.output)
        } else {
            None
        }
    }
}

impl Default for AudioGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}
