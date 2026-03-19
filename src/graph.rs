use std::collections::{HashMap, HashSet};

use egui_snarl::{InPinId, NodeId, Snarl};

use crate::audio::{AudioSystem, OutputStreamId};
use crate::effect_processor::{EffectNodeInfo, EffectNodeType, EffectProcessor};
use crate::nodes::{AudioNode, AudioOutputPort, ChannelBuffer};

/// アクティブノードの状態
enum ActiveNodeState {
    /// 入力ノード（mono状態を保持）
    Input { mono: bool },
    /// 出力ノード（ストリームIDとmono状態を保持）
    Output { stream_id: OutputStreamId, mono: bool },
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
            effect_processor: EffectProcessor::new(2),
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
        self.sample_rate = audio_system.config().sample_rate as f32;
        self.effect_processor.set_sample_rate(self.sample_rate);

        self.manage_streams(snarl, audio_system);
        self.update_effect_chain(snarl);
        self.update_spectrum(snarl);
    }

    /// 入出力ノードとGraphicEQのスペクトラムを更新
    fn update_spectrum(&self, snarl: &mut Snarl<AudioNode>) {
        use crate::nodes::FFT_SIZE;

        for (_node_id, node) in snarl.nodes_ids_mut() {
            match node {
                AudioNode::AudioInput(input_node) => {
                    if input_node.is_active {
                        if let Some(buffer) = input_node.buffers.output_buffers.first() {
                            let samples = buffer.lock().read(FFT_SIZE);
                            input_node.spectrum_display.update_from_samples(&samples);
                        }
                    }
                }
                AudioNode::AudioOutput(output_node) => {
                    if output_node.is_active {
                        if let Some(buffer) = output_node.buffers.output_buffers.first() {
                            let samples = buffer.lock().read(FFT_SIZE);
                            output_node.spectrum_display.update_from_samples(&samples);
                        }
                    }
                }
                AudioNode::GraphicEq(eq_node) => {
                    if eq_node.show_spectrum {
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
        if self.active_nodes.is_empty() {
            if self.effect_processor.is_running() {
                self.effect_processor.stop();
                self.effect_processor.clear_nodes();
            }
            return;
        }

        let sorted_nodes = self.topological_sort(snarl);
        let mut effect_nodes = Vec::new();

        for node_id in sorted_nodes {
            let node = &snarl[node_id];

            match node {
                // エフェクトノード: ソースバッファから直接読み取り、自身の出力バッファに書き込む
                AudioNode::AudioInput(_) => {
                    // 入力ノードはcpalが直接バッファに書き込むため処理不要
                }
                AudioNode::AudioOutput(output_node) => {
                    if output_node.is_active {
                        // 各チャンネルに対してCopyノードを作成
                        for ch in 0..output_node.channels() as usize {
                            if let Some(source_buffer) =
                                Self::get_source_buffer(snarl, node_id, ch)
                            {
                                if let Some(output_buffer) = output_node.channel_buffer(ch) {
                                    effect_nodes.push(EffectNodeInfo {
                                        node_type: EffectNodeType::Copy,
                                        source_buffers: vec![source_buffer],
                                        output_buffer,
                                    });
                                }
                            }
                        }
                    }
                }
                _ => {
                    if let Some(info) = self.build_effect_node_info(snarl, node_id, node) {
                        effect_nodes.push(info);
                    }
                }
            }
        }

        self.effect_processor.update_nodes(effect_nodes);

        if !self.effect_processor.is_running() {
            self.effect_processor.start();
        }
    }

    /// トポロジカルソートでノードの処理順序を決定
    fn topological_sort(&self, snarl: &Snarl<AudioNode>) -> Vec<NodeId> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

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
            AudioNode::WsolaPitchShift(pitch_node) => (
                EffectNodeType::WsolaPitchShift {
                    semitones: pitch_node.semitones,
                    phase_alignment_enabled: pitch_node.phase_alignment_enabled,
                    search_range_ratio: pitch_node.search_range_ratio,
                    correlation_length_ratio: pitch_node.correlation_length_ratio,
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
            AudioNode::WorldVocoder(vocoder_node) => (
                EffectNodeType::WorldVocoder {
                    pitch_mode: match vocoder_node.pitch_ui_mode {
                        crate::nodes::effects::PitchUIMode::Shift => {
                            crate::dsp::PitchMode::Shift(vocoder_node.pitch_semitones as f64)
                        }
                        crate::nodes::effects::PitchUIMode::Fixed => {
                            crate::dsp::PitchMode::Fixed(vocoder_node.fixed_pitch_hz as f64)
                        }
                    },
                    formant_semitones: vocoder_node.formant_semitones,
                    f0_method: vocoder_node.f0_method,
                    vocoder: vocoder_node.vocoder.clone(),
                },
                1,
            ),
            // 入出力ノードはここでは処理しない
            AudioNode::AudioInput(_) | AudioNode::AudioOutput(_) => return None,
        };

        // ソースバッファを収集（接続されたノードの出力バッファ）
        let mut source_buffers = Vec::new();
        for input_idx in 0..input_count {
            if let Some(buffer) = Self::get_source_buffer(snarl, node_id, input_idx) {
                source_buffers.push(buffer);
            }
        }

        // 出力バッファを取得
        let output_buffer = node.channel_buffer(0)?;

        Some(EffectNodeInfo {
            node_type,
            source_buffers,
            output_buffer,
        })
    }

    /// ストリームの開始/停止を管理
    fn manage_streams(&mut self, snarl: &mut Snarl<AudioNode>, audio_system: &mut AudioSystem) {
        let mut to_start_input: Vec<(NodeId, String, Vec<ChannelBuffer>, bool)> = Vec::new();
        let mut to_stop_input: Vec<NodeId> = Vec::new();
        let mut to_start_output: Vec<(NodeId, String, Vec<ChannelBuffer>, bool)> = Vec::new();
        let mut to_stop_output: Vec<(NodeId, OutputStreamId)> = Vec::new();
        // mono変更時にバッファ再構築が必要なノード（(NodeId, デバイスch数)）
        let mut mono_changed_inputs: Vec<NodeId> = Vec::new();
        let mut mono_changed_outputs: Vec<NodeId> = Vec::new();

        for (node_id, node) in snarl.node_ids() {
            match node {
                AudioNode::AudioInput(input_node) => {
                    if input_node.is_active && !self.active_nodes.contains_key(&node_id) {
                        to_start_input.push((
                            node_id,
                            input_node.device_name.clone(),
                            input_node.buffers.output_buffers.clone(),
                            input_node.mono,
                        ));
                    } else if input_node.is_active {
                        if let Some(ActiveNodeState::Input { mono }) =
                            self.active_nodes.get(&node_id)
                        {
                            if *mono != input_node.mono {
                                mono_changed_inputs.push(node_id);
                                to_stop_input.push(node_id);
                            }
                        }
                    } else if self.active_nodes.contains_key(&node_id) {
                        to_stop_input.push(node_id);
                    }
                }
                AudioNode::AudioOutput(output_node) => {
                    if output_node.is_active && !self.active_nodes.contains_key(&node_id) {
                        let buffers = output_node.buffers.output_buffers.clone();
                        to_start_output.push((
                            node_id,
                            output_node.device_name.clone(),
                            buffers,
                            output_node.mono,
                        ));
                    } else if output_node.is_active {
                        if let Some(ActiveNodeState::Output { stream_id, mono }) =
                            self.active_nodes.get(&node_id)
                        {
                            if *mono != output_node.mono {
                                mono_changed_outputs.push(node_id);
                                to_stop_output.push((node_id, *stream_id));
                            }
                        }
                    } else if !output_node.is_active {
                        if let Some(ActiveNodeState::Output { stream_id, .. }) =
                            self.active_nodes.get(&node_id)
                        {
                            to_stop_output.push((node_id, *stream_id));
                        }
                    }
                }
                _ => {}
            }
        }

        // Step 1: 停止処理
        for node_id in to_stop_input {
            audio_system.stop_input();
            self.active_nodes.remove(&node_id);
        }

        for (node_id, stream_id) in to_stop_output {
            audio_system.stop_output(stream_id);
            self.active_nodes.remove(&node_id);
        }

        // Step 2: mono変更されたノードのバッファを再構築してstart対象に追加
        for node_id in mono_changed_inputs {
            if let Some(node) = snarl.get_node_mut(node_id) {
                if let AudioNode::AudioInput(input_node) = node {
                    let ch = audio_system
                        .input_device_channels(&input_node.device_name)
                        .unwrap_or(2);
                    input_node.resize_buffers(ch);
                    to_start_input.push((
                        node_id,
                        input_node.device_name.clone(),
                        input_node.buffers.output_buffers.clone(),
                        input_node.mono,
                    ));
                }
            }
        }

        for node_id in mono_changed_outputs {
            if let Some(node) = snarl.get_node_mut(node_id) {
                if let AudioNode::AudioOutput(output_node) = node {
                    let ch = audio_system
                        .output_device_channels(&output_node.device_name)
                        .unwrap_or(2);
                    output_node.resize_buffers(ch);
                    to_start_output.push((
                        node_id,
                        output_node.device_name.clone(),
                        output_node.buffers.output_buffers.clone(),
                        output_node.mono,
                    ));
                }
            }
        }

        // Step 3: 開始処理
        for (node_id, device_name, buffers, mono) in to_start_input {
            match audio_system.start_input(&device_name, buffers) {
                Ok(channels) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes
                        .insert(node_id, ActiveNodeState::Input { mono });
                }
                Err(e) => {
                    eprintln!("Failed to start input: {}", e);
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
        }

        for (node_id, device_name, buffers, mono) in to_start_output {
            match audio_system.start_output(&device_name, buffers) {
                Ok((channels, stream_id)) => {
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_channels(channels);
                    }
                    self.active_nodes
                        .insert(node_id, ActiveNodeState::Output { stream_id, mono });
                }
                Err(e) => {
                    eprintln!("Failed to start output: {}", e);
                    if let Some(node) = snarl.get_node_mut(node_id) {
                        node.set_active(false);
                    }
                }
            }
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
