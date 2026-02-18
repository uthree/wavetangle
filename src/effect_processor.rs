//! エフェクト処理スレッドモジュール
//!
//! リアルタイムオーディオレートでエフェクトノードを処理する専用スレッド

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dsp::{BiquadCoeffs, CompressorParams};
use crate::nodes::{ChannelBuffer, FilterType};

/// ソースバッファのスナップショットを管理するための型
/// キー: バッファのアドレス、値: (スナップショットデータ, 消費済みフラグ)
type SourceSnapshot = HashMap<usize, (Vec<f32>, bool)>;

/// 処理対象のエフェクトノード情報
#[derive(Clone)]
pub struct EffectNodeInfo {
    /// ノードタイプ
    pub node_type: EffectNodeType,
    /// 接続されたソースノードの出力バッファ（データコピー元）
    pub source_buffers: Vec<ChannelBuffer>,
    /// ノード自身の入力バッファ（データコピー先、処理用）
    pub input_buffers: Vec<ChannelBuffer>,
    /// 出力バッファへの参照
    pub output_buffer: ChannelBuffer,
}

/// エフェクトノードタイプ
#[derive(Clone)]
pub enum EffectNodeType {
    Gain {
        gain: f32,
    },
    Add,
    Multiply,
    Filter {
        filter_type: FilterType,
        cutoff: f32,
        resonance: f32,
        state: Arc<Mutex<crate::dsp::BiquadState>>,
    },
    SpectrumAnalyzer {
        analyzer: Arc<Mutex<crate::dsp::SpectrumAnalyzer>>,
        spectrum: Arc<Mutex<Vec<f32>>>,
    },
    Compressor {
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
        makeup_gain: f32,
        state: Arc<Mutex<crate::dsp::CompressorState>>,
    },
    WsolaPitchShift {
        semitones: f32,
        phase_alignment_enabled: bool,
        search_range_ratio: f32,
        correlation_length_ratio: f32,
        pitch_shifter: Arc<Mutex<crate::dsp::PitchShifter>>,
    },
    TdPsolaPitchShift {
        pitch_shift: f32,
        formant_shift: f32,
        td_psola: Arc<Mutex<crate::dsp::TdPsolaPitchShifter>>,
    },
    GraphicEq {
        graphic_eq: Arc<Mutex<crate::dsp::GraphicEq>>,
    },
    /// パススルー - データをそのまま出力にコピー（出力ノードへのルーティング用）
    PassThrough,
}

/// エフェクトプロセッサー
/// 専用スレッドでエフェクトノードを処理
pub struct EffectProcessor {
    /// 処理スレッドハンドル
    thread_handle: Option<JoinHandle<()>>,
    /// 実行中フラグ
    running: Arc<AtomicBool>,
    /// 処理対象ノードリスト（スレッド間で共有）
    nodes: Arc<Mutex<Vec<EffectNodeInfo>>>,
    /// サンプルレート
    sample_rate: Arc<Mutex<f32>>,
    /// 処理間隔（ミリ秒）
    process_interval_ms: u64,
}

impl EffectProcessor {
    /// 新しいエフェクトプロセッサーを作成
    pub fn new(process_interval_ms: u64) -> Self {
        Self {
            thread_handle: None,
            running: Arc::new(AtomicBool::new(false)),
            nodes: Arc::new(Mutex::new(Vec::new())),
            sample_rate: Arc::new(Mutex::new(44100.0)),
            process_interval_ms,
        }
    }

    /// サンプルレートを設定
    pub fn set_sample_rate(&self, rate: f32) {
        *self.sample_rate.lock() = rate;
    }

    /// 処理対象ノードを更新
    pub fn update_nodes(&self, nodes: Vec<EffectNodeInfo>) {
        *self.nodes.lock() = nodes;
    }

    /// ノードリストをクリア
    pub fn clear_nodes(&self) {
        self.nodes.lock().clear();
    }

    /// 処理スレッドを開始
    pub fn start(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            return;
        }

        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();
        let nodes = self.nodes.clone();
        let sample_rate = self.sample_rate.clone();
        let interval_ms = self.process_interval_ms;

        let handle = thread::spawn(move || {
            let interval = Duration::from_millis(interval_ms);

            while running.load(Ordering::SeqCst) {
                let start = Instant::now();

                // ノードを処理
                let nodes_snapshot = nodes.lock().clone();
                let sr = *sample_rate.lock();

                // サンプルレートと処理間隔からブロックサイズを計算
                // 48kHz, 2ms -> 96 samples
                let base_block_size = ((sr * interval_ms as f32) / 1000.0).ceil() as usize;
                let max_block_size = base_block_size * 8; // 最大8倍まで

                // 各ノードを順番に処理（トポロジカル順序で渡されていると仮定）
                // 分岐時のデータ消費問題を防ぐため、スナップショット方式で処理

                // Phase 1: 全ソースバッファのスナップショットを作成
                // 同じソースバッファを複数ノードが参照している場合でも
                // データを一度だけ読み取り、共有する
                let mut source_snapshots: SourceSnapshot = HashMap::new();

                for node_info in &nodes_snapshot {
                    for source in &node_info.source_buffers {
                        let addr = Arc::as_ptr(source) as usize;
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            source_snapshots.entry(addr)
                        {
                            let buf = source.lock();
                            let available = buf.len().min(max_block_size);
                            if available > 0 {
                                let data = buf.read(available);
                                e.insert((data, false));
                            }
                        }
                    }
                }

                // Phase 2: 各ノードをスナップショットを使って処理
                for node_info in &nodes_snapshot {
                    // このノードのソースバッファの最小利用可能サンプル数を確認
                    let min_available = if node_info.source_buffers.is_empty() {
                        0
                    } else {
                        node_info
                            .source_buffers
                            .iter()
                            .map(|buf| {
                                let addr = Arc::as_ptr(buf) as usize;
                                source_snapshots
                                    .get(&addr)
                                    .map(|(data, _)| data.len())
                                    .unwrap_or(0)
                            })
                            .min()
                            .unwrap_or(0)
                    };

                    // 処理するサンプル数を決定
                    let actual_block_size = min_available.min(max_block_size);

                    // データがある場合のみ処理
                    if actual_block_size > 0 {
                        // スナップショットからノードの入力バッファへデータをコピー
                        Self::copy_source_to_input_from_snapshot(
                            node_info,
                            actual_block_size,
                            &source_snapshots,
                        );
                        // ノードを処理
                        Self::process_node(node_info, actual_block_size, sr);
                    }
                }

                // Phase 3: 使用したソースバッファからデータを消費
                for node_info in &nodes_snapshot {
                    for source in &node_info.source_buffers {
                        let addr = Arc::as_ptr(source) as usize;
                        if let Some((data, consumed)) = source_snapshots.get_mut(&addr) {
                            if !*consumed {
                                // まだ消費されていない場合のみ消費
                                source.lock().consume(data.len());
                                *consumed = true;
                            }
                        }
                    }
                }

                // 次の処理まで待機
                let elapsed = start.elapsed();
                if elapsed < interval {
                    thread::sleep(interval - elapsed);
                }
            }
        });

        self.thread_handle = Some(handle);
    }

    /// 処理スレッドを停止
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// スレッドが実行中か
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// スナップショットからノードの入力バッファへデータをコピー
    /// 分岐時に同じソースバッファが複数ノードで参照されていても
    /// 事前に取得したスナップショットから読み取るため、データの消失を防ぐ
    /// PassThroughの場合は直接出力バッファにコピー（入力バッファを経由しない）
    fn copy_source_to_input_from_snapshot(
        node_info: &EffectNodeInfo,
        block_size: usize,
        snapshots: &SourceSnapshot,
    ) {
        // PassThroughの場合は直接出力バッファにコピー
        if matches!(node_info.node_type, EffectNodeType::PassThrough) {
            if let Some(source) = node_info.source_buffers.first() {
                let addr = Arc::as_ptr(source) as usize;
                if let Some((data, _)) = snapshots.get(&addr) {
                    let len = block_size.min(data.len());
                    node_info.output_buffer.lock().push(&data[..len]);
                }
            }
            return;
        }

        // 通常のエフェクトノードの場合は入力バッファにコピー
        for (source, input) in node_info
            .source_buffers
            .iter()
            .zip(node_info.input_buffers.iter())
        {
            let addr = Arc::as_ptr(source) as usize;
            if let Some((data, _)) = snapshots.get(&addr) {
                let len = block_size.min(data.len());
                input.lock().push(&data[..len]);
            }
        }
    }

    /// 単一ノードを処理
    fn process_node(node_info: &EffectNodeInfo, block_size: usize, sample_rate: f32) {
        // PassThroughの場合は処理をスキップ（copy_source_to_inputで直接コピー済み）
        if matches!(node_info.node_type, EffectNodeType::PassThrough) {
            return;
        }

        // 入力データを読み取り（ノード自身の入力バッファから）
        let input_a = Self::read_from_input_buffer(&node_info.input_buffers, 0, block_size);

        // ノードタイプに応じた処理
        let output_data: Vec<f32> = match &node_info.node_type {
            EffectNodeType::Gain { gain } => input_a.iter().map(|&s| s * gain).collect(),
            EffectNodeType::Add => {
                let input_b = Self::read_from_input_buffer(&node_info.input_buffers, 1, block_size);
                input_a
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            EffectNodeType::Multiply => {
                let input_b = Self::read_from_input_buffer(&node_info.input_buffers, 1, block_size);
                input_a
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a * b)
                    .collect()
            }
            EffectNodeType::Filter {
                filter_type,
                cutoff,
                resonance,
                state,
            } => {
                let coeffs =
                    BiquadCoeffs::from_filter_type(*filter_type, sample_rate, *cutoff, *resonance);
                let mut state = state.lock();
                input_a.iter().map(|&s| state.process(s, &coeffs)).collect()
            }
            EffectNodeType::SpectrumAnalyzer { analyzer, spectrum } => {
                // FFT用にサンプルを蓄積
                {
                    let mut analyzer = analyzer.lock();
                    for &sample in &input_a {
                        analyzer.push_sample(sample);
                    }
                    // スペクトラムを計算
                    let spectrum_data = analyzer.compute_spectrum();
                    let mut spec = spectrum.lock();
                    if spec.len() == spectrum_data.len() {
                        spec.copy_from_slice(&spectrum_data);
                    }
                }
                // パススルー
                input_a
            }
            EffectNodeType::Compressor {
                threshold,
                ratio,
                attack,
                release,
                makeup_gain,
                state,
            } => {
                let params = CompressorParams {
                    threshold_db: *threshold,
                    ratio: *ratio,
                    attack_ms: *attack,
                    release_ms: *release,
                    makeup_db: *makeup_gain,
                    sample_rate,
                };
                let mut state = state.lock();
                input_a.iter().map(|&s| state.process(s, &params)).collect()
            }
            EffectNodeType::WsolaPitchShift {
                semitones,
                phase_alignment_enabled,
                search_range_ratio,
                correlation_length_ratio,
                pitch_shifter,
            } => {
                let mut shifter = pitch_shifter.lock();
                shifter.set_semitones(*semitones);
                // 位相アラインメントパラメータを更新
                shifter.set_phase_alignment(crate::dsp::PhaseAlignmentParams {
                    enabled: *phase_alignment_enabled,
                    search_range_ratio: *search_range_ratio,
                    correlation_length_ratio: *correlation_length_ratio,
                });
                let mut output = vec![0.0; input_a.len()];
                shifter.process(&input_a, &mut output);
                output
            }
            EffectNodeType::TdPsolaPitchShift {
                pitch_shift,
                formant_shift,
                td_psola,
            } => {
                let mut processor = td_psola.lock();
                processor.set_pitch_shift(*pitch_shift);
                processor.set_formant_shift(*formant_shift);

                let mut output = vec![0.0; input_a.len()];
                processor.process(&input_a, &mut output);
                output
            }
            EffectNodeType::GraphicEq { graphic_eq } => {
                let mut eq = graphic_eq.lock();
                let mut output = vec![0.0; input_a.len()];
                eq.process(&input_a, &mut output);
                output
            }
            EffectNodeType::PassThrough => {
                // この分岐には到達しない（早期リターン済み）
                unreachable!("PassThrough is handled in copy_source_to_input")
            }
        };

        // 出力バッファに追加
        node_info.output_buffer.lock().push(&output_data);
    }

    /// 入力バッファからサンプルを読み取り、消費する
    fn read_from_input_buffer(buffers: &[ChannelBuffer], index: usize, count: usize) -> Vec<f32> {
        if let Some(buffer) = buffers.get(index) {
            let mut buf = buffer.lock();
            let samples = buf.read(count);
            buf.consume(count);
            samples
        } else {
            vec![0.0; count]
        }
    }
}

impl Drop for EffectProcessor {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Default for EffectProcessor {
    fn default() -> Self {
        Self::new(5) // デフォルト5ms間隔
    }
}
