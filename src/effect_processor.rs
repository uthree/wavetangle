//! エフェクト処理スレッドモジュール
//!
//! リアルタイムオーディオレートでエフェクトノードを処理する専用スレッド

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dsp::{BiquadCoeffs, CompressorParams};
use crate::nodes::{ChannelBuffer, FilterType};

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
    PitchShift {
        semitones: f32,
        pitch_shifter: Arc<Mutex<crate::dsp::PitchShifter>>,
    },
    GraphicEq {
        graphic_eq: Arc<Mutex<crate::dsp::GraphicEq>>,
    },
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
                let block_size = ((sr * interval_ms as f32) / 1000.0).ceil() as usize;

                // すべてのソースバッファを収集（重複を除去）
                let mut all_source_buffers: Vec<ChannelBuffer> = Vec::new();
                for node_info in &nodes_snapshot {
                    for buf in &node_info.source_buffers {
                        // Arc::ptr_eqで重複チェック
                        if !all_source_buffers.iter().any(|b| Arc::ptr_eq(b, buf)) {
                            all_source_buffers.push(buf.clone());
                        }
                    }
                }

                // 最小利用可能サンプル数を確認
                let min_available = all_source_buffers
                    .iter()
                    .map(|buf| buf.lock().len())
                    .min()
                    .unwrap_or(0);

                // 処理するサンプル数を決定（利用可能な量を超えない）
                let actual_block_size = block_size.min(min_available);

                // データがある場合のみ処理
                if actual_block_size > 0 {
                    // ステップ1: ソースバッファからノードの入力バッファへデータをコピー
                    for node_info in &nodes_snapshot {
                        Self::copy_source_to_input(node_info, actual_block_size);
                    }

                    // ステップ2: すべてのノードを処理
                    for node_info in &nodes_snapshot {
                        Self::process_node(node_info, actual_block_size, sr);
                    }

                    // ステップ3: ソースバッファのデータを消費
                    for buf in &all_source_buffers {
                        buf.lock().consume(actual_block_size);
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

    /// ソースバッファからノードの入力バッファへデータをコピー
    fn copy_source_to_input(node_info: &EffectNodeInfo, block_size: usize) {
        for (source, input) in node_info
            .source_buffers
            .iter()
            .zip(node_info.input_buffers.iter())
        {
            // ソースバッファから読み取り（read - 状態を変更しない）
            let temp = source.lock().read(block_size);

            // 入力バッファに追加
            input.lock().push(&temp);
        }
    }

    /// 単一ノードを処理
    fn process_node(node_info: &EffectNodeInfo, block_size: usize, sample_rate: f32) {
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
            EffectNodeType::PitchShift {
                semitones,
                pitch_shifter,
            } => {
                let mut shifter = pitch_shifter.lock();
                shifter.set_semitones(*semitones);
                let mut output = vec![0.0; input_a.len()];
                shifter.process(&input_a, &mut output);
                output
            }
            EffectNodeType::GraphicEq { graphic_eq } => {
                let mut eq = graphic_eq.lock();
                let mut output = vec![0.0; input_a.len()];
                eq.process(&input_a, &mut output);
                output
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
