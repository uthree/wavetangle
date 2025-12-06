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
    /// 入力バッファへの参照
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
            let block_size = 256;
            let interval = Duration::from_millis(interval_ms);

            while running.load(Ordering::SeqCst) {
                let start = Instant::now();

                // ノードを処理
                let nodes_snapshot = nodes.lock().clone();
                let sr = *sample_rate.lock();

                // すべての入力バッファを収集（重複を除去）
                let mut input_buffers: Vec<ChannelBuffer> = Vec::new();
                for node_info in &nodes_snapshot {
                    for buf in &node_info.input_buffers {
                        // Arc::ptr_eqで重複チェック
                        if !input_buffers.iter().any(|b| Arc::ptr_eq(b, buf)) {
                            input_buffers.push(buf.clone());
                        }
                    }
                }

                // すべてのノードを処理（peekで読み取り）
                for node_info in &nodes_snapshot {
                    Self::process_node(node_info, block_size, sr);
                }

                // 処理後、すべての入力バッファを進める
                for buf in &input_buffers {
                    buf.lock().advance_read(block_size);
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

    /// 単一ノードを処理
    fn process_node(node_info: &EffectNodeInfo, block_size: usize, sample_rate: f32) {
        // 入力データを収集
        let input_a = Self::read_from_buffer(&node_info.input_buffers, 0, block_size);

        // ノードタイプに応じた処理
        let output_data: Vec<f32> = match &node_info.node_type {
            EffectNodeType::Gain { gain } => input_a.iter().map(|&s| s * gain).collect(),
            EffectNodeType::Add => {
                let input_b = Self::read_from_buffer(&node_info.input_buffers, 1, block_size);
                input_a
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            EffectNodeType::Multiply => {
                let input_b = Self::read_from_buffer(&node_info.input_buffers, 1, block_size);
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
        };

        // 出力バッファに書き込み
        let mut output = node_info.output_buffer.lock();
        output.write(&output_data);
    }

    /// バッファからサンプルを読み取り（peekを使用 - 読み取り位置を進めない）
    fn read_from_buffer(buffers: &[ChannelBuffer], index: usize, count: usize) -> Vec<f32> {
        let mut samples = vec![0.0; count];
        if let Some(buffer) = buffers.get(index) {
            let buf = buffer.lock();
            buf.peek(&mut samples);
        }
        samples
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
