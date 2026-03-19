//! エフェクト処理スレッドモジュール
//!
//! リアルタイムオーディオレートでエフェクトノードを処理する専用スレッド

use std::collections::HashSet;
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
    /// 接続されたソースノードの出力バッファ（データ読み取り元）
    pub source_buffers: Vec<ChannelBuffer>,
    /// ノード自身の出力バッファ（処理結果の書き込み先）
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
    GraphicEq {
        graphic_eq: Arc<Mutex<crate::dsp::GraphicEq>>,
    },
    WorldVocoder {
        pitch_semitones: f32,
        formant_semitones: f32,
        harmonic_gains: Vec<f32>,
        use_harmonic_gains: bool,
        f0_method: crate::dsp::F0Method,
        vocoder: Arc<Mutex<crate::dsp::WorldVocoder>>,
    },
    /// データをそのまま出力にコピー（出力ノードへのルーティング用）
    Copy,
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

                let nodes_snapshot = nodes.lock().clone();
                let sr = *sample_rate.lock();

                let base_block_size = ((sr * interval_ms as f32) / 1000.0).ceil() as usize;
                let max_block_size = base_block_size * 8;

                // 消費対象のソースバッファとサイズを追跡
                let mut consumed: HashSet<usize> = HashSet::new();
                let mut consume_list: Vec<(ChannelBuffer, usize)> = Vec::new();

                // Phase 1: 全ノードを順に処理（トポロジカル順序）
                // read()は非破壊なので、同じバッファを複数ノードが安全に読み取れる
                for node_info in &nodes_snapshot {
                    let min_available = if node_info.source_buffers.is_empty() {
                        0
                    } else {
                        node_info
                            .source_buffers
                            .iter()
                            .map(|buf| buf.lock().len())
                            .min()
                            .unwrap_or(0)
                    };

                    let actual_block_size = min_available.min(max_block_size);

                    if actual_block_size > 0 {
                        Self::process_node(node_info, actual_block_size, sr);

                        // 消費予約（同じバッファの重複消費を防ぐ）
                        for source in &node_info.source_buffers {
                            let addr = Arc::as_ptr(source) as usize;
                            if !consumed.contains(&addr) {
                                consumed.insert(addr);
                                consume_list.push((source.clone(), actual_block_size));
                            }
                        }
                    }
                }

                // Phase 2: 全ソースバッファからデータを一括消費
                for (buffer, size) in &consume_list {
                    buffer.lock().consume(*size);
                }

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

    /// ソースバッファからデータを読み取る（非破壊）
    fn read_source(source_buffers: &[ChannelBuffer], index: usize, count: usize) -> Vec<f32> {
        source_buffers
            .get(index)
            .map(|b| b.lock().read(count))
            .unwrap_or_else(|| vec![0.0; count])
    }

    /// 単一ノードを処理
    fn process_node(node_info: &EffectNodeInfo, block_size: usize, sample_rate: f32) {
        let input_a = Self::read_source(&node_info.source_buffers, 0, block_size);

        let output_data: Vec<f32> = match &node_info.node_type {
            EffectNodeType::Copy => input_a,
            EffectNodeType::Gain { gain } => input_a.iter().map(|&s| s * gain).collect(),
            EffectNodeType::Add => {
                let input_b = Self::read_source(&node_info.source_buffers, 1, block_size);
                input_a
                    .iter()
                    .zip(input_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            EffectNodeType::Multiply => {
                let input_b = Self::read_source(&node_info.source_buffers, 1, block_size);
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
                {
                    let mut analyzer = analyzer.lock();
                    for &sample in &input_a {
                        analyzer.push_sample(sample);
                    }
                    let spectrum_data = analyzer.compute_spectrum();
                    let mut spec = spectrum.lock();
                    if spec.len() == spectrum_data.len() {
                        spec.copy_from_slice(&spectrum_data);
                    }
                }
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
                shifter.set_phase_alignment(crate::dsp::PhaseAlignmentParams {
                    enabled: *phase_alignment_enabled,
                    search_range_ratio: *search_range_ratio,
                    correlation_length_ratio: *correlation_length_ratio,
                });
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
            EffectNodeType::WorldVocoder {
                pitch_semitones,
                formant_semitones,
                harmonic_gains,
                use_harmonic_gains,
                f0_method,
                vocoder,
            } => {
                let mut vocoder = vocoder.lock();
                vocoder.set_pitch_shift(*pitch_semitones as f64);
                vocoder.set_f0_method(*f0_method);
                if *use_harmonic_gains {
                    vocoder.set_formant_shift(0.0);
                    let gains_f64: Vec<f64> = harmonic_gains.iter().map(|&g| g as f64).collect();
                    vocoder.set_harmonic_gains(&gains_f64);
                } else {
                    vocoder.set_formant_shift(*formant_semitones as f64);
                    vocoder.set_harmonic_gains(&[]);
                }
                let mut output = vec![0.0; input_a.len()];
                vocoder.process(&input_a, &mut output);
                output
            }
        };

        node_info.output_buffer.lock().push(&output_data);
    }
}

impl Drop for EffectProcessor {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Default for EffectProcessor {
    fn default() -> Self {
        Self::new(5)
    }
}
