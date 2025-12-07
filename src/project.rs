//! プロジェクトの保存・読み込みモジュール

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use egui_snarl::Snarl;
use serde::{Deserialize, Serialize};

use crate::dsp::EqPoint;
use crate::nodes::{
    AddNode, AudioInputNode, AudioNode, AudioOutputNode, CompressorNode, FilterNode, FilterType,
    GainNode, GraphicEqNode, MultiplyNode, NodeType, PitchShiftNode, SpectrumAnalyzerNode,
};

/// プロジェクトファイルのバージョン
const PROJECT_VERSION: u32 = 1;

// デフォルト値関数（後方互換性のため）
fn default_channels() -> u16 {
    2
}
fn default_phase_alignment_enabled() -> bool {
    true
}
fn default_search_range_ratio() -> f32 {
    0.5
}
fn default_correlation_length_ratio() -> f32 {
    0.75
}

/// ノードの位置情報
#[derive(Clone, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
}

/// 接続情報
#[derive(Clone, Serialize, Deserialize)]
pub struct Connection {
    /// 出力ノードのインデックス
    pub from_node: usize,
    /// 出力ピンのインデックス
    pub from_output: usize,
    /// 入力ノードのインデックス
    pub to_node: usize,
    /// 入力ピンのインデックス
    pub to_input: usize,
}

/// シリアライズ可能なノードデータ
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SavedNode {
    AudioInput {
        device_name: String,
        #[serde(default = "default_channels")]
        channels: u16,
        show_spectrum: bool,
    },
    AudioOutput {
        device_name: String,
        #[serde(default = "default_channels")]
        channels: u16,
        show_spectrum: bool,
    },
    Gain {
        gain: f32,
    },
    Add,
    Multiply,
    Filter {
        filter_type: SavedFilterType,
        cutoff: f32,
        resonance: f32,
    },
    SpectrumAnalyzer,
    Compressor {
        threshold: f32,
        ratio: f32,
        attack: f32,
        release: f32,
        makeup_gain: f32,
    },
    PitchShift {
        semitones: f32,
        grain_size: usize,
        num_grains: usize,
        #[serde(default = "default_phase_alignment_enabled")]
        phase_alignment_enabled: bool,
        #[serde(default = "default_search_range_ratio")]
        search_range_ratio: f32,
        #[serde(default = "default_correlation_length_ratio")]
        correlation_length_ratio: f32,
    },
    GraphicEq {
        eq_points: Vec<SavedEqPoint>,
        show_spectrum: bool,
    },
}

/// シリアライズ可能なフィルタータイプ
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum SavedFilterType {
    Low,
    High,
    Band,
}

impl From<FilterType> for SavedFilterType {
    fn from(ft: FilterType) -> Self {
        match ft {
            FilterType::Low => SavedFilterType::Low,
            FilterType::High => SavedFilterType::High,
            FilterType::Band => SavedFilterType::Band,
        }
    }
}

impl From<SavedFilterType> for FilterType {
    fn from(ft: SavedFilterType) -> Self {
        match ft {
            SavedFilterType::Low => FilterType::Low,
            SavedFilterType::High => FilterType::High,
            SavedFilterType::Band => FilterType::Band,
        }
    }
}

/// シリアライズ可能なEQポイント
#[derive(Clone, Serialize, Deserialize)]
pub struct SavedEqPoint {
    pub freq: f32,
    pub gain_db: f32,
}

impl From<&EqPoint> for SavedEqPoint {
    fn from(p: &EqPoint) -> Self {
        Self {
            freq: p.freq,
            gain_db: p.gain_db,
        }
    }
}

impl From<&SavedEqPoint> for EqPoint {
    fn from(p: &SavedEqPoint) -> Self {
        EqPoint::new(p.freq, p.gain_db)
    }
}

/// プロジェクトファイル構造
#[derive(Serialize, Deserialize)]
pub struct ProjectFile {
    /// バージョン
    pub version: u32,
    /// ノードリスト
    pub nodes: Vec<SavedNode>,
    /// ノードの位置情報
    pub positions: Vec<NodePosition>,
    /// 接続リスト
    pub connections: Vec<Connection>,
}

impl ProjectFile {
    /// Snarlからプロジェクトファイルを作成
    pub fn from_snarl(snarl: &Snarl<AudioNode>) -> Self {
        let mut nodes = Vec::new();
        let mut positions = Vec::new();
        let mut node_id_to_index: HashMap<egui_snarl::NodeId, usize> = HashMap::new();

        // ノードを収集
        for (node_id, node) in snarl.node_ids() {
            let index = nodes.len();
            node_id_to_index.insert(node_id, index);
            let saved_node = match node.node_type() {
                NodeType::AudioInput => {
                    let n = node.as_any().downcast_ref::<AudioInputNode>().unwrap();
                    SavedNode::AudioInput {
                        device_name: n.device_name.clone(),
                        channels: n.channels,
                        show_spectrum: n.show_spectrum,
                    }
                }
                NodeType::AudioOutput => {
                    let n = node.as_any().downcast_ref::<AudioOutputNode>().unwrap();
                    SavedNode::AudioOutput {
                        device_name: n.device_name.clone(),
                        channels: n.channels,
                        show_spectrum: n.show_spectrum,
                    }
                }
                NodeType::Gain => {
                    let n = node.as_any().downcast_ref::<GainNode>().unwrap();
                    SavedNode::Gain { gain: n.gain }
                }
                NodeType::Add => SavedNode::Add,
                NodeType::Multiply => SavedNode::Multiply,
                NodeType::Filter => {
                    let n = node.as_any().downcast_ref::<FilterNode>().unwrap();
                    SavedNode::Filter {
                        filter_type: n.filter_type.into(),
                        cutoff: n.cutoff,
                        resonance: n.resonance,
                    }
                }
                NodeType::SpectrumAnalyzer => SavedNode::SpectrumAnalyzer,
                NodeType::Compressor => {
                    let n = node.as_any().downcast_ref::<CompressorNode>().unwrap();
                    SavedNode::Compressor {
                        threshold: n.threshold,
                        ratio: n.ratio,
                        attack: n.attack,
                        release: n.release,
                        makeup_gain: n.makeup_gain,
                    }
                }
                NodeType::PitchShift => {
                    let n = node.as_any().downcast_ref::<PitchShiftNode>().unwrap();
                    SavedNode::PitchShift {
                        semitones: n.semitones,
                        grain_size: n.grain_size,
                        num_grains: n.num_grains,
                        phase_alignment_enabled: n.phase_alignment_enabled,
                        search_range_ratio: n.search_range_ratio,
                        correlation_length_ratio: n.correlation_length_ratio,
                    }
                }
                NodeType::GraphicEq => {
                    let n = node.as_any().downcast_ref::<GraphicEqNode>().unwrap();
                    SavedNode::GraphicEq {
                        eq_points: n.eq_points.iter().map(SavedEqPoint::from).collect(),
                        show_spectrum: n.show_spectrum,
                    }
                }
            };

            nodes.push(saved_node);

            // 位置情報を取得
            let info = snarl.get_node_info(node_id).unwrap();
            positions.push(NodePosition {
                x: info.pos.x,
                y: info.pos.y,
            });
        }

        // 接続を収集
        let mut connections = Vec::new();
        for (out_pin, in_pin) in snarl.wires() {
            if let (Some(&from_idx), Some(&to_idx)) = (
                node_id_to_index.get(&out_pin.node),
                node_id_to_index.get(&in_pin.node),
            ) {
                connections.push(Connection {
                    from_node: from_idx,
                    from_output: out_pin.output,
                    to_node: to_idx,
                    to_input: in_pin.input,
                });
            }
        }

        Self {
            version: PROJECT_VERSION,
            nodes,
            positions,
            connections,
        }
    }

    /// プロジェクトファイルからSnarlを再構築
    pub fn to_snarl(&self) -> Snarl<AudioNode> {
        let mut snarl = Snarl::new();
        let mut node_ids = Vec::new();

        // ノードを作成
        for (i, saved_node) in self.nodes.iter().enumerate() {
            let pos = self
                .positions
                .get(i)
                .map(|p| egui::Pos2::new(p.x, p.y))
                .unwrap_or(egui::Pos2::new(100.0 + i as f32 * 200.0, 100.0));

            let audio_node: AudioNode = match saved_node {
                SavedNode::AudioInput {
                    device_name,
                    channels,
                    show_spectrum,
                } => {
                    let mut node = AudioInputNode::new(device_name.clone(), *channels);
                    node.show_spectrum = *show_spectrum;
                    Box::new(node)
                }
                SavedNode::AudioOutput {
                    device_name,
                    channels,
                    show_spectrum,
                } => {
                    let mut node = AudioOutputNode::new(device_name.clone(), *channels);
                    node.show_spectrum = *show_spectrum;
                    Box::new(node)
                }
                SavedNode::Gain { gain } => {
                    let mut node = GainNode::new();
                    node.gain = *gain;
                    Box::new(node)
                }
                SavedNode::Add => Box::new(AddNode::new()),
                SavedNode::Multiply => Box::new(MultiplyNode::new()),
                SavedNode::Filter {
                    filter_type,
                    cutoff,
                    resonance,
                } => {
                    let mut node = FilterNode::new();
                    node.filter_type = (*filter_type).into();
                    node.cutoff = *cutoff;
                    node.resonance = *resonance;
                    Box::new(node)
                }
                SavedNode::SpectrumAnalyzer => Box::new(SpectrumAnalyzerNode::new()),
                SavedNode::Compressor {
                    threshold,
                    ratio,
                    attack,
                    release,
                    makeup_gain,
                } => {
                    let mut node = CompressorNode::new();
                    node.threshold = *threshold;
                    node.ratio = *ratio;
                    node.attack = *attack;
                    node.release = *release;
                    node.makeup_gain = *makeup_gain;
                    Box::new(node)
                }
                SavedNode::PitchShift {
                    semitones,
                    grain_size,
                    num_grains,
                    phase_alignment_enabled,
                    search_range_ratio,
                    correlation_length_ratio,
                } => {
                    let mut node = PitchShiftNode::new();
                    node.semitones = *semitones;
                    node.grain_size = *grain_size;
                    node.num_grains = *num_grains;
                    node.phase_alignment_enabled = *phase_alignment_enabled;
                    node.search_range_ratio = *search_range_ratio;
                    node.correlation_length_ratio = *correlation_length_ratio;
                    // PitchShifterにもパラメータを反映
                    if let Some(mut shifter) = node.pitch_shifter.try_lock() {
                        shifter.set_grain_size(*grain_size);
                        shifter.set_num_grains(*num_grains);
                        shifter.set_phase_alignment(crate::dsp::PhaseAlignmentParams {
                            enabled: *phase_alignment_enabled,
                            search_range_ratio: *search_range_ratio,
                            correlation_length_ratio: *correlation_length_ratio,
                        });
                    }
                    Box::new(node)
                }
                SavedNode::GraphicEq {
                    eq_points,
                    show_spectrum,
                } => {
                    let mut node = GraphicEqNode::new();
                    node.eq_points = eq_points.iter().map(EqPoint::from).collect();
                    node.show_spectrum = *show_spectrum;
                    // EQカーブを更新
                    node.graphic_eq.lock().update_curve(&node.eq_points);
                    Box::new(node)
                }
            };

            let node_id = snarl.insert_node(pos, audio_node);
            node_ids.push(node_id);
        }

        // 接続を復元
        for conn in &self.connections {
            if let (Some(&from_id), Some(&to_id)) =
                (node_ids.get(conn.from_node), node_ids.get(conn.to_node))
            {
                let out_pin = egui_snarl::OutPinId {
                    node: from_id,
                    output: conn.from_output,
                };
                let in_pin = egui_snarl::InPinId {
                    node: to_id,
                    input: conn.to_input,
                };
                snarl.connect(out_pin, in_pin);
            }
        }

        snarl
    }

    /// ファイルに保存
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())
    }

    /// ファイルから読み込み
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let json = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::NodeBehavior;

    #[test]
    fn test_saved_node_serialization_with_channels() {
        // AudioInputノードのシリアライズ
        let saved = SavedNode::AudioInput {
            device_name: "Test Device".to_string(),
            channels: 1,
            show_spectrum: true,
        };
        let json = serde_json::to_string(&saved).unwrap();
        assert!(json.contains("\"channels\":1"));

        // デシリアライズ
        let restored: SavedNode = serde_json::from_str(&json).unwrap();
        if let SavedNode::AudioInput {
            device_name,
            channels,
            show_spectrum,
        } = restored
        {
            assert_eq!(device_name, "Test Device");
            assert_eq!(channels, 1);
            assert!(show_spectrum);
        } else {
            panic!("Expected AudioInput");
        }
    }

    #[test]
    fn test_saved_node_backward_compatibility() {
        // 古い形式（channelsフィールドなし）のJSONをデシリアライズ
        let old_json = r#"{"type":"AudioInput","device_name":"Test Device","show_spectrum":false}"#;
        let restored: SavedNode = serde_json::from_str(old_json).unwrap();
        if let SavedNode::AudioInput {
            device_name,
            channels,
            show_spectrum,
        } = restored
        {
            assert_eq!(device_name, "Test Device");
            assert_eq!(channels, 2); // デフォルト値
            assert!(!show_spectrum);
        } else {
            panic!("Expected AudioInput");
        }

        // AudioOutputも同様にテスト
        let old_json = r#"{"type":"AudioOutput","device_name":"Test Output","show_spectrum":true}"#;
        let restored: SavedNode = serde_json::from_str(old_json).unwrap();
        if let SavedNode::AudioOutput {
            device_name,
            channels,
            show_spectrum,
        } = restored
        {
            assert_eq!(device_name, "Test Output");
            assert_eq!(channels, 2); // デフォルト値
            assert!(show_spectrum);
        } else {
            panic!("Expected AudioOutput");
        }
    }

    #[test]
    fn test_project_file_roundtrip() {
        use egui_snarl::Snarl;

        // Snarlにノードを追加
        let mut snarl: Snarl<AudioNode> = Snarl::new();
        let input_node = AudioInputNode::new("Input Device".to_string(), 1);
        let output_node = AudioOutputNode::new("Output Device".to_string(), 2);

        let input_id = snarl.insert_node(egui::Pos2::new(100.0, 100.0), Box::new(input_node));
        let output_id = snarl.insert_node(egui::Pos2::new(300.0, 100.0), Box::new(output_node));

        // 接続を追加
        snarl.connect(
            egui_snarl::OutPinId {
                node: input_id,
                output: 0,
            },
            egui_snarl::InPinId {
                node: output_id,
                input: 0,
            },
        );

        // プロジェクトファイルに変換
        let project = ProjectFile::from_snarl(&snarl);

        // JSONにシリアライズしてデシリアライズ
        let json = serde_json::to_string(&project).unwrap();
        let restored_project: ProjectFile = serde_json::from_str(&json).unwrap();

        // Snarlに戻す
        let restored_snarl = restored_project.to_snarl();

        // ノード数を確認
        assert_eq!(restored_snarl.node_ids().count(), 2);

        // チャンネル数を確認
        for (_, node) in restored_snarl.node_ids() {
            match node.node_type() {
                NodeType::AudioInput => {
                    let n = node.as_any().downcast_ref::<AudioInputNode>().unwrap();
                    assert_eq!(n.channels, 1);
                    assert_eq!(n.output_count(), 1);
                }
                NodeType::AudioOutput => {
                    let n = node.as_any().downcast_ref::<AudioOutputNode>().unwrap();
                    assert_eq!(n.channels, 2);
                    assert_eq!(n.input_count(), 2);
                }
                _ => panic!("Unexpected node type"),
            }
        }
    }
}
