use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use egui::{Color32, Ui};
use egui_snarl::NodeId;
use parking_lot::Mutex;

// サブモジュールの宣言
pub mod analyzer;
pub mod effects;
pub mod io;
pub mod math;

// FFTサイズの定義（公開）
/// FFTサイズ
pub const FFT_SIZE: usize = 1024;
/// GraphicEQ用のFFTサイズ（dsp.rsのEQ_FFT_SIZEと同じ値）
pub const EQ_FFT_SIZE: usize = 2048;

// サブモジュールからの公開re-export
pub use analyzer::SpectrumAnalyzerNode;
pub use effects::{
    CompressorNode, FilterNode, FilterType, GainNode, GraphicEqNode, PitchShiftNode,
};
pub use io::{AudioInputNode, AudioOutputNode};
pub use math::{AddNode, MultiplyNode};

/// UI描画時に必要なコンテキスト
pub struct NodeUIContext<'a> {
    /// 入力デバイス名のリスト
    pub input_devices: &'a [String],
    /// 出力デバイス名のリスト
    pub output_devices: &'a [String],
    /// ノードID(ウィジェットの一意識別用)
    pub node_id: NodeId,
}

/// HSVからRGBに変換
pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color32 {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// スペクトラムを折れ線グラフで表示（dBスケール、周波数目盛付き）
pub(crate) fn show_spectrum_line(ui: &mut Ui, plot_id: &str, spectrum: &Arc<Mutex<Vec<f32>>>) {
    use egui_plot::{Line, Plot, PlotPoints};

    let spectrum_data = spectrum.lock();
    let point_count = 100;
    let spectrum_len = spectrum_data.len();

    // 周波数範囲（対数スケール）
    const MIN_FREQ: f64 = 20.0;
    const MAX_FREQ: f64 = 20000.0;
    const MIN_DB: f64 = -80.0;
    const MAX_DB: f64 = 0.0;

    // X座標を対数周波数に変換
    let x_to_freq = |x: f64| -> f64 { MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(x) };

    // ラインデータを作成（X: 0-1の正規化値、Y: dB値）
    let points: Vec<[f64; 2]> = (0..point_count)
        .map(|i| {
            let x = i as f64 / point_count as f64;
            // 対数スケールでインデックスをマッピング（低周波を細かく表示）
            let freq_idx = (x.powf(2.0) * spectrum_len as f64) as usize;
            let freq_idx = freq_idx.min(spectrum_len.saturating_sub(1));

            // マグニチュードをdBに変換
            let magnitude = if freq_idx < spectrum_data.len() {
                spectrum_data[freq_idx]
            } else {
                0.0
            };
            let db = if magnitude > 1e-6 {
                20.0 * (magnitude as f64).log10()
            } else {
                MIN_DB
            };
            let db_clamped = db.clamp(MIN_DB, MAX_DB);

            [x, db_clamped]
        })
        .collect();

    drop(spectrum_data); // ロックを解放

    // egui_plotで折れ線グラフを表示
    Plot::new(plot_id)
        .height(100.0)
        .width(200.0)
        .show_axes([true, true])
        .show_grid([true, true])
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .include_x(0.0)
        .include_x(1.0)
        .include_y(MIN_DB)
        .include_y(MAX_DB)
        .x_axis_formatter(move |grid_mark, _range| {
            let freq = x_to_freq(grid_mark.value);
            if freq >= 1000.0 {
                format!("{:.0}k", freq / 1000.0)
            } else {
                format!("{:.0}", freq)
            }
        })
        .y_axis_formatter(|grid_mark, _range| format!("{:.0}dB", grid_mark.value))
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("spectrum", PlotPoints::from(points))
                    .color(Color32::from_rgb(100, 200, 100))
                    .width(2.0),
            );
        });
}

/// EQポイント間を線形補間してゲインを取得
pub(crate) fn interpolate_eq_gain(points: &[crate::dsp::EqPoint], freq: f32) -> f32 {
    if points.is_empty() {
        return 0.0;
    }

    // 周波数が最小より小さい場合
    if freq <= points[0].freq {
        return points[0].gain_db;
    }

    // 周波数が最大より大きい場合
    if freq >= points[points.len() - 1].freq {
        return points[points.len() - 1].gain_db;
    }

    // 補間
    for i in 0..points.len() - 1 {
        if freq >= points[i].freq && freq <= points[i + 1].freq {
            let log_freq = freq.ln();
            let log_freq_low = points[i].freq.ln();
            let log_freq_high = points[i + 1].freq.ln();

            let t = (log_freq - log_freq_low) / (log_freq_high - log_freq_low);
            return points[i].gain_db + t * (points[i + 1].gain_db - points[i].gain_db);
        }
    }

    0.0
}

/// オーディオバッファ - シンプルなFIFO設計
///
/// 設計原則:
/// - `push()`: データを追加（状態変更）
/// - `read()`: データのコピーを取得（状態変更なし）
/// - `consume()`: 先頭からデータを削除（状態変更）
///
/// 読み取りは状態を変更しないため、複数のコンシューマーが
/// 同じバッファから安全にデータを読み取ることができる。
#[derive(Clone)]
pub struct AudioBuffer {
    data: VecDeque<f32>,
    capacity: usize,
}

impl AudioBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// サンプルを末尾に追加（プロデューサー用）
    /// 容量を超える場合は古いデータを自動的に削除
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.data.len() >= self.capacity {
                self.data.pop_front();
            }
            self.data.push_back(sample);
        }
    }

    /// 先頭からn個のサンプルのコピーを取得（状態変更なし）
    /// バッファが足りない場合は0.0でパディング
    pub fn read(&self, count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            if i < self.data.len() {
                result.push(self.data[i]);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// 先頭からn個のサンプルを削除（消費済みとしてマーク）
    /// グラフプロセッサーが全てのコンシューマー処理後に呼ぶ
    pub fn consume(&mut self, count: usize) {
        let n = count.min(self.data.len());
        self.data.drain(0..n);
    }

    /// 利用可能なサンプル数
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// チャンネルバッファ - 1チャンネル分のオーディオバッファ
pub type ChannelBuffer = Arc<Mutex<AudioBuffer>>;

/// 新しいチャンネルバッファを作成
pub fn new_channel_buffer(capacity: usize) -> ChannelBuffer {
    Arc::new(Mutex::new(AudioBuffer::new(capacity)))
}

/// ノードのピンタイプ
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PinType {
    Audio,
    // 将来のMIDIサポート用
    // Midi,
}

/// ノードの種類を識別するenum
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    AudioInput,
    AudioOutput,
    Gain,
    Add,
    Multiply,
    Filter,
    SpectrumAnalyzer,
    Compressor,
    PitchShift,
    GraphicEq,
}

// ============================================================================
// 分離されたトレイト定義
// ============================================================================

/// ノードの基本情報を提供するトレイト
/// すべてのノードが実装する必要がある
pub trait NodeBase: Any {
    /// ノードの種類を取得
    fn node_type(&self) -> NodeType;

    /// ノードのタイトル
    fn title(&self) -> &str;

    /// Anyとしての不変参照を取得（ダウンキャスト用）
    fn as_any(&self) -> &dyn Any;

    /// Anyとしての可変参照を取得（ダウンキャスト用）
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// オーディオ入力ポートを持つノードのトレイト
/// 入力ピン（接続を受け取る側）の機能を提供
/// デフォルト実装は「入力なし」を表す
pub trait AudioInputPort {
    /// 入力ピンの数
    fn input_count(&self) -> usize {
        0
    }

    /// 入力ピンのタイプ
    fn input_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    /// 入力ピンの名前
    fn input_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    /// 指定入力ピンのバッファを取得
    fn input_buffer(&self, _index: usize) -> Option<ChannelBuffer> {
        None
    }
}

/// オーディオ出力ポートを持つノードのトレイト
/// 出力ピン（接続を送り出す側）の機能を提供
/// デフォルト実装は「出力なし」を表す
#[allow(dead_code)]
pub trait AudioOutputPort {
    /// 出力ピンの数
    fn output_count(&self) -> usize {
        0
    }

    /// 出力ピンのタイプ
    fn output_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    /// 出力ピンの名前
    fn output_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    /// 指定チャンネルの出力バッファを取得
    fn channel_buffer(&self, _channel: usize) -> Option<ChannelBuffer> {
        None
    }

    /// オーディオチャンネル数を取得
    fn channels(&self) -> u16 {
        0
    }

    /// オーディオチャンネル数を設定（バッファも再作成）
    fn set_channels(&mut self, _channels: u16) {}
}

/// ノードのUI描画機能を提供するトレイト
#[allow(dead_code)]
pub trait NodeUI {
    /// アクティブ状態を取得
    fn is_active(&self) -> bool;

    /// アクティブ状態を設定
    fn set_active(&mut self, active: bool);

    /// ノードボディのUIを描画
    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext);
}

/// すべてのオーディオノードが実装する複合トレイト
/// NodeBase + AudioInputPort + AudioOutputPort + NodeUI を組み合わせる
///
/// 各ノードは必要なトレイトのみを実装し、
/// 不要な機能はデフォルト実装を使用できる：
/// - AudioInputNode: AudioOutputPortのみ実装（出力専用）
/// - AudioOutputNode: AudioInputPortのみ実装（入力専用）
/// - エフェクトノード: 両方実装（入出力あり）
pub trait NodeBehavior: NodeBase + AudioInputPort + AudioOutputPort + NodeUI {}

/// NodeBehaviorのブランケット実装
/// NodeBase + AudioInputPort + AudioOutputPort + NodeUI を実装した型は
/// 自動的にNodeBehaviorを実装する
impl<T: NodeBase + AudioInputPort + AudioOutputPort + NodeUI> NodeBehavior for T {}

/// デフォルトのリングバッファサイズ（サンプル数）
/// 4096 = 約85ms @ 48kHz（レイテンシと安定性のバランス）
pub const DEFAULT_RING_BUFFER_SIZE: usize = 4096;

/// as_any()とas_any_mut()の実装を生成するマクロ
macro_rules! impl_as_any {
    () => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    };
}

// マクロを各サブモジュールで使えるようにエクスポート
pub(crate) use impl_as_any;

/// マルチチャンネルオーディオのチャンネル名
/// サラウンド5.1chまで対応
const CHANNEL_NAMES: &[&str] = &["L", "R", "C", "LFE", "SL", "SR"];

/// チャンネルインデックスからチャンネル名を取得
pub(crate) fn channel_name(index: usize) -> Option<&'static str> {
    CHANNEL_NAMES.get(index).copied()
}

/// チャンネルバッファのサイズを調整
/// 変更があった場合はtrueを返す
pub(crate) fn resize_channel_buffers(
    channel_buffers: &mut Vec<ChannelBuffer>,
    current_channels: u16,
    new_channels: u16,
) -> bool {
    if current_channels == new_channels {
        return false;
    }

    let old_len = channel_buffers.len();
    let new_len = new_channels as usize;

    if new_len > old_len {
        for _ in old_len..new_len {
            channel_buffers.push(new_channel_buffer(DEFAULT_RING_BUFFER_SIZE));
        }
    } else {
        channel_buffers.truncate(new_len);
    }
    true
}

// ============================================================================
// AudioNode - Box<dyn NodeBehavior> 型エイリアス
// ============================================================================

/// オーディオグラフのノード（トレイトオブジェクト）
pub type AudioNode = Box<dyn NodeBehavior>;

/// AudioInputノードを作成
pub fn new_audio_input(device_name: String, channels: u16) -> AudioNode {
    Box::new(AudioInputNode::new(device_name, channels))
}

/// AudioOutputノードを作成
pub fn new_audio_output(device_name: String, channels: u16) -> AudioNode {
    Box::new(AudioOutputNode::new(device_name, channels))
}

/// Gainノードを作成
pub fn new_gain() -> AudioNode {
    Box::new(GainNode::new())
}

/// Addノードを作成
pub fn new_add() -> AudioNode {
    Box::new(AddNode::new())
}

/// Multiplyノードを作成
pub fn new_multiply() -> AudioNode {
    Box::new(MultiplyNode::new())
}

/// Filterノードを作成
pub fn new_filter() -> AudioNode {
    Box::new(FilterNode::new())
}

/// SpectrumAnalyzerノードを作成
pub fn new_spectrum_analyzer() -> AudioNode {
    Box::new(SpectrumAnalyzerNode::new())
}

/// Compressorノードを作成
pub fn new_compressor() -> AudioNode {
    Box::new(CompressorNode::new())
}

/// PitchShiftノードを作成
pub fn new_pitch_shift() -> AudioNode {
    Box::new(PitchShiftNode::new())
}

/// GraphicEqノードを作成
pub fn new_graphic_eq() -> AudioNode {
    Box::new(GraphicEqNode::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_input_node_channels() {
        // 1チャンネル（モノラル）
        let node = AudioInputNode::new("test_device".to_string(), 1);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.output_count(), 1);

        // 2チャンネル（ステレオ）
        let node = AudioInputNode::new("test_device".to_string(), 2);
        assert_eq!(node.channels, 2);
        assert_eq!(node.channel_buffers.len(), 2);
        assert_eq!(node.output_count(), 2);

        // 6チャンネル（5.1ch）
        let node = AudioInputNode::new("test_device".to_string(), 6);
        assert_eq!(node.channels, 6);
        assert_eq!(node.channel_buffers.len(), 6);
        assert_eq!(node.output_count(), 6);

        // 0チャンネルは1に補正される
        let node = AudioInputNode::new("test_device".to_string(), 0);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.output_count(), 1);
    }

    #[test]
    fn test_audio_output_node_channels() {
        // 1チャンネル（モノラル）
        let node = AudioOutputNode::new("test_device".to_string(), 1);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.input_count(), 1);

        // 2チャンネル（ステレオ）
        let node = AudioOutputNode::new("test_device".to_string(), 2);
        assert_eq!(node.channels, 2);
        assert_eq!(node.channel_buffers.len(), 2);
        assert_eq!(node.input_count(), 2);

        // 0チャンネルは1に補正される
        let node = AudioOutputNode::new("test_device".to_string(), 0);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.input_count(), 1);
    }

    #[test]
    fn test_audio_input_node_resize() {
        let mut node = AudioInputNode::new("test_device".to_string(), 2);
        assert_eq!(node.output_count(), 2);

        // チャンネル数を増やす
        node.resize_buffers(4);
        assert_eq!(node.channels, 4);
        assert_eq!(node.channel_buffers.len(), 4);
        assert_eq!(node.output_count(), 4);

        // チャンネル数を減らす
        node.resize_buffers(1);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.output_count(), 1);
    }

    #[test]
    fn test_audio_output_node_resize() {
        let mut node = AudioOutputNode::new("test_device".to_string(), 2);
        assert_eq!(node.input_count(), 2);

        // チャンネル数を増やす
        node.resize_buffers(4);
        assert_eq!(node.channels, 4);
        assert_eq!(node.channel_buffers.len(), 4);
        assert_eq!(node.input_count(), 4);

        // チャンネル数を減らす
        node.resize_buffers(1);
        assert_eq!(node.channels, 1);
        assert_eq!(node.channel_buffers.len(), 1);
        assert_eq!(node.input_count(), 1);
    }

    #[test]
    fn test_node_type() {
        let input = AudioInputNode::new("test".to_string(), 2);
        assert_eq!(input.node_type(), NodeType::AudioInput);

        let output = AudioOutputNode::new("test".to_string(), 2);
        assert_eq!(output.node_type(), NodeType::AudioOutput);

        let gain = GainNode::new();
        assert_eq!(gain.node_type(), NodeType::Gain);

        let filter = FilterNode::new();
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_factory_functions() {
        let input = new_audio_input("test".to_string(), 1);
        assert_eq!(input.node_type(), NodeType::AudioInput);
        assert_eq!(input.output_count(), 1);

        let output = new_audio_output("test".to_string(), 2);
        assert_eq!(output.node_type(), NodeType::AudioOutput);
        assert_eq!(output.input_count(), 2);

        let gain = new_gain();
        assert_eq!(gain.node_type(), NodeType::Gain);

        let filter = new_filter();
        assert_eq!(filter.node_type(), NodeType::Filter);
    }

    #[test]
    fn test_pin_names() {
        let node = AudioInputNode::new("test".to_string(), 6);
        assert_eq!(node.output_pin_name(0), Some("L"));
        assert_eq!(node.output_pin_name(1), Some("R"));
        assert_eq!(node.output_pin_name(2), Some("C"));
        assert_eq!(node.output_pin_name(3), Some("LFE"));
        assert_eq!(node.output_pin_name(4), Some("SL"));
        assert_eq!(node.output_pin_name(5), Some("SR"));
        assert_eq!(node.output_pin_name(6), None);
    }

    #[test]
    fn test_audio_input_node_trait_separation() {
        // AudioInputNodeは出力ポートのみを持つ（入力ポートは持たない）
        let node = AudioInputNode::new("test".to_string(), 2);

        // 入力ポートはデフォルト実装（0個）
        assert_eq!(node.input_count(), 0);
        assert_eq!(node.input_pin_type(0), None);
        assert_eq!(node.input_pin_name(0), None);
        assert!(node.input_buffer(0).is_none());

        // 出力ポートはチャンネル数に応じて存在
        assert_eq!(node.output_count(), 2);
        assert_eq!(node.output_pin_type(0), Some(PinType::Audio));
        assert_eq!(node.output_pin_name(0), Some("L"));
        assert!(node.channel_buffer(0).is_some());
    }

    #[test]
    fn test_audio_output_node_trait_separation() {
        // AudioOutputNodeは入力ポートのみを持つ（出力ポートは持たない）
        let node = AudioOutputNode::new("test".to_string(), 2);

        // 入力ポートはチャンネル数に応じて存在
        assert_eq!(node.input_count(), 2);
        assert_eq!(node.input_pin_type(0), Some(PinType::Audio));
        assert_eq!(node.input_pin_name(0), Some("L"));

        // 出力ポートはデフォルト実装（0個）
        assert_eq!(node.output_count(), 0);
        assert_eq!(node.output_pin_type(0), None);
        assert_eq!(node.output_pin_name(0), None);
    }

    #[test]
    fn test_effect_node_trait_separation() {
        // エフェクトノードは入力と出力の両方を持つ
        let gain = GainNode::new();
        assert_eq!(gain.input_count(), 1);
        assert_eq!(gain.output_count(), 1);
        assert_eq!(gain.input_pin_type(0), Some(PinType::Audio));
        assert_eq!(gain.output_pin_type(0), Some(PinType::Audio));
        assert_eq!(gain.input_pin_name(0), Some("In"));
        assert_eq!(gain.output_pin_name(0), Some("Out"));

        let filter = FilterNode::new();
        assert_eq!(filter.input_count(), 1);
        assert_eq!(filter.output_count(), 1);

        let compressor = CompressorNode::new();
        assert_eq!(compressor.input_count(), 1);
        assert_eq!(compressor.output_count(), 1);

        let pitch_shift = PitchShiftNode::new();
        assert_eq!(pitch_shift.input_count(), 1);
        assert_eq!(pitch_shift.output_count(), 1);

        let graphic_eq = GraphicEqNode::new();
        assert_eq!(graphic_eq.input_count(), 1);
        assert_eq!(graphic_eq.output_count(), 1);

        let spectrum = SpectrumAnalyzerNode::new();
        assert_eq!(spectrum.input_count(), 1);
        assert_eq!(spectrum.output_count(), 1);
    }

    #[test]
    fn test_multi_input_node_trait_separation() {
        // AddNodeとMultiplyNodeは2入力1出力
        let add = AddNode::new();
        assert_eq!(add.input_count(), 2);
        assert_eq!(add.output_count(), 1);
        assert_eq!(add.input_pin_name(0), Some("A"));
        assert_eq!(add.input_pin_name(1), Some("B"));
        assert_eq!(add.output_pin_name(0), Some("Out"));
        assert!(add.input_buffer(0).is_some());
        assert!(add.input_buffer(1).is_some());
        assert!(add.input_buffer(2).is_none());

        let multiply = MultiplyNode::new();
        assert_eq!(multiply.input_count(), 2);
        assert_eq!(multiply.output_count(), 1);
        assert_eq!(multiply.input_pin_name(0), Some("A"));
        assert_eq!(multiply.input_pin_name(1), Some("B"));
    }

    // ========================================================================
    // AudioBuffer テスト
    // ========================================================================

    #[test]
    fn test_audio_buffer_push_and_read() {
        let mut buffer = super::AudioBuffer::new(10);

        // 空のバッファからの読み取り
        let data = buffer.read(5);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // データをプッシュ
        buffer.push(&[1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3);

        // 読み取り（状態は変わらない）
        let data = buffer.read(3);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3); // 状態変更なし

        // 2回目の読み取りも同じ結果
        let data = buffer.read(3);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_audio_buffer_consume() {
        let mut buffer = super::AudioBuffer::new(10);
        buffer.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // 2サンプル消費
        buffer.consume(2);
        assert_eq!(buffer.len(), 3);

        // 残りのデータを確認
        let data = buffer.read(3);
        assert_eq!(data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_audio_buffer_capacity_overflow() {
        let mut buffer = super::AudioBuffer::new(5);

        // 容量を超えてプッシュ
        buffer.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // 最新の5サンプルのみ保持される
        assert_eq!(buffer.len(), 5);
        let data = buffer.read(5);
        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_audio_buffer_multiple_consumers() {
        // 複数のコンシューマーが同じデータを読み取れることをテスト
        let buffer = super::new_channel_buffer(10);

        // データをプッシュ
        {
            let mut buf = buffer.lock();
            buf.push(&[1.0, 2.0, 3.0, 4.0]);
        }

        // コンシューマー1が読み取り
        let data1 = {
            let buf = buffer.lock();
            buf.read(4)
        };

        // コンシューマー2も同じデータを読み取れる
        let data2 = {
            let buf = buffer.lock();
            buf.read(4)
        };

        assert_eq!(data1, data2);
        assert_eq!(data1, vec![1.0, 2.0, 3.0, 4.0]);

        // 消費後はデータがなくなる
        {
            let mut buf = buffer.lock();
            buf.consume(4);
        }

        let data3 = {
            let buf = buffer.lock();
            buf.read(4)
        };
        assert_eq!(data3, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ========================================================================
    // 音声処理シミュレーションテスト
    // ========================================================================

    #[test]
    fn test_gain_processing_simulation() {
        // GainNodeの処理をシミュレート
        let gain_node = GainNode::new();
        let gain_value = 0.5f32;

        // 入力バッファにテストデータを書き込み
        let input_buffer = gain_node.input_buffer(0).unwrap();
        {
            let mut buf = input_buffer.lock();
            buf.push(&[1.0, 0.5, -0.5, -1.0]);
        }

        // ゲイン処理をシミュレート（EffectProcessorが行う処理）
        let processed: Vec<f32> = {
            let buf = input_buffer.lock();
            buf.read(4).iter().map(|&s| s * gain_value).collect()
        };

        assert_eq!(processed, vec![0.5, 0.25, -0.25, -0.5]);
    }

    #[test]
    fn test_add_processing_simulation() {
        // AddNodeの処理をシミュレート
        let add_node = AddNode::new();

        // 入力A
        let input_a = add_node.input_buffer(0).unwrap();
        {
            let mut buf = input_a.lock();
            buf.push(&[1.0, 2.0, 3.0, 4.0]);
        }

        // 入力B
        let input_b = add_node.input_buffer(1).unwrap();
        {
            let mut buf = input_b.lock();
            buf.push(&[0.5, 0.5, 0.5, 0.5]);
        }

        // 加算処理をシミュレート
        let processed: Vec<f32> = {
            let buf_a = input_a.lock();
            let buf_b = input_b.lock();
            let data_a = buf_a.read(4);
            let data_b = buf_b.read(4);
            data_a
                .iter()
                .zip(data_b.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };

        assert_eq!(processed, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_multiply_processing_simulation() {
        // MultiplyNodeの処理をシミュレート（リングモジュレーション）
        let multiply_node = MultiplyNode::new();

        // 入力A（キャリア信号）
        let input_a = multiply_node.input_buffer(0).unwrap();
        {
            let mut buf = input_a.lock();
            buf.push(&[1.0, 0.5, -0.5, -1.0]);
        }

        // 入力B（モジュレータ信号）
        let input_b = multiply_node.input_buffer(1).unwrap();
        {
            let mut buf = input_b.lock();
            buf.push(&[1.0, 1.0, 1.0, 0.5]);
        }

        // 乗算処理をシミュレート
        let processed: Vec<f32> = {
            let buf_a = input_a.lock();
            let buf_b = input_b.lock();
            let data_a = buf_a.read(4);
            let data_b = buf_b.read(4);
            data_a
                .iter()
                .zip(data_b.iter())
                .map(|(&a, &b)| a * b)
                .collect()
        };

        assert_eq!(processed, vec![1.0, 0.5, -0.5, -0.5]);
    }

    #[test]
    fn test_node_chain_simulation() {
        // ノードチェーン: Source -> Gain(0.5) -> Output
        // 実際のEffectProcessorの処理フローをシミュレート

        let source_buffer = super::new_channel_buffer(100);
        let gain_node = GainNode::new();
        let output_buffer = super::new_channel_buffer(100);
        let gain_value = 0.5f32;

        // ソースにテストデータを書き込み（サイン波をシミュレート）
        let test_signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        {
            let mut buf = source_buffer.lock();
            buf.push(&test_signal);
        }

        // Phase 1: スナップショット作成
        let snapshot = {
            let buf = source_buffer.lock();
            buf.read(64)
        };

        // Phase 2: 処理（ソース -> Gain入力）
        {
            let input_buffer = gain_node.input_buffer(0).unwrap();
            let mut buf = input_buffer.lock();
            buf.push(&snapshot);
        }

        // Gain処理
        let processed: Vec<f32> = {
            let input_buffer = gain_node.input_buffer(0).unwrap();
            let buf = input_buffer.lock();
            buf.read(64).iter().map(|&s| s * gain_value).collect()
        };

        // 出力バッファに書き込み
        {
            let mut buf = output_buffer.lock();
            buf.push(&processed);
        }

        // Phase 3: 消費
        {
            let mut buf = source_buffer.lock();
            buf.consume(64);
        }

        // 検証
        let output_data = {
            let buf = output_buffer.lock();
            buf.read(64)
        };

        // 各サンプルが正しくゲイン処理されていることを確認
        for (i, &sample) in output_data.iter().enumerate() {
            let expected = (i as f32 * 0.1).sin() * 0.5;
            assert!(
                (sample - expected).abs() < 1e-6,
                "Sample {} mismatch: {} vs {}",
                i,
                sample,
                expected
            );
        }
    }

    #[test]
    fn test_signal_branching_simulation() {
        // 分岐テスト: Source -> [Output1, Output2]
        // 同じソースから複数の出力へ分岐する場合のテスト

        let source_buffer = super::new_channel_buffer(100);
        let output1_buffer = super::new_channel_buffer(100);
        let output2_buffer = super::new_channel_buffer(100);

        // ソースにデータを書き込み
        {
            let mut buf = source_buffer.lock();
            buf.push(&[1.0, 2.0, 3.0, 4.0]);
        }

        // スナップショット方式で読み取り（各出力が同じデータを受け取る）
        let snapshot = {
            let buf = source_buffer.lock();
            buf.read(4)
        };

        // 出力1にコピー
        {
            let mut buf = output1_buffer.lock();
            buf.push(&snapshot);
        }

        // 出力2にもコピー（同じデータ）
        {
            let mut buf = output2_buffer.lock();
            buf.push(&snapshot);
        }

        // 消費は一度だけ
        {
            let mut buf = source_buffer.lock();
            buf.consume(4);
        }

        // 両方の出力が同じデータを受け取ったことを確認
        let data1 = {
            let buf = output1_buffer.lock();
            buf.read(4)
        };
        let data2 = {
            let buf = output2_buffer.lock();
            buf.read(4)
        };

        assert_eq!(data1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data2, vec![1.0, 2.0, 3.0, 4.0]);

        // ソースは空になっている
        assert_eq!(source_buffer.lock().len(), 0);
    }
}
