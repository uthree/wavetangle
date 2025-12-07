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

/// すべてのノードが実装するトレイト
#[allow(dead_code)]
pub trait NodeBehavior: Any {
    /// ノードの種類を取得
    fn node_type(&self) -> NodeType;

    /// ノードのタイトル
    fn title(&self) -> &str;

    /// Anyとしての不変参照を取得（ダウンキャスト用）
    fn as_any(&self) -> &dyn Any;

    /// Anyとしての可変参照を取得（ダウンキャスト用）
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// 入力ピンの数
    fn input_count(&self) -> usize;

    /// 出力ピンの数
    fn output_count(&self) -> usize;

    /// 入力ピンのタイプ
    fn input_pin_type(&self, index: usize) -> Option<PinType>;

    /// 出力ピンのタイプ
    fn output_pin_type(&self, index: usize) -> Option<PinType>;

    /// 入力ピンの名前
    fn input_pin_name(&self, index: usize) -> Option<&str>;

    /// 出力ピンの名前
    fn output_pin_name(&self, index: usize) -> Option<&str>;

    /// 指定入力ピンのバッファを取得（エフェクトノード用）
    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer>;

    /// 指定チャンネルの出力バッファを取得
    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer>;

    /// オーディオチャンネル数を取得
    fn channels(&self) -> u16;

    /// オーディオチャンネル数を設定（バッファも再作成）
    fn set_channels(&mut self, channels: u16);

    /// アクティブ状態を取得
    fn is_active(&self) -> bool;

    /// アクティブ状態を設定
    fn set_active(&mut self, active: bool);

    /// ノードボディのUIを描画
    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext);
}

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
}
