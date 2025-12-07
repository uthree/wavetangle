use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use egui::{Color32, Ui};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points};
use egui_snarl::NodeId;
use parking_lot::Mutex;

use crate::dsp::EqPoint;

/// UI描画時に必要なコンテキスト
pub struct NodeUIContext<'a> {
    /// 入力デバイス名のリスト
    pub input_devices: &'a [String],
    /// 出力デバイス名のリスト
    pub output_devices: &'a [String],
    /// ノードID（ウィジェットの一意識別用）
    pub node_id: NodeId,
}

/// HSVからRGBに変換
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color32 {
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
fn show_spectrum_line(ui: &mut Ui, plot_id: &str, spectrum: &Arc<Mutex<Vec<f32>>>) {
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
fn interpolate_eq_gain(points: &[EqPoint], freq: f32) -> f32 {
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

    /// 先頭からn個のサンプルを読み取り、同時に消費する（アトミック操作）
    /// バッファが足りない場合は0.0でパディング
    /// readとconsumeの間に他のスレッドがpushすると競合が起きるため、
    /// この関数で1回のロックで両方を行う
    pub fn read_and_consume(&mut self, count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);
        let available = self.data.len().min(count);

        // 先頭からavailable個を取り出す（drainで同時に削除）
        for sample in self.data.drain(0..available) {
            result.push(sample);
        }

        // 不足分は0でパディング
        result.resize(count, 0.0);
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
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    };
}

/// マルチチャンネルオーディオのチャンネル名
/// サラウンド5.1chまで対応
const CHANNEL_NAMES: &[&str] = &["L", "R", "C", "LFE", "SL", "SR"];

/// チャンネルインデックスからチャンネル名を取得
fn channel_name(index: usize) -> Option<&'static str> {
    CHANNEL_NAMES.get(index).copied()
}

/// チャンネルバッファのサイズを調整
/// 変更があった場合はtrueを返す
fn resize_channel_buffers(
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
// Audio Input Node
// ============================================================================

/// オーディオ入力デバイスノード
pub struct AudioInputNode {
    pub device_name: String,
    /// チャンネルごとのバッファ
    pub channel_buffers: Vec<ChannelBuffer>,
    pub channels: u16,
    pub is_active: bool,
    /// スペクトラム表示を有効にするか
    pub show_spectrum: bool,
    /// スペクトラムデータ
    pub spectrum: Arc<Mutex<Vec<f32>>>,
    /// スペクトラムアナライザー（最初のチャンネルを解析）
    pub analyzer: Arc<Mutex<crate::dsp::SpectrumAnalyzer>>,
}

impl Clone for AudioInputNode {
    fn clone(&self) -> Self {
        Self {
            device_name: self.device_name.clone(),
            channel_buffers: self.channel_buffers.clone(),
            channels: self.channels,
            is_active: self.is_active,
            show_spectrum: self.show_spectrum,
            spectrum: self.spectrum.clone(),
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }
}

impl AudioInputNode {
    pub fn new(device_name: String, channels: u16) -> Self {
        let channels = channels.max(1); // 最低1チャンネル
        let channel_buffers = (0..channels)
            .map(|_| new_channel_buffer(DEFAULT_RING_BUFFER_SIZE))
            .collect();
        Self {
            device_name,
            channel_buffers,
            channels,
            is_active: false,
            show_spectrum: true,
            spectrum: Arc::new(Mutex::new(vec![0.0; FFT_SIZE / 2])),
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        if resize_channel_buffers(&mut self.channel_buffers, self.channels, channels) {
            self.channels = channels;
        }
    }
}

impl NodeBehavior for AudioInputNode {
    fn node_type(&self) -> NodeType {
        NodeType::AudioInput
    }

    fn title(&self) -> &str {
        "Audio Input"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        0
    }

    fn output_count(&self) -> usize {
        self.channel_buffers.len()
    }

    fn input_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channel_buffers.len() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
    }

    fn input_buffer(&self, _index: usize) -> Option<ChannelBuffer> {
        None // AudioInputは入力バッファを持たない
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        self.channel_buffers.get(channel).cloned()
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn set_channels(&mut self, channels: u16) {
        self.resize_buffers(channels);
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.is_active, "Active");
                if self.is_active {
                    ui.label(format!("{}ch", self.channels));
                }
            });

            egui::ComboBox::from_id_salt(format!("input_device_{:?}", ctx.node_id))
                .selected_text(self.device_name.as_str())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for dev in ctx.input_devices {
                        ui.selectable_value(&mut self.device_name, dev.clone(), dev);
                    }
                });

            // スペクトラム表示（チェックボックスで切り替え）
            if self.is_active {
                ui.checkbox(&mut self.show_spectrum, "Spectrum");
                if self.show_spectrum {
                    show_spectrum_line(
                        ui,
                        &format!("input_spectrum_{:?}", ctx.node_id),
                        &self.spectrum,
                    );
                }
            }
        });
    }
}

// ============================================================================
// Audio Output Node
// ============================================================================

/// オーディオ出力デバイスノード
pub struct AudioOutputNode {
    pub device_name: String,
    /// チャンネルごとのバッファ
    pub channel_buffers: Vec<ChannelBuffer>,
    pub channels: u16,
    pub is_active: bool,
    /// スペクトラム表示を有効にするか
    pub show_spectrum: bool,
    /// スペクトラムデータ
    pub spectrum: Arc<Mutex<Vec<f32>>>,
    /// スペクトラムアナライザー（最初のチャンネルを解析）
    pub analyzer: Arc<Mutex<crate::dsp::SpectrumAnalyzer>>,
}

impl Clone for AudioOutputNode {
    fn clone(&self) -> Self {
        Self {
            device_name: self.device_name.clone(),
            channel_buffers: self.channel_buffers.clone(),
            channels: self.channels,
            is_active: self.is_active,
            show_spectrum: self.show_spectrum,
            spectrum: self.spectrum.clone(),
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }
}

impl AudioOutputNode {
    pub fn new(device_name: String, channels: u16) -> Self {
        let channels = channels.max(1); // 最低1チャンネル
        let channel_buffers = (0..channels)
            .map(|_| new_channel_buffer(DEFAULT_RING_BUFFER_SIZE))
            .collect();
        Self {
            device_name,
            channel_buffers,
            channels,
            is_active: false,
            show_spectrum: true,
            spectrum: Arc::new(Mutex::new(vec![0.0; FFT_SIZE / 2])),
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        if resize_channel_buffers(&mut self.channel_buffers, self.channels, channels) {
            self.channels = channels;
        }
    }
}

impl NodeBehavior for AudioOutputNode {
    fn node_type(&self) -> NodeType {
        NodeType::AudioOutput
    }

    fn title(&self) -> &str {
        "Audio Output"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        self.channel_buffers.len()
    }

    fn output_count(&self) -> usize {
        0
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channel_buffers.len() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
    }

    fn output_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        // AudioOutputの入力はchannel_buffersと同じ（データを受け取る）
        self.channel_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        self.channel_buffers.get(channel).cloned()
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn set_channels(&mut self, channels: u16) {
        self.resize_buffers(channels);
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.is_active, "Active");
                if self.is_active {
                    ui.label(format!("{}ch", self.channels));
                }
            });

            egui::ComboBox::from_id_salt(format!("output_device_{:?}", ctx.node_id))
                .selected_text(self.device_name.as_str())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for dev in ctx.output_devices {
                        ui.selectable_value(&mut self.device_name, dev.clone(), dev);
                    }
                });

            // スペクトラム表示（チェックボックスで切り替え）
            if self.is_active {
                ui.checkbox(&mut self.show_spectrum, "Spectrum");
                if self.show_spectrum {
                    show_spectrum_line(
                        ui,
                        &format!("output_spectrum_{:?}", ctx.node_id),
                        &self.spectrum,
                    );
                }
            }
        });
    }
}

// ============================================================================
// Gain Node (Effect)
// ============================================================================

/// ゲインエフェクトノード
#[derive(Clone)]
pub struct GainNode {
    /// ゲイン値（倍率）
    pub gain: f32,
    /// 入力バッファ（各入力ピンに対応）
    pub input_buffers: Vec<ChannelBuffer>,
    /// 出力バッファ
    pub output_buffer: ChannelBuffer,
    /// アクティブ状態
    pub is_active: bool,
}

impl GainNode {
    pub fn new() -> Self {
        Self {
            gain: 1.0,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
        }
    }
}

impl Default for GainNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for GainNode {
    fn node_type(&self) -> NodeType {
        NodeType::Gain
    }

    fn title(&self) -> &str {
        "Gain"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {
        // GainNodeは常に1チャンネル
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Gain:");
            ui.add(egui::Slider::new(&mut self.gain, 0.0..=2.0).suffix("x"));

            // dB表示
            let db = 20.0 * self.gain.log10();
            if db.is_finite() {
                ui.label(format!("{:.1} dB", db));
            } else {
                ui.label("-∞ dB");
            }
        });
    }
}

// ============================================================================
// Add Node (Arithmetic)
// ============================================================================

/// 加算ノード - 2つの信号を加算
#[derive(Clone)]
pub struct AddNode {
    /// 入力バッファ（2入力）
    pub input_buffers: Vec<ChannelBuffer>,
    pub output_buffer: ChannelBuffer,
    pub is_active: bool,
}

impl AddNode {
    pub fn new() -> Self {
        Self {
            input_buffers: vec![
                new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
                new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            ], // 2入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
        }
    }
}

impl Default for AddNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for AddNode {
    fn node_type(&self) -> NodeType {
        NodeType::Add
    }

    fn title(&self) -> &str {
        "Add"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < 2 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("A"),
            1 => Some("B"),
            _ => None,
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("A + B");
        });
    }
}

// ============================================================================
// Multiply Node (Arithmetic)
// ============================================================================

/// 乗算ノード - 2つの信号を乗算（リングモジュレーション）
#[derive(Clone)]
pub struct MultiplyNode {
    /// 入力バッファ（2入力）
    pub input_buffers: Vec<ChannelBuffer>,
    pub output_buffer: ChannelBuffer,
    pub is_active: bool,
}

impl MultiplyNode {
    pub fn new() -> Self {
        Self {
            input_buffers: vec![
                new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
                new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            ], // 2入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
        }
    }
}

impl Default for MultiplyNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for MultiplyNode {
    fn node_type(&self) -> NodeType {
        NodeType::Multiply
    }

    fn title(&self) -> &str {
        "Multiply"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < 2 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("A"),
            1 => Some("B"),
            _ => None,
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("A × B");
        });
    }
}

// ============================================================================
// Filter Node (Effect)
// ============================================================================

/// フィルタータイプ
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterType {
    Low,
    High,
    Band,
}

/// フィルターノード - ローパス/ハイパス/バンドパスフィルター
#[derive(Clone)]
pub struct FilterNode {
    pub filter_type: FilterType,
    /// カットオフ周波数 (Hz)
    pub cutoff: f32,
    /// レゾナンス (Q値)
    pub resonance: f32,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    pub output_buffer: ChannelBuffer,
    pub is_active: bool,
    /// Biquadフィルター状態
    pub biquad_state: Arc<Mutex<crate::dsp::BiquadState>>,
    /// 現在のフィルター係数（キャッシュ用）
    #[allow(dead_code)]
    pub biquad_coeffs: Arc<Mutex<Option<crate::dsp::BiquadCoeffs>>>,
}

impl FilterNode {
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::Low,
            cutoff: 1000.0,
            resonance: 0.707,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            biquad_state: Arc::new(Mutex::new(crate::dsp::BiquadState::new())),
            biquad_coeffs: Arc::new(Mutex::new(None)),
        }
    }

    /// フィルター係数を更新
    #[allow(dead_code)]
    pub fn update_coeffs(&self, sample_rate: f32) {
        let coeffs = crate::dsp::BiquadCoeffs::from_filter_type(
            self.filter_type,
            sample_rate,
            self.cutoff,
            self.resonance,
        );
        *self.biquad_coeffs.lock() = Some(coeffs);
    }
}

impl Default for FilterNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for FilterNode {
    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn title(&self) -> &str {
        "Filter"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt(format!("filter_type_{:?}", ctx.node_id))
                .selected_text(match self.filter_type {
                    FilterType::Low => "Low Pass",
                    FilterType::High => "High Pass",
                    FilterType::Band => "Band Pass",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.filter_type, FilterType::Low, "Low Pass");
                    ui.selectable_value(&mut self.filter_type, FilterType::High, "High Pass");
                    ui.selectable_value(&mut self.filter_type, FilterType::Band, "Band Pass");
                });

            ui.label("Cutoff:");
            ui.add(
                egui::Slider::new(&mut self.cutoff, 20.0..=20000.0)
                    .logarithmic(true)
                    .suffix(" Hz"),
            );

            ui.label("Q:");
            ui.add(egui::Slider::new(&mut self.resonance, 0.1..=10.0));
        });
    }
}

// ============================================================================
// Spectrum Analyzer Node
// ============================================================================

/// FFTサイズ
pub const FFT_SIZE: usize = 1024;
/// GraphicEQ用のFFTサイズ（dsp.rsのEQ_FFT_SIZEと同じ値）
pub const EQ_FFT_SIZE: usize = 2048;

/// スペクトラムアナライザーノード
pub struct SpectrumAnalyzerNode {
    /// スペクトラムデータ（マグニチュード）
    pub spectrum: Arc<Mutex<Vec<f32>>>,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    pub output_buffer: ChannelBuffer,
    pub is_active: bool,
    /// FFTアナライザー（スレッドセーフ）
    pub analyzer: Arc<Mutex<crate::dsp::SpectrumAnalyzer>>,
}

impl Clone for SpectrumAnalyzerNode {
    fn clone(&self) -> Self {
        Self {
            spectrum: self.spectrum.clone(),
            input_buffers: self.input_buffers.clone(),
            output_buffer: self.output_buffer.clone(),
            is_active: self.is_active,
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }
}

impl SpectrumAnalyzerNode {
    pub fn new() -> Self {
        Self {
            spectrum: Arc::new(Mutex::new(vec![0.0; FFT_SIZE / 2])),
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
        }
    }

    /// スペクトラムを更新
    #[allow(dead_code)]
    pub fn update_spectrum(&self) {
        let mut analyzer = self.analyzer.lock();
        let spectrum_data = analyzer.compute_spectrum();
        let mut spectrum = self.spectrum.lock();
        spectrum.copy_from_slice(&spectrum_data);
    }
}

impl Default for SpectrumAnalyzerNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for SpectrumAnalyzerNode {
    fn node_type(&self) -> NodeType {
        NodeType::SpectrumAnalyzer
    }

    fn title(&self) -> &str {
        "Spectrum"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            // スペクトラムデータを取得
            let spectrum = self.spectrum.lock();
            let bar_count = 48; // 表示するバーの数

            // バーデータを作成
            let bars: Vec<Bar> = (0..bar_count)
                .map(|i| {
                    // 対数スケールでインデックスをマッピング（低周波を細かく表示）
                    let freq_ratio = i as f32 / bar_count as f32;
                    let freq_idx = (freq_ratio.powf(2.0) * (FFT_SIZE / 2) as f32) as usize;
                    let freq_idx = freq_idx.min(spectrum.len().saturating_sub(1));

                    // マグニチュードを正規化（対数スケール、dB変換）
                    let magnitude = if freq_idx < spectrum.len() {
                        spectrum[freq_idx]
                    } else {
                        0.0
                    };
                    let db = if magnitude > 1e-6 {
                        20.0 * magnitude.log10()
                    } else {
                        -80.0
                    };
                    // -80dB〜0dBを0.0〜1.0にマッピング
                    let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0) as f64;

                    // 周波数に応じたグラデーションカラー（低周波=緑、高周波=シアン）
                    let hue = 120.0 + freq_ratio * 60.0; // 緑(120)→シアン(180)
                    let sat = 0.7 + normalized as f32 * 0.3;
                    let val = 0.3 + normalized as f32 * 0.7;
                    let color = hsv_to_rgb(hue, sat, val);

                    Bar::new(i as f64, normalized).fill(color).width(0.85)
                })
                .collect();

            drop(spectrum); // ロックを解放

            // egui_plotでバーチャートを表示
            Plot::new(format!("spectrum_{:?}", ctx.node_id))
                .height(100.0)
                .width(220.0)
                .show_axes([false, false])
                .show_grid([false, false])
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .include_y(0.0)
                .include_y(1.0)
                .show_background(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new("spectrum", bars));
                });
        });
    }
}

// ============================================================================
// Compressor Node
// ============================================================================

/// コンプレッサーノード - ダイナミックレンジ圧縮
#[derive(Clone)]
pub struct CompressorNode {
    /// スレッショルド (dB)
    pub threshold: f32,
    /// レシオ (1:n)
    pub ratio: f32,
    /// アタックタイム (ms)
    pub attack: f32,
    /// リリースタイム (ms)
    pub release: f32,
    /// メイクアップゲイン (dB)
    pub makeup_gain: f32,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    pub output_buffer: ChannelBuffer,
    pub is_active: bool,
    /// コンプレッサー状態
    pub compressor_state: Arc<Mutex<crate::dsp::CompressorState>>,
    /// 現在のゲインリダクション (dB) - メーター表示用
    #[allow(dead_code)]
    pub gain_reduction: Arc<Mutex<f32>>,
}

impl CompressorNode {
    pub fn new() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack: 10.0,
            release: 100.0,
            makeup_gain: 0.0,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            compressor_state: Arc::new(Mutex::new(crate::dsp::CompressorState::new())),
            gain_reduction: Arc::new(Mutex::new(0.0)),
        }
    }
}

impl Default for CompressorNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for CompressorNode {
    fn node_type(&self) -> NodeType {
        NodeType::Compressor
    }

    fn title(&self) -> &str {
        "Compressor"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Threshold:");
            ui.add(egui::Slider::new(&mut self.threshold, -60.0..=0.0).suffix(" dB"));

            ui.label("Ratio:");
            ui.add(egui::Slider::new(&mut self.ratio, 1.0..=20.0).suffix(":1"));

            ui.label("Attack:");
            ui.add(egui::Slider::new(&mut self.attack, 0.1..=100.0).suffix(" ms"));

            ui.label("Release:");
            ui.add(egui::Slider::new(&mut self.release, 10.0..=1000.0).suffix(" ms"));

            ui.label("Makeup:");
            ui.add(egui::Slider::new(&mut self.makeup_gain, 0.0..=24.0).suffix(" dB"));
        });
    }
}

// ============================================================================
// PitchShift Node - PSOLAピッチシフト
// ============================================================================

/// ピッチシフトノード
pub struct PitchShiftNode {
    /// ピッチシフト量（半音単位、-12〜+12）
    pub semitones: f32,
    /// グレインサイズ（サンプル数、128〜8192）
    pub grain_size: usize,
    /// グレイン数（2〜16）
    pub num_grains: usize,
    /// 位相アラインメントを有効にするか
    pub phase_alignment_enabled: bool,
    /// 位相アラインメントの探索範囲（グレインサイズに対する割合、0.1〜1.0）
    pub search_range_ratio: f32,
    /// 位相アラインメントの相関長（グレインサイズに対する割合、0.1〜1.0）
    pub correlation_length_ratio: f32,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    /// 出力バッファ
    pub output_buffer: ChannelBuffer,
    /// アクティブ状態
    pub is_active: bool,
    /// ピッチシフター（スレッドセーフ）
    pub pitch_shifter: Arc<Mutex<crate::dsp::PitchShifter>>,
}

impl Clone for PitchShiftNode {
    fn clone(&self) -> Self {
        Self {
            semitones: self.semitones,
            grain_size: self.grain_size,
            num_grains: self.num_grains,
            phase_alignment_enabled: self.phase_alignment_enabled,
            search_range_ratio: self.search_range_ratio,
            correlation_length_ratio: self.correlation_length_ratio,
            input_buffers: self.input_buffers.clone(),
            output_buffer: self.output_buffer.clone(),
            is_active: self.is_active,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::with_params(
                44100.0,
                self.grain_size,
                self.num_grains,
                crate::dsp::DEFAULT_PITCH_BUFFER_SIZE,
            ))),
        }
    }
}

impl PitchShiftNode {
    pub fn new() -> Self {
        let default_params = crate::dsp::PhaseAlignmentParams::default();
        Self {
            semitones: 0.0,
            grain_size: crate::dsp::DEFAULT_GRAIN_SIZE,
            num_grains: crate::dsp::DEFAULT_NUM_GRAINS,
            phase_alignment_enabled: default_params.enabled,
            search_range_ratio: default_params.search_range_ratio,
            correlation_length_ratio: default_params.correlation_length_ratio,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::new(44100.0))),
        }
    }
}

impl Default for PitchShiftNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for PitchShiftNode {
    fn node_type(&self) -> NodeType {
        NodeType::PitchShift
    }

    fn title(&self) -> &str {
        "Pitch Shift"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.set_max_width(150.0);
            ui.label("Semitones:");
            ui.add(egui::Slider::new(&mut self.semitones, -12.0..=12.0).suffix(" st"));

            // セント表示
            let cents = (self.semitones.fract() * 100.0).round() as i32;
            let semitones_int = self.semitones.trunc() as i32;
            if cents != 0 {
                ui.label(format!("{:+} semitones, {:+} cents", semitones_int, cents));
            } else {
                ui.label(format!("{:+} semitones", semitones_int));
            }

            ui.separator();

            // グレインサイズ（2のべき乗で調整）
            ui.label("Grain Size:");
            let mut grain_size_log = (self.grain_size as f32).log2();
            if ui
                .add(egui::Slider::new(&mut grain_size_log, 7.0..=13.0).show_value(false))
                .changed()
            {
                let new_grain_size = 2_usize.pow(grain_size_log.round() as u32);
                if new_grain_size != self.grain_size {
                    self.grain_size = new_grain_size;
                    if let Some(mut shifter) = self.pitch_shifter.try_lock() {
                        shifter.set_grain_size(new_grain_size);
                    }
                }
            }
            ui.label(format!("{} samples", self.grain_size));

            // グレイン数
            ui.label("Num Grains:");
            let mut num_grains = self.num_grains as i32;
            if ui.add(egui::Slider::new(&mut num_grains, 2..=16)).changed() {
                let new_num_grains = num_grains as usize;
                if new_num_grains != self.num_grains {
                    self.num_grains = new_num_grains;
                    if let Some(mut shifter) = self.pitch_shifter.try_lock() {
                        shifter.set_num_grains(new_num_grains);
                    }
                }
            }

            // 推定レイテンシ表示（サンプルレート48kHz想定）
            let latency_samples = self.grain_size * self.num_grains / 2;
            let latency_ms = latency_samples as f32 / 48.0;
            ui.label(format!("Latency: ~{:.1} ms", latency_ms));

            ui.separator();

            // 位相アラインメント設定
            ui.collapsing("Phase Alignment", |ui| {
                ui.checkbox(&mut self.phase_alignment_enabled, "Enabled");

                ui.add_enabled_ui(self.phase_alignment_enabled, |ui| {
                    ui.label("Search Range:");
                    ui.add(
                        egui::Slider::new(&mut self.search_range_ratio, 0.1..=1.0)
                            .suffix("x")
                            .fixed_decimals(2),
                    );

                    ui.label("Correlation Length:");
                    ui.add(
                        egui::Slider::new(&mut self.correlation_length_ratio, 0.1..=1.0)
                            .suffix("x")
                            .fixed_decimals(2),
                    );
                });
            });
        });
    }
}

// ============================================================================
// GraphicEqNode - グラフィックイコライザー
// ============================================================================

/// グラフィックイコライザーノード（カーブエディター付き）
pub struct GraphicEqNode {
    /// EQカーブのコントロールポイント
    pub eq_points: Vec<crate::dsp::EqPoint>,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    /// 出力バッファ
    pub output_buffer: ChannelBuffer,
    /// アクティブ状態
    pub is_active: bool,
    /// グラフィックEQプロセッサー（スレッドセーフ）
    pub graphic_eq: Arc<Mutex<crate::dsp::GraphicEq>>,
    /// スペクトラム表示を有効にするか
    pub show_spectrum: bool,
    /// スペクトラムデータ（入力信号）
    pub spectrum: Arc<Mutex<Vec<f32>>>,
}

impl Clone for GraphicEqNode {
    fn clone(&self) -> Self {
        Self {
            eq_points: self.eq_points.clone(),
            input_buffers: self.input_buffers.clone(),
            output_buffer: self.output_buffer.clone(),
            is_active: self.is_active,
            graphic_eq: Arc::new(Mutex::new(crate::dsp::GraphicEq::new(44100.0))),
            show_spectrum: self.show_spectrum,
            spectrum: self.spectrum.clone(),
        }
    }
}

impl GraphicEqNode {
    pub fn new() -> Self {
        // デフォルトの5ポイントEQカーブ
        let eq_points = vec![
            crate::dsp::EqPoint::new(50.0, 0.0),
            crate::dsp::EqPoint::new(200.0, 0.0),
            crate::dsp::EqPoint::new(1000.0, 0.0),
            crate::dsp::EqPoint::new(5000.0, 0.0),
            crate::dsp::EqPoint::new(15000.0, 0.0),
        ];
        Self {
            eq_points,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)],
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            graphic_eq: Arc::new(Mutex::new(crate::dsp::GraphicEq::new(44100.0))),
            show_spectrum: true,
            spectrum: Arc::new(Mutex::new(vec![0.0; EQ_FFT_SIZE / 2])),
        }
    }

    /// EQカーブを更新（UIから呼ばれる）
    #[allow(dead_code)]
    pub fn update_eq_curve(&mut self) {
        // ポイントを周波数でソート
        self.eq_points
            .sort_by(|a, b| a.freq.partial_cmp(&b.freq).unwrap());
        self.graphic_eq.lock().update_curve(&self.eq_points);
    }
}

impl Default for GraphicEqNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for GraphicEqNode {
    fn node_type(&self) -> NodeType {
        NodeType::GraphicEq
    }

    fn title(&self) -> &str {
        "Graphic EQ"
    }

    impl_as_any!();

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        if channel == 0 {
            Some(self.output_buffer.clone())
        } else {
            None
        }
    }

    fn channels(&self) -> u16 {
        1
    }

    fn set_channels(&mut self, _channels: u16) {}

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            // スペクトラム表示トグル
            ui.checkbox(&mut self.show_spectrum, "Show Spectrum");

            // 周波数範囲
            const MIN_FREQ: f64 = 20.0;
            const MAX_FREQ: f64 = 20000.0;
            const MIN_GAIN: f64 = -24.0;
            const MAX_GAIN: f64 = 24.0;

            // 周波数を対数スケールのX座標に変換
            let freq_to_x =
                |freq: f64| -> f64 { (freq / MIN_FREQ).ln() / (MAX_FREQ / MIN_FREQ).ln() };
            let x_to_freq = |x: f64| -> f64 { MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(x) };

            // EQカーブを描画するためのポイントを生成
            let curve_points: Vec<[f64; 2]> = (0..=100)
                .map(|i| {
                    let x = i as f64 / 100.0;
                    let freq = x_to_freq(x) as f32;

                    // ポイント間を線形補間してゲインを計算
                    let gain = interpolate_eq_gain(&self.eq_points, freq);
                    [x, gain as f64]
                })
                .collect();

            // コントロールポイントの座標
            let control_points: Vec<[f64; 2]> = self
                .eq_points
                .iter()
                .map(|p| [freq_to_x(p.freq as f64), p.gain_db as f64])
                .collect();

            // スペクトラムデータを取得してプロット座標に変換
            let spectrum_points: Vec<[f64; 2]> = if self.show_spectrum {
                let spectrum_data = self.spectrum.lock();
                let spectrum_len = spectrum_data.len();
                (0..100)
                    .map(|i| {
                        let x = i as f64 / 100.0;
                        // 対数周波数スケールでスペクトラムインデックスを計算
                        let freq_idx = (x.powf(2.0) * spectrum_len as f64) as usize;
                        let freq_idx = freq_idx.min(spectrum_len.saturating_sub(1));

                        let magnitude = if freq_idx < spectrum_data.len() {
                            spectrum_data[freq_idx]
                        } else {
                            0.0
                        };

                        // dBに変換（-80dB〜0dBを-24〜+24dBにマッピング）
                        let db = if magnitude > 1e-6 {
                            20.0 * (magnitude as f64).log10()
                        } else {
                            -80.0
                        };
                        // -80dB〜0dBを-24〜+24dBにスケール
                        let scaled_db = (db + 80.0) / 80.0 * 48.0 - 24.0;
                        let clamped_db = scaled_db.clamp(MIN_GAIN, MAX_GAIN);

                        [x, clamped_db]
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // プロット表示
            let plot_response = Plot::new(format!("graphic_eq_{:?}", ctx.node_id))
                .height(150.0)
                .width(280.0)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_axes([true, true])
                .show_grid([true, true])
                .include_x(0.0)
                .include_x(1.0)
                .include_y(MIN_GAIN)
                .include_y(MAX_GAIN)
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
                    // スペクトラム（背景として表示）
                    if self.show_spectrum && !spectrum_points.is_empty() {
                        plot_ui.line(
                            Line::new("spectrum", PlotPoints::from(spectrum_points.clone()))
                                .color(Color32::from_rgb(100, 200, 100))
                                .width(1.5),
                        );
                    }

                    // 0dBライン
                    plot_ui.line(
                        Line::new("zero", PlotPoints::from(vec![[0.0, 0.0], [1.0, 0.0]]))
                            .color(Color32::from_gray(100))
                            .width(1.0),
                    );

                    // EQカーブ
                    plot_ui.line(
                        Line::new("eq_curve", PlotPoints::from(curve_points))
                            .color(Color32::from_rgb(100, 200, 255))
                            .width(2.0),
                    );

                    // コントロールポイント
                    plot_ui.points(
                        Points::new("eq_points", PlotPoints::from(control_points.clone()))
                            .radius(6.0)
                            .color(Color32::from_rgb(255, 200, 100))
                            .filled(true),
                    );
                });

            // ドラッグでポイントを移動
            if let Some(pointer_pos) = plot_response.response.hover_pos() {
                let plot_bounds = plot_response.transform.bounds();
                let plot_rect = plot_response.response.rect;

                // ポインタ位置をプロット座標に変換
                let pointer_x = ((pointer_pos.x - plot_rect.left()) / plot_rect.width()
                    * plot_bounds.width() as f32
                    + plot_bounds.min()[0] as f32) as f64;
                let pointer_y = ((1.0 - (pointer_pos.y - plot_rect.top()) / plot_rect.height())
                    * plot_bounds.height() as f32
                    + plot_bounds.min()[1] as f32) as f64;

                // クリック/ドラッグ処理
                let is_primary_down = ui.input(|i| i.pointer.primary_down());
                let is_clicked = plot_response.response.clicked();

                if is_primary_down || is_clicked {
                    // 最も近いポイントを探す
                    let mut closest_idx = None;
                    let mut closest_dist = f64::MAX;

                    for (idx, point) in control_points.iter().enumerate() {
                        let dx = (point[0] - pointer_x) * plot_rect.width() as f64;
                        let dy = (point[1] - pointer_y) / plot_bounds.height()
                            * plot_rect.height() as f64;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < closest_dist && dist < 20.0 {
                            closest_dist = dist;
                            closest_idx = Some(idx);
                        }
                    }

                    // ポイントを移動
                    if let Some(idx) = closest_idx {
                        let new_gain = pointer_y.clamp(MIN_GAIN, MAX_GAIN) as f32;
                        self.eq_points[idx].gain_db = new_gain;

                        // GraphicEqの周波数ゲインカーブを更新
                        let mut eq = self.graphic_eq.lock();
                        eq.update_curve(&self.eq_points);
                    }
                }
            }

            // ポイント一覧（周波数ラベル）
            ui.horizontal(|ui| {
                for point in &self.eq_points {
                    let freq_str = if point.freq >= 1000.0 {
                        format!("{:.1}k", point.freq / 1000.0)
                    } else {
                        format!("{:.0}", point.freq)
                    };
                    ui.label(format!("{}:{:+.1}dB", freq_str, point.gain_db));
                }
            });

            // リセットボタン
            if ui.button("Reset").clicked() {
                for point in &mut self.eq_points {
                    point.gain_db = 0.0;
                }
                let mut eq = self.graphic_eq.lock();
                eq.update_curve(&self.eq_points);
            }
        });
    }
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
