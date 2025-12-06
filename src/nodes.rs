use std::sync::Arc;

use parking_lot::Mutex;

/// リングバッファ - 音声データの低遅延転送用
#[derive(Clone)]
pub struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
        }
    }

    /// データを書き込む
    pub fn write(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.data[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    /// データを読み込む（読み込んだ分だけ進む）
    pub fn read(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    /// データを読み込む（読み取り位置を進めない - 複数のコンシューマー用）
    pub fn peek(&self, output: &mut [f32]) {
        let mut pos = self.read_pos;
        for sample in output.iter_mut() {
            *sample = self.data[pos];
            pos = (pos + 1) % self.capacity;
        }
    }

    /// 読み取り位置を指定サンプル数だけ進める
    pub fn advance_read(&mut self, count: usize) {
        self.read_pos = (self.read_pos + count) % self.capacity;
    }

    /// 利用可能なサンプル数
    #[allow(dead_code)]
    pub fn available(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }
}

/// チャンネルバッファ - 1チャンネル分のリングバッファ
pub type ChannelBuffer = Arc<Mutex<RingBuffer>>;

/// 新しいチャンネルバッファを作成
pub fn new_channel_buffer(capacity: usize) -> ChannelBuffer {
    Arc::new(Mutex::new(RingBuffer::new(capacity)))
}

/// ノードのピンタイプ
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PinType {
    Audio,
}

/// ノードの種類（UIでの表示用）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum NodeCategory {
    Input,
    Output,
    Effect,
}

/// ノードのサイズ（幅, 高さ）
#[derive(Clone, Copy, Debug)]
pub struct NodeSize {
    pub width: f32,
    pub height: f32,
}

impl Default for NodeSize {
    fn default() -> Self {
        Self {
            width: 200.0,
            height: 150.0,
        }
    }
}

/// すべてのノードが実装するトレイト
#[allow(dead_code)]
pub trait NodeBehavior {
    /// ノードのタイトル
    fn title(&self) -> &str;

    /// ノードのカテゴリ
    fn category(&self) -> NodeCategory;

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

    /// ノードのサイズを取得
    fn node_size(&self) -> NodeSize;

    /// ノードのサイズを設定
    fn set_node_size(&mut self, size: NodeSize);
}

/// デフォルトのリングバッファサイズ（サンプル数）
pub const DEFAULT_RING_BUFFER_SIZE: usize = 8192;

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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: self.node_size,
        }
    }
}

impl AudioInputNode {
    pub fn new(device_name: String) -> Self {
        let channels = 2u16; // デフォルトはステレオ
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
            node_size: NodeSize::default(),
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    /// 注意: 既存のバッファを保持し、必要に応じて追加/削除のみ行う
    pub fn resize_buffers(&mut self, channels: u16) {
        if self.channels == channels {
            return; // チャンネル数が同じなら何もしない
        }

        let old_len = self.channel_buffers.len();
        let new_len = channels as usize;

        if new_len > old_len {
            // チャンネルを追加
            for _ in old_len..new_len {
                self.channel_buffers
                    .push(new_channel_buffer(DEFAULT_RING_BUFFER_SIZE));
            }
        } else {
            // チャンネルを削除
            self.channel_buffers.truncate(new_len);
        }
        self.channels = channels;
    }
}

impl NodeBehavior for AudioInputNode {
    fn title(&self) -> &str {
        "Audio Input"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Input
    }

    fn input_count(&self) -> usize {
        0
    }

    fn output_count(&self) -> usize {
        self.channels as usize
    }

    fn input_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channels as usize {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("L"),
            1 => Some("R"),
            2 => Some("C"),
            3 => Some("LFE"),
            4 => Some("SL"),
            5 => Some("SR"),
            _ => None,
        }
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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: self.node_size,
        }
    }
}

impl AudioOutputNode {
    pub fn new(device_name: String) -> Self {
        let channels = 2u16; // デフォルトはステレオ
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
            node_size: NodeSize::default(),
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    /// 注意: 既存のバッファを保持し、必要に応じて追加/削除のみ行う
    pub fn resize_buffers(&mut self, channels: u16) {
        if self.channels == channels {
            return; // チャンネル数が同じなら何もしない
        }

        let old_len = self.channel_buffers.len();
        let new_len = channels as usize;

        if new_len > old_len {
            // チャンネルを追加
            for _ in old_len..new_len {
                self.channel_buffers
                    .push(new_channel_buffer(DEFAULT_RING_BUFFER_SIZE));
            }
        } else {
            // チャンネルを削除
            self.channel_buffers.truncate(new_len);
        }
        self.channels = channels;
    }
}

impl NodeBehavior for AudioOutputNode {
    fn title(&self) -> &str {
        "Audio Output"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Output
    }

    fn input_count(&self) -> usize {
        self.channels as usize
    }

    fn output_count(&self) -> usize {
        0
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channels as usize {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("L"),
            1 => Some("R"),
            2 => Some("C"),
            3 => Some("LFE"),
            4 => Some("SL"),
            5 => Some("SR"),
            _ => None,
        }
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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
}

impl GainNode {
    pub fn new() -> Self {
        Self {
            gain: 1.0,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            node_size: NodeSize::default(),
        }
    }
}

impl Default for GainNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for GainNode {
    fn title(&self) -> &str {
        "Gain"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: NodeSize::default(),
        }
    }
}

impl Default for AddNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for AddNode {
    fn title(&self) -> &str {
        "Add"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: NodeSize::default(),
        }
    }
}

impl Default for MultiplyNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for MultiplyNode {
    fn title(&self) -> &str {
        "Multiply"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: NodeSize::default(),
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
    fn title(&self) -> &str {
        "Filter"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
}

impl Clone for SpectrumAnalyzerNode {
    fn clone(&self) -> Self {
        Self {
            spectrum: self.spectrum.clone(),
            input_buffers: self.input_buffers.clone(),
            output_buffer: self.output_buffer.clone(),
            is_active: self.is_active,
            analyzer: Arc::new(Mutex::new(crate::dsp::SpectrumAnalyzer::new())),
            node_size: self.node_size,
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
            node_size: NodeSize::default(),
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
    fn title(&self) -> &str {
        "Spectrum"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: NodeSize::default(),
        }
    }
}

impl Default for CompressorNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for CompressorNode {
    fn title(&self) -> &str {
        "Compressor"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
    }
}

// ============================================================================
// PitchShift Node - PSOLAピッチシフト
// ============================================================================

/// ピッチシフトノード
pub struct PitchShiftNode {
    /// ピッチシフト量（半音単位、-12〜+12）
    pub semitones: f32,
    /// 入力バッファ（1入力）
    pub input_buffers: Vec<ChannelBuffer>,
    /// 出力バッファ
    pub output_buffer: ChannelBuffer,
    /// アクティブ状態
    pub is_active: bool,
    /// ピッチシフター（スレッドセーフ）
    pub pitch_shifter: Arc<Mutex<crate::dsp::PitchShifter>>,
    /// ノードのサイズ
    pub node_size: NodeSize,
}

impl Clone for PitchShiftNode {
    fn clone(&self) -> Self {
        Self {
            semitones: self.semitones,
            input_buffers: self.input_buffers.clone(),
            output_buffer: self.output_buffer.clone(),
            is_active: self.is_active,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::new(44100.0))),
            node_size: self.node_size,
        }
    }
}

impl PitchShiftNode {
    pub fn new() -> Self {
        Self {
            semitones: 0.0,
            input_buffers: vec![new_channel_buffer(DEFAULT_RING_BUFFER_SIZE)], // 1入力
            output_buffer: new_channel_buffer(DEFAULT_RING_BUFFER_SIZE),
            is_active: false,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::new(44100.0))),
            node_size: NodeSize::default(),
        }
    }
}

impl Default for PitchShiftNode {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeBehavior for PitchShiftNode {
    fn title(&self) -> &str {
        "Pitch Shift"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
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
    /// ノードのサイズ
    pub node_size: NodeSize,
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
            node_size: self.node_size,
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
            node_size: NodeSize::default(),
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
    fn title(&self) -> &str {
        "Graphic EQ"
    }

    fn category(&self) -> NodeCategory {
        NodeCategory::Effect
    }

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

    fn node_size(&self) -> NodeSize {
        self.node_size
    }

    fn set_node_size(&mut self, size: NodeSize) {
        self.node_size = size;
    }
}

// ============================================================================
// AudioNode Enum - egui-snarlで使用するラッパー
// ============================================================================

/// オーディオグラフのノード（enumラッパー）
#[derive(Clone)]
pub enum AudioNode {
    AudioInput(AudioInputNode),
    AudioOutput(AudioOutputNode),
    Gain(GainNode),
    Add(AddNode),
    Multiply(MultiplyNode),
    Filter(FilterNode),
    SpectrumAnalyzer(SpectrumAnalyzerNode),
    Compressor(CompressorNode),
    PitchShift(PitchShiftNode),
    GraphicEq(GraphicEqNode),
}

/// enumバリアントに対してtraitメソッドをデリゲートするマクロ
macro_rules! delegate_node_behavior {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            AudioNode::AudioInput(node) => node.$method($($arg),*),
            AudioNode::AudioOutput(node) => node.$method($($arg),*),
            AudioNode::Gain(node) => node.$method($($arg),*),
            AudioNode::Add(node) => node.$method($($arg),*),
            AudioNode::Multiply(node) => node.$method($($arg),*),
            AudioNode::Filter(node) => node.$method($($arg),*),
            AudioNode::SpectrumAnalyzer(node) => node.$method($($arg),*),
            AudioNode::Compressor(node) => node.$method($($arg),*),
            AudioNode::PitchShift(node) => node.$method($($arg),*),
            AudioNode::GraphicEq(node) => node.$method($($arg),*),
        }
    };
}

impl NodeBehavior for AudioNode {
    fn title(&self) -> &str {
        delegate_node_behavior!(self, title)
    }

    fn category(&self) -> NodeCategory {
        delegate_node_behavior!(self, category)
    }

    fn input_count(&self) -> usize {
        delegate_node_behavior!(self, input_count)
    }

    fn output_count(&self) -> usize {
        delegate_node_behavior!(self, output_count)
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        delegate_node_behavior!(self, input_pin_type, index)
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        delegate_node_behavior!(self, output_pin_type, index)
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        delegate_node_behavior!(self, input_pin_name, index)
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        delegate_node_behavior!(self, output_pin_name, index)
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        delegate_node_behavior!(self, input_buffer, index)
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        delegate_node_behavior!(self, channel_buffer, channel)
    }

    fn channels(&self) -> u16 {
        delegate_node_behavior!(self, channels)
    }

    fn set_channels(&mut self, channels: u16) {
        match self {
            AudioNode::AudioInput(node) => node.set_channels(channels),
            AudioNode::AudioOutput(node) => node.set_channels(channels),
            AudioNode::Gain(node) => node.set_channels(channels),
            AudioNode::Add(node) => node.set_channels(channels),
            AudioNode::Multiply(node) => node.set_channels(channels),
            AudioNode::Filter(node) => node.set_channels(channels),
            AudioNode::SpectrumAnalyzer(node) => node.set_channels(channels),
            AudioNode::Compressor(node) => node.set_channels(channels),
            AudioNode::PitchShift(node) => node.set_channels(channels),
            AudioNode::GraphicEq(node) => node.set_channels(channels),
        }
    }

    fn is_active(&self) -> bool {
        delegate_node_behavior!(self, is_active)
    }

    fn set_active(&mut self, active: bool) {
        match self {
            AudioNode::AudioInput(node) => node.set_active(active),
            AudioNode::AudioOutput(node) => node.set_active(active),
            AudioNode::Gain(node) => node.set_active(active),
            AudioNode::Add(node) => node.set_active(active),
            AudioNode::Multiply(node) => node.set_active(active),
            AudioNode::Filter(node) => node.set_active(active),
            AudioNode::SpectrumAnalyzer(node) => node.set_active(active),
            AudioNode::Compressor(node) => node.set_active(active),
            AudioNode::PitchShift(node) => node.set_active(active),
            AudioNode::GraphicEq(node) => node.set_active(active),
        }
    }

    fn node_size(&self) -> NodeSize {
        delegate_node_behavior!(self, node_size)
    }

    fn set_node_size(&mut self, size: NodeSize) {
        match self {
            AudioNode::AudioInput(node) => node.set_node_size(size),
            AudioNode::AudioOutput(node) => node.set_node_size(size),
            AudioNode::Gain(node) => node.set_node_size(size),
            AudioNode::Add(node) => node.set_node_size(size),
            AudioNode::Multiply(node) => node.set_node_size(size),
            AudioNode::Filter(node) => node.set_node_size(size),
            AudioNode::SpectrumAnalyzer(node) => node.set_node_size(size),
            AudioNode::Compressor(node) => node.set_node_size(size),
            AudioNode::PitchShift(node) => node.set_node_size(size),
            AudioNode::GraphicEq(node) => node.set_node_size(size),
        }
    }
}

impl AudioNode {
    /// AudioInputノードを作成
    pub fn new_audio_input(device_name: String) -> Self {
        Self::AudioInput(AudioInputNode::new(device_name))
    }

    /// AudioOutputノードを作成
    pub fn new_audio_output(device_name: String) -> Self {
        Self::AudioOutput(AudioOutputNode::new(device_name))
    }

    /// AudioInputノードへの参照を取得
    #[allow(dead_code)]
    pub fn as_audio_input(&self) -> Option<&AudioInputNode> {
        match self {
            AudioNode::AudioInput(node) => Some(node),
            _ => None,
        }
    }

    /// AudioInputノードへの可変参照を取得
    #[allow(dead_code)]
    pub fn as_audio_input_mut(&mut self) -> Option<&mut AudioInputNode> {
        match self {
            AudioNode::AudioInput(node) => Some(node),
            _ => None,
        }
    }

    /// AudioOutputノードへの参照を取得
    #[allow(dead_code)]
    pub fn as_audio_output(&self) -> Option<&AudioOutputNode> {
        match self {
            AudioNode::AudioOutput(node) => Some(node),
            _ => None,
        }
    }

    /// AudioOutputノードへの可変参照を取得
    #[allow(dead_code)]
    pub fn as_audio_output_mut(&mut self) -> Option<&mut AudioOutputNode> {
        match self {
            AudioNode::AudioOutput(node) => Some(node),
            _ => None,
        }
    }
}
