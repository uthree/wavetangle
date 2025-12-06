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

    /// 指定チャンネルのバッファを取得
    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer>;

    /// オーディオチャンネル数を取得
    fn channels(&self) -> u16;

    /// オーディオチャンネル数を設定（バッファも再作成）
    fn set_channels(&mut self, channels: u16);

    /// アクティブ状態を取得
    fn is_active(&self) -> bool;

    /// アクティブ状態を設定
    fn set_active(&mut self, active: bool);
}

/// デフォルトのリングバッファサイズ（サンプル数）
pub const DEFAULT_RING_BUFFER_SIZE: usize = 8192;

// ============================================================================
// Audio Input Node
// ============================================================================

/// オーディオ入力デバイスノード
#[derive(Clone)]
pub struct AudioInputNode {
    pub device_name: String,
    /// チャンネルごとのバッファ
    pub channel_buffers: Vec<ChannelBuffer>,
    pub channels: u16,
    pub is_active: bool,
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
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        self.channels = channels;
        self.channel_buffers = (0..channels)
            .map(|_| new_channel_buffer(DEFAULT_RING_BUFFER_SIZE))
            .collect();
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
}

// ============================================================================
// Audio Output Node
// ============================================================================

/// オーディオ出力デバイスノード
#[derive(Clone)]
pub struct AudioOutputNode {
    pub device_name: String,
    /// チャンネルごとのバッファ
    pub channel_buffers: Vec<ChannelBuffer>,
    pub channels: u16,
    pub is_active: bool,
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
        }
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        self.channels = channels;
        self.channel_buffers = (0..channels)
            .map(|_| new_channel_buffer(DEFAULT_RING_BUFFER_SIZE))
            .collect();
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
}

// ============================================================================
// AudioNode Enum - egui-snarlで使用するラッパー
// ============================================================================

/// オーディオグラフのノード（enumラッパー）
#[derive(Clone)]
pub enum AudioNode {
    AudioInput(AudioInputNode),
    AudioOutput(AudioOutputNode),
}

/// enumバリアントに対してtraitメソッドをデリゲートするマクロ
macro_rules! delegate_node_behavior {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            AudioNode::AudioInput(node) => node.$method($($arg),*),
            AudioNode::AudioOutput(node) => node.$method($($arg),*),
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
        }
    }

    fn is_active(&self) -> bool {
        delegate_node_behavior!(self, is_active)
    }

    fn set_active(&mut self, active: bool) {
        match self {
            AudioNode::AudioInput(node) => node.set_active(active),
            AudioNode::AudioOutput(node) => node.set_active(active),
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
