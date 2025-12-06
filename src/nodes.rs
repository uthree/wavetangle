use std::sync::Arc;

use parking_lot::Mutex;

/// オーディオバッファ - ノード間でデータを共有するため
pub type AudioBuffer = Arc<Mutex<Vec<f32>>>;

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

    /// オーディオバッファへの参照（あれば）
    fn buffer(&self) -> Option<&AudioBuffer>;

    /// アクティブ状態を取得
    fn is_active(&self) -> bool;

    /// アクティブ状態を設定
    fn set_active(&mut self, active: bool);
}

// ============================================================================
// Audio Input Node
// ============================================================================

/// オーディオ入力デバイスノード
#[derive(Clone)]
pub struct AudioInputNode {
    pub device_name: String,
    pub buffer: AudioBuffer,
    pub is_active: bool,
}

impl AudioInputNode {
    pub fn new(device_name: String) -> Self {
        Self {
            device_name,
            buffer: Arc::new(Mutex::new(Vec::new())),
            is_active: false,
        }
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
        1
    }

    fn input_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("Out")
        } else {
            None
        }
    }

    fn buffer(&self) -> Option<&AudioBuffer> {
        Some(&self.buffer)
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
    pub buffer: AudioBuffer,
    pub is_active: bool,
}

impl AudioOutputNode {
    pub fn new(device_name: String) -> Self {
        Self {
            device_name,
            buffer: Arc::new(Mutex::new(Vec::new())),
            is_active: false,
        }
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
        1
    }

    fn output_count(&self) -> usize {
        0
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_type(&self, _index: usize) -> Option<PinType> {
        None
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some("In")
        } else {
            None
        }
    }

    fn output_pin_name(&self, _index: usize) -> Option<&str> {
        None
    }

    fn buffer(&self) -> Option<&AudioBuffer> {
        Some(&self.buffer)
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

    fn buffer(&self) -> Option<&AudioBuffer> {
        delegate_node_behavior!(self, buffer)
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
