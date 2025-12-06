use std::sync::Arc;

use parking_lot::Mutex;

/// オーディオバッファ - ノード間でデータを共有するため
pub type AudioBuffer = Arc<Mutex<Vec<f32>>>;

/// ノードのピンタイプ
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PinType {
    AudioInput,
    AudioOutput,
}

/// オーディオグラフのノード
#[derive(Clone)]
pub enum AudioNode {
    /// オーディオ入力デバイスノード
    AudioInput {
        device_name: String,
        buffer: AudioBuffer,
        is_active: bool,
    },
    /// オーディオ出力デバイスノード
    AudioOutput {
        device_name: String,
        buffer: AudioBuffer,
        is_active: bool,
    },
}

impl AudioNode {
    pub fn new_audio_input(device_name: String) -> Self {
        Self::AudioInput {
            device_name,
            buffer: Arc::new(Mutex::new(Vec::new())),
            is_active: false,
        }
    }

    pub fn new_audio_output(device_name: String) -> Self {
        Self::AudioOutput {
            device_name,
            buffer: Arc::new(Mutex::new(Vec::new())),
            is_active: false,
        }
    }

    /// ノードのタイトルを取得
    pub fn title(&self) -> &str {
        match self {
            AudioNode::AudioInput { .. } => "Audio Input",
            AudioNode::AudioOutput { .. } => "Audio Output",
        }
    }

    /// ノードの入力ピン数を取得
    pub fn input_count(&self) -> usize {
        match self {
            AudioNode::AudioInput { .. } => 0,
            AudioNode::AudioOutput { .. } => 1,
        }
    }

    /// ノードの出力ピン数を取得
    pub fn output_count(&self) -> usize {
        match self {
            AudioNode::AudioInput { .. } => 1,
            AudioNode::AudioOutput { .. } => 0,
        }
    }

    /// ピンタイプを取得
    pub fn pin_type(&self, is_input: bool, _index: usize) -> PinType {
        if is_input {
            PinType::AudioInput
        } else {
            PinType::AudioOutput
        }
    }

    /// デバイス名を取得
    #[allow(dead_code)]
    pub fn device_name(&self) -> &str {
        match self {
            AudioNode::AudioInput { device_name, .. } => device_name,
            AudioNode::AudioOutput { device_name, .. } => device_name,
        }
    }

    /// デバイス名を設定
    #[allow(dead_code)]
    pub fn set_device_name(&mut self, name: String) {
        match self {
            AudioNode::AudioInput { device_name, .. } => *device_name = name,
            AudioNode::AudioOutput { device_name, .. } => *device_name = name,
        }
    }

    /// アクティブ状態を取得
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        match self {
            AudioNode::AudioInput { is_active, .. } => *is_active,
            AudioNode::AudioOutput { is_active, .. } => *is_active,
        }
    }

    /// アクティブ状態を設定
    pub fn set_active(&mut self, active: bool) {
        match self {
            AudioNode::AudioInput { is_active, .. } => *is_active = active,
            AudioNode::AudioOutput { is_active, .. } => *is_active = active,
        }
    }

    /// バッファを取得
    pub fn buffer(&self) -> Option<&AudioBuffer> {
        match self {
            AudioNode::AudioInput { buffer, .. } => Some(buffer),
            AudioNode::AudioOutput { buffer, .. } => Some(buffer),
        }
    }
}
