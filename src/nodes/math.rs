use egui::Ui;

use super::{
    impl_as_any, new_channel_buffer, AudioInputPort, AudioOutputPort, ChannelBuffer, NodeBase,
    NodeType, NodeUI, NodeUIContext, PinType, DEFAULT_RING_BUFFER_SIZE,
};

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

// AddNodeのトレイト実装（2入力1出力）
impl NodeBase for AddNode {
    fn node_type(&self) -> NodeType {
        NodeType::Add
    }

    fn title(&self) -> &str {
        "Add"
    }

    impl_as_any!();
}

impl AudioInputPort for AddNode {
    fn input_count(&self) -> usize {
        2
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < 2 {
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

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }
}

impl AudioOutputPort for AddNode {
    fn output_count(&self) -> usize {
        1
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
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
}

impl NodeUI for AddNode {
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

// MultiplyNodeのトレイト実装（2入力1出力）
impl NodeBase for MultiplyNode {
    fn node_type(&self) -> NodeType {
        NodeType::Multiply
    }

    fn title(&self) -> &str {
        "Multiply"
    }

    impl_as_any!();
}

impl AudioInputPort for MultiplyNode {
    fn input_count(&self) -> usize {
        2
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < 2 {
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

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.input_buffers.get(index).cloned()
    }
}

impl AudioOutputPort for MultiplyNode {
    fn output_count(&self) -> usize {
        1
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index == 0 {
            Some(PinType::Audio)
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
}

impl NodeUI for MultiplyNode {
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
