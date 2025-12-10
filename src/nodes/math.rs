use egui::Ui;

use super::{
    impl_as_any, impl_input_port_nb, impl_single_output_port_nb, AudioInputPort, AudioOutputPort,
    ChannelBuffer, NodeBase, NodeBuffers, NodeType, NodeUI, NodeUIContext, PinType,
};

// ============================================================================
// Add Node (Arithmetic)
// ============================================================================

/// 加算ノード - 2つの信号を加算
#[derive(Clone)]
pub struct AddNode {
    /// バッファ管理（2入力1出力）
    pub buffers: NodeBuffers,
    pub is_active: bool,
}

impl AddNode {
    pub fn new() -> Self {
        Self {
            buffers: NodeBuffers::multi_input(2),
            is_active: false,
        }
    }
}

impl Default for AddNode {
    fn default() -> Self {
        Self::new()
    }
}

// AddNodeのトレイト実装（2入力1出力、マクロ使用）
impl NodeBase for AddNode {
    fn node_type(&self) -> NodeType {
        NodeType::Add
    }

    fn title(&self) -> &str {
        "Add"
    }

    impl_as_any!();
}

impl_input_port_nb!(AddNode, ["A", "B"]);
impl_single_output_port_nb!(AddNode);

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
    /// バッファ管理（2入力1出力）
    pub buffers: NodeBuffers,
    pub is_active: bool,
}

impl MultiplyNode {
    pub fn new() -> Self {
        Self {
            buffers: NodeBuffers::multi_input(2),
            is_active: false,
        }
    }
}

impl Default for MultiplyNode {
    fn default() -> Self {
        Self::new()
    }
}

// MultiplyNodeのトレイト実装（2入力1出力、マクロ使用）
impl NodeBase for MultiplyNode {
    fn node_type(&self) -> NodeType {
        NodeType::Multiply
    }

    fn title(&self) -> &str {
        "Multiply"
    }

    impl_as_any!();
}

impl_input_port_nb!(MultiplyNode, ["A", "B"]);
impl_single_output_port_nb!(MultiplyNode);

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
