use egui::Ui;

use super::{
    channel_name, impl_as_any, AudioInputPort, AudioOutputPort, ChannelBuffer, NodeBase,
    NodeBuffers, NodeType, NodeUI, NodeUIContext, PinType, SpectrumDisplay, FFT_SIZE,
};

// ============================================================================
// Audio Input Node
// ============================================================================

/// オーディオ入力デバイスノード
#[derive(Clone)]
pub struct AudioInputNode {
    pub device_name: String,
    /// バッファ管理（出力専用）
    pub buffers: NodeBuffers,
    pub is_active: bool,
    /// スペクトラム表示
    pub spectrum_display: SpectrumDisplay,
}

impl AudioInputNode {
    pub fn new(device_name: String, channels: u16) -> Self {
        Self {
            device_name,
            buffers: NodeBuffers::output_only(channels),
            is_active: false,
            spectrum_display: SpectrumDisplay::with_analyzer(FFT_SIZE),
        }
    }

    /// チャンネル数を取得
    pub fn channels(&self) -> u16 {
        self.buffers.output_count() as u16
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        let channels = channels.max(1);
        self.buffers.resize_outputs(channels as usize);
    }
}

// AudioInputNodeのトレイト実装
// 出力専用ノード: NodeBase + AudioOutputPort + NodeUI
// AudioInputPortはデフォルト実装を使用（入力ピンなし）

impl NodeBase for AudioInputNode {
    fn node_type(&self) -> NodeType {
        NodeType::AudioInput
    }

    fn title(&self) -> &str {
        "Audio Input"
    }

    impl_as_any!();
}

/// AudioInputNodeは入力ピンを持たない（デフォルト実装を使用）
impl AudioInputPort for AudioInputNode {}

/// AudioInputNodeは出力ピンを持つ
impl AudioOutputPort for AudioInputNode {
    fn output_count(&self) -> usize {
        self.buffers.output_count()
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.buffers.output_count() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
    }

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        self.buffers.output_buffer(channel)
    }

    fn channels(&self) -> u16 {
        self.buffers.output_count() as u16
    }

    fn set_channels(&mut self, channels: u16) {
        self.resize_buffers(channels);
    }
}

impl NodeUI for AudioInputNode {
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
                    ui.label(format!("{}ch", self.channels()));
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
                ui.checkbox(&mut self.spectrum_display.enabled, "Spectrum");
                if self.spectrum_display.enabled {
                    self.spectrum_display
                        .show_line(ui, &format!("input_spectrum_{:?}", ctx.node_id));
                }
            }
        });
    }
}

// ============================================================================
// Audio Output Node
// ============================================================================

/// オーディオ出力デバイスノード
#[derive(Clone)]
pub struct AudioOutputNode {
    pub device_name: String,
    /// バッファ管理（入力専用）
    pub buffers: NodeBuffers,
    pub is_active: bool,
    /// スペクトラム表示
    pub spectrum_display: SpectrumDisplay,
}

impl AudioOutputNode {
    pub fn new(device_name: String, channels: u16) -> Self {
        Self {
            device_name,
            buffers: NodeBuffers::input_only(channels),
            is_active: false,
            spectrum_display: SpectrumDisplay::with_analyzer(FFT_SIZE),
        }
    }

    /// チャンネル数を取得
    pub fn channels(&self) -> u16 {
        self.buffers.input_count() as u16
    }

    /// チャンネル数に合わせてバッファを再作成
    pub fn resize_buffers(&mut self, channels: u16) {
        let channels = channels.max(1);
        self.buffers.resize_inputs(channels as usize);
    }
}

// AudioOutputNodeのトレイト実装
// 入力専用ノード: NodeBase + AudioInputPort + NodeUI
// AudioOutputPortは内部バッファアクセス用にchannel_bufferとchannelsのみ実装

impl NodeBase for AudioOutputNode {
    fn node_type(&self) -> NodeType {
        NodeType::AudioOutput
    }

    fn title(&self) -> &str {
        "Audio Output"
    }

    impl_as_any!();
}

/// AudioOutputNodeは入力ピンを持つ
impl AudioInputPort for AudioOutputNode {
    fn input_count(&self) -> usize {
        self.buffers.input_count()
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.buffers.input_count() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        self.buffers.input_buffer(index)
    }
}

/// AudioOutputNodeは出力ピンを持たないが、
/// 内部バッファへのアクセスとチャンネル管理が必要
impl AudioOutputPort for AudioOutputNode {
    // output_count, output_pin_type, output_pin_nameはデフォルト（0, None）を使用

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        // 出力ノードでは入力バッファを返す（データ送出用）
        self.buffers.input_buffer(channel)
    }

    fn channels(&self) -> u16 {
        self.buffers.input_count() as u16
    }

    fn set_channels(&mut self, channels: u16) {
        self.resize_buffers(channels);
    }
}

impl NodeUI for AudioOutputNode {
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
                    ui.label(format!("{}ch", self.channels()));
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
                ui.checkbox(&mut self.spectrum_display.enabled, "Spectrum");
                if self.spectrum_display.enabled {
                    self.spectrum_display
                        .show_line(ui, &format!("output_spectrum_{:?}", ctx.node_id));
                }
            }
        });
    }
}
