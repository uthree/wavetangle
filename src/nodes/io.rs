use std::sync::Arc;

use egui::Ui;
use parking_lot::Mutex;

use super::{
    channel_name, impl_as_any, new_channel_buffer, resize_channel_buffers, show_spectrum_line,
    AudioInputPort, AudioOutputPort, ChannelBuffer, NodeBase, NodeType, NodeUI, NodeUIContext,
    PinType, DEFAULT_RING_BUFFER_SIZE, FFT_SIZE,
};

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
        self.channel_buffers.len()
    }

    fn output_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channel_buffers.len() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn output_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
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
        self.channel_buffers.len()
    }

    fn input_pin_type(&self, index: usize) -> Option<PinType> {
        if index < self.channel_buffers.len() {
            Some(PinType::Audio)
        } else {
            None
        }
    }

    fn input_pin_name(&self, index: usize) -> Option<&str> {
        channel_name(index)
    }

    fn input_buffer(&self, index: usize) -> Option<ChannelBuffer> {
        // AudioOutputの入力はchannel_buffersと同じ（データを受け取る）
        self.channel_buffers.get(index).cloned()
    }
}

/// AudioOutputNodeは出力ピンを持たないが、
/// 内部バッファへのアクセスとチャンネル管理が必要
impl AudioOutputPort for AudioOutputNode {
    // output_count, output_pin_type, output_pin_nameはデフォルト（0, None）を使用

    fn channel_buffer(&self, channel: usize) -> Option<ChannelBuffer> {
        self.channel_buffers.get(channel).cloned()
    }

    fn channels(&self) -> u16 {
        self.channels
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
