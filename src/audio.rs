use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, Host, SampleRate, Stream, StreamConfig};
use parking_lot::Mutex;

use crate::nodes::AudioBuffer;

/// 一般的なサンプルレートの選択肢
pub const SAMPLE_RATES: &[u32] = &[22050, 44100, 48000, 88200, 96000, 192000];

/// 一般的なバッファサイズの選択肢
pub const BUFFER_SIZES: &[u32] = &[64, 128, 256, 512, 1024, 2048, 4096];

/// オーディオ設定
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            buffer_size: 512,
        }
    }
}

/// オーディオシステム - デバイスの管理とストリームの制御
pub struct AudioSystem {
    host: Host,
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    input_buffer: AudioBuffer,
    output_buffer: AudioBuffer,
    config: AudioConfig,
}

impl AudioSystem {
    pub fn new() -> Self {
        let host = cpal::default_host();
        Self {
            host,
            input_stream: None,
            output_stream: None,
            input_buffer: Arc::new(Mutex::new(Vec::new())),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
            config: AudioConfig::default(),
        }
    }

    /// 現在のオーディオ設定を取得
    pub fn config(&self) -> AudioConfig {
        self.config
    }

    /// オーディオ設定を更新（ストリームが停止している時のみ有効）
    pub fn set_config(&mut self, config: AudioConfig) {
        self.config = config;
    }

    /// 利用可能な入力デバイス名のリストを取得
    pub fn input_device_names(&self) -> Vec<String> {
        self.host
            .input_devices()
            .map(|devices| devices.filter_map(|d| d.name().ok()).collect::<Vec<_>>())
            .unwrap_or_default()
    }

    /// 利用可能な出力デバイス名のリストを取得
    pub fn output_device_names(&self) -> Vec<String> {
        self.host
            .output_devices()
            .map(|devices| devices.filter_map(|d| d.name().ok()).collect::<Vec<_>>())
            .unwrap_or_default()
    }

    /// 名前からデバイスを取得
    fn get_input_device(&self, name: &str) -> Option<Device> {
        self.host
            .input_devices()
            .ok()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
    }

    fn get_output_device(&self, name: &str) -> Option<Device> {
        self.host
            .output_devices()
            .ok()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
    }

    /// 設定からStreamConfigを作成
    fn build_stream_config(&self, device: &Device, is_input: bool) -> Result<StreamConfig, String> {
        let default_config = if is_input {
            device.default_input_config()
        } else {
            device.default_output_config()
        }
        .map_err(|e| format!("Failed to get default config: {}", e))?;

        // デフォルトのチャンネル数を使用
        let channels = default_config.channels();

        Ok(StreamConfig {
            channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: BufferSize::Fixed(self.config.buffer_size),
        })
    }

    /// 入力ストリームを開始
    pub fn start_input(&mut self, device_name: &str, buffer: AudioBuffer) -> Result<(), String> {
        let device = self
            .get_input_device(device_name)
            .ok_or_else(|| format!("Input device '{}' not found", device_name))?;

        let stream_config = self.build_stream_config(&device, true)?;
        self.input_buffer = buffer.clone();

        let buffer_clone = buffer;
        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = buffer_clone.lock();
                    buf.clear();
                    buf.extend_from_slice(data);
                },
                |err| eprintln!("Input stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play input stream: {}", e))?;
        self.input_stream = Some(stream);
        Ok(())
    }

    /// 出力ストリームを開始
    pub fn start_output(&mut self, device_name: &str, buffer: AudioBuffer) -> Result<(), String> {
        let device = self
            .get_output_device(device_name)
            .ok_or_else(|| format!("Output device '{}' not found", device_name))?;

        let stream_config = self.build_stream_config(&device, false)?;
        self.output_buffer = buffer.clone();

        let buffer_clone = buffer;
        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let buf = buffer_clone.lock();
                    for (i, sample) in data.iter_mut().enumerate() {
                        *sample = buf.get(i % buf.len().max(1)).copied().unwrap_or(0.0);
                    }
                },
                |err| eprintln!("Output stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play output stream: {}", e))?;
        self.output_stream = Some(stream);
        Ok(())
    }

    /// 入力ストリームを停止
    pub fn stop_input(&mut self) {
        self.input_stream = None;
    }

    /// 出力ストリームを停止
    pub fn stop_output(&mut self) {
        self.output_stream = None;
    }

    /// 入力バッファへの参照を取得
    #[allow(dead_code)]
    pub fn input_buffer(&self) -> &AudioBuffer {
        &self.input_buffer
    }

    /// 出力バッファへの参照を取得
    #[allow(dead_code)]
    pub fn output_buffer(&self) -> &AudioBuffer {
        &self.output_buffer
    }
}

impl Default for AudioSystem {
    fn default() -> Self {
        Self::new()
    }
}
