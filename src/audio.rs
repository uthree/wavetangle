use std::collections::HashMap;
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, Host, SampleRate, Stream, StreamConfig};

use crate::nodes::ChannelBuffer;

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

/// 出力ストリームID
pub type OutputStreamId = usize;

/// オーディオシステム - デバイスの管理とストリームの制御
pub struct AudioSystem {
    host: Host,
    input_stream: Option<Stream>,
    /// 複数の出力ストリームを管理（ID -> Stream）
    output_streams: HashMap<OutputStreamId, Stream>,
    /// 次の出力ストリームID
    next_output_id: OutputStreamId,
    config: AudioConfig,
}

impl AudioSystem {
    pub fn new() -> Self {
        let host = cpal::default_host();
        Self {
            host,
            input_stream: None,
            output_streams: HashMap::new(),
            next_output_id: 0,
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

    /// 入力デバイスのデフォルトチャンネル数を取得
    pub fn input_device_channels(&self, name: &str) -> Option<u16> {
        let device = self.get_input_device(name)?;
        device.default_input_config().ok().map(|c| c.channels())
    }

    /// 出力デバイスのデフォルトチャンネル数を取得
    pub fn output_device_channels(&self, name: &str) -> Option<u16> {
        let device = self.get_output_device(name)?;
        device.default_output_config().ok().map(|c| c.channels())
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

    /// 入力ストリームを開始し、チャンネル数を返す
    /// channel_buffers: チャンネルごとのリングバッファ
    pub fn start_input(
        &mut self,
        device_name: &str,
        channel_buffers: Vec<ChannelBuffer>,
    ) -> Result<u16, String> {
        let device = self
            .get_input_device(device_name)
            .ok_or_else(|| format!("Input device '{}' not found", device_name))?;

        let stream_config = self.build_stream_config(&device, true)?;
        let channels = stream_config.channels;

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let num_channels = channel_buffers.len();
                    if num_channels == 0 {
                        return;
                    }

                    // インターリーブされたデータをチャンネルごとに分離
                    let frame_count = data.len() / channels as usize;
                    for ch in 0..num_channels.min(channels as usize) {
                        let mut buf = channel_buffers[ch].lock();
                        let samples: Vec<f32> = (0..frame_count)
                            .map(|frame| data[frame * channels as usize + ch])
                            .collect();
                        buf.push(&samples);
                    }
                },
                |err| eprintln!("Input stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play input stream: {}", e))?;
        self.input_stream = Some(stream);
        Ok(channels)
    }

    /// 出力ストリームを開始し、チャンネル数とストリームIDを返す
    /// channel_buffers: 出力ノード自身のバッファ（データはeffect_processorでコピー済み）
    pub fn start_output(
        &mut self,
        device_name: &str,
        channel_buffers: Vec<ChannelBuffer>,
    ) -> Result<(u16, OutputStreamId), String> {
        let device = self
            .get_output_device(device_name)
            .ok_or_else(|| format!("Output device '{}' not found", device_name))?;

        let stream_config = self.build_stream_config(&device, false)?;
        let channels = stream_config.channels;

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let num_channels = channel_buffers.len();
                    if num_channels == 0 {
                        data.fill(0.0);
                        return;
                    }

                    let frame_count = data.len() / channels as usize;

                    // ユニークなバッファを特定（consume用）
                    let mut unique_buffers: Vec<ChannelBuffer> = Vec::new();
                    for ch in 0..channels as usize {
                        let source_ch = ch.min(num_channels - 1);
                        let buf = &channel_buffers[source_ch];
                        if !unique_buffers.iter().any(|b| Arc::ptr_eq(b, buf)) {
                            unique_buffers.push(buf.clone());
                        }
                    }

                    // 各チャンネルからデータを読み取ってインターリーブ
                    for ch in 0..channels as usize {
                        let source_ch = ch.min(num_channels - 1);
                        let buf = channel_buffers[source_ch].lock();
                        let samples = buf.read(frame_count);

                        for (frame, &sample) in samples.iter().enumerate() {
                            data[frame * channels as usize + ch] = sample;
                        }
                    }

                    // 全チャンネル読み取り後、ユニークバッファごとに1回だけconsume
                    for buf in &unique_buffers {
                        buf.lock().consume(frame_count);
                    }
                },
                |err| eprintln!("Output stream error: {}", err),
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to play output stream: {}", e))?;

        // ストリームをHashMapに追加
        let stream_id = self.next_output_id;
        self.next_output_id += 1;
        self.output_streams.insert(stream_id, stream);

        Ok((channels, stream_id))
    }

    /// 入力ストリームを停止
    pub fn stop_input(&mut self) {
        self.input_stream = None;
    }

    /// 指定したIDの出力ストリームを停止
    pub fn stop_output(&mut self, stream_id: OutputStreamId) {
        self.output_streams.remove(&stream_id);
    }
}

impl Default for AudioSystem {
    fn default() -> Self {
        Self::new()
    }
}
