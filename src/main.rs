mod audio;
mod graph;
mod nodes;
mod viewer;

use eframe::egui;
use egui_snarl::ui::SnarlStyle;
use egui_snarl::Snarl;

use crate::audio::{AudioSystem, BUFFER_SIZES, SAMPLE_RATES};
use crate::graph::AudioGraphProcessor;
use crate::nodes::AudioNode;
use crate::viewer::AudioGraphViewer;

/// アプリケーションのメイン状態
struct WavetangleApp {
    snarl: Snarl<AudioNode>,
    snarl_style: SnarlStyle,
    audio_system: AudioSystem,
    graph_processor: AudioGraphProcessor,
    /// キャッシュされたデバイスリスト
    input_devices: Vec<String>,
    output_devices: Vec<String>,
}

impl WavetangleApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let audio_system = AudioSystem::new();
        let input_devices = audio_system.input_device_names();
        let output_devices = audio_system.output_device_names();
        Self {
            snarl: Snarl::new(),
            snarl_style: SnarlStyle::default(),
            audio_system,
            graph_processor: AudioGraphProcessor::new(),
            input_devices,
            output_devices,
        }
    }

    /// アクティブなオーディオノードがあるかチェック
    fn has_active_audio(&self) -> bool {
        self.graph_processor.has_active_streams()
    }
}

impl eframe::App for WavetangleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // オーディオグラフの処理
        self.graph_processor
            .process(&mut self.snarl, &mut self.audio_system);

        // メニューバー
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            #[allow(deprecated)]
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Clear Graph").clicked() {
                        self.snarl = Snarl::new();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        ui.close();
                    }
                });
            });
        });

        // サイドパネル - デバイス情報と設定
        egui::SidePanel::left("device_panel")
            .resizable(true)
            .min_width(220.0)
            .show(ctx, |ui| {
                // オーディオ設定セクション
                ui.heading("Audio Settings");
                ui.separator();

                let mut config = self.audio_system.config();
                let is_active = self.has_active_audio();

                ui.add_enabled_ui(!is_active, |ui| {
                    // サンプルレート選択
                    ui.horizontal(|ui| {
                        ui.label("Sample Rate:");
                        egui::ComboBox::from_id_salt("sample_rate")
                            .selected_text(format!("{} Hz", config.sample_rate))
                            .show_ui(ui, |ui| {
                                for &rate in SAMPLE_RATES {
                                    ui.selectable_value(
                                        &mut config.sample_rate,
                                        rate,
                                        format!("{} Hz", rate),
                                    );
                                }
                            });
                    });

                    // バッファサイズ選択
                    ui.horizontal(|ui| {
                        ui.label("Buffer Size:");
                        egui::ComboBox::from_id_salt("buffer_size")
                            .selected_text(format!("{}", config.buffer_size))
                            .show_ui(ui, |ui| {
                                for &size in BUFFER_SIZES {
                                    ui.selectable_value(
                                        &mut config.buffer_size,
                                        size,
                                        format!("{}", size),
                                    );
                                }
                            });
                    });
                });

                if is_active {
                    ui.label("(Stop audio to change settings)");
                }

                // 設定が変更されたら適用
                if config != self.audio_system.config() {
                    self.audio_system.set_config(config);
                }

                ui.add_space(10.0);

                // デバイス一覧セクション
                ui.heading("Audio Devices");
                ui.separator();

                ui.collapsing("Input Devices", |ui| {
                    for name in &self.input_devices {
                        ui.label(format!("  {}", name));
                    }
                });

                ui.collapsing("Output Devices", |ui| {
                    for name in &self.output_devices {
                        ui.label(format!("  {}", name));
                    }
                });

                ui.separator();
                if ui.button("Refresh Devices").clicked() {
                    self.input_devices = self.audio_system.input_device_names();
                    self.output_devices = self.audio_system.output_device_names();
                }

                ui.add_space(10.0);

                // 操作説明セクション
                ui.heading("Instructions");
                ui.separator();
                ui.label("Right-click to add nodes");
                ui.label("Drag from pins to connect");
                ui.label("Right-click node to delete");
            });

        // メインパネル - ノードグラフ
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut viewer = AudioGraphViewer::with_devices(
                self.input_devices.clone(),
                self.output_devices.clone(),
            );
            self.snarl
                .show(&mut viewer, &self.snarl_style, "audio_graph", ui);
        });

        // オーディオがアクティブな場合のみ高頻度で更新
        if self.has_active_audio() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("Wavetangle - Audio Graph Editor"),
        ..Default::default()
    };

    eframe::run_native(
        "Wavetangle",
        options,
        Box::new(|cc| Ok(Box::new(WavetangleApp::new(cc)))),
    )
}
