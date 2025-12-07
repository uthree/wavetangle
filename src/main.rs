mod audio;
mod dsp;
mod effect_processor;
mod graph;
mod nodes;
mod project;
mod viewer;

use std::path::PathBuf;

use eframe::egui;
use egui_snarl::ui::SnarlStyle;
use egui_snarl::Snarl;

use crate::audio::{AudioSystem, BUFFER_SIZES, SAMPLE_RATES};
use crate::graph::AudioGraphProcessor;
use crate::nodes::AudioNode;
use crate::project::ProjectFile;
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
    /// Audio Settings ウィンドウの表示状態
    show_audio_settings: bool,
    /// 現在開いているファイルのパス
    current_file_path: Option<PathBuf>,
    /// ステータスメッセージ
    status_message: Option<(String, std::time::Instant)>,
}

impl WavetangleApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // 日本語フォントの設定
        let mut fonts = egui::FontDefinitions::default();

        // システムフォントから日本語対応フォントを追加
        #[cfg(target_os = "macos")]
        {
            if let Ok(font_data) = std::fs::read("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
            {
                fonts.font_data.insert(
                    "japanese".to_owned(),
                    std::sync::Arc::new(egui::FontData::from_owned(font_data)),
                );
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .push("japanese".to_owned());
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(font_data) = std::fs::read("C:\\Windows\\Fonts\\msgothic.ttc") {
                fonts.font_data.insert(
                    "japanese".to_owned(),
                    std::sync::Arc::new(egui::FontData::from_owned(font_data)),
                );
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .push("japanese".to_owned());
            }
        }

        cc.egui_ctx.set_fonts(fonts);

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
            show_audio_settings: false,
            current_file_path: None,
            status_message: None,
        }
    }

    /// ステータスメッセージを設定
    fn set_status(&mut self, message: &str) {
        self.status_message = Some((message.to_string(), std::time::Instant::now()));
    }

    /// プロジェクトを開く
    fn open_project(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Wavetangle Project", &["wtg", "json"])
            .pick_file()
        {
            match ProjectFile::load_from_file(&path) {
                Ok(project) => {
                    // すべてのストリームを停止
                    self.deactivate_all();
                    self.graph_processor.stop_all_streams();

                    // グラフを読み込み
                    self.snarl = project.to_snarl();
                    self.current_file_path = Some(path.clone());
                    self.set_status(&format!("Opened: {}", path.display()));
                }
                Err(e) => {
                    self.set_status(&format!("Error opening file: {}", e));
                }
            }
        }
    }

    /// プロジェクトを保存（上書き）
    fn save_project(&mut self) {
        if let Some(path) = &self.current_file_path.clone() {
            self.save_project_to(path.clone());
        } else {
            self.save_project_as();
        }
    }

    /// 名前をつけて保存
    fn save_project_as(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Wavetangle Project", &["wtg"])
            .set_file_name("untitled.wtg")
            .save_file()
        {
            self.save_project_to(path);
        }
    }

    /// 指定されたパスに保存
    fn save_project_to(&mut self, path: PathBuf) {
        let project = ProjectFile::from_snarl(&self.snarl);
        match project.save_to_file(&path) {
            Ok(()) => {
                self.current_file_path = Some(path.clone());
                self.set_status(&format!("Saved: {}", path.display()));
            }
            Err(e) => {
                self.set_status(&format!("Error saving file: {}", e));
            }
        }
    }

    /// アクティブなオーディオノードがあるかチェック
    fn has_active_audio(&self) -> bool {
        self.graph_processor.has_active_streams()
    }

    /// すべてのノードをアクティブ化
    fn activate_all(&mut self) {
        for (_node_id, node) in self.snarl.nodes_ids_mut() {
            node.set_active(true);
        }
    }

    /// すべてのノードを非アクティブ化
    fn deactivate_all(&mut self) {
        for (_node_id, node) in self.snarl.nodes_ids_mut() {
            node.set_active(false);
        }
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
                    if ui.button("New").clicked() {
                        self.deactivate_all();
                        self.graph_processor.stop_all_streams();
                        self.snarl = Snarl::new();
                        self.current_file_path = None;
                        self.set_status("New project created");
                        ui.close();
                    }
                    if ui.button("Open...").clicked() {
                        self.open_project();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Save").clicked() {
                        self.save_project();
                        ui.close();
                    }
                    if ui.button("Save As...").clicked() {
                        self.save_project_as();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Transport", |ui| {
                    if ui.button("Activate All").clicked() {
                        self.activate_all();
                        ui.close();
                    }
                    if ui.button("Deactivate All").clicked() {
                        self.deactivate_all();
                        ui.close();
                    }
                });

                ui.menu_button("Audio", |ui| {
                    if ui.button("Settings...").clicked() {
                        self.show_audio_settings = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Refresh Devices").clicked() {
                        self.input_devices = self.audio_system.input_device_names();
                        self.output_devices = self.audio_system.output_device_names();
                        ui.close();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        ui.close();
                    }
                });
            });
        });

        // Audio Settings ウィンドウ（独立したOSウィンドウ）
        if self.show_audio_settings {
            let mut config = self.audio_system.config();
            let is_active = self.has_active_audio();
            let input_devices = self.input_devices.clone();
            let output_devices = self.output_devices.clone();

            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("audio_settings"),
                egui::ViewportBuilder::default()
                    .with_title("Audio Settings")
                    .with_inner_size([300.0, 250.0])
                    .with_resizable(false),
                |ctx, _class| {
                    egui::CentralPanel::default().show(ctx, |ui| {
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

                        ui.add_space(10.0);
                        ui.separator();

                        // デバイス一覧セクション
                        ui.collapsing("Input Devices", |ui| {
                            for name in &input_devices {
                                ui.label(name);
                            }
                        });

                        ui.collapsing("Output Devices", |ui| {
                            for name in &output_devices {
                                ui.label(name);
                            }
                        });
                    });

                    // ウィンドウが閉じられたかチェック
                    if ctx.input(|i| i.viewport().close_requested()) {
                        self.show_audio_settings = false;
                    }
                },
            );

            // 設定が変更されたら適用
            if config != self.audio_system.config() {
                self.audio_system.set_config(config);
            }
        }

        // ステータスバー
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // ファイル名表示
                if let Some(path) = &self.current_file_path {
                    let file_name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "Unknown".to_string());
                    ui.label(format!("File: {}", file_name));
                } else {
                    ui.label("File: (Untitled)");
                }

                ui.separator();

                // ステータスメッセージ（3秒間表示）
                if let Some((msg, time)) = &self.status_message {
                    if time.elapsed().as_secs() < 3 {
                        ui.label(msg);
                    } else {
                        self.status_message = None;
                    }
                }
            });
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
