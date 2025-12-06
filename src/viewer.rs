use std::sync::Arc;

use egui::{Color32, Ui};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, Points};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};
use parking_lot::Mutex;

use crate::dsp::EqPoint;
use crate::nodes::{AudioNode, FilterType, NodeBehavior, PinType, FFT_SIZE};

/// ピンタイプに応じた色を取得
fn pin_color(pin_type: PinType) -> Color32 {
    match pin_type {
        PinType::Audio => Color32::from_rgb(100, 200, 100),
    }
}

/// HSVからRGBに変換
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color32 {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// スペクトラムを折れ線グラフで表示
fn show_spectrum_line(ui: &mut Ui, plot_id: &str, spectrum: &Arc<Mutex<Vec<f32>>>) {
    let spectrum_data = spectrum.lock();
    let point_count = 100;
    let spectrum_len = spectrum_data.len();

    // ラインデータを作成
    let points: Vec<[f64; 2]> = (0..point_count)
        .map(|i| {
            let x = i as f64 / point_count as f64;
            // 対数スケールでインデックスをマッピング（低周波を細かく表示）
            let freq_idx = (x.powf(2.0) * spectrum_len as f64) as usize;
            let freq_idx = freq_idx.min(spectrum_len.saturating_sub(1));

            // マグニチュードを正規化（対数スケール、dB変換）
            let magnitude = if freq_idx < spectrum_data.len() {
                spectrum_data[freq_idx]
            } else {
                0.0
            };
            let db = if magnitude > 1e-6 {
                20.0 * (magnitude as f64).log10()
            } else {
                -80.0
            };
            // -80dB〜0dBを0.0〜1.0にマッピング
            let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0);

            [x, normalized]
        })
        .collect();

    drop(spectrum_data); // ロックを解放

    // egui_plotで折れ線グラフを表示
    Plot::new(plot_id)
        .height(80.0)
        .width(180.0)
        .show_axes([false, true])
        .show_grid([true, true])
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("spectrum", PlotPoints::from(points))
                    .color(Color32::from_rgb(100, 200, 100))
                    .width(1.5),
            );
        });
}

/// オーディオグラフビューアー
pub struct AudioGraphViewer {
    pub input_devices: Vec<String>,
    pub output_devices: Vec<String>,
}

impl AudioGraphViewer {
    /// キャッシュされたデバイスリストから作成
    pub fn with_devices(input_devices: Vec<String>, output_devices: Vec<String>) -> Self {
        Self {
            input_devices,
            output_devices,
        }
    }
}

impl SnarlViewer<AudioNode> for AudioGraphViewer {
    fn title(&mut self, node: &AudioNode) -> String {
        node.title().to_string()
    }

    fn inputs(&mut self, node: &AudioNode) -> usize {
        node.input_count()
    }

    fn outputs(&mut self, node: &AudioNode) -> usize {
        node.output_count()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];

        // ピン名を表示
        if let Some(name) = node.input_pin_name(pin.id.input) {
            ui.label(name);
        }

        // ピンタイプに応じた色
        let color = node
            .input_pin_type(pin.id.input)
            .map(pin_color)
            .unwrap_or(Color32::GRAY);

        PinInfo::circle().with_fill(color)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];

        // ピン名を表示
        if let Some(name) = node.output_pin_name(pin.id.output) {
            ui.label(name);
        }

        // ピンタイプに応じた色
        let color = node
            .output_pin_type(pin.id.output)
            .map(pin_color)
            .unwrap_or(Color32::GRAY);

        PinInfo::circle().with_fill(color)
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) {
        let node = &mut snarl[node_id];

        match node {
            AudioNode::AudioInput(input_node) => {
                self.show_audio_input_body(node_id, input_node, ui);
            }
            AudioNode::AudioOutput(output_node) => {
                self.show_audio_output_body(node_id, output_node, ui);
            }
            AudioNode::Gain(gain_node) => {
                self.show_gain_body(node_id, gain_node, ui);
            }
            AudioNode::Add(add_node) => {
                self.show_add_body(node_id, add_node, ui);
            }
            AudioNode::Multiply(multiply_node) => {
                self.show_multiply_body(node_id, multiply_node, ui);
            }
            AudioNode::Filter(filter_node) => {
                self.show_filter_body(node_id, filter_node, ui);
            }
            AudioNode::SpectrumAnalyzer(spectrum_node) => {
                self.show_spectrum_body(node_id, spectrum_node, ui);
            }
            AudioNode::Compressor(compressor_node) => {
                self.show_compressor_body(node_id, compressor_node, ui);
            }
            AudioNode::PitchShift(pitch_shift_node) => {
                self.show_pitch_shift_body(node_id, pitch_shift_node, ui);
            }
            AudioNode::GraphicEq(graphic_eq_node) => {
                self.show_graphic_eq_body(node_id, graphic_eq_node, ui);
            }
        }
    }

    fn has_body(&mut self, _node: &AudioNode) -> bool {
        true
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<AudioNode>) {
        let from_node = &snarl[from.id.node];
        let to_node = &snarl[to.id.node];

        let from_type = from_node.output_pin_type(from.id.output);
        let to_type = to_node.input_pin_type(to.id.input);

        // 同じピンタイプ同士のみ接続を許可
        if from_type.is_some() && from_type == to_type {
            // 既存の接続を削除（単一入力の場合）
            for &remote in &to.remotes {
                snarl.disconnect(remote, to.id);
            }
            snarl.connect(from.id, to.id);
        }
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<AudioNode>) {
        snarl.disconnect(from.id, to.id);
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<AudioNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut Ui, snarl: &mut Snarl<AudioNode>) {
        ui.label("Add Node");
        ui.separator();

        ui.menu_button("Input", |ui| {
            if self.input_devices.is_empty() {
                ui.label("No input devices");
            } else {
                for device_name in &self.input_devices.clone() {
                    if ui.button(device_name).clicked() {
                        snarl.insert_node(pos, AudioNode::new_audio_input(device_name.clone()));
                        ui.close();
                    }
                }
            }
        });

        ui.menu_button("Output", |ui| {
            if self.output_devices.is_empty() {
                ui.label("No output devices");
            } else {
                for device_name in &self.output_devices.clone() {
                    if ui.button(device_name).clicked() {
                        snarl.insert_node(pos, AudioNode::new_audio_output(device_name.clone()));
                        ui.close();
                    }
                }
            }
        });

        ui.menu_button("Effect", |ui| {
            if ui.button("Gain").clicked() {
                snarl.insert_node(pos, AudioNode::Gain(crate::nodes::GainNode::new()));
                ui.close();
            }
            if ui.button("Filter").clicked() {
                snarl.insert_node(pos, AudioNode::Filter(crate::nodes::FilterNode::new()));
                ui.close();
            }
            if ui.button("Compressor").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::Compressor(crate::nodes::CompressorNode::new()),
                );
                ui.close();
            }
            if ui.button("Pitch Shift").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::PitchShift(crate::nodes::PitchShiftNode::new()),
                );
                ui.close();
            }
            if ui.button("Graphic EQ").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::GraphicEq(crate::nodes::GraphicEqNode::new()),
                );
                ui.close();
            }
        });

        ui.menu_button("Math", |ui| {
            if ui.button("Add").clicked() {
                snarl.insert_node(pos, AudioNode::Add(crate::nodes::AddNode::new()));
                ui.close();
            }
            if ui.button("Multiply").clicked() {
                snarl.insert_node(pos, AudioNode::Multiply(crate::nodes::MultiplyNode::new()));
                ui.close();
            }
        });

        ui.menu_button("Analysis", |ui| {
            if ui.button("Spectrum Analyzer").clicked() {
                snarl.insert_node(
                    pos,
                    AudioNode::SpectrumAnalyzer(crate::nodes::SpectrumAnalyzerNode::new()),
                );
                ui.close();
            }
        });
    }

    fn has_node_menu(&mut self, _node: &AudioNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<AudioNode>,
    ) {
        if ui.button("Delete").clicked() {
            snarl.remove_node(node_id);
            ui.close();
        }
    }
}

// ノードタイプ別のUI表示ヘルパーメソッド
impl AudioGraphViewer {
    fn show_audio_input_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::AudioInputNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut node.is_active, "Active");
                if node.is_active {
                    ui.label(format!("{}ch", node.channels));
                }
            });

            egui::ComboBox::from_id_salt(format!("input_device_{:?}", node_id))
                .selected_text(node.device_name.as_str())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for dev in &self.input_devices {
                        ui.selectable_value(&mut node.device_name, dev.clone(), dev);
                    }
                });

            // スペクトラム表示（チェックボックスで切り替え）
            if node.is_active {
                ui.checkbox(&mut node.show_spectrum, "Spectrum");
                if node.show_spectrum {
                    show_spectrum_line(
                        ui,
                        &format!("input_spectrum_{:?}", node_id),
                        &node.spectrum,
                    );
                }
            }
        });
    }

    fn show_audio_output_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::AudioOutputNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut node.is_active, "Active");
                if node.is_active {
                    ui.label(format!("{}ch", node.channels));
                }
            });

            egui::ComboBox::from_id_salt(format!("output_device_{:?}", node_id))
                .selected_text(node.device_name.as_str())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for dev in &self.output_devices {
                        ui.selectable_value(&mut node.device_name, dev.clone(), dev);
                    }
                });

            // スペクトラム表示（チェックボックスで切り替え）
            if node.is_active {
                ui.checkbox(&mut node.show_spectrum, "Spectrum");
                if node.show_spectrum {
                    show_spectrum_line(
                        ui,
                        &format!("output_spectrum_{:?}", node_id),
                        &node.spectrum,
                    );
                }
            }
        });
    }

    fn show_gain_body(&self, _node_id: NodeId, node: &mut crate::nodes::GainNode, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.label("Gain:");
            ui.add(egui::Slider::new(&mut node.gain, 0.0..=2.0).suffix("x"));

            // dB表示
            let db = 20.0 * node.gain.log10();
            if db.is_finite() {
                ui.label(format!("{:.1} dB", db));
            } else {
                ui.label("-∞ dB");
            }
        });
    }

    fn show_add_body(&self, _node_id: NodeId, _node: &mut crate::nodes::AddNode, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.label("A + B");
        });
    }

    fn show_multiply_body(
        &self,
        _node_id: NodeId,
        _node: &mut crate::nodes::MultiplyNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            ui.label("A × B");
        });
    }

    fn show_filter_body(&self, node_id: NodeId, node: &mut crate::nodes::FilterNode, ui: &mut Ui) {
        ui.vertical(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt(format!("filter_type_{:?}", node_id))
                .selected_text(match node.filter_type {
                    FilterType::Low => "Low Pass",
                    FilterType::High => "High Pass",
                    FilterType::Band => "Band Pass",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut node.filter_type, FilterType::Low, "Low Pass");
                    ui.selectable_value(&mut node.filter_type, FilterType::High, "High Pass");
                    ui.selectable_value(&mut node.filter_type, FilterType::Band, "Band Pass");
                });

            ui.label("Cutoff:");
            ui.add(
                egui::Slider::new(&mut node.cutoff, 20.0..=20000.0)
                    .logarithmic(true)
                    .suffix(" Hz"),
            );

            ui.label("Q:");
            ui.add(egui::Slider::new(&mut node.resonance, 0.1..=10.0));
        });
    }

    fn show_spectrum_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::SpectrumAnalyzerNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            // スペクトラムデータを取得
            let spectrum = node.spectrum.lock();
            let bar_count = 48; // 表示するバーの数

            // バーデータを作成
            let bars: Vec<Bar> = (0..bar_count)
                .map(|i| {
                    // 対数スケールでインデックスをマッピング（低周波を細かく表示）
                    let freq_ratio = i as f32 / bar_count as f32;
                    let freq_idx = (freq_ratio.powf(2.0) * (FFT_SIZE / 2) as f32) as usize;
                    let freq_idx = freq_idx.min(spectrum.len().saturating_sub(1));

                    // マグニチュードを正規化（対数スケール、dB変換）
                    let magnitude = if freq_idx < spectrum.len() {
                        spectrum[freq_idx]
                    } else {
                        0.0
                    };
                    let db = if magnitude > 1e-6 {
                        20.0 * magnitude.log10()
                    } else {
                        -80.0
                    };
                    // -80dB〜0dBを0.0〜1.0にマッピング
                    let normalized = ((db + 80.0) / 80.0).clamp(0.0, 1.0) as f64;

                    // 周波数に応じたグラデーションカラー（低周波=緑、高周波=シアン）
                    let hue = 120.0 + freq_ratio * 60.0; // 緑(120)→シアン(180)
                    let sat = 0.7 + normalized as f32 * 0.3;
                    let val = 0.3 + normalized as f32 * 0.7;
                    let color = hsv_to_rgb(hue, sat, val);

                    Bar::new(i as f64, normalized).fill(color).width(0.85)
                })
                .collect();

            drop(spectrum); // ロックを解放

            // egui_plotでバーチャートを表示
            Plot::new(format!("spectrum_{:?}", node_id))
                .height(100.0)
                .width(220.0)
                .show_axes([false, false])
                .show_grid([false, false])
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .include_y(0.0)
                .include_y(1.0)
                .show_background(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new("spectrum", bars));
                });
        });
    }

    fn show_compressor_body(
        &self,
        _node_id: NodeId,
        node: &mut crate::nodes::CompressorNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            ui.label("Threshold:");
            ui.add(egui::Slider::new(&mut node.threshold, -60.0..=0.0).suffix(" dB"));

            ui.label("Ratio:");
            ui.add(egui::Slider::new(&mut node.ratio, 1.0..=20.0).suffix(":1"));

            ui.label("Attack:");
            ui.add(egui::Slider::new(&mut node.attack, 0.1..=100.0).suffix(" ms"));

            ui.label("Release:");
            ui.add(egui::Slider::new(&mut node.release, 10.0..=1000.0).suffix(" ms"));

            ui.label("Makeup:");
            ui.add(egui::Slider::new(&mut node.makeup_gain, 0.0..=24.0).suffix(" dB"));
        });
    }

    fn show_pitch_shift_body(
        &self,
        _node_id: NodeId,
        node: &mut crate::nodes::PitchShiftNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            ui.label("Semitones:");
            ui.add(egui::Slider::new(&mut node.semitones, -12.0..=12.0).suffix(" st"));

            // セント表示
            let cents = (node.semitones.fract() * 100.0).round() as i32;
            let semitones_int = node.semitones.trunc() as i32;
            if cents != 0 {
                ui.label(format!("{:+} semitones, {:+} cents", semitones_int, cents));
            } else {
                ui.label(format!("{:+} semitones", semitones_int));
            }
        });
    }

    fn show_graphic_eq_body(
        &self,
        node_id: NodeId,
        node: &mut crate::nodes::GraphicEqNode,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            // スペクトラム表示トグル
            ui.checkbox(&mut node.show_spectrum, "Show Spectrum");

            // 周波数範囲
            const MIN_FREQ: f64 = 20.0;
            const MAX_FREQ: f64 = 20000.0;
            const MIN_GAIN: f64 = -24.0;
            const MAX_GAIN: f64 = 24.0;

            // 周波数を対数スケールのX座標に変換
            let freq_to_x =
                |freq: f64| -> f64 { (freq / MIN_FREQ).ln() / (MAX_FREQ / MIN_FREQ).ln() };
            let x_to_freq = |x: f64| -> f64 { MIN_FREQ * (MAX_FREQ / MIN_FREQ).powf(x) };

            // EQカーブを描画するためのポイントを生成
            let curve_points: Vec<[f64; 2]> = (0..=100)
                .map(|i| {
                    let x = i as f64 / 100.0;
                    let freq = x_to_freq(x) as f32;

                    // ポイント間を線形補間してゲインを計算
                    let gain = Self::interpolate_eq_gain(&node.eq_points, freq);
                    [x, gain as f64]
                })
                .collect();

            // コントロールポイントの座標
            let control_points: Vec<[f64; 2]> = node
                .eq_points
                .iter()
                .map(|p| [freq_to_x(p.freq as f64), p.gain_db as f64])
                .collect();

            // スペクトラムデータを取得してプロット座標に変換
            let spectrum_points: Vec<[f64; 2]> = if node.show_spectrum {
                let spectrum_data = node.spectrum.lock();
                let spectrum_len = spectrum_data.len();
                (0..100)
                    .map(|i| {
                        let x = i as f64 / 100.0;
                        // 対数周波数スケールでスペクトラムインデックスを計算
                        let freq_idx = (x.powf(2.0) * spectrum_len as f64) as usize;
                        let freq_idx = freq_idx.min(spectrum_len.saturating_sub(1));

                        let magnitude = if freq_idx < spectrum_data.len() {
                            spectrum_data[freq_idx]
                        } else {
                            0.0
                        };

                        // dBに変換（-80dB〜0dBを-24〜+24dBにマッピング）
                        let db = if magnitude > 1e-6 {
                            20.0 * (magnitude as f64).log10()
                        } else {
                            -80.0
                        };
                        // -80dB〜0dBを-24〜+24dBにスケール
                        let scaled_db = (db + 80.0) / 80.0 * 48.0 - 24.0;
                        let clamped_db = scaled_db.clamp(MIN_GAIN, MAX_GAIN);

                        [x, clamped_db]
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // プロット表示
            let plot_response = Plot::new(format!("graphic_eq_{:?}", node_id))
                .height(150.0)
                .width(280.0)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_axes([false, true])
                .show_grid([true, true])
                .include_x(0.0)
                .include_x(1.0)
                .include_y(MIN_GAIN)
                .include_y(MAX_GAIN)
                .x_axis_label("Freq")
                .y_axis_label("dB")
                .show(ui, |plot_ui| {
                    // スペクトラム（背景として表示）
                    if node.show_spectrum && !spectrum_points.is_empty() {
                        plot_ui.line(
                            Line::new("spectrum", PlotPoints::from(spectrum_points.clone()))
                                .color(Color32::from_rgb(100, 200, 100))
                                .width(1.5),
                        );
                    }

                    // 0dBライン
                    plot_ui.line(
                        Line::new("zero", PlotPoints::from(vec![[0.0, 0.0], [1.0, 0.0]]))
                            .color(Color32::from_gray(100))
                            .width(1.0),
                    );

                    // EQカーブ
                    plot_ui.line(
                        Line::new("eq_curve", PlotPoints::from(curve_points))
                            .color(Color32::from_rgb(100, 200, 255))
                            .width(2.0),
                    );

                    // コントロールポイント
                    plot_ui.points(
                        Points::new("eq_points", PlotPoints::from(control_points.clone()))
                            .radius(6.0)
                            .color(Color32::from_rgb(255, 200, 100))
                            .filled(true),
                    );
                });

            // ドラッグでポイントを移動
            if let Some(pointer_pos) = plot_response.response.hover_pos() {
                let plot_bounds = plot_response.transform.bounds();
                let plot_rect = plot_response.response.rect;

                // ポインタ位置をプロット座標に変換
                let pointer_x = ((pointer_pos.x - plot_rect.left()) / plot_rect.width()
                    * plot_bounds.width() as f32
                    + plot_bounds.min()[0] as f32) as f64;
                let pointer_y = ((1.0 - (pointer_pos.y - plot_rect.top()) / plot_rect.height())
                    * plot_bounds.height() as f32
                    + plot_bounds.min()[1] as f32) as f64;

                // クリック/ドラッグ処理
                let is_primary_down = ui.input(|i| i.pointer.primary_down());
                let is_clicked = plot_response.response.clicked();

                if is_primary_down || is_clicked {
                    // 最も近いポイントを探す
                    let mut closest_idx = None;
                    let mut closest_dist = f64::MAX;

                    for (idx, point) in control_points.iter().enumerate() {
                        let dx = (point[0] - pointer_x) * plot_rect.width() as f64;
                        let dy = (point[1] - pointer_y) / plot_bounds.height()
                            * plot_rect.height() as f64;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist < closest_dist && dist < 20.0 {
                            closest_dist = dist;
                            closest_idx = Some(idx);
                        }
                    }

                    // ポイントを移動
                    if let Some(idx) = closest_idx {
                        let new_gain = pointer_y.clamp(MIN_GAIN, MAX_GAIN) as f32;
                        node.eq_points[idx].gain_db = new_gain;

                        // GraphicEqの周波数ゲインカーブを更新
                        let mut eq = node.graphic_eq.lock();
                        eq.update_curve(&node.eq_points);
                    }
                }
            }

            // ポイント一覧（周波数ラベル）
            ui.horizontal(|ui| {
                for point in &node.eq_points {
                    let freq_str = if point.freq >= 1000.0 {
                        format!("{:.1}k", point.freq / 1000.0)
                    } else {
                        format!("{:.0}", point.freq)
                    };
                    ui.label(format!("{}:{:+.1}dB", freq_str, point.gain_db));
                }
            });

            // リセットボタン
            if ui.button("Reset").clicked() {
                for point in &mut node.eq_points {
                    point.gain_db = 0.0;
                }
                let mut eq = node.graphic_eq.lock();
                eq.update_curve(&node.eq_points);
            }
        });
    }

    /// EQポイント間を線形補間してゲインを取得
    fn interpolate_eq_gain(points: &[EqPoint], freq: f32) -> f32 {
        if points.is_empty() {
            return 0.0;
        }

        // 周波数が最小より小さい場合
        if freq <= points[0].freq {
            return points[0].gain_db;
        }

        // 周波数が最大より大きい場合
        if freq >= points[points.len() - 1].freq {
            return points[points.len() - 1].gain_db;
        }

        // 補間
        for i in 0..points.len() - 1 {
            if freq >= points[i].freq && freq <= points[i + 1].freq {
                let log_freq = freq.ln();
                let log_freq_low = points[i].freq.ln();
                let log_freq_high = points[i + 1].freq.ln();

                let t = (log_freq - log_freq_low) / (log_freq_high - log_freq_low);
                return points[i].gain_db + t * (points[i + 1].gain_db - points[i].gain_db);
            }
        }

        0.0
    }
}
