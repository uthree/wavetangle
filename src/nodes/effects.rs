use std::sync::Arc;

use egui::{Color32, Ui};
use egui_plot::{Line, Plot, PlotPoints, Points};
use parking_lot::Mutex;

use super::{
    impl_as_any, impl_input_port_nb, impl_single_output_port_nb, interpolate_eq_gain,
    AudioInputPort, AudioOutputPort, ChannelBuffer, NodeBase, NodeBuffers, NodeType, NodeUI,
    NodeUIContext, PinType, EQ_FFT_SIZE,
};

// ============================================================================
// Gain Node (Effect)
// ============================================================================

/// ゲインエフェクトノード
#[derive(Clone)]
pub struct GainNode {
    /// ゲイン値（倍率）
    pub gain: f32,
    /// バッファ管理
    pub buffers: NodeBuffers,
    /// アクティブ状態
    pub is_active: bool,
}

impl GainNode {
    pub fn new() -> Self {
        Self {
            gain: 1.0,
            buffers: NodeBuffers::single_io(),
            is_active: false,
        }
    }
}

impl Default for GainNode {
    fn default() -> Self {
        Self::new()
    }
}

// GainNodeのトレイト実装（NodeBuffers対応マクロを使用）
impl NodeBase for GainNode {
    fn node_type(&self) -> NodeType {
        NodeType::Gain
    }

    fn title(&self) -> &str {
        "Gain"
    }

    impl_as_any!();
}

impl_input_port_nb!(GainNode, ["In"]);
impl_single_output_port_nb!(GainNode);

impl NodeUI for GainNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Gain:");
            ui.add(egui::Slider::new(&mut self.gain, 0.0..=2.0).suffix("x"));

            // dB表示
            let db = 20.0 * self.gain.log10();
            if db.is_finite() {
                ui.label(format!("{:.1} dB", db));
            } else {
                ui.label("-∞ dB");
            }
        });
    }
}

// ============================================================================
// Filter Node (Effect)
// ============================================================================

/// フィルタータイプ
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterType {
    Low,
    High,
    Band,
}

/// フィルターノード - ローパス/ハイパス/バンドパスフィルター
#[derive(Clone)]
pub struct FilterNode {
    pub filter_type: FilterType,
    /// カットオフ周波数 (Hz)
    pub cutoff: f32,
    /// レゾナンス (Q値)
    pub resonance: f32,
    /// バッファ管理
    pub buffers: NodeBuffers,
    pub is_active: bool,
    /// Biquadフィルター状態
    pub biquad_state: Arc<Mutex<crate::dsp::BiquadState>>,
}

impl FilterNode {
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::Low,
            cutoff: 1000.0,
            resonance: 0.707,
            buffers: NodeBuffers::single_io(),
            is_active: false,
            biquad_state: Arc::new(Mutex::new(crate::dsp::BiquadState::new())),
        }
    }
}

impl Default for FilterNode {
    fn default() -> Self {
        Self::new()
    }
}

// FilterNodeのトレイト実装（NodeBuffers対応マクロを使用）
impl NodeBase for FilterNode {
    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn title(&self) -> &str {
        "Filter"
    }

    impl_as_any!();
}

impl_input_port_nb!(FilterNode, ["In"]);
impl_single_output_port_nb!(FilterNode);

impl NodeUI for FilterNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt(format!("filter_type_{:?}", ctx.node_id))
                .selected_text(match self.filter_type {
                    FilterType::Low => "Low Pass",
                    FilterType::High => "High Pass",
                    FilterType::Band => "Band Pass",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.filter_type, FilterType::Low, "Low Pass");
                    ui.selectable_value(&mut self.filter_type, FilterType::High, "High Pass");
                    ui.selectable_value(&mut self.filter_type, FilterType::Band, "Band Pass");
                });

            ui.label("Cutoff:");
            ui.add(
                egui::Slider::new(&mut self.cutoff, 20.0..=20000.0)
                    .logarithmic(true)
                    .suffix(" Hz"),
            );

            ui.label("Q:");
            ui.add(egui::Slider::new(&mut self.resonance, 0.1..=10.0));
        });
    }
}

// ============================================================================
// Compressor Node
// ============================================================================

/// コンプレッサーノード - ダイナミックレンジ圧縮
#[derive(Clone)]
pub struct CompressorNode {
    /// スレッショルド (dB)
    pub threshold: f32,
    /// レシオ (1:n)
    pub ratio: f32,
    /// アタックタイム (ms)
    pub attack: f32,
    /// リリースタイム (ms)
    pub release: f32,
    /// メイクアップゲイン (dB)
    pub makeup_gain: f32,
    /// バッファ管理
    pub buffers: NodeBuffers,
    pub is_active: bool,
    /// コンプレッサー状態
    pub compressor_state: Arc<Mutex<crate::dsp::CompressorState>>,
}

impl CompressorNode {
    pub fn new() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack: 10.0,
            release: 100.0,
            makeup_gain: 0.0,
            buffers: NodeBuffers::single_io(),
            is_active: false,
            compressor_state: Arc::new(Mutex::new(crate::dsp::CompressorState::new())),
        }
    }
}

impl Default for CompressorNode {
    fn default() -> Self {
        Self::new()
    }
}

// CompressorNodeのトレイト実装（NodeBuffers対応マクロを使用）
impl NodeBase for CompressorNode {
    fn node_type(&self) -> NodeType {
        NodeType::Compressor
    }

    fn title(&self) -> &str {
        "Compressor"
    }

    impl_as_any!();
}

impl_input_port_nb!(CompressorNode, ["In"]);
impl_single_output_port_nb!(CompressorNode);

impl NodeUI for CompressorNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.label("Threshold:");
            ui.add(egui::Slider::new(&mut self.threshold, -60.0..=0.0).suffix(" dB"));

            ui.label("Ratio:");
            ui.add(egui::Slider::new(&mut self.ratio, 1.0..=20.0).suffix(":1"));

            ui.label("Attack:");
            ui.add(egui::Slider::new(&mut self.attack, 0.1..=100.0).suffix(" ms"));

            ui.label("Release:");
            ui.add(egui::Slider::new(&mut self.release, 10.0..=1000.0).suffix(" ms"));

            ui.label("Makeup:");
            ui.add(egui::Slider::new(&mut self.makeup_gain, 0.0..=24.0).suffix(" dB"));
        });
    }
}

// ============================================================================
// PitchShift Node - PSOLAピッチシフト
// ============================================================================

/// ピッチシフトノード
pub struct PitchShiftNode {
    /// ピッチシフト量（半音単位、-12〜+12）
    pub semitones: f32,
    /// グレインサイズ（サンプル数、128〜8192）
    pub grain_size: usize,
    /// グレイン数（2〜16）
    pub num_grains: usize,
    /// 位相アラインメントを有効にするか
    pub phase_alignment_enabled: bool,
    /// 位相アラインメントの探索範囲（グレインサイズに対する割合、0.1〜1.0）
    pub search_range_ratio: f32,
    /// 位相アラインメントの相関長（グレインサイズに対する割合、0.1〜1.0）
    pub correlation_length_ratio: f32,
    /// バッファ管理
    pub buffers: NodeBuffers,
    /// アクティブ状態
    pub is_active: bool,
    /// ピッチシフター（スレッドセーフ）
    pub pitch_shifter: Arc<Mutex<crate::dsp::PitchShifter>>,
}

impl Clone for PitchShiftNode {
    fn clone(&self) -> Self {
        Self {
            semitones: self.semitones,
            grain_size: self.grain_size,
            num_grains: self.num_grains,
            phase_alignment_enabled: self.phase_alignment_enabled,
            search_range_ratio: self.search_range_ratio,
            correlation_length_ratio: self.correlation_length_ratio,
            buffers: self.buffers.clone(),
            is_active: self.is_active,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::with_params(
                44100.0,
                self.grain_size,
                self.num_grains,
                crate::dsp::DEFAULT_PITCH_BUFFER_SIZE,
            ))),
        }
    }
}

impl PitchShiftNode {
    pub fn new() -> Self {
        let default_params = crate::dsp::PhaseAlignmentParams::default();
        Self {
            semitones: 0.0,
            grain_size: crate::dsp::DEFAULT_GRAIN_SIZE,
            num_grains: crate::dsp::DEFAULT_NUM_GRAINS,
            phase_alignment_enabled: default_params.enabled,
            search_range_ratio: default_params.search_range_ratio,
            correlation_length_ratio: default_params.correlation_length_ratio,
            buffers: NodeBuffers::single_io(),
            is_active: false,
            pitch_shifter: Arc::new(Mutex::new(crate::dsp::PitchShifter::new(44100.0))),
        }
    }
}

impl Default for PitchShiftNode {
    fn default() -> Self {
        Self::new()
    }
}

// PitchShiftNodeのトレイト実装（NodeBuffers対応マクロを使用）
impl NodeBase for PitchShiftNode {
    fn node_type(&self) -> NodeType {
        NodeType::PitchShift
    }

    fn title(&self) -> &str {
        "Pitch Shift"
    }

    impl_as_any!();
}

impl_input_port_nb!(PitchShiftNode, ["In"]);
impl_single_output_port_nb!(PitchShiftNode);

impl NodeUI for PitchShiftNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, _ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            ui.set_max_width(150.0);
            ui.label("Semitones:");
            ui.add(egui::Slider::new(&mut self.semitones, -12.0..=12.0).suffix(" st"));

            // セント表示
            let cents = (self.semitones.fract() * 100.0).round() as i32;
            let semitones_int = self.semitones.trunc() as i32;
            if cents != 0 {
                ui.label(format!("{:+} semitones, {:+} cents", semitones_int, cents));
            } else {
                ui.label(format!("{:+} semitones", semitones_int));
            }

            ui.separator();

            // グレインサイズ（2のべき乗で調整）
            ui.label("Grain Size:");
            let mut grain_size_log = (self.grain_size as f32).log2();
            if ui
                .add(egui::Slider::new(&mut grain_size_log, 7.0..=13.0).show_value(false))
                .changed()
            {
                let new_grain_size = 2_usize.pow(grain_size_log.round() as u32);
                if new_grain_size != self.grain_size {
                    self.grain_size = new_grain_size;
                    if let Some(mut shifter) = self.pitch_shifter.try_lock() {
                        shifter.set_grain_size(new_grain_size);
                    }
                }
            }
            ui.label(format!("{} samples", self.grain_size));

            // グレイン数
            ui.label("Num Grains:");
            let mut num_grains = self.num_grains as i32;
            if ui.add(egui::Slider::new(&mut num_grains, 2..=16)).changed() {
                let new_num_grains = num_grains as usize;
                if new_num_grains != self.num_grains {
                    self.num_grains = new_num_grains;
                    if let Some(mut shifter) = self.pitch_shifter.try_lock() {
                        shifter.set_num_grains(new_num_grains);
                    }
                }
            }

            // 推定レイテンシ表示（サンプルレート48kHz想定）
            let latency_samples = self.grain_size * self.num_grains / 2;
            let latency_ms = latency_samples as f32 / 48.0;
            ui.label(format!("Latency: ~{:.1} ms", latency_ms));

            ui.separator();

            // 位相アラインメント設定
            ui.collapsing("Phase Alignment", |ui| {
                ui.checkbox(&mut self.phase_alignment_enabled, "Enabled");

                ui.add_enabled_ui(self.phase_alignment_enabled, |ui| {
                    ui.label("Search Range:");
                    ui.add(
                        egui::Slider::new(&mut self.search_range_ratio, 0.1..=1.0)
                            .suffix("x")
                            .fixed_decimals(2),
                    );

                    ui.label("Correlation Length:");
                    ui.add(
                        egui::Slider::new(&mut self.correlation_length_ratio, 0.1..=1.0)
                            .suffix("x")
                            .fixed_decimals(2),
                    );
                });
            });
        });
    }
}

// ============================================================================
// GraphicEqNode - グラフィックイコライザー
// ============================================================================

/// グラフィックイコライザーノード（カーブエディター付き）
pub struct GraphicEqNode {
    /// EQカーブのコントロールポイント
    pub eq_points: Vec<crate::dsp::EqPoint>,
    /// バッファ管理
    pub buffers: NodeBuffers,
    /// アクティブ状態
    pub is_active: bool,
    /// グラフィックEQプロセッサー（スレッドセーフ）
    pub graphic_eq: Arc<Mutex<crate::dsp::GraphicEq>>,
    /// スペクトラム表示を有効にするか
    pub show_spectrum: bool,
    /// スペクトラムデータ（入力信号）
    pub spectrum: Arc<Mutex<Vec<f32>>>,
}

impl Clone for GraphicEqNode {
    fn clone(&self) -> Self {
        Self {
            eq_points: self.eq_points.clone(),
            buffers: self.buffers.clone(),
            is_active: self.is_active,
            graphic_eq: Arc::new(Mutex::new(crate::dsp::GraphicEq::new(44100.0))),
            show_spectrum: self.show_spectrum,
            spectrum: self.spectrum.clone(),
        }
    }
}

impl GraphicEqNode {
    pub fn new() -> Self {
        // デフォルトの5ポイントEQカーブ
        let eq_points = vec![
            crate::dsp::EqPoint::new(50.0, 0.0),
            crate::dsp::EqPoint::new(200.0, 0.0),
            crate::dsp::EqPoint::new(1000.0, 0.0),
            crate::dsp::EqPoint::new(5000.0, 0.0),
            crate::dsp::EqPoint::new(15000.0, 0.0),
        ];
        Self {
            eq_points,
            buffers: NodeBuffers::single_io(),
            is_active: false,
            graphic_eq: Arc::new(Mutex::new(crate::dsp::GraphicEq::new(44100.0))),
            show_spectrum: true,
            spectrum: Arc::new(Mutex::new(vec![0.0; EQ_FFT_SIZE / 2])),
        }
    }
}

impl Default for GraphicEqNode {
    fn default() -> Self {
        Self::new()
    }
}

// GraphicEqNodeのトレイト実装（NodeBuffers対応マクロを使用）
impl NodeBase for GraphicEqNode {
    fn node_type(&self) -> NodeType {
        NodeType::GraphicEq
    }

    fn title(&self) -> &str {
        "Graphic EQ"
    }

    impl_as_any!();
}

impl_input_port_nb!(GraphicEqNode, ["In"]);
impl_single_output_port_nb!(GraphicEqNode);

impl NodeUI for GraphicEqNode {
    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    fn show_body(&mut self, ui: &mut Ui, ctx: &NodeUIContext) {
        ui.vertical(|ui| {
            // スペクトラム表示トグル
            ui.checkbox(&mut self.show_spectrum, "Show Spectrum");

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
                    let gain = interpolate_eq_gain(&self.eq_points, freq);
                    [x, gain as f64]
                })
                .collect();

            // コントロールポイントの座標
            let control_points: Vec<[f64; 2]> = self
                .eq_points
                .iter()
                .map(|p| [freq_to_x(p.freq as f64), p.gain_db as f64])
                .collect();

            // スペクトラムデータを取得してプロット座標に変換
            let spectrum_points: Vec<[f64; 2]> = if self.show_spectrum {
                let spectrum_data = self.spectrum.lock();
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
            let plot_response = Plot::new(format!("graphic_eq_{:?}", ctx.node_id))
                .height(150.0)
                .width(280.0)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_axes([true, true])
                .show_grid([true, true])
                .include_x(0.0)
                .include_x(1.0)
                .include_y(MIN_GAIN)
                .include_y(MAX_GAIN)
                .x_axis_formatter(move |grid_mark, _range| {
                    let freq = x_to_freq(grid_mark.value);
                    if freq >= 1000.0 {
                        format!("{:.0}k", freq / 1000.0)
                    } else {
                        format!("{:.0}", freq)
                    }
                })
                .y_axis_formatter(|grid_mark, _range| format!("{:.0}dB", grid_mark.value))
                .show(ui, |plot_ui| {
                    // スペクトラム（背景として表示）
                    if self.show_spectrum && !spectrum_points.is_empty() {
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
                        self.eq_points[idx].gain_db = new_gain;

                        // GraphicEqの周波数ゲインカーブを更新
                        let mut eq = self.graphic_eq.lock();
                        eq.update_curve(&self.eq_points);
                    }
                }
            }

            // ポイント一覧（周波数ラベル）
            ui.horizontal(|ui| {
                for point in &self.eq_points {
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
                for point in &mut self.eq_points {
                    point.gain_db = 0.0;
                }
                let mut eq = self.graphic_eq.lock();
                eq.update_curve(&self.eq_points);
            }
        });
    }
}
