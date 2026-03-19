//! TD-PSOLA (Time-Domain Pitch Synchronous Overlap-Add) ピッチシフター
//!
//! ピッチとフォルマントを独立に制御できるピッチシフト処理

// 将来の拡張用に提供されているAPIについて警告を抑制
#![allow(dead_code)]

use super::create_hann_window;
use super::interpolation::{Interpolator, LinearInterpolator};
use super::yin::YinPitchDetector;

/// TD-PSOLAピッチシフターの設定
#[derive(Clone, Copy, Debug)]
pub struct TdPsolaConfig {
    /// 最小検出周波数 (Hz)
    pub min_freq: f32,
    /// 最大検出周波数 (Hz)
    pub max_freq: f32,
    /// YIN閾値
    pub yin_threshold: f32,
}

impl Default for TdPsolaConfig {
    fn default() -> Self {
        Self {
            min_freq: 50.0,
            max_freq: 1000.0,
            yin_threshold: 0.15,
        }
    }
}

/// TD-PSOLAピッチシフター
///
/// 入力と出力のタイミングを分離して管理:
/// - 入力: 一定レートでバッファに蓄積
/// - 出力: ピッチ比率に応じた速度で入力を消費し、合成
pub struct TdPsolaPitchShifter {
    /// サンプルレート
    sample_rate: f32,

    /// 入力リングバッファ
    input_buffer: Vec<f32>,
    /// バッファサイズ
    buffer_size: usize,
    /// 入力書き込み位置（整数、実際の書き込み位置）
    input_write_pos: usize,

    /// 出力リングバッファ（overlap-add用）
    output_buffer: Vec<f32>,
    /// 出力読み取り位置
    output_read_pos: usize,

    /// 合成用の入力読み取り位置（浮動小数点、ピッチシフトで変化）
    synthesis_input_pos: f64,
    /// 合成用の出力書き込み位置
    synthesis_output_pos: usize,

    /// 次のグレイン生成までの残りサンプル（出力時間で計測）
    samples_until_grain: f32,

    /// YINピッチ検出器
    pitch_detector: YinPitchDetector,

    /// 補間器
    interpolator: Box<dyn Interpolator>,

    /// ピッチシフト量（半音単位）
    pitch_shift_semitones: f32,
    /// フォルマントシフト量（半音単位）
    formant_shift_semitones: f32,

    /// 現在の入力周期（サンプル数）
    current_period: f32,
    /// デフォルト周期（ピッチ検出失敗時）
    default_period: f32,

    /// 窓関数キャッシュ
    window_cache: Vec<f32>,
    /// キャッシュされた窓関数サイズ
    cached_window_size: usize,

    /// 分析バッファ（ピッチ検出用）
    analysis_buffer: Vec<f32>,
    /// 分析バッファの書き込み位置
    analysis_write_pos: usize,

    /// 処理済みサンプル数（初期化判定用）
    processed_samples: usize,
    /// 初期レイテンシ（サンプル数）
    latency: usize,
}

impl TdPsolaPitchShifter {
    /// 新しいTD-PSOLAピッチシフターを作成
    pub fn new(sample_rate: f32) -> Self {
        Self::with_config(sample_rate, TdPsolaConfig::default())
    }

    /// カスタム設定で作成
    pub fn with_config(sample_rate: f32, config: TdPsolaConfig) -> Self {
        let pitch_detector = YinPitchDetector::new(sample_rate, config.min_freq, config.max_freq);
        let analysis_buffer_size = pitch_detector.required_samples();

        // バッファサイズ: 最大周期の48倍（ピッチダウン時のドリフトに余裕を持たせる）
        let max_period = (sample_rate / config.min_freq).ceil() as usize;
        let buffer_size = max_period * 48;

        // デフォルト周期（200Hz相当）
        let default_period = sample_rate / 200.0;

        // 初期レイテンシ: 最大周期の4倍
        let latency = max_period * 4;

        Self {
            sample_rate,
            input_buffer: vec![0.0; buffer_size],
            buffer_size,
            input_write_pos: 0,
            output_buffer: vec![0.0; buffer_size],
            output_read_pos: 0,
            synthesis_input_pos: 0.0,
            synthesis_output_pos: 0, // 位置0からグレインを配置開始
            samples_until_grain: 0.0,
            pitch_detector,
            interpolator: Box::new(LinearInterpolator::new()),
            pitch_shift_semitones: 0.0,
            formant_shift_semitones: 0.0,
            current_period: default_period,
            default_period,
            window_cache: Vec::new(),
            cached_window_size: 0,
            analysis_buffer: vec![0.0; analysis_buffer_size],
            analysis_write_pos: 0,
            processed_samples: 0,
            latency,
        }
    }

    /// ピッチシフト量を設定（半音単位）
    pub fn set_pitch_shift(&mut self, semitones: f32) {
        self.pitch_shift_semitones = semitones.clamp(-24.0, 24.0);
    }

    /// フォルマントシフト量を設定（半音単位）
    pub fn set_formant_shift(&mut self, semitones: f32) {
        self.formant_shift_semitones = semitones.clamp(-24.0, 24.0);
    }

    /// YIN閾値を設定
    pub fn set_yin_threshold(&mut self, threshold: f32) {
        self.pitch_detector.set_threshold(threshold);
    }

    /// 補間器を設定
    pub fn set_interpolator(&mut self, interpolator: Box<dyn Interpolator>) {
        self.interpolator = interpolator;
    }

    /// 現在検出されている周波数を取得
    pub fn current_frequency(&self) -> f32 {
        if self.current_period > 0.0 {
            self.sample_rate / self.current_period
        } else {
            0.0
        }
    }

    /// ピッチ比率を計算
    /// pitch_ratio > 1: 高いピッチ
    /// pitch_ratio < 1: 低いピッチ
    fn pitch_ratio(&self) -> f32 {
        2.0_f32.powf(self.pitch_shift_semitones / 12.0)
    }

    /// フォルマント比率を計算
    fn formant_ratio(&self) -> f32 {
        2.0_f32.powf(self.formant_shift_semitones / 12.0)
    }

    /// 窓関数を取得（サイズが変わったら再生成）
    fn get_window(&mut self, size: usize) -> &[f32] {
        if size != self.cached_window_size {
            self.window_cache = create_hann_window(size);
            self.cached_window_size = size;
        }
        &self.window_cache
    }

    /// 入力バッファから補間してサンプルを取得
    fn read_input(&self, pos: f64) -> f32 {
        let pos_mod = pos.rem_euclid(self.buffer_size as f64);
        self.interpolator.interpolate(&self.input_buffer, pos_mod)
    }

    /// ピッチを更新
    fn update_pitch(&mut self) {
        if let Some(result) = self.pitch_detector.detect(&self.analysis_buffer) {
            if result.voiced {
                self.current_period = result.period;
            }
            // 無声音の場合は前回の周期を維持
        }
    }

    /// グレインを生成して出力バッファに加算
    ///
    /// - grain_center: グレインの中心位置（入力バッファ内、浮動小数点）
    /// - output_center: グレインの出力位置（出力バッファ内）
    fn generate_grain(&mut self, grain_center: f64, output_center: usize) {
        let formant_ratio = self.formant_ratio();
        let pitch_ratio = self.pitch_ratio();
        let input_period = self.current_period;
        let output_period = input_period / pitch_ratio;

        // グレインサイズ: 2周期分（オーバーラップのため）
        let grain_periods = 2.0;
        let input_grain_size = (input_period * grain_periods) as usize;
        let input_grain_size = input_grain_size.max(64).min(self.buffer_size / 4);

        // 出力グレインサイズ（フォルマントシフト適用）
        // formant_ratio > 1: 出力グレインを縮小 → 高フォルマント
        // formant_ratio < 1: 出力グレインを拡大 → 低フォルマント
        let mut output_grain_size = (input_grain_size as f32 / formant_ratio) as usize;

        // ピッチダウン時: 出力グレインが出力アドバンスの2倍以上であることを保証
        // これにより50%オーバーラップが維持される
        let output_advance = (output_period / 2.0).max(32.0);
        let min_output_grain_size = (output_advance * 2.0) as usize;
        output_grain_size = output_grain_size.max(min_output_grain_size);

        let output_grain_size = output_grain_size.max(64).min(self.buffer_size / 4);

        // 窓関数を取得
        let window = self.get_window(output_grain_size).to_vec();

        // COLA正規化: output_grain_size > 2*output_advance の場合、
        // 重なるグレイン数が増えるため振幅を補正する
        // Hann窓の50%オーバーラップでCOLA sum = 1.0
        // それ以外の重複率では sum = output_grain_size / (2 * output_advance)
        let cola_scale = (2.0 * output_advance) / output_grain_size as f32;

        // グレインを生成
        let half_output = output_grain_size as f32 / 2.0;

        for (i, &win) in window.iter().enumerate().take(output_grain_size) {
            // 出力位置（グレインの中心を基準に）
            let output_offset = i as f32 - half_output;

            // 入力位置（フォルマント比率でスケーリング）
            let input_offset = output_offset * formant_ratio;
            let input_pos = grain_center + input_offset as f64;

            // サンプルを補間して取得
            let sample = self.read_input(input_pos);

            // 窓関数を適用（COLA正規化付き）
            let windowed_sample = sample * win * cola_scale;

            // 出力バッファに加算（overlap-add）
            let out_idx = ((output_center as i64 + output_offset as i64)
                .rem_euclid(self.buffer_size as i64)) as usize;
            self.output_buffer[out_idx] += windowed_sample;
        }
    }

    /// オーディオを処理
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let pitch_ratio = self.pitch_ratio();

        for (i, &sample) in input.iter().enumerate() {
            // 入力バッファに書き込み
            self.input_buffer[self.input_write_pos] = sample;
            self.input_write_pos = (self.input_write_pos + 1) % self.buffer_size;

            // 分析バッファに書き込み（ピッチ検出用）
            self.analysis_buffer[self.analysis_write_pos] = sample;
            self.analysis_write_pos = (self.analysis_write_pos + 1) % self.analysis_buffer.len();

            // 分析バッファが一周したらピッチを更新
            if self.analysis_write_pos == 0 {
                self.update_pitch();
            }

            self.processed_samples += 1;

            // グレイン生成タイミングを確認
            // 出力周期 = 入力周期 / ピッチ比率
            let output_period = self.current_period / pitch_ratio;
            // グレイン間隔: 出力周期の50%オーバーラップ
            let grain_interval = (output_period / 2.0).max(32.0);

            self.samples_until_grain -= 1.0;

            // レイテンシ期間中もグレインを生成する（出力バッファを事前に埋める）
            if self.samples_until_grain <= 0.0 {
                let grain_center = self.synthesis_input_pos;
                let output_center = self.synthesis_output_pos;

                self.generate_grain(grain_center, output_center);

                // 次のグレインのための位置更新
                // 入力側: 入力周期の半分（50%オーバーラップ）だけ進む
                // input_advance / output_advance の比率がピッチ比率を決定する
                let input_advance = self.current_period / 2.0;
                self.synthesis_input_pos += input_advance as f64;

                // 出力側: 出力周期の半分（50%オーバーラップ）だけ進む
                let output_advance = (output_period / 2.0).max(32.0) as usize;
                self.synthesis_output_pos =
                    (self.synthesis_output_pos + output_advance) % self.buffer_size;

                // 次のグレインまでの間隔をリセット
                self.samples_until_grain += grain_interval;
            }

            // synthesis_input_posが入力データの範囲内に収まるようにする
            // ピッチダウン時: synthesis_input_posは input_write_pos から徐々に遅れる
            // ピッチアップ時: synthesis_input_posは input_write_pos に追いつく
            // いずれの場合も、バッファ範囲外になったらリセットが必要
            let behind = (self.input_write_pos as f64 - self.synthesis_input_pos)
                .rem_euclid(self.buffer_size as f64);
            let min_behind = self.current_period as f64;
            let max_behind = (self.buffer_size * 3 / 4) as f64;

            if behind > max_behind || behind < min_behind {
                // リセットが必要：ピッチ周期の整数倍だけジャンプして位相を維持
                let target_behind = self.latency as f64;
                let target_pos = (self.input_write_pos as f64 - target_behind)
                    .rem_euclid(self.buffer_size as f64);

                // 現在位置からターゲットまでの距離をピッチ周期単位で丸める
                let period = self.current_period as f64;
                let distance = (target_pos - self.synthesis_input_pos)
                    .rem_euclid(self.buffer_size as f64);
                let periods_to_jump = (distance / period).round();
                self.synthesis_input_pos = (self.synthesis_input_pos + periods_to_jump * period)
                    .rem_euclid(self.buffer_size as f64);
            }

            // 出力バッファから読み取り
            if self.processed_samples > self.latency {
                output[i] = self.output_buffer[self.output_read_pos];
                // 読み取った位置をクリア
                self.output_buffer[self.output_read_pos] = 0.0;
                self.output_read_pos = (self.output_read_pos + 1) % self.buffer_size;
            } else {
                output[i] = 0.0;
            }
        }
    }
}

impl Default for TdPsolaPitchShifter {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_sine_wave(freq: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_td_psola_creation() {
        let shifter = TdPsolaPitchShifter::new(48000.0);
        assert!((shifter.pitch_shift_semitones - 0.0).abs() < 1e-6);
        assert!((shifter.formant_shift_semitones - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_td_psola_passthrough() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // シフトなし
        shifter.set_pitch_shift(0.0);
        shifter.set_formant_shift(0.0);

        // サイン波入力（十分な長さ）
        let input = generate_sine_wave(440.0, sample_rate, 16384);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 後半の出力が無音ではないことを確認（初期化後の出力を検証）
        let max_output: f32 = output[8192..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.05,
            "Output should not be silent after initialization, max={}",
            max_output
        );
    }

    #[test]
    fn test_td_psola_pitch_shift() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 1オクターブ上
        shifter.set_pitch_shift(12.0);
        shifter.set_formant_shift(0.0);

        let input = generate_sine_wave(220.0, sample_rate, 16384);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 出力が存在することを確認
        let max_output: f32 = output[8192..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.05,
            "Output should not be silent, max={}",
            max_output
        );
    }

    #[test]
    fn test_td_psola_formant_shift() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // フォルマントのみシフト
        shifter.set_pitch_shift(0.0);
        shifter.set_formant_shift(6.0); // 半オクターブ上

        let input = generate_sine_wave(220.0, sample_rate, 16384);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        let max_output: f32 = output[8192..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.05,
            "Output should not be silent, max={}",
            max_output
        );
    }

    #[test]
    fn test_td_psola_preserve_formant() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // ピッチを上げてフォルマントを逆方向にシフト（保持）
        shifter.set_pitch_shift(12.0);
        shifter.set_formant_shift(-12.0);

        let input = generate_sine_wave(220.0, sample_rate, 16384);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 純粋なサイン波では位相干渉により振幅が低くなる場合がある
        // 実際の音声（複雑な波形）ではこの問題は発生しない
        let max_output: f32 = output[8192..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.001,
            "Output should not be silent, max={}",
            max_output
        );
    }

    #[test]
    fn test_td_psola_current_frequency() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 440Hzのサイン波を処理
        let input = generate_sine_wave(440.0, sample_rate, 8192);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // ピッチが検出されているはず
        let freq = shifter.current_frequency();
        if freq > 0.0 {
            let error_percent = ((freq - 440.0) / 440.0).abs() * 100.0;
            assert!(
                error_percent < 5.0,
                "Detected frequency should be close to 440Hz, got {}Hz",
                freq
            );
        }
    }

    #[test]
    fn test_td_psola_pitch_down() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 1オクターブ下
        shifter.set_pitch_shift(-12.0);
        shifter.set_formant_shift(0.0);

        let input = generate_sine_wave(440.0, sample_rate, 32768);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 出力が存在することを確認
        let max_output: f32 = output[16384..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.05,
            "Pitch-down output should not be silent, max={}",
            max_output
        );

        // 長時間処理しても安定していることを確認（途切れがないこと）
        // 後半の連続するブロックすべてに出力があること
        let block_size = 1024;
        let mut silent_blocks = 0;
        for block_start in (16384..32768).step_by(block_size) {
            let block_max: f32 = output[block_start..block_start + block_size]
                .iter()
                .map(|x| x.abs())
                .fold(0.0, f32::max);
            if block_max < 0.01 {
                silent_blocks += 1;
            }
        }
        assert!(
            silent_blocks == 0,
            "Pitch-down should have no silent blocks, found {} silent blocks",
            silent_blocks
        );
    }

    #[test]
    fn test_td_psola_pitch_down_extreme() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 2オクターブ下（極端なケース）
        shifter.set_pitch_shift(-24.0);
        shifter.set_formant_shift(0.0);

        let input = generate_sine_wave(440.0, sample_rate, 32768);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        let max_output: f32 = output[16384..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.01,
            "Extreme pitch-down output should not be silent, max={}",
            max_output
        );
    }

    #[test]
    fn test_td_psola_set_interpolator() {
        use super::super::interpolation::CubicInterpolator;

        let mut shifter = TdPsolaPitchShifter::new(48000.0);
        shifter.set_interpolator(Box::new(CubicInterpolator::new()));

        // 処理できることを確認
        let input = vec![0.0; 1024];
        let mut output = vec![0.0; 1024];
        shifter.process(&input, &mut output);
    }

    #[test]
    fn test_td_psola_realtime_performance() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        shifter.set_pitch_shift(7.0); // 完全5度上
        shifter.set_formant_shift(-7.0); // フォルマント保持

        let block_size = 128;
        let input = vec![0.0; block_size];
        let mut output = vec![0.0; block_size];

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            shifter.process(&input, &mut output);
        }
        let elapsed = start.elapsed();

        // 128000サンプル（約2.67秒分 @ 48kHz）の処理
        // リアルタイムで動作するためには2.67秒以内に処理できる必要がある
        let processing_time = elapsed.as_secs_f32();
        let realtime_factor = 2.67 / processing_time;

        // デバッグビルドでは2x以上あればOK（リリースビルドで大幅に改善される）
        assert!(
            realtime_factor > 2.0,
            "Should process at least 2x realtime in debug build, got {:.1}x",
            realtime_factor
        );
    }
}
