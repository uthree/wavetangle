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
    /// グレインあたりの周期数
    pub periods_per_grain: usize,
}

impl Default for TdPsolaConfig {
    fn default() -> Self {
        Self {
            min_freq: 50.0,
            max_freq: 1000.0,
            yin_threshold: 0.15,
            periods_per_grain: 2,
        }
    }
}

/// TD-PSOLAピッチシフター
pub struct TdPsolaPitchShifter {
    /// サンプルレート
    sample_rate: f32,

    /// 入力リングバッファ
    input_buffer: Vec<f32>,
    /// バッファサイズ
    buffer_size: usize,
    /// 入力書き込み位置
    write_pos: usize,

    /// 出力リングバッファ（overlap-add用）
    output_buffer: Vec<f32>,
    /// 出力読み取り位置
    output_read_pos: usize,
    /// 出力書き込み位置
    output_write_pos: usize,

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

    /// グレインあたりの周期数
    periods_per_grain: usize,

    /// 窓関数キャッシュ
    window_cache: Vec<f32>,
    /// キャッシュされた窓関数サイズ
    cached_window_size: usize,

    /// 次のグレイン開始までの残りサンプル数
    samples_until_next_grain: f32,

    /// 分析バッファ（ピッチ検出用）
    analysis_buffer: Vec<f32>,
    /// 分析バッファの書き込み位置
    analysis_write_pos: usize,

    /// 初期レイテンシフラグ
    initialized: bool,
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

        // バッファサイズ: 最大周期の8倍程度
        let max_period = (sample_rate / config.min_freq).ceil() as usize;
        let buffer_size = max_period * 8;

        // デフォルト周期（200Hz相当）
        let default_period = sample_rate / 200.0;

        Self {
            sample_rate,
            input_buffer: vec![0.0; buffer_size],
            buffer_size,
            write_pos: 0,
            output_buffer: vec![0.0; buffer_size],
            output_read_pos: 0,
            output_write_pos: buffer_size / 2, // レイテンシ分オフセット
            pitch_detector,
            interpolator: Box::new(LinearInterpolator::new()),
            pitch_shift_semitones: 0.0,
            formant_shift_semitones: 0.0,
            current_period: default_period,
            default_period,
            periods_per_grain: config.periods_per_grain,
            window_cache: Vec::new(),
            cached_window_size: 0,
            samples_until_next_grain: 0.0,
            analysis_buffer: vec![0.0; analysis_buffer_size],
            analysis_write_pos: 0,
            initialized: false,
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
    fn pitch_ratio(&self) -> f32 {
        // pitch_ratio > 1: 高いピッチ（グレイン間隔を狭める）
        // pitch_ratio < 1: 低いピッチ（グレイン間隔を広げる）
        2.0_f32.powf(self.pitch_shift_semitones / 12.0)
    }

    /// フォルマント比率を計算
    fn formant_ratio(&self) -> f32 {
        // フォルマント比率 > 1: グレインを縮小読み取り → 高いフォルマント
        // フォルマント比率 < 1: グレインを拡大読み取り → 低いフォルマント
        // semitones > 0 で高いフォルマントにしたいので、正の指数を使用
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
        self.interpolator.interpolate(&self.input_buffer, pos)
    }

    /// ピッチを更新
    fn update_pitch(&mut self) {
        if let Some(result) = self.pitch_detector.detect(&self.analysis_buffer) {
            if result.voiced {
                self.current_period = result.period;
            }
            // 無声音の場合は前回の周期を維持（YinPitchDetectorが処理済み）
        }
    }

    /// グレインを生成して出力バッファに加算
    fn generate_grain(&mut self) {
        let input_period = self.current_period;
        let formant_ratio = self.formant_ratio();

        // グレインサイズ（入力側）: 周期 × periods_per_grain
        let input_grain_size = (input_period * self.periods_per_grain as f32) as usize;
        let input_grain_size = input_grain_size.max(64).min(self.buffer_size / 4);

        // フォルマントシフト後のグレインサイズ
        // formant_ratio > 1: グレインを縮小（高いフォルマント）
        // formant_ratio < 1: グレインを拡大（低いフォルマント）
        let output_grain_size = (input_grain_size as f32 / formant_ratio) as usize;
        let output_grain_size = output_grain_size.max(64).min(self.buffer_size / 4);

        // 窓関数を取得
        let window = self.get_window(output_grain_size).to_vec();

        // グレインの中心位置（入力バッファ内）
        // 書き込み位置から適切な距離（レイテンシ）を取る
        let latency_samples = input_grain_size * 2;
        let grain_center =
            (self.write_pos as f64 - latency_samples as f64).rem_euclid(self.buffer_size as f64);

        // グレインを生成
        let half_grain = output_grain_size as f32 / 2.0;
        for (i, &win) in window.iter().enumerate().take(output_grain_size) {
            // 出力位置（グレインの中心を基準に）
            let output_offset = i as f32 - half_grain;

            // 入力位置（フォルマント比率でスケーリング）
            let input_offset = output_offset * formant_ratio;
            let input_pos =
                (grain_center + input_offset as f64).rem_euclid(self.buffer_size as f64);

            // サンプルを補間して取得
            let sample = self.read_input(input_pos);

            // 窓関数を適用
            let windowed_sample = sample * win;

            // 出力バッファに加算（overlap-add）
            let output_pos = (self.output_write_pos + i) % self.buffer_size;
            self.output_buffer[output_pos] += windowed_sample;
        }
    }

    /// オーディオを処理
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let pitch_ratio = self.pitch_ratio();

        // 出力周期 = 入力周期 / ピッチ比率
        // ピッチ比率 > 1: 出力周期短い（高いピッチ）
        // ピッチ比率 < 1: 出力周期長い（低いピッチ）

        for (i, &sample) in input.iter().enumerate() {
            // 入力バッファに書き込み
            self.input_buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.buffer_size;

            // 分析バッファに書き込み（ピッチ検出用）
            self.analysis_buffer[self.analysis_write_pos] = sample;
            self.analysis_write_pos = (self.analysis_write_pos + 1) % self.analysis_buffer.len();

            // 分析バッファが一周したらピッチを更新
            if self.analysis_write_pos == 0 {
                self.update_pitch();
            }

            // グレイン生成タイミング
            self.samples_until_next_grain -= 1.0;

            if self.samples_until_next_grain <= 0.0 {
                // 十分なサンプルがある場合のみグレインを生成
                if self.write_pos > self.buffer_size / 4 || self.initialized {
                    self.generate_grain();
                    self.initialized = true;
                }

                // 次のグレイン間隔 = 出力周期 × periods_per_grain / 2（50%オーバーラップ）
                let output_period = self.current_period / pitch_ratio;
                let grain_interval = output_period * self.periods_per_grain as f32 / 2.0;
                self.samples_until_next_grain += grain_interval.max(32.0);

                // 出力書き込み位置を進める
                self.output_write_pos =
                    (self.output_write_pos + grain_interval as usize) % self.buffer_size;
            }

            // 出力バッファから読み取り
            if self.initialized {
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
        let input = generate_sine_wave(440.0, sample_rate, 8192);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 後半の出力が無音ではないことを確認（初期化後の出力を検証）
        let max_output: f32 = output[4096..].iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_output > 0.05,
            "Output should not be silent after initialization"
        );
    }

    #[test]
    fn test_td_psola_pitch_shift() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 1オクターブ上
        shifter.set_pitch_shift(12.0);
        shifter.set_formant_shift(0.0);

        let input = generate_sine_wave(220.0, sample_rate, 8192);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        // 出力が存在することを確認
        let max_output: f32 = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_output > 0.05, "Output should not be silent");
    }

    #[test]
    fn test_td_psola_formant_shift() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // フォルマントのみシフト
        shifter.set_pitch_shift(0.0);
        shifter.set_formant_shift(6.0); // 半オクターブ上

        let input = generate_sine_wave(220.0, sample_rate, 8192);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        let max_output: f32 = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_output > 0.05, "Output should not be silent");
    }

    #[test]
    fn test_td_psola_preserve_formant() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // ピッチを上げてフォルマントを逆方向にシフト（保持）
        shifter.set_pitch_shift(12.0);
        shifter.set_formant_shift(-12.0);

        let input = generate_sine_wave(220.0, sample_rate, 8192);
        let mut output = vec![0.0; input.len()];

        shifter.process(&input, &mut output);

        let max_output: f32 = output.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_output > 0.05, "Output should not be silent");
    }

    #[test]
    fn test_td_psola_current_frequency() {
        let sample_rate = 48000.0;
        let mut shifter = TdPsolaPitchShifter::new(sample_rate);

        // 440Hzのサイン波を処理
        let input = generate_sine_wave(440.0, sample_rate, 4096);
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
