//! WORLDアルゴリズムベースの声質変換DSPモジュール
//!
//! 入力音声を分析(F0, スペクトル包絡, 非周期性指標)し、
//! ピッチシフトとフォルマントシフトを適用して再合成する。

use ndarray::Array2;

/// WORLDボコーダーのデフォルトフレーム周期 (ms)
const FRAME_PERIOD: f64 = 5.0;

/// WORLDボコーダー処理
///
/// ブロック単位で音声を蓄積し、WORLDの分析→変換→再合成を行う。
/// overlap-add方式で出力を生成し、レイテンシを最小化する。
pub struct WorldVocoder {
    /// サンプルレート
    sample_rate: i32,
    /// ピッチシフト量（半音）
    pitch_shift_semitones: f64,
    /// フォルマントシフト量（半音）
    formant_shift_semitones: f64,
    /// 入力バッファ（f64精度で蓄積）
    input_buffer: Vec<f64>,
    /// 出力バッファ（再合成済みサンプル）
    output_buffer: Vec<f32>,
    /// 処理ブロックサイズ（サンプル数）
    block_size: usize,
    /// ホップサイズ（新しいブロックごとにシフトする量）
    hop_size: usize,
    /// フェードイン/アウト用の窓関数
    fade_window: Vec<f64>,
    /// 前回のブロックの末尾部分（クロスフェード用）
    prev_tail: Vec<f64>,
}

impl WorldVocoder {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate as i32;
        // 分析に十分なブロックサイズ（約100ms）
        let block_size = (sample_rate * 0.1) as usize;
        // 50%オーバーラップ
        let hop_size = block_size / 2;

        // クロスフェード窓を生成
        let overlap = block_size - hop_size;
        let fade_window: Vec<f64> = (0..overlap)
            .map(|i| {
                let t = i as f64 / overlap as f64;
                0.5 * (1.0 - (std::f64::consts::PI * t).cos())
            })
            .collect();

        Self {
            sample_rate: sr,
            pitch_shift_semitones: 0.0,
            formant_shift_semitones: 0.0,
            input_buffer: Vec::with_capacity(block_size * 2),
            output_buffer: Vec::new(),
            block_size,
            hop_size,
            fade_window,
            prev_tail: Vec::new(),
        }
    }

    pub fn set_pitch_shift(&mut self, semitones: f64) {
        self.pitch_shift_semitones = semitones;
    }

    pub fn set_formant_shift(&mut self, semitones: f64) {
        self.formant_shift_semitones = semitones;
    }

    /// 入力サンプルを処理して出力サンプルを生成
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // 入力をf64に変換してバッファに追加
        for &s in input {
            self.input_buffer.push(s as f64);
        }

        // 十分なサンプルが溜まったら処理
        while self.input_buffer.len() >= self.block_size {
            let block: Vec<f64> = self.input_buffer[..self.block_size].to_vec();
            self.input_buffer.drain(..self.hop_size);

            let processed = self.process_block(&block);
            self.overlap_add(&processed);
        }

        // 出力バッファから必要な分だけコピー
        let copy_len = output.len().min(self.output_buffer.len());
        for i in 0..copy_len {
            output[i] = self.output_buffer[i];
        }
        // 残りは無音
        for i in copy_len..output.len() {
            output[i] = 0.0;
        }
        // 使用分を消費
        if copy_len > 0 {
            self.output_buffer.drain(..copy_len);
        }
    }

    /// 1ブロック分のWORLD分析→変換→再合成
    fn process_block(&self, block: &[f64]) -> Vec<f64> {
        let fs = self.sample_rate;
        let f0_floor = 71.0;
        let fft_size = world_dsp::get_fft_size_for_cheaptrick(fs, f0_floor);

        // F0推定（YINを使用、高速）
        let yin = world_dsp::Yin::new(fs);
        let (temporal_positions, f0) = world_dsp::F0Estimator::estimate(&yin, block);

        // スペクトル包絡推定
        let cheaptrick = world_dsp::CheapTrick::new(fs, fft_size);
        let spectrogram = cheaptrick.estimate(block, &temporal_positions, &f0);

        // 非周期性指標推定
        let d4c = world_dsp::D4C::new(fs, fft_size);
        let aperiodicity = d4c.estimate(block, &temporal_positions, &f0);

        // パラメータ変換
        let pitch_ratio = 2.0_f64.powf(self.pitch_shift_semitones / 12.0);
        let formant_ratio = 2.0_f64.powf(self.formant_shift_semitones / 12.0);

        // ピッチシフト: F0を変更
        let modified_f0: Vec<f64> = f0
            .iter()
            .map(|&v| if v > 0.0 { v * pitch_ratio } else { 0.0 })
            .collect();

        // フォルマントシフト: スペクトル包絡を周波数方向にシフト
        let modified_spectrogram = if (formant_ratio - 1.0).abs() > 1e-6 {
            self.shift_spectrogram(&spectrogram, formant_ratio)
        } else {
            spectrogram
        };

        // 再合成
        let synthesizer = world_dsp::Synthesizer::new(FRAME_PERIOD, fs, fft_size);
        let result = synthesizer.synthesize(&modified_f0, &modified_spectrogram, &aperiodicity);

        result.to_vec()
    }

    /// スペクトル包絡を周波数方向にシフト（フォルマント操作）
    fn shift_spectrogram(&self, spectrogram: &Array2<f64>, ratio: f64) -> Array2<f64> {
        let (num_frames, freq_bins) = spectrogram.dim();
        let mut shifted = Array2::zeros((num_frames, freq_bins));

        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                // 元の周波数位置（シフト後の位置から逆算）
                let src_bin = bin as f64 / ratio;
                let src_idx = src_bin.floor() as usize;
                let frac = src_bin - src_idx as f64;

                if src_idx + 1 < freq_bins {
                    // 線形補間
                    shifted[[frame, bin]] = spectrogram[[frame, src_idx]] * (1.0 - frac)
                        + spectrogram[[frame, src_idx + 1]] * frac;
                } else if src_idx < freq_bins {
                    shifted[[frame, bin]] = spectrogram[[frame, src_idx]];
                }
                // 範囲外は0のまま
            }
        }

        shifted
    }

    /// overlap-addで出力バッファに追加
    fn overlap_add(&mut self, processed: &[f64]) {
        if processed.is_empty() {
            return;
        }

        let overlap_len = self.block_size - self.hop_size;

        if self.prev_tail.is_empty() {
            // 最初のブロック: hop_size分だけ出力、残りは保存
            for i in 0..processed.len().min(self.hop_size) {
                self.output_buffer.push(processed[i] as f32);
            }
            if processed.len() > self.hop_size {
                self.prev_tail = processed[self.hop_size..].to_vec();
            }
        } else {
            // クロスフェード部分
            let cross_len = overlap_len.min(self.prev_tail.len()).min(processed.len());
            for i in 0..cross_len {
                let fade_in = if i < self.fade_window.len() {
                    self.fade_window[i]
                } else {
                    1.0
                };
                let fade_out = 1.0 - fade_in;
                let mixed = self.prev_tail[i] * fade_out + processed[i] * fade_in;
                self.output_buffer.push(mixed as f32);
            }

            // クロスフェード後の新しいデータ（hop_size分まで）
            let new_start = cross_len;
            let new_end = processed.len().min(self.hop_size + cross_len);
            for i in new_start..new_end {
                self.output_buffer.push(processed[i] as f32);
            }

            // 残りは次のブロック用に保存
            if processed.len() > self.hop_size {
                self.prev_tail = processed[self.hop_size..].to_vec();
            } else {
                self.prev_tail.clear();
            }
        }
    }
}
