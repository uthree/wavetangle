//! WSOLAベースのピッチシフター

use super::create_hann_window;

/// デフォルトのバッファサイズ
pub const DEFAULT_PITCH_BUFFER_SIZE: usize = 16384;
/// デフォルトのグレインサイズ
pub const DEFAULT_GRAIN_SIZE: usize = 1024;
/// デフォルトのグレイン数
pub const DEFAULT_NUM_GRAINS: usize = 4;

/// 位相アラインメントの探索パラメータ
#[derive(Clone, Copy, Debug)]
pub struct PhaseAlignmentParams {
    /// 探索範囲（グレインサイズに対する割合、0.0〜1.0）
    pub search_range_ratio: f32,
    /// 相関計算に使うサンプル数（グレインサイズに対する割合、0.0〜1.0）
    pub correlation_length_ratio: f32,
    /// 位相アラインメントを有効にするか
    pub enabled: bool,
}

impl Default for PhaseAlignmentParams {
    fn default() -> Self {
        Self {
            search_range_ratio: 0.5,        // グレインサイズの50%
            correlation_length_ratio: 0.75, // グレインサイズの75%
            enabled: true,
        }
    }
}

/// グラニュラー合成ベースのピッチシフター
pub struct PitchShifter {
    /// 入力リングバッファ
    input_buffer: Vec<f32>,
    /// バッファサイズ
    buffer_size: usize,
    /// グレインサイズ
    grain_size: usize,
    /// グレイン数
    num_grains: usize,
    /// 入力書き込み位置
    write_pos: usize,
    /// 各グレインの読み取り位置（浮動小数点）
    grain_read_pos: Vec<f64>,
    /// 各グレインの位相（0.0〜1.0）
    grain_phase: Vec<f64>,
    /// ピッチシフト量（1.0 = 変化なし、2.0 = 1オクターブ上）
    pitch_ratio: f32,
    /// サンプルカウンター
    sample_count: usize,
    /// 窓関数
    window: Vec<f32>,
    /// 位相アラインメントパラメータ
    phase_alignment: PhaseAlignmentParams,
}

impl PitchShifter {
    /// デフォルトパラメータで作成
    pub fn new(_sample_rate: f32) -> Self {
        Self::with_params(
            _sample_rate,
            DEFAULT_GRAIN_SIZE,
            DEFAULT_NUM_GRAINS,
            DEFAULT_PITCH_BUFFER_SIZE,
        )
    }

    /// カスタムパラメータで作成
    pub fn with_params(
        _sample_rate: f32,
        grain_size: usize,
        num_grains: usize,
        buffer_size: usize,
    ) -> Self {
        // パラメータの妥当性を確認
        let grain_size = grain_size.clamp(128, 8192);
        let num_grains = num_grains.clamp(2, 16);
        // バッファサイズはグレインサイズ×グレイン数×4以上必要
        let min_buffer_size = grain_size * num_grains * 4;
        let buffer_size = buffer_size.max(min_buffer_size);

        let window = create_hann_window(grain_size);

        // グレインを均等に配置（位相）
        let mut grain_phase = vec![0.0; num_grains];
        for (i, phase) in grain_phase.iter_mut().enumerate() {
            *phase = i as f64 / num_grains as f64;
        }

        // 初期書き込み位置
        let write_pos = buffer_size / 2;

        // グレインの読み取り位置を書き込み位置の手前に設定
        let grain_spacing = grain_size / num_grains;
        let mut grain_read_pos = vec![0.0; num_grains];
        for (i, read_pos) in grain_read_pos.iter_mut().enumerate() {
            let offset = grain_size + (i * grain_spacing);
            *read_pos = (write_pos as f64 - offset as f64).rem_euclid(buffer_size as f64);
        }

        Self {
            input_buffer: vec![0.0; buffer_size],
            buffer_size,
            grain_size,
            num_grains,
            write_pos,
            grain_read_pos,
            grain_phase,
            pitch_ratio: 1.0,
            sample_count: 0,
            window,
            phase_alignment: PhaseAlignmentParams::default(),
        }
    }

    /// 位相アラインメントパラメータを設定
    pub fn set_phase_alignment(&mut self, params: PhaseAlignmentParams) {
        self.phase_alignment = params;
    }

    /// グレインサイズを設定（内部状態がリセットされる）
    pub fn set_grain_size(&mut self, grain_size: usize) {
        let grain_size = grain_size.clamp(128, 8192);
        if grain_size != self.grain_size {
            self.rebuild_with_params(grain_size, self.num_grains, self.buffer_size);
        }
    }

    /// グレイン数を設定（内部状態がリセットされる）
    pub fn set_num_grains(&mut self, num_grains: usize) {
        let num_grains = num_grains.clamp(2, 16);
        if num_grains != self.num_grains {
            self.rebuild_with_params(self.grain_size, num_grains, self.buffer_size);
        }
    }

    /// パラメータを変更して内部バッファを再構築
    fn rebuild_with_params(&mut self, grain_size: usize, num_grains: usize, buffer_size: usize) {
        let min_buffer_size = grain_size * num_grains * 4;
        let buffer_size = buffer_size.max(min_buffer_size);

        self.grain_size = grain_size;
        self.num_grains = num_grains;
        self.buffer_size = buffer_size;

        // バッファを再割り当て
        self.input_buffer = vec![0.0; buffer_size];

        self.window = create_hann_window(grain_size);

        // グレイン配列を再割り当て
        self.grain_phase = vec![0.0; num_grains];
        self.grain_read_pos = vec![0.0; num_grains];

        // 状態をリセット
        self.write_pos = buffer_size / 2;
        self.sample_count = 0;

        // グレインを均等に再配置
        for (i, phase) in self.grain_phase.iter_mut().enumerate() {
            *phase = i as f64 / num_grains as f64;
        }

        let grain_spacing = grain_size / num_grains;
        for (i, read_pos) in self.grain_read_pos.iter_mut().enumerate() {
            let offset = grain_size + (i * grain_spacing);
            *read_pos = (self.write_pos as f64 - offset as f64).rem_euclid(buffer_size as f64);
        }
    }

    /// ピッチシフト量を設定
    /// 半音単位でピッチを設定（-12 = 1オクターブ下、+12 = 1オクターブ上）
    pub fn set_semitones(&mut self, semitones: f32) {
        self.pitch_ratio = 2.0_f32.powf(semitones / 12.0);
    }

    /// 線形補間でサンプルを取得
    fn interpolate(&self, pos: f64) -> f32 {
        let pos_mod = pos.rem_euclid(self.buffer_size as f64);
        let idx0 = pos_mod.floor() as usize;
        let idx1 = (idx0 + 1) % self.buffer_size;
        let frac = (pos_mod - idx0 as f64) as f32;

        self.input_buffer[idx0] * (1.0 - frac) + self.input_buffer[idx1] * frac
    }

    /// 2つの位置間の正規化相互相関を計算（サブサンプリングで高速化）
    /// 戻り値は-1.0〜1.0の範囲（正規化されているため信号レベルに依存しない）
    fn compute_correlation(&self, pos1: f64, pos2: f64, length: usize) -> f32 {
        const SUBSAMPLE_STEP: usize = 4; // 4サンプルごとに計算
        let mut correlation = 0.0f32;
        let mut energy1 = 0.0f32;
        let mut energy2 = 0.0f32;

        for i in (0..length).step_by(SUBSAMPLE_STEP) {
            let sample1 = self.interpolate(pos1 + i as f64);
            let sample2 = self.interpolate(pos2 + i as f64);
            correlation += sample1 * sample2;
            energy1 += sample1 * sample1;
            energy2 += sample2 * sample2;
        }

        // 正規化（ゼロ除算を防ぐ）
        let denominator = (energy1 * energy2).sqrt();
        if denominator > 1e-10 {
            correlation / denominator
        } else {
            0.0
        }
    }

    /// 参照グレインとの相互相関が最大になる位置を探索（WSOLA風アルゴリズム）
    fn find_best_alignment(&self, base_pos: f64, reference_pos: f64) -> f64 {
        // パラメータから探索範囲と相関長を計算
        let search_range =
            (self.grain_size as f32 * self.phase_alignment.search_range_ratio) as i32;
        let correlation_length =
            (self.grain_size as f32 * self.phase_alignment.correlation_length_ratio) as usize;

        // 探索範囲に応じてステップサイズを調整
        let coarse_step = (search_range / 64).max(4) as usize;
        let fine_step = (coarse_step / 4).max(1);

        let mut best_pos = base_pos;
        let mut best_correlation = f32::MIN;

        // 粗い探索
        for offset in (-search_range..=search_range).step_by(coarse_step) {
            let candidate_pos = base_pos + offset as f64;
            let correlation =
                self.compute_correlation(candidate_pos, reference_pos, correlation_length);

            if correlation > best_correlation {
                best_correlation = correlation;
                best_pos = candidate_pos;
            }
        }

        // 細かい探索（粗い探索で見つかった位置の周辺をより細かく探索）
        let fine_base = best_pos;
        let fine_range = coarse_step as i32;
        for offset in (-fine_range..=fine_range).step_by(fine_step) {
            let candidate_pos = fine_base + offset as f64;
            let correlation =
                self.compute_correlation(candidate_pos, reference_pos, correlation_length);

            if correlation > best_correlation {
                best_correlation = correlation;
                best_pos = candidate_pos;
            }
        }

        best_pos.rem_euclid(self.buffer_size as f64)
    }

    /// 現在最も位相が進んでいるグレインのインデックスを取得
    fn find_reference_grain(&self, exclude_idx: usize) -> Option<usize> {
        let mut best_idx = None;
        let mut best_phase = -1.0;

        for i in 0..self.num_grains {
            if i == exclude_idx {
                continue;
            }
            // 位相が0.3〜0.7の範囲にあるグレインを優先（窓関数の中央部分）
            let phase = self.grain_phase[i];
            if phase > 0.3 && phase < 0.7 && phase > best_phase {
                best_phase = phase;
                best_idx = Some(i);
            }
        }

        // 見つからなければ、除外以外で最大の位相を持つものを選択
        if best_idx.is_none() {
            for i in 0..self.num_grains {
                if i != exclude_idx && self.grain_phase[i] > best_phase {
                    best_phase = self.grain_phase[i];
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }

    /// サンプルを処理
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let grain_size_f = self.grain_size as f64;
        let phase_increment = 1.0 / grain_size_f;

        for (i, out_sample) in output.iter_mut().enumerate() {
            // 入力をバッファに書き込み
            self.input_buffer[self.write_pos] = input[i];
            self.write_pos = (self.write_pos + 1) % self.buffer_size;

            // 各グレインからの出力を合成
            let mut sum = 0.0f32;

            for grain_idx in 0..self.num_grains {
                let phase = self.grain_phase[grain_idx];

                // 窓関数の値を取得
                let window_pos = (phase * grain_size_f) as usize;
                let window_val = if window_pos < self.grain_size {
                    self.window[window_pos]
                } else {
                    0.0
                };

                // 補間してサンプルを取得
                let sample = self.interpolate(self.grain_read_pos[grain_idx]);
                sum += sample * window_val;

                // 読み取り位置を進める（ピッチ比率に応じて）
                self.grain_read_pos[grain_idx] += self.pitch_ratio as f64;
                if self.grain_read_pos[grain_idx] >= self.buffer_size as f64 {
                    self.grain_read_pos[grain_idx] -= self.buffer_size as f64;
                }

                // 位相を進める
                self.grain_phase[grain_idx] += phase_increment;

                // グレインが終了したら次の位置にリセット
                if self.grain_phase[grain_idx] >= 1.0 {
                    self.grain_phase[grain_idx] -= 1.0;

                    // ベースとなる開始位置（書き込み位置の手前）
                    let offset = (self.grain_size * self.num_grains / 2) as f64;
                    let base_pos =
                        (self.write_pos as f64 - offset).rem_euclid(self.buffer_size as f64);

                    // 参照グレインを探して位相アラインメントを適用（有効な場合のみ）
                    let aligned_pos = if self.phase_alignment.enabled {
                        if let Some(ref_idx) = self.find_reference_grain(grain_idx) {
                            let ref_pos = self.grain_read_pos[ref_idx];
                            self.find_best_alignment(base_pos, ref_pos)
                        } else {
                            base_pos
                        }
                    } else {
                        base_pos
                    };

                    self.grain_read_pos[grain_idx] = aligned_pos;
                }
            }

            // 正規化（グレイン数で割る）
            *out_sample = sum / (self.num_grains as f32 / 2.0);
        }

        self.sample_count += input.len();
    }
}

impl Default for PitchShifter {
    fn default() -> Self {
        Self::new(44100.0)
    }
}
