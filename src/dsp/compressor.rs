//! コンプレッサー

/// コンプレッサーパラメータ
#[derive(Clone, Copy)]
pub struct CompressorParams {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_db: f32,
    pub sample_rate: f32,
}

/// コンプレッサー状態
#[derive(Clone)]
pub struct CompressorState {
    /// エンベロープ
    pub envelope: f32,
}

impl Default for CompressorState {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorState {
    pub fn new() -> Self {
        // 初期エンベロープを十分低い値（-120dB）に設定
        // 0.0（0dB）だと閾値より高くなり、最初からゲインリダクションが適用されてしまう
        Self { envelope: -120.0 }
    }

    /// 1サンプル処理
    pub fn process(&mut self, input: f32, params: &CompressorParams) -> f32 {
        let CompressorParams {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            sample_rate,
        } = *params;
        // 入力レベルをdBに変換
        let input_abs = input.abs();
        let input_db = if input_abs > 1e-6 {
            20.0 * input_abs.log10()
        } else {
            -120.0
        };

        // アタック/リリース係数
        let attack_coeff = (-1.0 / (attack_ms * sample_rate / 1000.0)).exp();
        let release_coeff = (-1.0 / (release_ms * sample_rate / 1000.0)).exp();

        // エンベロープフォロワー
        if input_db > self.envelope {
            self.envelope = attack_coeff * self.envelope + (1.0 - attack_coeff) * input_db;
        } else {
            self.envelope = release_coeff * self.envelope + (1.0 - release_coeff) * input_db;
        }

        // ゲインリダクション計算
        let gain_reduction_db = if self.envelope > threshold_db {
            (threshold_db - self.envelope) * (1.0 - 1.0 / ratio)
        } else {
            0.0
        };

        // 出力ゲイン（線形）
        let output_gain = 10.0_f32.powf((gain_reduction_db + makeup_db) / 20.0);

        input * output_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor() {
        let mut state = CompressorState::new();
        let params = CompressorParams {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_db: 0.0,
            sample_rate: 44100.0,
        };

        // 閾値以下の信号のテスト：初期状態ではエンベロープが低いのでゲインリダクションなし
        // まず小さい信号を複数回処理してエンベロープを安定させる
        for _ in 0..100 {
            state.process(0.01, &params); // -40dB、閾値より十分小さい
        }
        let output = state.process(0.01, &params);
        // 閾値以下では圧縮されない（makeup_gain=0なので入力≒出力）
        assert!(
            (output - 0.01).abs() < 0.005,
            "Expected ~0.01, got {}",
            output
        );

        // 閾値以上の信号（十分な時間エンベロープを追従させる）
        // 10msアタック @ 44100Hz = 441サンプルで63%到達
        // エンベロープが-120dBから0dBまで上昇するには約5-6時定数必要
        let mut state2 = CompressorState::new();
        for _ in 0..5000 {
            state2.process(1.0, &params);
        }
        let output = state2.process(1.0, &params);
        // エンベロープが0dB近くになり、閾値(-20dB)を超えているので圧縮される
        assert!(
            output < 0.95, // 4:1レシオで約6dBのゲインリダクションを期待
            "Expected compression, got {}",
            output
        );
    }
}
