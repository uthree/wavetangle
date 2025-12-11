use std::fs;
use std::path::PathBuf;

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// アプリケーション設定
#[derive(Serialize, Deserialize, Default)]
pub struct AppConfig {
    /// 最後に開いたファイルのパス
    pub last_opened_file: Option<PathBuf>,
}

impl AppConfig {
    /// 設定ファイルのパスを取得
    fn config_path() -> Option<PathBuf> {
        ProjectDirs::from("", "", "Wavetangle").map(|dirs| dirs.config_dir().join("config.json"))
    }

    /// 設定を読み込み
    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(&path).ok())
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default()
    }

    /// 設定を保存
    pub fn save(&self) {
        if let Some(path) = Self::config_path() {
            // ディレクトリが存在しない場合は作成
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(content) = serde_json::to_string_pretty(self) {
                let _ = fs::write(&path, content);
            }
        }
    }

    /// 最後に開いたファイルを設定して保存
    pub fn set_last_opened(&mut self, path: Option<PathBuf>) {
        self.last_opened_file = path;
        self.save();
    }
}
