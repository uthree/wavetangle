# Wavetangle アーキテクチャ仕様書

## 概要

Wavetangleは、egui-snarlを使用したノードベースのオーディオグラフエディタです。
リアルタイムでオーディオの入出力を接続・ルーティングできます。

## モジュール構成

```
src/
├── main.rs      # アプリケーションエントリーポイント、eframeアプリ実装
├── nodes.rs     # オーディオノードの定義
├── audio.rs     # cpalを使用したオーディオシステム
├── graph.rs     # オーディオグラフの処理ロジック
└── viewer.rs    # egui-snarlのSnarlViewer実装
```

## 主要コンポーネント

### NodeBehavior trait (nodes.rs)
すべてのノードが実装するトレイト。共通インターフェースを定義：
- `title()`: ノードのタイトル
- `category()`: ノードのカテゴリ（Input/Output/Effect）
- `input_count()`, `output_count()`: ピン数
- `input_pin_type()`, `output_pin_type()`: ピンタイプ
- `input_pin_name()`, `output_pin_name()`: ピン名
- `buffer()`: オーディオバッファ
- `is_active()`, `set_active()`: アクティブ状態

### AudioNode (nodes.rs)
オーディオグラフのノードを表すenum。個別の構造体をラップ：
- `AudioInput(AudioInputNode)`: オーディオ入力デバイスノード
- `AudioOutput(AudioOutputNode)`: オーディオ出力デバイスノード

`delegate_node_behavior!`マクロでtraitメソッドをデリゲート。
新しいノードタイプを追加する際は：
1. 構造体を定義
2. `NodeBehavior`トレイトを実装
3. `AudioNode` enumにバリアントを追加
4. マクロにバリアントを追加

### AudioConfig (audio.rs)
オーディオストリームの設定を保持する構造体。
- `sample_rate`: サンプリングレート (22050〜192000 Hz)
- `buffer_size`: バッファサイズ (64〜4096 samples)

### AudioSystem (audio.rs)
cpalを使用したオーディオデバイス管理システム。
- デバイス列挙
- 入力/出力ストリームの開始・停止
- バッファを介したデータの受け渡し
- カスタムサンプルレート・バッファサイズの設定

### AudioGraphProcessor (graph.rs)
オーディオグラフの接続を処理し、ノード間のデータルーティングを管理。
- 接続されたノード間でバッファを共有
- ストリームのライフサイクル管理

### AudioGraphViewer (viewer.rs)
egui-snarlのSnarlViewerトレイトを実装。
- ノードのUI表示
- ピン接続のロジック
- コンテキストメニュー（ノード追加・削除）

## データフロー

1. `AudioInput`ノードがデバイスから音声データを取得
2. データはArc<Mutex<Vec<f32>>>バッファに格納
3. グラフ接続に基づき、データが接続先ノードにコピー
4. `AudioOutput`ノードがバッファからデータを読み取り、デバイスに出力

## 依存ライブラリ

- **eframe/egui**: GUIフレームワーク
- **egui-snarl**: ノードグラフエディタ
- **cpal**: クロスプラットフォームオーディオI/O
- **parking_lot**: 高性能mutex実装
- **ndarray**: 将来的な信号処理用（現在未使用）

## 今後の拡張予定

- 音声エフェクトノード（ゲイン、フィルタ等）
- ファイル入出力ノード
- シグナルジェネレータノード
- グラフの保存・読み込み
