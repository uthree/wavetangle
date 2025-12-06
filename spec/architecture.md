# Wavetangle アーキテクチャ仕様書

## 概要

Wavetangleは、egui-snarlを使用したノードベースのオーディオグラフエディタです。
リアルタイムでオーディオの入出力を接続・ルーティングできます。

## モジュール構成

```
src/
├── main.rs              # アプリケーションエントリーポイント、eframeアプリ実装
├── nodes.rs             # オーディオノードの定義
├── audio.rs             # cpalを使用したオーディオシステム
├── dsp.rs               # DSPアルゴリズム（フィルター、コンプレッサー、FFT）
├── effect_processor.rs  # エフェクト処理専用スレッド
├── graph.rs             # オーディオグラフの処理ロジック
├── pipeline.rs          # パイプライン並列処理（ロックフリーSPSCバッファ）
└── viewer.rs            # egui-snarlのSnarlViewer実装
```

## 主要コンポーネント

### RingBuffer (nodes.rs)
音声データの低遅延転送用リングバッファ。
- `write()`: データを書き込む
- `read()`: データを読み込む（読み取り位置が進む）
- `available()`: 利用可能なサンプル数

### ChannelBuffer (nodes.rs)
`Arc<Mutex<RingBuffer>>` - チャンネルごとの共有バッファ。

### NodeBehavior trait (nodes.rs)
すべてのノードが実装するトレイト。共通インターフェースを定義：
- `title()`: ノードのタイトル
- `category()`: ノードのカテゴリ（Input/Output/Effect）
- `input_count()`, `output_count()`: ピン数（チャンネル数に連動）
- `input_pin_type()`, `output_pin_type()`: ピンタイプ
- `input_pin_name()`, `output_pin_name()`: ピン名（L, R, C, LFE, SL, SR）
- `channel_buffer()`: 指定チャンネルの出力バッファを取得
- `input_buffer()`: 指定入力ピンのバッファを取得（エフェクトノード用）
- `channels()`, `set_channels()`: チャンネル数（ストリーム開始時に設定、ピン数も更新）
- `is_active()`, `set_active()`: アクティブ状態

### AudioNode (nodes.rs)
オーディオグラフのノードを表すenum。個別の構造体をラップ：
- `AudioInput(AudioInputNode)`: オーディオ入力デバイスノード（出力ピン = チャンネル数、スペクトラム表示統合）
- `AudioOutput(AudioOutputNode)`: オーディオ出力デバイスノード（入力ピン = チャンネル数、スペクトラム表示統合）
- `Gain(GainNode)`: ゲインエフェクトノード（1入力1出力、ゲインスライダー付き）
- `Add(AddNode)`: 加算ノード（2入力1出力、A + B）
- `Multiply(MultiplyNode)`: 乗算ノード（2入力1出力、A × B、リングモジュレーション用）
- `Filter(FilterNode)`: フィルターノード（1入力1出力、Low/High/Band Pass、カットオフ周波数、Q値）
- `SpectrumAnalyzer(SpectrumAnalyzerNode)`: スペクトラムアナライザー（1入力1出力、FFTでスペクトラム表示）
- `Compressor(CompressorNode)`: コンプレッサー（1入力1出力、Threshold、Ratio、Attack、Release、Makeup Gain）
- `PitchShift(PitchShiftNode)`: ピッチシフター（1入力1出力、PSOLAアルゴリズム、-12〜+12半音）
- `GraphicEq(GraphicEqNode)`: グラフィックEQ（1入力1出力、FFTベースの周波数ゲイン調整、egui_plotによるカーブエディタUI、入力スペクトラム表示統合）

`delegate_node_behavior!`マクロでtraitメソッドをデリゲート。
新しいノードタイプを追加する際は：
1. 構造体を定義（`channel_buffers: Vec<ChannelBuffer>`または`input_buffer`/`output_buffer`を含む）
2. `NodeBehavior`トレイトを実装
3. `AudioNode` enumにバリアントを追加
4. マクロにバリアントを追加
5. `viewer.rs`の`show_body`と`show_graph_menu`を更新

### AudioConfig (audio.rs)
オーディオストリームの設定を保持する構造体。
- `sample_rate`: サンプリングレート (22050〜192000 Hz)
- `buffer_size`: バッファサイズ (64〜4096 samples)

### AudioSystem (audio.rs)
cpalを使用したオーディオデバイス管理システム。
- デバイス列挙
- 入力ストリーム：デバイスからデータを取得し、デインターリーブしてチャンネルバッファに書き込む
- 出力ストリーム：チャンネルバッファから読み取り、インターリーブしてデバイスに出力
- カスタムサンプルレート・バッファサイズの設定

### AudioGraphProcessor (graph.rs)
オーディオグラフの接続を処理し、ノード間のデータルーティングを管理。
- 出力ノード起動時に、接続された入力ノードのチャンネルバッファを直接参照
- ストリームのライフサイクル管理
- チャンネルごとの独立したルーティング
- トポロジカルソートによるエフェクトノードの処理順序決定
- EffectProcessorとの連携によるリアルタイムエフェクト処理

### EffectProcessor (effect_processor.rs)
専用スレッドでエフェクトノードをリアルタイム処理。コピーベースのバッファアーキテクチャを採用。
- 2ms間隔で処理スレッドが動作
- `EffectNodeInfo`: 処理対象ノードの情報
  - `source_buffers`: 接続元ノードの出力バッファ（データコピー元）
  - `input_buffers`: ノード自身の入力バッファ（データコピー先、処理用）
  - `output_buffer`: ノード自身の出力バッファ
- `EffectNodeType`: エフェクトタイプのenum（Gain, Add, Multiply, Filter, SpectrumAnalyzer, Compressor, PitchShift, GraphicEq）
- 処理フロー:
  1. ソースバッファからノードの入力バッファへデータをコピー
  2. 入力バッファから読み取り、DSP処理を行い、出力バッファに書き込み
  3. ソースバッファの読み取り位置を進める
- スレッドセーフなDSP状態管理（`Arc<Mutex<>>`でUI/処理スレッド間共有）

### AudioGraphViewer (viewer.rs)
egui-snarlのSnarlViewerトレイトを実装。
- ノードのUI表示
- 動的なピン数（チャンネル数に応じて変化）
- ピン接続のロジック（同じPinType同士のみ）
- コンテキストメニュー（ノード追加・削除）

### Pipeline (pipeline.rs)
パイプライン並列処理のための基盤。将来的にエフェクトノードをパイプライン処理する。
- `SpscProducer`/`SpscConsumer`: ロックフリーSPSCリングバッファ（ringbufクレート使用）
- `ProcessingNode`トレイト: スレッド上でオーディオ処理を行うノードのインターフェース
- `NodeThread`: 処理スレッドの管理（開始、停止、状態確認）
- `PipelineBuilder`: ノードをチェーンしてパイプラインを構築
- `Pipeline`: 実行中のパイプライン管理

## データフロー

1. `AudioInput`ノードがデバイスからインターリーブされた音声データを取得
2. データはチャンネルごとに分離され、各`ChannelBuffer`（リングバッファ）に書き込まれる
3. `EffectProcessor`スレッドがトポロジカル順序でエフェクトノードを処理
   - ソースノードの出力バッファからエフェクトノードの入力バッファへデータをコピー
   - エフェクトノードは自身の入力バッファから読み取り、DSP処理を行い、出力バッファに書き込む
   - フィルター、コンプレッサー等はDSP状態を保持してサンプル間で連続性を維持
   - 処理後、ソースバッファの読み取り位置を進める
4. `AudioOutput`ノード起動時、接続された入力ピンのチャンネルバッファを参照
5. 出力コールバックで各チャンネルバッファから読み取り、インターリーブしてデバイスに出力

### コピーベースアーキテクチャの利点
- 各ノードが独自のバッファを持つため、スレッド間の状態共有を最小化
- ノードの追加・削除時にバッファ参照が壊れない
- デバッグとトラブルシューティングが容易

## DSP処理 (dsp.rs)

エフェクトノードのオーディオ処理アルゴリズムを実装：
- `BiquadCoeffs`: Biquadフィルター係数（LowPass/HighPass/BandPass）
- `BiquadState`: フィルター状態（1サンプル処理）
- `CompressorParams`: コンプレッサーパラメータ
- `CompressorState`: コンプレッサー状態（エンベロープフォロワー、初期値-120dB）
- `SpectrumAnalyzer`: FFTベースのスペクトラム解析
  - Hann窓、1024点FFT
  - 指数移動平均によるスムージング（係数0.8）
  - egui_plotでバーチャート表示（48バンド、対数周波数スケール）
- `PitchShifter`: グラニュラー合成ベースのピッチシフト
  - 4グレインオーバーラップ、1024サンプルグレインサイズ
  - ハン窓による滑らかなクロスフェード
  - 線形補間による高品質な再サンプリング
- `GraphicEq`: FFTベースのグラフィックイコライザー
  - 2048点FFT、50%オーバーラップ
  - EqPoint構造体でコントロールポイント管理（周波数とゲイン）
  - 対数周波数スケールで線形補間
  - egui_plotによるカーブエディタUI（ドラッグでポイント移動可能）

## 依存ライブラリ

- **eframe/egui**: GUIフレームワーク
- **egui-snarl**: ノードグラフエディタ
- **egui_plot**: プロット表示（スペクトラムアナライザー用）
- **cpal**: クロスプラットフォームオーディオI/O
- **parking_lot**: 高性能mutex実装
- **ringbuf**: ロックフリーSPSCリングバッファ（パイプライン並列処理用）
- **ndarray**: 将来的な信号処理用（現在未使用）
- **rustfft**: FFT実装（スペクトラムアナライザー用）

## 今後の拡張予定

- ファイル入出力ノード
- シグナルジェネレータノード（オシレーター）
- ディレイ・リバーブノード
- グラフの保存・読み込み
