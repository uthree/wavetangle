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
├── project.rs           # プロジェクトの保存・読み込み
└── viewer.rs            # egui-snarlのSnarlViewer実装
```

## 主要コンポーネント

### AudioBuffer (nodes.rs)
音声データの転送用FIFOバッファ（VecDequeベース）。
- `push()`: データを末尾に追加（容量超過時は先頭を削除）
- `read()`: データのコピーを取得（状態変更なし - 複数コンシューマー対応）
- `read_and_consume()`: 読み取りと消費をアトミックに実行（競合状態防止）
- `consume()`: 先頭からデータを削除
- `len()`: 利用可能なサンプル数

設計原則:
- `read()`は状態を変更しない（複数コンシューマー対応）
- プロデューサー/コンシューマー間の競合を防ぐため、エフェクト処理では
  `read_and_consume()`を使用してアトミックに操作する

### ChannelBuffer (nodes.rs)
`Arc<Mutex<AudioBuffer>>` - チャンネルごとの共有バッファ。

### NodeBehavior trait (nodes.rs)
すべてのノードが実装するトレイト。共通インターフェースを定義：
- `title()`: ノードのタイトル
- `input_count()`, `output_count()`: ピン数（チャンネル数に連動）
- `input_pin_type()`, `output_pin_type()`: ピンタイプ
- `input_pin_name()`, `output_pin_name()`: ピン名（L, R, C, LFE, SL, SR）
- `channel_buffer()`: 指定チャンネルの出力バッファを取得
- `input_buffer()`: 指定入力ピンのバッファを取得（エフェクトノード用）
- `channels()`, `set_channels()`: チャンネル数（ストリーム開始時に設定、ピン数も更新）
- `is_active()`, `set_active()`: アクティブ状態

### ヘルパー関数 (nodes.rs)
コード重複を削減するための共通関数：
- `channel_name()`: チャンネルインデックスからチャンネル名を取得（L, R, C, LFE, SL, SR）
- `resize_channel_buffers()`: チャンネルバッファのサイズを調整

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
- 複数出力ストリーム：`HashMap<OutputStreamId, Stream>`で複数の出力ノードを同時にサポート
- 各出力ノードは独自のバッファを持ち、ストリームIDで管理
- カスタムサンプルレート・バッファサイズの設定

### AudioGraphProcessor (graph.rs)
オーディオグラフの接続を処理し、ノード間のデータルーティングを管理。
- 出力ノードは常に自身のバッファを使用（複数出力の分岐をサポート）
- ストリームのライフサイクル管理（ストリームIDで出力を個別管理）
- チャンネルごとの独立したルーティング
- トポロジカルソートによるエフェクトノードの処理順序決定
- 出力ノードへのデータルーティングはPassThroughエフェクトとして処理
- EffectProcessorとの連携によるリアルタイムエフェクト処理

### EffectProcessor (effect_processor.rs)
専用スレッドでエフェクトノードをリアルタイム処理。コピーベースのバッファアーキテクチャを採用。
- 2ms間隔で処理スレッドが動作（利用可能なデータ量に応じて動的にブロックサイズを調整）
- `EffectNodeInfo`: 処理対象ノードの情報
  - `source_buffers`: 接続元ノードの出力バッファ（データコピー元）
  - `input_buffers`: ノード自身の入力バッファ（データコピー先、処理用）
  - `output_buffer`: ノード自身の出力バッファ
- `EffectNodeType`: エフェクトタイプのenum（Gain, Add, Multiply, Filter, SpectrumAnalyzer, Compressor, PitchShift, GraphicEq, PassThrough）
- 処理フロー:
  1. ソースバッファから`read_and_consume()`でアトミックにデータを読み取り・消費し、入力バッファへコピー
  2. 入力バッファから読み取り、DSP処理を行い、出力バッファに書き込み
  3. PassThrough（出力ノードへのルーティング）はソースから直接出力バッファにコピー
- スレッドセーフなDSP状態管理（`Arc<Mutex<>>`でUI/処理スレッド間共有）
- バッファ蓄積防止のため、利用可能なデータをできるだけ多く処理（最大8倍まで）

### AudioGraphViewer (viewer.rs)
egui-snarlのSnarlViewerトレイトを実装。
- ノードのUI表示
- 動的なピン数（チャンネル数に応じて変化）
- ピン接続のロジック（同じPinType同士のみ）
- コンテキストメニュー（ノード追加・削除）

## データフロー

### コピーベースアーキテクチャ
すべてのノード間接続はコピーベースで動作する。バッファを直接共有せず、データを常にコピーすることで：
- 複数の出力への分岐が安全に行える（consume()の競合なし）
- 各ノードが独自のバッファを持つため、スレッド間の状態共有を最小化
- ノードの追加・削除時にバッファ参照が壊れない
- デバッグとトラブルシューティングが容易

### 処理フロー
1. `AudioInput`ノードがデバイスからインターリーブされた音声データを取得
2. データはチャンネルごとに分離され、各`ChannelBuffer`（リングバッファ）に書き込まれる
3. `EffectProcessor`スレッドがトポロジカル順序で全ノードを処理
   - ソースノードの出力バッファから`read_and_consume()`でデータを取得
   - 自身の入力バッファ（または直接出力バッファ）にコピー
   - DSP処理を行い、出力バッファに書き込む
   - 出力ノードへのルーティングもPassThroughとして同様に処理
4. `AudioOutput`ノードは自身のバッファからデータを読み取り、デバイスに出力

## DSP処理 (dsp.rs)

エフェクトノードのオーディオ処理アルゴリズムを実装。

### ヘルパー関数
- `create_hann_window()`: 指定サイズのHann窓を生成（FFT、ピッチシフト、EQで共通使用）

### Biquadフィルター
- `BiquadCoeffs`: Biquadフィルター係数（LowPass/HighPass/BandPass）
- `BiquadState`: フィルター状態（1サンプル処理）
- `CompressorParams`: コンプレッサーパラメータ
- `CompressorState`: コンプレッサー状態（エンベロープフォロワー、初期値-120dB）
- `SpectrumAnalyzer`: FFTベースのスペクトラム解析
  - Hann窓、1024点FFT
  - 指数移動平均によるスムージング（係数0.8）
  - egui_plotでバーチャート表示（48バンド、対数周波数スケール）
- `PitchShifter`: グラニュラー合成ベースのピッチシフト
  - 動的に調整可能なパラメータ（グレインサイズ、グレイン数）
  - デフォルト: 4グレインオーバーラップ、1024サンプルグレインサイズ
  - ハン窓による滑らかなクロスフェード
  - 線形補間による高品質な再サンプリング
  - 相互相関による位相アラインメント（エコー低減）
  - 遅延-品質のトレードオフをUIで調整可能
- `GraphicEq`: FFTベースのグラフィックイコライザー
  - 2048点FFT、50%オーバーラップ
  - EqPoint構造体でコントロールポイント管理（周波数とゲイン）
  - 対数周波数スケールで線形補間
  - egui_plotによるカーブエディタUI（ドラッグでポイント移動可能）

## プロジェクトファイル (project.rs)

グラフの保存・読み込み機能を提供。JSON形式でシリアライズ。

### ProjectFile
- `nodes`: 各ノードのパラメータ（デバイス名、ゲイン値、フィルター設定など）
- `positions`: ノードの位置情報
- `connections`: ノード間の接続情報
- ファイル拡張子: `.wtg`

### 対応メニュー
- File > New: 新規プロジェクト
- File > Open...: プロジェクトを開く（rfdファイルダイアログ使用）
- File > Save: 上書き保存
- File > Save As...: 名前をつけて保存

## 依存ライブラリ

- **eframe/egui**: GUIフレームワーク
- **egui-snarl**: ノードグラフエディタ（serde機能有効）
- **egui_plot**: プロット表示（スペクトラムアナライザー用）
- **cpal**: クロスプラットフォームオーディオI/O
- **parking_lot**: 高性能mutex実装
- **rustfft**: FFT実装（スペクトラムアナライザー用）
- **serde/serde_json**: シリアライズ・デシリアライズ
- **rfd**: ネイティブファイルダイアログ

## 今後の拡張予定

- ファイル入出力ノード
- シグナルジェネレータノード（オシレーター）
- ディレイ・リバーブノード
