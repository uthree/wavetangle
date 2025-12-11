# Wavetangle アーキテクチャ仕様書

## 概要

Wavetangleは、egui-snarlを使用したノードベースのオーディオグラフエディタです。
リアルタイムでオーディオの入出力を接続・ルーティングできます。

## モジュール構成

```
src/
├── main.rs              # アプリケーションエントリーポイント、eframeアプリ実装
├── nodes/               # オーディオノードの定義
│   ├── mod.rs           # 共通型、トレイト、ファクトリ関数、テスト
│   ├── io.rs            # AudioInputNode, AudioOutputNode
│   ├── effects.rs       # GainNode, FilterNode, CompressorNode, PitchShiftNode, GraphicEqNode
│   ├── math.rs          # AddNode, MultiplyNode
│   └── analyzer.rs      # SpectrumAnalyzerNode
├── audio.rs             # cpalを使用したオーディオシステム
├── dsp.rs               # DSPアルゴリズム（フィルター、コンプレッサー、FFT）
├── effect_processor.rs  # エフェクト処理専用スレッド
├── graph.rs             # オーディオグラフの処理ロジック
├── project.rs           # プロジェクトの保存・読み込み
├── config.rs            # アプリケーション設定（最後に開いたファイルなど）
└── viewer.rs            # egui-snarlのSnarlViewer実装
```

## 主要コンポーネント

### AudioBuffer (nodes/mod.rs)
音声データの転送用FIFOバッファ（VecDequeベース）。
- `push()`: データを末尾に追加（容量超過時は先頭を削除）
- `read()`: データのコピーを取得（状態変更なし - 複数コンシューマー対応）
- `consume()`: 先頭からデータを削除
- `len()`: 利用可能なサンプル数

設計原則:
- `read()`は状態を変更しない（複数コンシューマー対応）
- 分岐処理ではスナップショット方式を採用し、全コンシューマーが読み取り後に一度だけ`consume()`を実行

### ChannelBuffer (nodes/mod.rs)
`Arc<Mutex<AudioBuffer>>` - チャンネルごとの共有バッファ。

### NodeBuffers (nodes/mod.rs)
ノードのバッファ管理を統一する構造体。入出力ノードと中間ノードで共通のインターフェースを提供。
- `input_buffers`: 入力ピンごとのバッファ
- `output_buffers`: 出力ピンごとのバッファ
- `single_io()`: 1入力1出力ノード用（GainNode, FilterNode等）
- `multi_input(n)`: N入力1出力ノード用（AddNode, MultiplyNode等）
- `input_only(ch)`: 入力専用ノード用（AudioOutputNode）
- `output_only(ch)`: 出力専用ノード用（AudioInputNode）
- `resize_inputs()`/`resize_outputs()`: バッファ数の動的変更

### SpectrumDisplay (nodes/mod.rs)
IOノードのスペクトラム表示機能をカプセル化する構造体。
- `enabled`: 表示有効フラグ
- `spectrum`: スペクトラムデータ（FFTサイズ/2のf32配列）
- `analyzer`: スペクトラムアナライザー（オプション）
- `update_from_samples()`: サンプルデータからスペクトラムを更新
- `show_line()`: 折れ線グラフでスペクトラムを表示

### NodeUIContext (nodes/mod.rs)
UI描画時に必要なコンテキストを保持する構造体：
- `input_devices`: 入力デバイス名のリスト
- `output_devices`: 出力デバイス名のリスト
- `node_id`: ウィジェットの一意識別用ノードID

### ノードトレイト設計 (nodes/mod.rs)
ノードの機能は4つの独立したトレイトに分割されており、入力専用/出力専用/中間ノードを適切に表現できる：

#### NodeBase trait
ノードの基本情報を提供するコアトレイト：
- `node_type()`: ノードの型を返す（NodeType enum）
- `title()`: ノードのタイトル
- `as_any()`, `as_any_mut()`: ダウンキャスト用のAny参照を取得

#### AudioInputPort trait
オーディオ入力ポートを持つノード向けトレイト（デフォルト実装あり）：
- `input_count()`: 入力ピン数（デフォルト: 0）
- `input_pin_type()`: 入力ピンタイプ
- `input_pin_name()`: 入力ピン名
- `input_buffer()`: 指定入力ピンのバッファを取得

#### AudioOutputPort trait
オーディオ出力ポートを持つノード向けトレイト（デフォルト実装あり）：
- `output_count()`: 出力ピン数（デフォルト: 0）
- `output_pin_type()`: 出力ピンタイプ
- `output_pin_name()`: 出力ピン名
- `channel_buffer()`: 指定チャンネルの出力バッファを取得
- `channels()`, `set_channels()`: チャンネル数

#### NodeUI trait
ノードのUI描画機能を提供するトレイト：
- `is_active()`, `set_active()`: アクティブ状態
- `show_body()`: ノードボディのUI描画（NodeUIContextを受け取る）

#### NodeBehavior trait（スーパートレイト）
上記4トレイトを統合したスーパートレイト。ブランケット実装により自動的に付与される：
```rust
pub trait NodeBehavior: NodeBase + AudioInputPort + AudioOutputPort + NodeUI {}
impl<T: NodeBase + AudioInputPort + AudioOutputPort + NodeUI> NodeBehavior for T {}
```

このトレイト分割により：
- `AudioInputNode`: `AudioOutputPort`のみ実装（出力専用）
- `AudioOutputNode`: `AudioInputPort`のみ実装（入力専用）
- エフェクトノード: 両方のポートトレイトを実装（中間ノード）
- 将来的にMIDIポート用トレイトを追加可能

### NodeType enum (nodes/mod.rs)
ノードの型を識別するためのenum：
- `AudioInput`, `AudioOutput`, `Gain`, `Add`, `Multiply`, `Filter`
- `SpectrumAnalyzer`, `Compressor`, `PitchShift`, `GraphicEq`

ランタイムで型を判別し、`as_any()`と組み合わせて具体型にダウンキャストする際に使用。

### ヘルパー関数 (nodes/mod.rs)
コード重複を削減するための共通関数：
- `channel_name()`: チャンネルインデックスからチャンネル名を取得（L, R, C, LFE, SL, SR）
- `new_channel_buffer()`: 新しいチャンネルバッファを作成

### AudioNode (nodes/mod.rs)
`Box<dyn NodeBehavior>`の型エイリアス。動的ディスパッチによりノードを管理。

利用可能なノード型（各サブモジュールで定義）：
- `AudioInputNode` (io.rs): オーディオ入力デバイスノード（出力ピン = チャンネル数、スペクトラム表示統合）
- `AudioOutputNode` (io.rs): オーディオ出力デバイスノード（入力ピン = チャンネル数、スペクトラム表示統合）
- `GainNode` (effects.rs): ゲインエフェクトノード（1入力1出力、ゲインスライダー付き）
- `AddNode` (math.rs): 加算ノード（2入力1出力、A + B）
- `MultiplyNode` (math.rs): 乗算ノード（2入力1出力、A × B、リングモジュレーション用）
- `FilterNode` (effects.rs): フィルターノード（1入力1出力、Low/High/Band Pass、カットオフ周波数、Q値）
- `SpectrumAnalyzerNode` (analyzer.rs): スペクトラムアナライザー（1入力1出力、FFTでスペクトラム表示）
- `CompressorNode` (effects.rs): コンプレッサー（1入力1出力、Threshold、Ratio、Attack、Release、Makeup Gain）
- `PitchShiftNode` (effects.rs): ピッチシフター（1入力1出力、PSOLAアルゴリズム、-12〜+12半音）
- `GraphicEqNode` (effects.rs): グラフィックEQ（1入力1出力、FFTベースの周波数ゲイン調整、egui_plotによるカーブエディタUI、入力スペクトラム表示統合）

ファクトリ関数でノードを生成（nodes/mod.rsで定義）：
- `new_audio_input(device_name, channels)`, `new_audio_output(device_name, channels)`: チャンネル数を指定して生成
- `new_gain()`, `new_add()`, `new_multiply()`, `new_filter()`, `new_spectrum_analyzer()`
- `new_compressor()`, `new_pitch_shift()`, `new_graphic_eq()`

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
- 処理フロー（スナップショット方式）:
  1. **Phase 1 - スナップショット作成**: 全ソースバッファから`read()`でデータを読み取り、スナップショットを作成
     - 同じソースバッファを複数ノードが参照していても、データは一度だけ読み取る
     - `HashMap<バッファアドレス, (データ, 消費済みフラグ)>`で管理
  2. **Phase 2 - ノード処理**: スナップショットから入力バッファへコピーし、DSP処理を実行
     - PassThrough（出力ノードへのルーティング）はソースから直接出力バッファにコピー
  3. **Phase 3 - データ消費**: 使用したソースバッファから`consume()`でデータを削除（各バッファ一度だけ）
- スレッドセーフなDSP状態管理（`Arc<Mutex<>>`でUI/処理スレッド間共有）
- バッファ蓄積防止のため、利用可能なデータをできるだけ多く処理（最大8倍まで）

スナップショット方式の利点:
- 複数の出力ノードへの分岐時にデータが消失しない（従来の`read_and_consume()`では最初の消費者がデータを取り、後続が空になる問題があった）
- 各ソースバッファのデータは全コンシューマーが読み取った後に一度だけ消費される

### AudioGraphViewer (viewer.rs)
egui-snarlのSnarlViewerトレイトを実装。
- ノードのUI表示（各ノードの`show_body()`にデリゲート）
- 動的なピン数（チャンネル数に応じて変化）
- ピン接続のロジック（同じPinType同士のみ）
- コンテキストメニュー（ノード追加・削除）

新しいノードタイプを追加する際は：
1. 構造体を定義（`buffers: NodeBuffers`フィールドを含む）
2. 以下の4トレイトを実装：
   - `NodeBase`: `node_type()`, `title()`, `as_any()`, `as_any_mut()` （`impl_as_any!()`マクロ使用可）
   - `AudioInputPort`: 入力ポートがある場合は実装（`impl_input_port_nb!()`マクロ使用可）
   - `AudioOutputPort`: 出力ポートがある場合は実装（`impl_single_output_port_nb!()`マクロ使用可）
   - `NodeUI`: `is_active()`, `set_active()`, `show_body()`
3. `NodeType` enumにバリアントを追加
4. ファクトリ関数を追加（`new_xxx() -> AudioNode`）
5. `viewer.rs`の`show_graph_menu`を更新（ノード追加メニュー）
6. `project.rs`のシリアライズ/デシリアライズを更新
7. 必要に応じて`graph.rs`と`effect_processor.rs`を更新

エフェクトノード用のマクロ（NodeBuffers対応）：
- `impl_input_port_nb!(NodeType, ["Pin1", "Pin2", ...])`: 任意の入力ピン名でAudioInputPortを実装
- `impl_single_output_port_nb!(NodeType)`: 1出力ピン("Out")でAudioOutputPortを実装

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
3. `EffectProcessor`スレッドがスナップショット方式で全ノードを処理
   - Phase 1: 全ソースバッファからスナップショットを作成（各バッファ一度だけ）
   - Phase 2: スナップショットから入力バッファへコピー、DSP処理、出力バッファに書き込み
   - Phase 3: 使用したソースバッファからデータを消費（各バッファ一度だけ）
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

## アプリケーション設定 (config.rs)

アプリケーション全体の設定を管理。`directories`クレートを使用してOSごとの設定ディレクトリにJSON形式で保存。

### AppConfig
- `last_opened_file`: 最後に開いたファイルのパス
- アプリ起動時に設定を読み込み、最後に開いたファイルが存在すれば自動的に復元
- ファイルを開く/保存する際に設定を自動更新
- 新規プロジェクト作成時に設定をクリア

設定ファイルの保存場所:
- macOS: `~/Library/Application Support/Wavetangle/config.json`
- Linux: `~/.config/Wavetangle/config.json`
- Windows: `%APPDATA%\Wavetangle\config\config.json`

## 依存ライブラリ

- **eframe/egui**: GUIフレームワーク
- **egui-snarl**: ノードグラフエディタ（serde機能有効）
- **egui_plot**: プロット表示（スペクトラムアナライザー用）
- **cpal**: クロスプラットフォームオーディオI/O
- **parking_lot**: 高性能mutex実装
- **rustfft**: FFT実装（スペクトラムアナライザー用）
- **serde/serde_json**: シリアライズ・デシリアライズ
- **rfd**: ネイティブファイルダイアログ
- **directories**: OS固有の設定ディレクトリの取得

## 今後の拡張予定

- ファイル入出力ノード
- シグナルジェネレータノード（オシレーター）
- ディレイ・リバーブノード
