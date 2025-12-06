//! パイプライン並列処理モジュール
//!
//! 各ノードを独立したスレッドで実行し、ロックフリーSPSCバッファで接続する

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;

/// SPSCバッファのデフォルト容量（サンプル数）
pub const DEFAULT_BUFFER_CAPACITY: usize = 8192;

/// ロックフリーSPSCプロデューサー
pub type SpscProducer = ringbuf::HeapProd<f32>;

/// ロックフリーSPSCコンシューマー
pub type SpscConsumer = ringbuf::HeapCons<f32>;

/// SPSCバッファのペアを作成
pub fn create_spsc_pair(capacity: usize) -> (SpscProducer, SpscConsumer) {
    let rb = HeapRb::<f32>::new(capacity);
    rb.split()
}

/// 処理ノードのトレイト
/// 独自スレッドでオーディオ処理を行うノードが実装する
pub trait ProcessingNode: Send + 'static {
    /// ノード名（デバッグ用）
    fn name(&self) -> &str;

    /// 1フレーム分の処理を実行
    /// input: 入力サンプル（None = 入力なし）
    /// output: 出力バッファ
    /// 戻り値: 出力されたサンプル数
    fn process(&mut self, input: Option<&[f32]>, output: &mut [f32]) -> usize;

    /// 処理のブロックサイズ（一度に処理するサンプル数）
    fn block_size(&self) -> usize {
        256
    }
}

/// ノードの処理スレッドを管理
pub struct NodeThread {
    /// スレッドハンドル
    handle: Option<JoinHandle<()>>,
    /// 停止フラグ
    running: Arc<AtomicBool>,
}

impl NodeThread {
    /// 新しい処理スレッドを開始
    pub fn spawn<N: ProcessingNode>(
        mut node: N,
        mut input: Option<SpscConsumer>,
        mut output: Option<SpscProducer>,
    ) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let handle = thread::spawn(move || {
            let block_size = node.block_size();
            let mut input_buf = vec![0.0f32; block_size];
            let mut output_buf = vec![0.0f32; block_size];

            while running_clone.load(Ordering::Relaxed) {
                // 入力からデータを読み取る
                let input_slice = if let Some(ref mut cons) = input {
                    let available = cons.occupied_len();
                    if available >= block_size {
                        let count = cons.pop_slice(&mut input_buf);
                        Some(&input_buf[..count])
                    } else {
                        // データが足りない場合は少し待つ
                        thread::yield_now();
                        continue;
                    }
                } else {
                    None
                };

                // 処理を実行
                let output_count = node.process(input_slice, &mut output_buf);

                // 出力にデータを書き込む
                if let Some(ref mut prod) = output {
                    if output_count > 0 {
                        // バッファに空きがあれば書き込む
                        let _written = prod.push_slice(&output_buf[..output_count]);
                    }
                }

                // 入力がない場合（ジェネレータノード等）は少し待つ
                if input.is_none() {
                    thread::yield_now();
                }
            }
        });

        Self {
            handle: Some(handle),
            running,
        }
    }

    /// スレッドを停止
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// スレッドが実行中か
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for NodeThread {
    fn drop(&mut self) {
        self.stop();
    }
}

/// パイプラインのビルダー
/// ノードを連結してパイプラインを構築する
pub struct PipelineBuilder {
    /// 現在の出力プロデューサー（次のノードの入力になる）
    current_output: Option<SpscConsumer>,
    /// 開始されたスレッド
    threads: Vec<NodeThread>,
}

impl PipelineBuilder {
    /// 新しいパイプラインビルダーを作成
    pub fn new() -> Self {
        Self {
            current_output: None,
            threads: Vec::new(),
        }
    }

    /// ソースノード（入力なし）を追加
    pub fn source<N: ProcessingNode>(mut self, node: N) -> Self {
        let (prod, cons) = create_spsc_pair(DEFAULT_BUFFER_CAPACITY);
        let thread = NodeThread::spawn(node, None, Some(prod));
        self.threads.push(thread);
        self.current_output = Some(cons);
        self
    }

    /// 処理ノードを追加（前のノードの出力を入力とする）
    pub fn then<N: ProcessingNode>(mut self, node: N) -> Self {
        let input = self.current_output.take();
        let (prod, cons) = create_spsc_pair(DEFAULT_BUFFER_CAPACITY);
        let thread = NodeThread::spawn(node, input, Some(prod));
        self.threads.push(thread);
        self.current_output = Some(cons);
        self
    }

    /// シンクノード（出力なし）を追加してパイプラインを完成
    pub fn sink<N: ProcessingNode>(mut self, node: N) -> Pipeline {
        let input = self.current_output.take();
        let thread = NodeThread::spawn(node, input, None);
        self.threads.push(thread);

        Pipeline {
            threads: self.threads,
        }
    }

    /// 最終出力を取得してパイプラインを完成
    pub fn build(self) -> (Pipeline, Option<SpscConsumer>) {
        let pipeline = Pipeline {
            threads: self.threads,
        };
        (pipeline, self.current_output)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 実行中のパイプライン
pub struct Pipeline {
    threads: Vec<NodeThread>,
}

impl Pipeline {
    /// すべてのスレッドを停止
    pub fn stop(&mut self) {
        for thread in &mut self.threads {
            thread.stop();
        }
    }

    /// パイプラインが実行中か
    pub fn is_running(&self) -> bool {
        self.threads.iter().any(|t| t.is_running())
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用のパススルーノード
    struct PassthroughNode;

    impl ProcessingNode for PassthroughNode {
        fn name(&self) -> &str {
            "Passthrough"
        }

        fn process(&mut self, input: Option<&[f32]>, output: &mut [f32]) -> usize {
            if let Some(input) = input {
                let count = input.len().min(output.len());
                output[..count].copy_from_slice(&input[..count]);
                count
            } else {
                0
            }
        }
    }

    /// テスト用のゲインノード
    struct GainNode {
        gain: f32,
    }

    impl ProcessingNode for GainNode {
        fn name(&self) -> &str {
            "Gain"
        }

        fn process(&mut self, input: Option<&[f32]>, output: &mut [f32]) -> usize {
            if let Some(input) = input {
                let count = input.len().min(output.len());
                for i in 0..count {
                    output[i] = input[i] * self.gain;
                }
                count
            } else {
                0
            }
        }
    }

    #[test]
    fn test_spsc_pair() {
        let (mut prod, mut cons) = create_spsc_pair(16);

        // 書き込み
        assert!(prod.try_push(1.0).is_ok());
        assert!(prod.try_push(2.0).is_ok());

        // 読み取り
        assert_eq!(cons.try_pop(), Some(1.0));
        assert_eq!(cons.try_pop(), Some(2.0));
        assert_eq!(cons.try_pop(), None);
    }

    #[test]
    fn test_push_slice() {
        let (mut prod, mut cons) = create_spsc_pair(16);

        let data = [1.0, 2.0, 3.0, 4.0];
        let written = prod.push_slice(&data);
        assert_eq!(written, 4);

        let mut output = [0.0; 4];
        let read = cons.pop_slice(&mut output);
        assert_eq!(read, 4);
        assert_eq!(output, data);
    }
}
