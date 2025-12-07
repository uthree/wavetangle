# wavetangle
Rust製リアルタイムオーディオ処理ツール  
パイプライン並列化によりノードごとに別スレッドで処理されるため、ノードの数が増えても安定して動作するのが特徴。  
仮想オーディオミキサーやボイスチェンジャーとしての利用を想定。 

## スクリーンショット
![スクリーンショット](./assets/images/wavetangle_screenshot.png)

## 機能
- オーディオのリアルタイム入出力
- イコライザーとフィルター
- ピッチシフト

## 実行方法
1. このリポジトリをクローン
```sh
git clone https://github.com/uthree/wavetangle
```

2. cargoでビルド、実行
```sh
cd wavetangle
cargo run --release
```
