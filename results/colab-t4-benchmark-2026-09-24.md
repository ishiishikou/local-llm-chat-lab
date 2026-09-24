# Colab T4 ベンチマーク（2026-09-24）

`local-llm-chat-lab` の既存検証済み構成と同じ Qwen3.5-4B GGUF を、Google Colab の Tesla T4 上で測定した。

## 条件

- Model: `bartowski/Qwen_Qwen3.5-4B-GGUF`
- GGUF: `Qwen_Qwen3.5-4B-Q4_K_M.gguf`
- Model size: `3,013,027,808 bytes`
- SHA256: `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`
- llama.cpp: `b10809` / commit `5266f24`
- GPU: Tesla T4
- GPU memory reported by `nvidia-smi`: 15,360 MiB
- CUDA driver runtime reported by `nvidia-smi`: CUDA 13.0
- CUDA compiler: 13.3
- benchmark: `llama-bench`
- prompt processing: `pp512`
- text generation: `tg64`
- repetitions: 3
- CPU comparison and GPU comparison are on the same Colab VM

## 結果

| 項目 | CPU寄り `ngl=0` | T4 GPU `ngl=99` | GPU / CPU |
| --- | ---: | ---: | ---: |
| `pp512` | 506.05 ± 7.38 tok/s | 1460.62 ± 58.30 tok/s | 約2.89倍 |
| `tg64` | 2.27 ± 0.02 tok/s | 52.78 ± 0.24 tok/s | 約23.25倍 |
| benchmark wall time | 92.803 s | 7.839 s | 約11.84倍高速 |
| peak VRAM | 1175 MiB | 3449 MiB | - |

## 所見

チャット用途に直接効く生成速度 `tg64` は、CPU寄りの 2.27 tok/s から T4 フルオフロード時の 52.78 tok/s まで上がった。Qwen3.5-4B Q4_K_M は、T4 では対話用途として十分実用的な生成速度になった。

フルオフロード時のピークVRAMは約3.4 GiBだったため、今回の 4B Q4_K_M と context 2048 程度には T4 のVRAM容量上かなり余裕がある。

## 初回実行で見つかった注意点

最初のベンチスクリプトでは、最後に `llama-cli` で実チャット生成を確認した際、Qwenのchat templateによりconversation modeが自動有効化され、1回答生成後も次のユーザー入力を待って終了しなかった。

性能値である `llama-bench` のCPU/GPU比較はその前に正常完了している。

再発防止として `scripts/colab-gpu-benchmark.py` では実チャット確認に以下を明示する。

- `--conversation`
- `--single-turn`
- `--simple-io`

これにより、事前指定したpromptへの1ターン生成後に終了する。

## 実行方法

`chatgpt-remote-runtime` の Colab CLI が利用できる環境では、例えば次のように実行する。

```bash
colab run --gpu T4 --timeout 1800 scripts/colab-gpu-benchmark.py
```

セッションを調査目的で残す必要がなければ `--keep` は付けない。付けた場合は作業終了後に必ず `colab stop` で停止する。

## 補足

この結果は Colab の特定時点の Tesla T4 VM 上での実測値であり、ColabのVM構成や負荷によって変動し得る。
