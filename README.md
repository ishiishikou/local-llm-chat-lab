# local-llm-chat-lab

ChatGPT の通常コンテナ内でローカルLLMを起動し、必要に応じて AGMSG 経由で ChatGPT とローカルLLMを会話させるための実験用リポジトリです。

モデル本体や大容量バイナリは Git にコミットしません。GitHub Actions の標準 GitHub-hosted runner で一時 artifact を生成し、GitHub connector 経由でコンテナへ搬入する構成を前提にしています。

## 現在の検証済み構成

- Model: `bartowski/Qwen_Qwen3.5-4B-GGUF`
- GGUF: `Qwen_Qwen3.5-4B-Q4_K_M.gguf`
- GGUF size: `3,013,027,808 bytes`
- GGUF SHA256: `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`
- llama.cpp: `b10809`
- llama.cpp archive SHA256: `5e34434ddc6d03cd1584f403201aff0d4bd1a5793a72ff7e286532dfd1e4b941`
- AGMSG: `1.2.3`（検証時 commit `948b1003868b9f29567132084346368200c36963`）
- CPU inference: 4 threads / GPU offload 0
- 4 GiB cgroup 環境で Q4_K_M + context 2048 の起動を確認済み

## 構成

```text
.github/workflows/qwen-model-bridge.yml    Qwen GGUF を Range 取得して artifact 化
.github/workflows/agmsg-runtime-bridge.yml AGMSG + sqlite3 runtime を artifact 化
.github/bridge-request.env                 Qwen part 指定用
.github/agmsg-bridge-trigger               AGMSG bridge のPRトリガー用
scripts/reassemble-qwen.sh                 part 0..6 の復元・SHA256検証・llama.cpp展開
scripts/install-agmsg-runtime.sh           AGMSG runtime artifact の導入
scripts/setup-agmsg-team.sh                chatgpt / qwen のAGMSGチーム作成
scripts/qwen-agmsg-once.sh                 Qwen inbox → llama.cpp → AGMSG reply を1回実行
AGENTS.md                                  AIエージェント向け運用ルール
```

## ChatGPT コンテナへのQwen搬入

GitHub connector からこのリポジトリを操作する場合、コンテナ内の `git clone` や外向きネットワーク成否で GitHub の可否を判断しません。GitHub connector 自体を確認して使用します。

1. `main` から `bridge/...` の一時ブランチを作る。
2. `.github/bridge-request.env` の `PART` を `0` にして draft PR を `main` 向けに作る。
3. `Qwen Model Bridge` が完了したら artifact `qwen4b-part-0` を GitHub connector の artifact download で `/mnt/data` へ搬入する。
4. 同じ一時ブランチで `PART=1` へ更新し、以後 `6` まで繰り返す。
5. 各runは前回の `qwen4b-part-*` artifact を削除してから次を生成するため、GitHub側へ3GB全体を同時保持しない。
6. 全7 ZIP が揃ったら `scripts/reassemble-qwen.sh` を実行する。
7. SHA256 が期待値と一致した場合だけ推論を開始する。

手動操作の場合は `Qwen Model Bridge` の `workflow_dispatch` から part 0〜6 を指定しても構いません。

## AGMSG の搬入

`AGMSG Runtime Bridge` は標準 `ubuntu-latest` 上で AGMSG ソースと対応する `sqlite3` / `libsqlite3.so.0` を1つの小さな artifact にまとめます。

ChatGPT connector から起動する場合は、一時 `bridge/...` ブランチで `.github/agmsg-bridge-trigger` の値を変更してPRを更新します。artifact を `/mnt/data/agmsg-container-bridge.zip` として搬入した後、次を実行します。

```bash
bash scripts/install-agmsg-runtime.sh /mnt/data/agmsg-container-bridge.zip
bash scripts/setup-agmsg-team.sh
```

## Qwen とAGMSGで1往復

Qwen宛てにメッセージを送った後、次を実行します。

```bash
bash scripts/qwen-agmsg-once.sh
```

既定では以下を使用します。

- Team: `qwen-chat`
- Local LLM agent: `qwen`
- Peer: `chatgpt`
- Context: 2048
- Threads: 4
- Reasoning: off

環境変数 `TEAM`, `QWEN_AGENT`, `PEER_AGENT`, `QWEN_MODEL`, `LLAMA_CLI`, `QWEN_CONTEXT`, `QWEN_THREADS`, `QWEN_MAX_TOKENS` で変更できます。

## 注意

- 有料CPU/GPUジョブや larger runner は使用しない。
- GitHub Actions は public repository の標準 GitHub-hosted runner のみを想定する。
- GGUF はGitリポジトリへコミットしない。
- 4 GiB RAM環境では 2048 context でも余裕が小さい。まず1024でロード確認し、2048へ上げる。
- Q4_K_M が実ロードで失敗した場合にのみ、より小さい量子化を検討する。
