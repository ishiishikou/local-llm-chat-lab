# AGENTS.md

このリポジトリは、ChatGPT通常コンテナへローカルLLMとAGMSG環境を安全に再構築するための実験用リポジトリです。

## 必須運用ルール

- GitHub操作が必要な場合、コンテナ内の `git clone` や外向き通信結果でGitHub可否を判断しない。必ずGitHub connector自体を確認する。
- 有料サービスを使用しない。Hugging Face Jobs、有料GPU/CPUジョブ、GitHub larger runnerなどを起動しない。
- GitHub Actionsは、このpublic repositoryの標準GitHub-hosted `ubuntu-latest` runnerのみを使用する。
- GGUF、llama.cpp archive、AGMSG runtimeなどの大容量/生成物はGitへコミットしない。Actions artifact経由で一時搬送する。
- モデル搬入作業では `main` を都度書き換えない。`main` から `bridge/...` 一時ブランチを作り、draft PRでbridge workflowを起動し、PRはmergeしない。
- artifactをコンテナへ取得したら、次のpart生成前にGitHub側の前artifactを削除する。workflowにもcleanupを実装している。
- 作業終了時は一時artifactを削除し、一時PRをcloseする。一時ブランチも不要なら削除する。

## Qwen3.5-4B 検証済み値

- repository: `bartowski/Qwen_Qwen3.5-4B-GGUF`
- file: `Qwen_Qwen3.5-4B-Q4_K_M.gguf`
- size: `3013027808`
- SHA256: `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`
- chunk size: `440401920` bytes
- parts: `0..6`

## llama.cpp 検証済み値

- release: `b10809`
- archive: `llama-b10809-bin-ubuntu-x64.tar.gz`
- SHA256: `5e34434ddc6d03cd1584f403201aff0d4bd1a5793a72ff7e286532dfd1e4b941`

## 推論時の初期設定

4 GiB cgroup / swapなし / CPU-only環境を想定する。

1. Q4_K_Mを必ず最初に実ロードする。
2. `threads=4`, `context=1024`, `GPU offload=0` でロード確認する。
3. 成功後、必要ならcontext 2048へ上げる。
4. 4096は4 GiB環境ではOOMリスクが高いため、明確な必要性がなければ試さない。
5. Q4_K_Mが実際にロード失敗した場合にのみ、IQ4_XS等の小さい量子化を検討する。

## AGMSG

検証時はAGMSG `1.2.3`、commit `948b1003868b9f29567132084346368200c36963` を使用した。

既定team:

- team: `qwen-chat`
- ChatGPT identity: `chatgpt`
- Qwen identity: `qwen`

QwenはAGMSGのnative CLI agentではないため、`scripts/qwen-agmsg-once.sh` が adapter として inbox → llama.cpp → send を担当する。

## SHA256ゲート

復元後のGGUF SHA256が期待値と一致しない限り、モデルを起動しない。Range境界、part数、連結順序を確認してからやり直す。
