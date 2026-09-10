#!/usr/bin/env bash
set -euo pipefail

TEAM="${TEAM:-qwen-chat}"
CHATGPT_AGENT="${CHATGPT_AGENT:-chatgpt}"
QWEN_AGENT="${QWEN_AGENT:-qwen}"
PROJECT="${AGMSG_PROJECT:-/mnt/data}"
SKILL="${AGMSG_SKILL:-$HOME/.agents/skills/agmsg}"
RUNTIME_BIN="${AGMSG_RUNTIME_BIN:-/mnt/data/agmsg-runtime/bin}"

if [ -d "$RUNTIME_BIN" ]; then
  export PATH="$RUNTIME_BIN:$PATH"
fi

if [ ! -x "$SKILL/scripts/join.sh" ]; then
  echo "AGMSG is not installed at $SKILL" >&2
  exit 1
fi

mkdir -p "$PROJECT"

bash "$SKILL/scripts/join.sh" "$TEAM" "$CHATGPT_AGENT" codex "$PROJECT"
bash "$SKILL/scripts/join.sh" "$TEAM" "$QWEN_AGENT" codex "$PROJECT"

cat <<EOF
AGMSG team ready.
Team:     $TEAM
ChatGPT:  $CHATGPT_AGENT
Qwen:     $QWEN_AGENT
Project:  $PROJECT
EOF
