#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <message>" >&2
  exit 1
fi

TEAM="${TEAM:-qwen-chat}"
QWEN_AGENT="${QWEN_AGENT:-qwen}"
PEER_AGENT="${PEER_AGENT:-chatgpt}"
SKILL="${AGMSG_SKILL:-$HOME/.agents/skills/agmsg}"
RUNTIME_BIN="${AGMSG_RUNTIME_BIN:-/mnt/data/agmsg-runtime/bin}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MESSAGE="$*"

if [ -d "$RUNTIME_BIN" ]; then
  export PATH="$RUNTIME_BIN:$PATH"
fi

bash "$SKILL/scripts/send.sh" "$TEAM" "$PEER_AGENT" "$QWEN_AGENT" "$MESSAGE"

# qwen-agmsg-once.sh prints the generated response and also stores it in the
# peer's AGMSG inbox.
bash "$SCRIPT_DIR/qwen-agmsg-once.sh"
