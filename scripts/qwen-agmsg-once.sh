#!/usr/bin/env bash
set -euo pipefail

TEAM="${TEAM:-qwen-chat}"
QWEN_AGENT="${QWEN_AGENT:-qwen}"
PEER_AGENT="${PEER_AGENT:-chatgpt}"
SKILL="${AGMSG_SKILL:-$HOME/.agents/skills/agmsg}"
RUNTIME_BIN="${AGMSG_RUNTIME_BIN:-/mnt/data/agmsg-runtime/bin}"
MODEL="${QWEN_MODEL:-/mnt/data/qwen4b/Qwen_Qwen3.5-4B-Q4_K_M.gguf}"
CONTEXT="${QWEN_CONTEXT:-2048}"
THREADS="${QWEN_THREADS:-4}"
MAX_TOKENS="${QWEN_MAX_TOKENS:-240}"
TEMP="${QWEN_TEMP:-0.2}"
REASONING="${QWEN_REASONING:-off}"

if [ -d "$RUNTIME_BIN" ]; then
  export PATH="$RUNTIME_BIN:$PATH"
fi

if [ ! -x "$SKILL/scripts/inbox.sh" ] || [ ! -x "$SKILL/scripts/send.sh" ]; then
  echo "AGMSG is not installed at $SKILL" >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "Qwen model not found: $MODEL" >&2
  exit 1
fi

LLAMA="${LLAMA_CLI:-}"
if [ -z "$LLAMA" ]; then
  LLAMA="$(find /mnt/data/qwen4b -name llama-cli -type f | head -1)"
fi
if [ -z "$LLAMA" ] || [ ! -x "$LLAMA" ]; then
  echo "llama-cli not found" >&2
  exit 1
fi

INBOX="$(bash "$SKILL/scripts/inbox.sh" "$TEAM" "$QWEN_AGENT")"
if grep -q '^No new messages\.$' <<<"$INBOX"; then
  echo "$INBOX"
  exit 0
fi

# Collect unread messages sent by the configured peer.
PROMPT="$(printf '%s\n' "$INBOX" \
  | sed -n "s/^  \[[^]]*\] ${PEER_AGENT}: //p" \
  | sed 's/\\n/\n/g')"

if [ -z "$PROMPT" ]; then
  echo "No parseable message from $PEER_AGENT" >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

"$LLAMA" \
  --offline \
  -m "$MODEL" \
  -t "$THREADS" \
  -tb "$THREADS" \
  -c "$CONTEXT" \
  -n "$MAX_TOKENS" \
  --temp "$TEMP" \
  --single-turn \
  --simple-io \
  --no-display-prompt \
  --reasoning "$REASONING" \
  -p "$PROMPT" \
  >"$TMP" 2>/dev/null

# llama-cli simple-io output used in the verified container includes a `> `
# separator before the model text and a `[ Prompt: ... ]` stats line after it.
ANSWER="$(awk '
  BEGIN { p=0 }
  /^> / { p=1; next }
  p && /^\[ Prompt:/ { exit }
  p { print }
' "$TMP" | sed '/^[[:space:]]*$/d')"

# Fallback for builds whose simple-io formatting differs.
if [ -z "$ANSWER" ]; then
  ANSWER="$(sed '/^\[ Prompt:/,$d' "$TMP" | sed '/^[[:space:]]*$/d')"
fi

if [ -z "$ANSWER" ]; then
  echo "Qwen produced no parseable answer" >&2
  exit 1
fi

bash "$SKILL/scripts/send.sh" "$TEAM" "$QWEN_AGENT" "$PEER_AGENT" "$ANSWER"
printf '%s\n' "$ANSWER"
