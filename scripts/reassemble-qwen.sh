#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-/mnt/data}"
DEST_DIR="${2:-/mnt/data/qwen4b}"
MODEL_NAME="Qwen_Qwen3.5-4B-Q4_K_M.gguf"
MODEL_SIZE=3013027808
MODEL_SHA256=13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983
CHUNK_SIZE=440401920
LAST_PART_SIZE=370616288
LLAMA_ARCHIVE=llama-b10809-bin-ubuntu-x64.tar.gz
LLAMA_SHA256=5e34434ddc6d03cd1584f403201aff0d4bd1a5793a72ff7e286532dfd1e4b941

mkdir -p "$DEST_DIR"

for i in 0 1 2 3 4 5 6; do
  zip="$SOURCE_DIR/qwen4b-part-${i}.zip"
  part="$DEST_DIR/qwen4b.part.${i}"

  if [ ! -f "$zip" ]; then
    echo "Missing artifact ZIP: $zip" >&2
    exit 1
  fi

  unzip -oq "$zip" "qwen4b.part.${i}" -d "$DEST_DIR"

  expected="$CHUNK_SIZE"
  if [ "$i" -eq 6 ]; then
    expected="$LAST_PART_SIZE"
  fi
  actual="$(stat -c%s "$part")"
  if [ "$actual" -ne "$expected" ]; then
    echo "Part $i size mismatch: expected=$expected actual=$actual" >&2
    exit 1
  fi

done

# llama.cpp runtime is bundled only with part 0.
unzip -oq "$SOURCE_DIR/qwen4b-part-0.zip" "$LLAMA_ARCHIVE" -d "$DEST_DIR"
echo "$LLAMA_SHA256  $DEST_DIR/$LLAMA_ARCHIVE" | sha256sum -c -

model_tmp="$DEST_DIR/${MODEL_NAME}.tmp"
model="$DEST_DIR/$MODEL_NAME"
rm -f "$model_tmp"
: > "$model_tmp"
for i in 0 1 2 3 4 5 6; do
  cat "$DEST_DIR/qwen4b.part.${i}" >> "$model_tmp"
done

actual_size="$(stat -c%s "$model_tmp")"
if [ "$actual_size" -ne "$MODEL_SIZE" ]; then
  echo "Combined model size mismatch: expected=$MODEL_SIZE actual=$actual_size" >&2
  exit 1
fi

actual_sha="$(sha256sum "$model_tmp" | awk '{print $1}')"
if [ "$actual_sha" != "$MODEL_SHA256" ]; then
  echo "Combined model SHA256 mismatch" >&2
  echo "expected: $MODEL_SHA256" >&2
  echo "actual:   $actual_sha" >&2
  echo "Inference is blocked until this is fixed." >&2
  exit 1
fi

mv -f "$model_tmp" "$model"

tar -xzf "$DEST_DIR/$LLAMA_ARCHIVE" -C "$DEST_DIR"
LLAMA_CLI="$(find "$DEST_DIR" -name llama-cli -type f | head -1)"
if [ -z "$LLAMA_CLI" ]; then
  echo "llama-cli was not found after extraction" >&2
  exit 1
fi
chmod +x "$LLAMA_CLI"

cat <<EOF
Qwen restore complete.
Model:      $model
Size:       $actual_size bytes
SHA256:     $actual_sha (matched)
llama-cli:  $LLAMA_CLI

Recommended first load test:
  "$LLAMA_CLI" --offline -m "$model" -t 4 -tb 4 -c 1024 -n 8 \\
    --single-turn --simple-io --no-display-prompt \\
    -p "日本の首都はどこですか？一言で答えてください。"
EOF
