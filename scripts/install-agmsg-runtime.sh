#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ZIP="${1:-/mnt/data/agmsg-container-bridge.zip}"
RUNTIME_DIR="${AGMSG_RUNTIME_DIR:-/mnt/data/agmsg-runtime}"
BIN_DIR="$RUNTIME_DIR/bin"

if [ ! -f "$ARTIFACT_ZIP" ]; then
  echo "Missing AGMSG artifact ZIP: $ARTIFACT_ZIP" >&2
  exit 1
fi

rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR" "$BIN_DIR"
unzip -oq "$ARTIFACT_ZIP" -d "$RUNTIME_DIR"

(
  cd "$RUNTIME_DIR"
  sha256sum -c SHA256SUMS.txt
)

tar -xzf "$RUNTIME_DIR/agmsg-source.tar.gz" -C "$RUNTIME_DIR"
cp "$RUNTIME_DIR/sqlite3.real" "$BIN_DIR/sqlite3.real"
cp "$RUNTIME_DIR/libsqlite3.so.0" "$BIN_DIR/libsqlite3.so.0"
chmod +x "$BIN_DIR/sqlite3.real"

cat > "$BIN_DIR/sqlite3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$HERE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/sqlite3.real" "$@"
EOF
chmod +x "$BIN_DIR/sqlite3"

export PATH="$BIN_DIR:$PATH"

printf 'Bundled sqlite3: '
sqlite3 --version
printf 'AGMSG source version: '
cat "$RUNTIME_DIR/AGMSG_VERSION.txt"
printf 'AGMSG source commit: '
cat "$RUNTIME_DIR/AGMSG_COMMIT.txt"

bash "$RUNTIME_DIR/agmsg/install.sh" --cmd agmsg --agent-type codex

SKILL="$HOME/.agents/skills/agmsg"
if [ ! -f "$SKILL/VERSION" ]; then
  echo "AGMSG installation did not produce $SKILL/VERSION" >&2
  exit 1
fi

cat <<EOF
AGMSG install complete.
Skill: $SKILL
Version: $(cat "$SKILL/VERSION")

For this shell, prepend the bundled sqlite runtime when invoking AGMSG directly:
  export PATH="$BIN_DIR:\$PATH"
EOF
