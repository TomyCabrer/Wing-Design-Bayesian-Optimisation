#!/usr/bin/env bash
# Get AeroBO onto this machine and start it. macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/TomyCabrer/AeroBO/main/installer/install.sh | bash
#
# Downloads the app into ~/AeroBO (override with AEROBO_DIR=...) and runs the
# launcher, which builds its own private Python environment there. Nothing is
# installed system-wide and nothing is written outside that folder; removing
# it removes everything.
#
# `git clone` is preferred over the zip so that later updates are `git pull`.

set -euo pipefail

REPO="${AEROBO_REPO:-https://github.com/TomyCabrer/AeroBO}"
DIR="${AEROBO_DIR:-$HOME/AeroBO}"
BRANCH="${AEROBO_BRANCH:-main}"

say() { printf '  %s\n' "$*"; }

printf '\n  AeroBO — install\n\n'

if [ -d "$DIR/.git" ]; then
  say "updating the copy already in $DIR"
  git -C "$DIR" pull --ff-only || say "could not update — starting the copy you have"
elif [ -d "$DIR" ] && [ -f "$DIR/launch.py" ]; then
  say "using the copy already in $DIR"
elif command -v git >/dev/null 2>&1; then
  say "downloading into $DIR"
  if ! git clone --depth 1 --branch "$BRANCH" "$REPO" "$DIR" 2>/dev/null; then
    printf '\n  Could not download it.\n\n'
    printf '  If the repository is still private, GitHub will refuse an anonymous\n'
    printf '  clone. Either ask for access and then run:\n\n'
    printf '      git clone %s "%s"\n\n' "$REPO" "$DIR"
    printf '  or download the zip from the repository page, unpack it, and\n'
    printf '  double-click AeroBO.command (macOS) or run ./AeroBO.sh (Linux).\n\n'
    exit 1
  fi
else
  say "downloading into $DIR (no git here — fetching the zip)"
  TMP="$(mktemp -d)"
  curl -fsSL "$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP/app.tgz" \
    || { printf '\n  Could not download it (the repository may still be private).\n\n'; exit 1; }
  mkdir -p "$DIR"
  tar -xzf "$TMP/app.tgz" -C "$DIR" --strip-components=1
  rm -rf "$TMP"
fi

chmod +x "$DIR/AeroBO.command" "$DIR/AeroBO.sh" "$DIR/installer/bootstrap.sh" 2>/dev/null || true
say "starting it — the first run installs what it needs and takes a few minutes"
printf '\n'
exec "$DIR/installer/bootstrap.sh" "$@"
