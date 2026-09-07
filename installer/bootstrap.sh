#!/usr/bin/env bash
# Build (once) and then start the AeroBO app on macOS and Linux.
#
# AeroBO.command and AeroBO.sh in the app's top folder both call this. It is
# the whole install: there is nothing for the user to have installed first,
# not even Python.
#
#   1. find or fetch `uv` (Astral's installer, into the app folder — never
#      into the user's system, so deleting the app folder deletes all of it)
#   2. make .venv-app with a managed CPython 3.11
#   3. install installer/requirements-core.txt  — must succeed
#   4. install installer/requirements-bo.txt    — best effort, see below
#   5. run launch.py
#
# Step 4 is allowed to fail. PyTorch publishes no macOS x86_64 wheel and its
# arm64 wheel is tagged macosx_14_0, so an Intel Mac and a Mac on Ventura or
# older cannot get one. That costs the Bayesian optimiser and nothing else —
# api.py imports optimize.bo inside the function that runs it — so the app
# still opens and launch.py's preflight says out loud what is missing.
#
# Re-running is cheap: a stamp file records which locks the venv was built
# from, and the install steps are skipped when they have not changed. Pass
# --reinstall to force them anyway.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

VENV="$APP_DIR/.venv-app"
UV_DIR="$APP_DIR/.aerobo-tools"
CORE="$APP_DIR/installer/requirements-core.txt"
BO="$APP_DIR/installer/requirements-bo.txt"
STAMP="$VENV/.aerobo-stamp"
PY_VERSION="3.11"

# EVERYTHING uv writes goes inside the app folder. Left to itself uv keeps a
# wheel cache in ~/.cache/uv, a receipt in ~/.config/uv and any Python it has
# to download in ~/.local/share/uv — measured at 872 MB for one install of
# this app, outside the folder the user thinks is the app and invisible when
# they delete it. Pointed here instead, "delete the folder" is the whole
# uninstall. The installs below also pass --no-cache, so the wheel cache is
# never written at all rather than written somewhere tidier: it is a download
# accelerator for a step that happens once.
export UV_CACHE_DIR="$UV_DIR/cache"
export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export XDG_CONFIG_HOME="$UV_DIR/config"
export XDG_DATA_HOME="$UV_DIR/share"

REINSTALL=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--reinstall" ]; then REINSTALL=1; else ARGS+=("$a"); fi
done

say() { printf '  %s\n' "$*"; }
die() {
  printf '\n  AeroBO could not start: %s\n\n' "$*" >&2
  # A double-clicked .command closes its window the instant this exits, taking
  # the reason with it, so hold it open when there is a terminal to hold.
  if [ -t 0 ]; then read -r -p "  Press return to close. " _ || true; fi
  exit 1
}

printf '\n  AeroBO\n\n'

# ---------------------------------------------------------------- uv --------
UV=""
for cand in "$UV_DIR/uv" "$HOME/.local/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then UV="$cand"; break; fi
done
if [ -z "$UV" ]; then
  say "first run — fetching the installer (uv, ~35 MB)"
  command -v curl >/dev/null 2>&1 || die "curl is not installed."
  mkdir -p "$UV_DIR"
  # UV_INSTALL_DIR keeps it inside the app folder; NO_MODIFY_PATH keeps it out
  # of the user's shell profile. Nothing here escapes this directory.
  if ! curl -LsSf https://astral.sh/uv/install.sh \
       | env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; then
    die "could not download uv. Check the network, then try again."
  fi
  UV="$UV_DIR/uv"
  [ -x "$UV" ] || die "uv did not land in $UV_DIR."
fi

# ------------------------------------------------------------- venv ---------
LOCK_ID="$(cat "$CORE" "$BO" | shasum -a 256 2>/dev/null | cut -c1-16 || cksum "$CORE" "$BO")"
if [ ! -x "$VENV/bin/python" ]; then
  say "building the environment (Python $PY_VERSION)"
  "$UV" venv --python "$PY_VERSION" "$VENV" >/dev/null \
    || die "could not create the Python environment."
  REINSTALL=1
fi
if [ "$REINSTALL" = "1" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$LOCK_ID" ]; then
  say "installing what the app needs (a few hundred MB, once)"
  VIRTUAL_ENV="$VENV" "$UV" pip install --no-cache --python "$VENV/bin/python" -r "$CORE" \
    || die "the core install failed. The message above says why."
  if ! VIRTUAL_ENV="$VENV" "$UV" pip install --no-cache --python "$VENV/bin/python" -r "$BO" 2>/dev/null; then
    say "note: no PyTorch for this machine — Bayesian optimisation will be off."
    say "      Everything else, including the GA and SLSQP searches, still runs."
  fi
  printf '%s' "$LOCK_ID" > "$STAMP"
fi

# -------------------------------------------------------------- run ---------
exec "$VENV/bin/python" "$APP_DIR/launch.py" ${ARGS+"${ARGS[@]}"}
