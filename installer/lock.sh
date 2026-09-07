#!/usr/bin/env bash
# Recompile the app's two lock files from their .in files.
#
#   ./installer/lock.sh
#
# Run this after changing requirements-core.in or requirements-bo.in, and
# commit the .txt files with them: the .txt is what a user's machine installs,
# so an uncommitted recompile is a version nobody can reproduce.
#
# --universal resolves for every platform at once and writes environment
# markers, which is what lets one file serve macOS, Windows and Linux. The BO
# lock is compiled AGAINST the core lock so the packages they share (numpy,
# scipy, ...) resolve to the versions core already pinned rather than to a
# second, conflicting set.

set -euo pipefail
cd "$(dirname "$0")/.."

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
[ -x "$UV" ] || { echo "uv not found — see https://astral.sh/uv"; exit 1; }

"$UV" pip compile --universal --python-version 3.11 \
      installer/requirements-core.in -o installer/requirements-core.txt

"$UV" pip compile --universal --python-version 3.11 \
      --constraint installer/requirements-core.txt \
      installer/requirements-bo.in -o installer/requirements-bo.txt

# A lock that cannot be installed is worse than no lock. Resolve it, without
# installing, for each platform the app claims to support.
for p in aarch64-apple-darwin x86_64-apple-darwin \
         x86_64-unknown-linux-gnu x86_64-pc-windows-msvc; do
  "$UV" pip install --dry-run --python-platform "$p" --python-version 3.11 \
        -r installer/requirements-core.txt >/dev/null \
    && echo "  core resolves on $p" \
    || { echo "  CORE DOES NOT RESOLVE ON $p"; exit 1; }
done
echo "  (the BO lock is allowed to fail on x86_64-apple-darwin — no wheel exists)"
