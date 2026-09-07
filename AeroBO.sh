#!/usr/bin/env bash
# Start AeroBO on Linux:  ./AeroBO.sh
#
# Opens in your browser. The native-window backend (pywebview) needs GTK and
# WebKit system packages that this installer will not install for you, so the
# Linux app is a browser app by design.
cd "$(dirname "$0")" || exit 1
exec ./installer/bootstrap.sh "$@"
