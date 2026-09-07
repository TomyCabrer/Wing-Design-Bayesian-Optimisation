#!/usr/bin/env bash
# Double-click this to start AeroBO on a Mac.
#
# A .command file opens in Terminal with the HOME directory as its working
# directory, not this folder — hence the cd. Everything else is in
# installer/bootstrap.sh, which AeroBO.sh (Linux) runs too.
cd "$(dirname "$0")" || exit 1
exec ./installer/bootstrap.sh "$@"
