# Getting AeroBO

There is nothing to install first. Not Python either: the launcher fetches a
private copy of everything it needs into its own folder, and deleting that
folder removes every trace of it.

## macOS

1. Get the app: download the zip from the
   [latest release](https://github.com/TomyCabrer/Wing-Design-Bayesian-Optimisation/releases/latest)
   and unzip it, or `git clone https://github.com/TomyCabrer/Wing-Design-Bayesian-Optimisation.git`.
2. Double-click **`AeroBO.command`**.

The first run takes a few minutes and about 880 MB: it fetches `uv`, builds a
Python 3.11 environment and installs the solver stack. Every run after that
starts in about a tenth of a second.

macOS may say *"AeroBO.command cannot be opened because it is from an
unidentified developer"* — this is Gatekeeper, and it applies to any script
that was not downloaded from the App Store. Right-click the file, choose
**Open**, then **Open** again. You only have to do it once.

If double-clicking does nothing at all, the unzip dropped the file's
executable bit. Open Terminal in the folder and run this once:

```bash
chmod +x AeroBO.command AeroBO.sh installer/*.sh
```

## Windows

1. Get the app: download the zip from the
   [latest release](https://github.com/TomyCabrer/Wing-Design-Bayesian-Optimisation/releases/latest)
   and unzip it, or `git clone https://github.com/TomyCabrer/Wing-Design-Bayesian-Optimisation.git`.
2. Double-click **`AeroBO.bat`**.

SmartScreen may warn about an unrecognised app: **More info ▸ Run anyway**.

## Linux

Get the app the same way — the release zip, or a clone — then:

```bash
./AeroBO.sh
```

Opens in your browser. The native-window backend needs GTK and WebKit system
packages that this installer will not install for you, so the Linux app is a
browser app by design.

## One line, if you prefer

```bash
curl -fsSL https://raw.githubusercontent.com/TomyCabrer/Wing-Design-Bayesian-Optimisation/main/installer/install.sh | bash
```

```powershell
irm https://raw.githubusercontent.com/TomyCabrer/Wing-Design-Bayesian-Optimisation/main/installer/install.ps1 | iex
```

Both put the app in `~/AeroBO` and start it. Running the same line again
later updates the copy you have.

---

## What it will tell you at startup

The launcher prints a three-line report before the window opens. Two of those
lines can come back short, and neither is a broken install.

### "no PyTorch on this machine"

Bayesian optimisation and the adjoint solver are off. **Everything else
runs** — the genetic algorithm, SLSQP, the penalty method, Sobol/DOE and the
grid search, on constrained problems as well as unconstrained ones — and the
strategy list on the wing stage drops what it cannot run and says why.

This is not something you did wrong. PyTorch has published no macOS x86_64
wheel since 2.2.2, and its Apple-silicon wheel is tagged `macosx_14_0`, so:

| machine | Bayesian optimisation |
|---|---|
| Apple silicon, macOS 14+ | yes |
| Apple silicon, macOS 13 or older | no |
| Intel Mac | no |
| Windows / Linux, x86-64 | yes |

### "no XFOIL found"

Designing an aerofoil *section* drives the real XFOIL binary. That is stage 2,
and it is also any wing or tail family whose name says `(XFOIL)` or
`(coupled)` — those solve the section inside the wing search rather than
reading one off the shelf.

Every other family needs nothing installed: they read the pre-computed polar
tables in `data/airfoils`, and so do the results, controls and flight stages.

| | |
|---|---|
| macOS | `brew install xfoil` |
| Linux | `sudo apt install xfoil` |
| Windows | download `xfoil.exe` from [web.mit.edu/drela/Public/web/xfoil](https://web.mit.edu/drela/Public/web/xfoil/) and put it in the app's `bin/` folder |

You can also point the app at a copy anywhere: set `AEROBO_XFOIL_BIN` to its
full path, or drop the binary in `bin/` inside the app folder, which the
launcher checks before `PATH`.

## Running it by hand

```bash
python launch.py               # native window where there is one
python launch.py --browser     # always a browser tab
python launch.py --port 9000   # busy ports are skipped automatically
python launch.py --v3          # the four-stage pipeline, without controls and flight
python launch.py --check       # print the startup report and exit
```

`./installer/bootstrap.sh --reinstall` rebuilds the environment from the lock
files if it ever gets into a strange state.

## What gets put where

Everything, without exception, inside the app folder:

| | |
|---|---|
| `.venv-app/` | the Python environment — about 850 MB |
| `.aerobo-tools/` | `uv` (35 MB), plus any Python it had to download |
| `results/` | XFOIL and evaluation caches and your saved runs, written as you use it |

With the app's own ~40 MB of source and data, a working install is about
1 GB.

Nothing outside that folder is touched and neither is your shell profile —
`uv`'s wheel cache, its receipt and its Python downloads are all redirected
into `.aerobo-tools/`, which is not where they would go by default. So
deleting the folder is the entire uninstall.

## If you are here to work on the code, not to use it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,bo]"
.venv/bin/pytest
```

`installer/requirements-core.in` explains the app's dependency set and how to
recompile the locks with `installer/lock.sh`.
