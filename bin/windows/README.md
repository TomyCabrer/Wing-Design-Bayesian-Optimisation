# XFOIL for Windows

`xfoil.exe` here is Mark Drela's own Windows build of XFOIL 6.99 (GPL), taken
unchanged from `XFOIL6.99.zip` on <https://web.mit.edu/drela/Public/web/xfoil/>
(the zip also holds the source, `Xfoil699src.zip`).

| file | SHA-256 |
|---|---|
| `XFOIL6.99.zip` as downloaded | `e13e8fe5cc38d8ac2626e9d3b17643bdcfaa63791619f042afdaa7cd103bcb08` |
| `xfoil.exe` (1,002,125 bytes, dated 2013-05-22 in the zip) | `c17342f84ae260c2b11a74cd0e2fb8189a5f8954c6bb7a8467a0f27055c7faea` |

It is a 32-bit console program, which 64-bit Windows and Windows on ARM both
run. It needs no installer and no other files. The app drives it through its
standard input with plotting switched off (`PLOP G`) as its first command, so
it should never open a plot window.

**Not yet run on Windows by this project.** It is not the compiler build
behind the published results, so its numbers will be close to them, not
identical (see `../macos-x86_64/README.md` for how close a rebuild lands).

**Downloaded as a zip:** Windows marks downloaded files with a
`Zone.Identifier` stream; the solver removes it from this file before first
use.
