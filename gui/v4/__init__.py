"""AeroBO V4 — V3, plus the two stages that turn a scored design into a
flyable one.

V4 IS V3. Not a fork of it: :mod:`gui.v4.session` re-exports every name
:mod:`gui.v3.session` defines and overrides a handful of them, and
:mod:`gui.v4.app` executes ``gui/v3/app.py``'s own source as a second module
object with its ``session`` global rebound. So the V3 shell keeps exactly
the four stages it has always had — open it and there is no Controls tab and
no Flight tab anywhere — while V4 gets every fix the V3 chrome ever receives
without a line of it being copied.

    1 Mission -> 2 Airfoil -> (2.5 Airfoil) -> 3 Wing -> 4 Results
                                                          |
                                          5 Controls  <---+
                                                |
                                          6 Flight  <-----+

...where there is something to fly. On the TRACK the pipeline ends at
4 Results: a car rear wing is bolted to a car, so it has no control surface
to cut and no free-flight degrees of freedom, and both stages are hidden and
locked (:func:`gui.v4.session.stage_visible`). An airfoil-only session ends
at 2 Airfoil for the same kind of reason — it has no vehicle at all.

Stage 5 cuts ailerons out of the planform the search produced, hinges the
stabiliser, and adds the VERTICAL surface the lattice never had — before it,
``Cn_beta`` was exactly zero for every configuration this package builds.
Stage 6 flies the result: six degrees of freedom on stage 5's derivative
deck, live in a three.js view, flown from the keyboard.

Launch::

    .venv/bin/python gui/nice_app_v4.py            # native window
    .venv/bin/python gui/nice_app_v4.py --browser  # browser on :8769

Importing this package is side-effect free, exactly as V3's is.
"""
