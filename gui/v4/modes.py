"""Re-export of :mod:`aerobo.modes`. The namer moved into the engine.

Everything that used to be written here now lives in ``src/aerobo/modes.py``,
unchanged, and this module binds the SAME OBJECTS under the same names.

WHY IT MOVED. :mod:`aerobo.handling` gates the optimiser on these modes, and a
constraint the search is run under cannot import from a shell. It was already
a problem before the gate: ``scripts/gate_ladder.py`` reached into the GUI
(``sys.path.insert(0, "gui"); from v4 import modes``) to measure the ladder
that this package publishes, so a study of the engine depended on a view.

WHY THE FILE STAYS. V4 reads ``modes`` in four places and a test asks for it
by this path (``from gui.v4 import modes``). Re-exporting costs nothing and
means no caller moves — which is also the guarantee: there is one definition,
so the shell and the engine cannot drift into two answers about one aeroplane,
which is the whole reason ``classify`` takes a ``Cn_beta`` in the first place.

Object identity is load-bearing and asserted:
``test_the_margin_is_the_engines_formula_not_the_shells`` requires
``gui.v4.modes.spiral_margin is aerobo.dynamics.spiral_margin``. Bind, never
wrap — an adapter here would pass that test by accident today and stop being
one formula tomorrow.
"""

from __future__ import annotations

from aerobo.modes import *                                   # noqa: F401,F403

# ...and the names ``import *`` will not carry: it skips anything leading with
# an underscore. Both are read from outside — ``scripts/gate_ladder.py`` asks
# ``md._dyn.weathercocks`` so that the script and the namer share ONE
# predicate about whether a design weathercocks.
from aerobo.modes import _ZERO, _dyn                         # noqa: F401
from aerobo.modes import __all__                             # noqa: F401
