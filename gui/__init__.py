"""The AeroBO shell — a thin presentation layer over ``aerobo.api``.

This package contains no physics and no optimiser logic: every run is
assembled as an ``aerobo.api.RunConfig`` and executed through
``aerobo.api.run``. ``gui/v4`` is the seven-stage shell that ``launch.py``
starts; ``gui/v3`` is the four-stage pipeline it extends; ``nice_app.py``,
``metrics.py`` and ``diagnose.py`` hold the helpers both share.

Launch::

    python launch.py
"""
