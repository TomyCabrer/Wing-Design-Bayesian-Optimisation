"""AeroBO V3 — the pipeline shell: mission → airfoil → wing → results.

A light CAE-desktop front end (object tree, properties, tabbed work area,
output log, status bar) over the SAME thin API the other shells use: it
holds zero physics and zero optimiser logic, builds an
``aerobo.api.RunConfig`` and calls ``aerobo.api.run``.

Launch::

    .venv/bin/python gui/nice_app_v3.py            # native window
    .venv/bin/python gui/nice_app_v3.py --browser  # browser on :8767

Importing this package (and every module under it) is side-effect free: no
server, no window, no nicegui and no aerobo import happens until a build
function runs.
"""
