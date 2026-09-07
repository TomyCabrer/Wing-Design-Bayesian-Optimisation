"""Bridge: a scored design report -> a flyable lattice, deck and aircraft.

STATED IS NOT FLOWN. The solver builds its :class:`vlm.VLM` inside
``evaluate`` and throws it away, so anything downstream that wants to FLY a
stored design has to rebuild one — and a rebuild is exactly where a shell
starts quietly flying a different aeroplane from the one it scored.

So this module rebuilds, and then CHECKS ITSELF. :class:`Fidelity` re-derives
the lift-curve slope, the neutral point and the static margin from the
rebuilt lattice and compares them with the numbers the report recorded when
the design was actually evaluated. The comparison travels with the model and
is meant to be shown, not asserted away: a 3 % disagreement in ``CL_alpha`` is
something a user should see next to the simulation, not something this module
should hide behind a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import (drag as _drag, dynamics as dyn, fin as _fin,
               polar as _polar, sixdof as sd)
from .geometry import Wing
from .vlm import VLM, SecondWing, TailSurface, VerticalSurface

__all__ = ["DEFAULT_V_MS", "Fidelity", "FlightModel",
           "SPIRAL_DIHEDRAL_MAX_DEG", "SpiralFix",
           "build_flight_model", "dihedral_for_spiral",
           "ControlsSpec", "QUARTER"]

#: where a surface's bound vortex sits on its own chord. The lattice
#: places every surface by this line and the shell asks for the leading
#: edge, so the two are one multiplication apart and the constant is
#: named rather than written 0.25 in four places.
QUARTER = 0.25

#: the speed a rebuild flies when NOTHING states one — not a design speed
#: and not a default anybody chose, only the last resort for a report that
#: carries no flight point at all. Every family that states its own
#: (``breakdown["V"]``) is flown at its own instead, and the assumption note
#: says which happened.
DEFAULT_V_MS = 45.0


@dataclass(frozen=True)
class Fidelity:
    """Rebuilt versus flown, for the three numbers that decide the dynamics."""

    rows: tuple = ()               # (name, flown, rebuilt, rel_error)

    @property
    def worst(self) -> float:
        return max((abs(r[3]) for r in self.rows if r[3] is not None),
                   default=0.0)

    @property
    def ok(self) -> bool:
        """Under 2 % on every comparable row. NOT a gate — a label."""
        return self.worst < 0.02

    def as_text(self) -> str:
        if not self.rows:
            return "nothing in the report to check the rebuild against"
        parts = []
        for name, flown, got, err in self.rows:
            if err is None:
                parts.append(f"{name}: rebuilt {got:.4g} (not in the report)")
            else:
                parts.append(f"{name}: flown {flown:.4g}, rebuilt {got:.4g} "
                             f"({100 * err:+.2f} %)")
        return "; ".join(parts)


@dataclass
class ControlsSpec:
    """Stage 5's answers, in engineering units rather than shell dicts."""

    aileron: bool = True
    aileron_span: tuple = (0.60, 0.98)
    aileron_chord_frac: float = 0.25
    flap: bool = False
    flap_span: tuple = (0.05, 0.55)
    flap_chord_frac: float = 0.30
    elevator: bool = True
    elevator_chord_frac: float = 0.40
    fin: bool = True
    fin_height_m: float | None = None      # None -> sized from the span
    fin_chord_m: float | None = None
    #: THE FIN'S LEADING EDGE [m], not its quarter chord.
    #:
    #: The lattice places a :class:`vlm.VerticalSurface` by the QUARTER-CHORD
    #: line, because that is the arm that makes ``Cn_beta``. That is the
    #: right internal coordinate and the wrong thing to ask a user for: a
    #: quarter chord is a station you cannot see on a drawing, it moves when
    #: the chord changes even though the surface has not been moved, and
    #: nothing else in this package is stated that way. So the ANSWER is the
    #: leading edge and the conversion happens once, HERE, after the chord
    #: default has resolved — which the shell could not do, because a blank
    #: chord means a default this function owns.
    #:
    #: None -> the leading edge that puts the quarter chord at the tail
    #: station, which is exactly where a blank used to put it.
    fin_x_le_m: float | None = None
    fin_ventral: bool = False
    #: IS THERE A RUDDER AT ALL. Separate from :attr:`fin`, because a
    #: vertical surface and a hinge in it are two different things: a
    #: hydrofoil's vertical IS the mast — a strut the run sizes, charges drag
    #: on and reports — and nothing on it is hinged. Before this, any design
    #: with a vertical got a rudder column it was never asked about, and
    #: ``rudder_chord_frac=0.0`` gives a column of ZEROS rather than no
    #: column, which is a control the deck says exists and the aircraft has
    #: not got. True by default, so every aircraft build is unchanged.
    rudder: bool = True
    rudder_chord_frac: float = 0.40

    #: WHERE THE THRUST VECTOR STARTS, and it is a stage-5 question.
    #:
    #: It used to be asked in stage 6, beside the flight condition, which put
    #: a property of the AIRFRAME among the levers of a particular flight.
    #: It belongs here with the other "what is this thing flown with"
    #: answers, and it is asked as a pair rather than as one number, because
    #: the interesting case is not a value:
    #:
    #: ``thrust_through_cg`` True is thrust through the CG — no pitching
    #: moment at all, which is the honest default for a design whose
    #: propulsion this package does not model. False turns
    #: ``thrust_z_below_cg_m`` on: metres BELOW the CG, positive, and the
    #: throttle becomes a pitch input (:class:`sixdof.Propulsion`).
    #:
    #: There is no x or y offset to answer. A thrust line ahead of or behind
    #: the CG makes no moment (``r x F`` is zero when both lie along x), and
    #: a y offset is asymmetric thrust, which needs a second engine to be a
    #: question at all. :class:`sixdof.Propulsion` says so in one place.
    thrust_through_cg: bool = True
    thrust_z_below_cg_m: float = 0.0

    @property
    def thrust_z_offset_m(self) -> float:
        """The offset :class:`sixdof.Propulsion` is built with — the whole
        rule, once, so the shell cannot hold half of it."""
        return (0.0 if self.thrust_through_cg
                else float(self.thrust_z_below_cg_m or 0.0))


@dataclass
class FlightModel:
    model: VLM
    deck: object
    aircraft: sd.Aircraft
    fidelity: Fidelity
    #: what had to be assumed because the report does not carry it
    assumptions: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    #: the state the deck is LINEARISED about [rad]. Cl_r and Cn_p are
    #: proportional to the lift at this state, so it is part of the answer.
    alpha_trim: float = 0.0
    i_t_trim: float = 0.0


#: how many chord-law coefficients the rebuild is allowed to fit. Three is
#: the order the published families search over; more would fit the
#: cosine-station noise of the report's own sampling rather than the law.
_CHORD_FIT_ORDER = 3


def _planform_from_report(geom: dict, name: str = "wing",
                          b_hint: float | None = None,
                          S_hint: float | None = None
                          ) -> tuple[float, float, float, tuple, float]:
    """``(b, S, taper, chord_coeffs, rms)`` — the planform the solver BUILT.

    The report writes the wing out as its own ``(y, chord)`` arrays, which
    ARE the planform: whatever chord law, free planform or inverse taper the
    search produced is in those numbers. Reading only the two end chords and
    calling the result a straight taper is how a design gets scored as one
    aeroplane and flown as another.

    Two things this fixes and one it measures.

    * **The taper is no longer clipped to 1.** It was ``clip(tip/root, 1e-3,
      1.0)``, so a planform whose tip chord exceeds its root — which the free
      planform and the "ends" chord law can both produce — was silently
      turned into a constant chord.
    * **The tip chord is read at the widest WING station**, not at the widest
      station of any kind. On a wing with a canted tip device the outermost
      strip belongs to the device, so its chord was being read as the wing's
      tip chord.
    * **The chord law is FITTED**, over the same polynomial basis
      :class:`geometry.Wing` uses, and kept only if it actually reduces the
      residual. What is returned with it is the RMS disagreement between the
      rebuilt chord distribution and the reported one, as a fraction of the
      mean chord — a number :class:`Fidelity` carries, so a planform this
      function cannot reproduce is reported beside the simulation instead of
      being flown quietly.
    """
    from scipy.optimize import least_squares

    from .geometry import Wing

    taper, coeffs, rms = 1.0, (), 0.0
    surfs = geom.get("surfaces") or []
    named = next((s for s in surfs if s.get("name") == name), None)
    src = named if named is not None else geom
    # THE HINTS WIN. A tandem's two wings are both under ``surfaces`` and the
    # report's top-level ``b``/``S`` are the PAIR's, so the caller states each
    # surface's own from the breakdown, which carries them exactly. Without a
    # hint the resolution is unchanged: the surface's own scalars, then the
    # report's, then — last — a span derived from the chord array, which the
    # cosine stations make short by a fraction of a per cent.
    b = float(b_hint or src.get("b") or geom.get("b") or 0.0)
    S = float(S_hint or src.get("S") or geom.get("S") or 0.0)
    chord = np.asarray(src.get("chord") or [], dtype=float)
    y = np.asarray(src.get("y") or [], dtype=float)
    if chord.size >= 2 and y.size == chord.size:
        if not b:
            b = 2.0 * float(np.abs(y).max())
        if not S:
            S = float(np.trapezoid(chord, y))
    elif chord.size >= 2 and not b:
        b = 0.0
    if not (b > 0 and S > 0):
        raise ValueError(
            "this report carries no planform (no span/area and no spanwise "
            "chord array), so there is nothing to fly")
    if not (chord.size >= 2 and y.size == chord.size):
        return b, S, taper, coeffs, 0.0

    eta = np.abs(2.0 * y / b)
    keep = eta <= 1.0 + 1e-9              # a tip device is not the wing
    if keep.sum() < 2:
        keep = np.ones_like(eta, dtype=bool)
    ey, ec = eta[keep], chord[keep]
    # THE TAPER IS AN EXTRAPOLATION, NOT A RATIO OF TWO SAMPLES. The report
    # writes its chord at the lifting line's COSINE stations, which never
    # reach either end: measured on the shipped tail design they run from
    # eta = 0.0262 to 0.9997, so ``chord[tip]/chord[root]`` is the ratio of
    # two interior chords and misses the true taper by enough to leave a
    # 0.36 % residual on a planform that is a straight taper to 2.6e-16.
    # The trapezoid's chord is exactly affine in eta, so a least-squares
    # line through the stations gives both ends without inventing anything.
    if ey.size >= 2 and float(np.ptp(ey)) > 1e-9:
        slope, c0 = np.polyfit(ey, ec, 1)
        root, tip = float(c0), float(c0 + slope)
    else:
        root = tip = float(ec[0])
    if root > 0:
        taper = float(max(tip / root, 1e-3))

    def _rms(w) -> float:
        return float(np.sqrt(np.mean((w.chord(y[keep]) - ec) ** 2))
                     / max(float(np.mean(ec)), 1e-12))

    try:
        base = Wing(b=b, S=S, taper=taper)
    except ValueError:
        return b, S, taper, coeffs, 0.0
    rms = _rms(base)

    # ...and only then the law, and only where a straight taper genuinely
    # is not the planform. Measured on the shipped tail design, whose wing
    # IS a plain trapezoid: the extrapolated taper comes out at 0.600000,
    # which is the design variable, and reproduces the reported chord array
    # to 1.6e-16 of the mean chord — bit-exact. Nothing is fitted and no
    # coefficients are invented. (The two sampled end chords gave 0.6065 and
    # a 3.6e-3 residual.) Below this gate a fit would be fitting arithmetic.
    if rms > 1e-3:
        def _resid(k):
            try:
                return Wing(b=b, S=S, taper=taper,
                            chord_coeffs=tuple(k)).chord(y[keep]) - ec
            except ValueError:
                return np.full(ec.shape, 1e3)
        try:
            fit = least_squares(_resid, np.zeros(_CHORD_FIT_ORDER),
                                bounds=(-2.0, 2.0), xtol=1e-10, max_nfev=200)
            cand = Wing(b=b, S=S, taper=taper, chord_coeffs=tuple(fit.x))
            if _rms(cand) < 0.5 * rms:
                coeffs, rms = tuple(float(v) for v in fit.x), _rms(cand)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return b, S, taper, coeffs, rms


def build_flight_model(report: dict, spec: ControlsSpec | None = None,
                       *, V: float | None = None, rho: float | None = None,
                       mass_kg: float | None = None,
                       x_cg_m: float | None = None,
                       CD0: float | None = None,
                       oswald_e: float | None = None,
                       inertia: sd.Inertia | None = None,
                       body_length_m: float | None = None) -> FlightModel:
    """Rebuild a flyable model from ``api.design_report``'s output."""
    spec = spec or ControlsSpec()
    geom = report.get("geometry") or {}
    bd = report.get("breakdown") or {}
    tail = geom.get("tail") or {}
    notes: list[str] = []

    # ---- THE POINT THIS IS FLOWN AT, and it is the DESIGN'S unless a caller
    # states otherwise. It used to default to 45 m/s outright, so a design
    # scored at 14.6 was rebuilt at 9.5x the dynamic pressure and — carrying
    # no weight in its report — had a mass BACK-SOLVED there: 632 kg against
    # the 66.6 the design flies. The static signs do not move with that, and
    # every rate derivative, every modal time constant and the trim ATTITUDE
    # (which the spiral criterion is not free of — dynamics.spiral_margin)
    # do. The note says which point was used, on every build, because a
    # rebuild at a speed nobody chose is exactly the failure that hid here.
    if V is None:
        V = bd.get("V")
        if V:
            V = float(V)
            notes.append(f"flown at the design's own point, V = {V:.4g} m/s, "
                         f"as its report states it.")
        else:
            V = DEFAULT_V_MS
            notes.append(
                f"the report states no speed, so V = {V:.4g} m/s was ASSUMED "
                f"— the design was scored somewhere else, and every rate "
                f"derivative and modal time below is at this speed, not at "
                f"the design's.")
    else:
        V = float(V)
        notes.append(f"flown at V = {V:.4g} m/s, as the caller asked.")
    if rho is None:
        rho = float(bd.get("rho") or 1.225)
    else:
        rho = float(rho)

    # ---- IS THIS A TANDEM? Two surfaces named front/rear, staggered.
    # A tandem's report carries no ``tail`` block at all, so this function
    # used to raise "this report carries no planform" on one and V4 could
    # not fly the family AT ALL. The pair is a wing and a SecondWing (not a
    # wing and a rectangle): the rear surface has its own planform and its
    # incidence is geometry, because a tandem is trimmed in lift alone.
    _names = {s.get("name") for s in (geom.get("surfaces") or [])}
    # TWO REPORT SHAPES, ONE PAIR. The lifting-line `tandem` writes its wings
    # as a two-entry ``surfaces`` list; the NONPLANAR pair (tandemvlm.py)
    # writes the front at the top level and the rear under
    # ``second_surface`` — the same block ``cad.second_wing_surfaces`` reads.
    # Handling only the first left the lattice family, which is the one that
    # can carry a cant, unable to be flown.
    _second_sf = geom.get("second_surface") or {}
    tandem = ({"front", "rear"} <= _names
              or str(_second_sf.get("name") or "") == "rear")
    _pair_listed = {"front", "rear"} <= _names

    if _pair_listed:
        b, S, taper, chord_coeffs, plan_rms = _planform_from_report(
            geom, "front", b_hint=bd.get("b_front"), S_hint=bd.get("S_front"))
    elif tandem:
        # the nonplanar pair's FRONT wing is the report's own top level
        b, S, taper, chord_coeffs, plan_rms = _planform_from_report(
            geom, "wing", b_hint=bd.get("b_front"), S_hint=bd.get("S_front"))
    else:
        b, S, taper, chord_coeffs, plan_rms = _planform_from_report(geom)
    # THE TWIST, and it is in ``geometry`` — NOT in the breakdown. Reading
    # only ``bd`` returned 0.0 for both, so every rebuild flew an UNTWISTED
    # wing while the design carried its washout. Measured on `tail + winglet`
    # (twist_tip_deg = -2.0): the rebuilt lattice's zero-lift angle was
    # -1.6789 deg against the solver's -0.8091, and reading the twist closes
    # that to -0.0004 — the entire remaining disagreement between the two
    # instruments, and the reason ``Cl_r`` (the one derivative that depends on
    # the base LOADING rather than on shape) was 1.24x the scored value.
    #
    # Both dicts are asked, breakdown first, because a family that puts it
    # there should still be believed.
    def _twist(key: str) -> float:
        v = bd.get(key)
        if v is None:
            v = geom.get(key)
        return float(v or 0.0)

    tw_r, tw_t = _twist("twist_root_deg"), _twist("twist_tip_deg")
    # SWEEP AND THICKNESS ARE THE DESIGN'S, where the report states them.
    # They were dropped: every rebuild flew an unswept 12 %-thick wing
    # whatever had been scored.
    sweep = float(bd.get("sweep_deg", geom.get("sweep_deg", 0.0)) or 0.0)
    tc = float(bd.get("tc", geom.get("tc", 0.0)) or 0.0)
    # THE WING'S dihedral, which is not the tail's: ``geometry["tail"]
    # ["dihedral_deg"]`` is a V-tail's panel cant and is read further down.
    # A report that states neither leaves both at zero, which is what every
    # design published before this key existed actually flew.
    dih = float(bd.get("wing_dihedral_deg",
                       geom.get("wing_dihedral_deg", 0.0)) or 0.0)
    wing = Wing(b=b, S=S, taper=taper, twist_root_deg=tw_r,
                twist_tip_deg=tw_t, chord_coeffs=chord_coeffs,
                sweep_deg=sweep, dihedral_deg=dih,
                **({"tc": tc} if tc > 0 else {}))
    if chord_coeffs:
        notes.append(
            f"the report states no chord law, only the chord array it "
            f"produced, so the law was FITTED to that array over the same "
            f"polynomial basis the search uses: coefficients "
            f"{tuple(round(v, 4) for v in chord_coeffs)}, residual "
            f"{100 * plan_rms:.3f} % of the mean chord.")
    if plan_rms > 0.01:
        notes.append(
            f"the rebuilt planform does NOT reproduce the reported chord "
            f"distribution: {100 * plan_rms:.2f} % RMS of the mean chord. "
            f"The design being flown is not the one that was scored — a "
            f"canted tip device is the usual cause, and no report this "
            f"package writes describes one.")

    # A V-TAIL IS FLOWN AS A V-TAIL. The report states the cant and the
    # package has always known what it means (tail.lifting_area), but the
    # rebuild drew a FLAT surface of the panel area and then bolted on a fin
    # that the design does not have — so a V-tail reached stage 6 as a
    # conventional aeroplane with too much tail and an invented vertical.
    # ---- THE SECTION. The rebuild flew a SYMMETRIC one.
    # ``VLM``'s defaults are ``a = 2 pi`` and ``alpha_L0 = 0`` and nothing here
    # overrode them, so every rebuilt design flew an uncambered wing while the
    # solver flew the design's own polar. Measured on `tail + winglet` at its
    # box centre: at the SAME alpha and tail incidence the rebuild returned
    # CL 0.412244 against the scored 0.500000 — a 17.55 % lift gap — while
    # CL_alpha agreed to 0.62 %. A parallel lift curve at the wrong offset is
    # the signature of a missing zero-lift angle, and NACA 2412's is -2.21 deg.
    #
    # Everything proportional to base lift rides on this: Cl_r and Cn_p in the
    # deck, the trim state, and what stage 6 actually flies.
    #
    # The report names its section but does not carry the two numbers, so they
    # are resolved from the name where it is one this package can rebuild, and
    # ASSUMED — loudly — where it is not.
    sec_a = sec_aL0 = None
    _pol_name = str(bd.get("polar") or "")
    try:
        _dflt = _polar.default_polar()
        if _pol_name and _pol_name == getattr(_dflt, "name", None):
            sec_a, sec_aL0 = float(_dflt.a_lin), float(_dflt.alpha_L0)
    except Exception:                                  # noqa: BLE001
        sec_a = sec_aL0 = None
    if sec_a is None:
        notes.append(
            f"the report names its section {_pol_name or '(unnamed)'!r} but "
            f"carries neither its lift-curve slope nor its zero-lift angle, "
            f"and this is not a section the rebuild can reconstruct — so it "
            f"flies a SYMMETRIC one at a = 2 pi. Everything proportional to "
            f"lift (Cl_r, Cn_p, the trim state) is off by whatever the "
            f"section's camber is worth.")
    else:
        notes.append(
            f"the section is the design's own {_pol_name}: lift-curve slope "
            f"{sec_a:.4f} /rad, zero-lift angle "
            f"{np.rad2deg(sec_aL0):+.2f} deg. Flying the VLM's symmetric "
            f"default instead costs 17.6 % of the lift at trim, and with it "
            f"every derivative proportional to it.")

    v_tail = str(tail.get("type") or "") == "v_tail"
    tail_cant = float(tail.get("dihedral_deg") or 0.0) if v_tail else 0.0

    # ---- the tandem's REAR WING, at the stagger the pair was scored with.
    second = None
    if tandem:
        _rear_geom = (geom if _pair_listed
                      else {"surfaces": [_second_sf], "b": _second_sf.get("b")})
        b2, S2, tp2, cc2, rms2 = _planform_from_report(
            _rear_geom, "rear", b_hint=bd.get("b_rear"),
            S_hint=bd.get("S_rear"))
        rear = Wing(b=b2, S=S2, taper=tp2, chord_coeffs=cc2,
                    sweep_deg=sweep, dihedral_deg=dih)
        second = SecondWing(
            wing=rear,
            x=float(bd.get("dx", _second_sf.get("x_offset", 0.0)) or 0.0),
            z=float(bd.get("dz", _second_sf.get("z_offset", 0.0)) or 0.0),
            N=40)
        plan_rms = max(plan_rms, rms2)
        notes.append(
            f"this is a TANDEM: the rear wing is rebuilt from its own "
            f"reported chord distribution ({b2:.3g} m span, {S2:.3g} m2) at "
            f"the scored stagger dx = {second.x:.3g} m, dz = {second.z:.3g} "
            f"m. The pair's DECALAGE lives in the wings' twist, which no "
            f"tandem report writes out — so both surfaces are flown at the "
            f"twist stated for the front one, and the CL_alpha row below is "
            f"where that shows.")

    tail_surface = None
    if tail:
        S_t = float(tail.get("S_t") or tail.get("S_lift") or 0.0) or 1e-3
        # ``b_t`` in the report is the HORIZONTAL PROJECTION (it is
        # sqrt(AR * S_lift), and S_lift is S_t cos^2 G). The lattice is given
        # the PANEL span, because the panels are what it draws — divide the
        # projection back out rather than handing a canted surface a
        # projected aspect ratio, which would shrink it twice.
        b_panel = (float(tail.get("b_t") or 1.0)
                   / max(np.cos(np.deg2rad(tail_cant)), 1e-9))
        # THE SECOND SURFACE FLIES ITS OWN SECTION, and on a downloading
        # stabiliser that is the wing's mounted INVERTED (the report says so:
        # ``section_inverted``, and ``polar_tail`` is named "(inverted)").
        # Handing it the wing's camber makes it LIFT where the design has it
        # pushing down — measured as the rebuilt CL overshooting the scored
        # one by +27 % once the wing's section was passed correctly.
        t_a, t_aL0 = sec_a, sec_aL0
        if sec_aL0 is not None and bool(tail.get("section_inverted")):
            t_aL0 = -sec_aL0
        tail_surface = TailSurface(
            a=t_a, alpha_L0=t_aL0,
            S=S_t,
            x=float(tail.get("dist_m") or tail.get("l_t") or 0.0),
            z=float(tail.get("dz_m") or 0.0),
            AR=float(b_panel ** 2 / max(S_t, 1e-9)),
            dihedral_deg=tail_cant,
            N=20)

    # a tandem states no mac: its reference chord is the PAIR's area over
    # the pair's span, which is what its own coefficients are quoted on.
    _S_pair = float(bd.get("Sref") or S) if tandem else S
    mac = float(tail.get("mac") or bd.get("mac") or (_S_pair / b))
    x_cg = float(x_cg_m if x_cg_m is not None
                 else tail.get("x_cg", bd.get("x_cg", 0.0)) or 0.0)
    if x_cg_m is None and not (tail.get("x_cg") or bd.get("x_cg")):
        # A CG ON THE QUARTER-CHORD LINE IS AN ASSUMPTION, and it decides
        # the static margin, which decides whether the aeroplane is stable.
        # Everything else invented here (the fin, the mass, CD0, e) is
        # recorded; this was the one that was not.
        notes.append(
            "the report states no CG, so it is taken at x = 0 — the wing's "
            "quarter-chord line. The static margin shown is a consequence "
            "of that assumption, not a measurement; move the CG lever in "
            "stage 6 and watch it change.")

    # ---- the vertical surface. It is now IN THE REPORT for every layout
    # that has one (``api.design_report`` -> ``fin.size_fin``, the same law
    # the drag book charges), so the default stopped being an invention and
    # became a reading. What is still assumed is what a user OVERRODE in
    # stage 5, and what a report predating the fin block cannot supply.
    stated = _fin.fin_from_report(geom)
    # ...AND THE DESIGN MAY STATE THAT IT HAS NONE, which a missing block
    # does not. Absent means "this family does not report a fin" and the
    # assumed one below is the honest answer; an explicit None means the
    # design was asked and said no (``fin.states_no_fin``). Reading them the
    # same way is how an aeroplane with the fin switched off came to be
    # rebuilt with an INVENTED fin at 12 % of span — larger than the one it
    # had been charged for, so the switch RAISED Cn_beta 0.1114 -> 0.1351
    # instead of zeroing it.
    none_by_design = _fin.states_no_fin(geom)
    vertical = None
    # A V-TAIL CARRIES NO FIN — that is its whole raison d'etre, and
    # ``tail.py:1739`` has always said so for the drag book while this
    # function bolted one on anyway. It is a property of the DESIGN, so it
    # overrides the stage-5 toggle rather than being asked again. So is the
    # answer the mission gave, for the same reason: stage 5 must not fit a
    # surface stage 1 deleted and stage 3 neither charged nor weighed.
    want_fin = bool(spec.fin) and not v_tail and not none_by_design
    if spec.fin and v_tail:
        notes.append(
            "this is a V-tail, so no separate fin was built: its yaw "
            "stiffness comes from the cant of its own panels "
            f"({tail_cant:.1f} deg). The fin switch in stage 5 does not "
            "apply to this design.")
    elif spec.fin and none_by_design:
        notes.append(
            "this design carries no vertical stabiliser — the mission said "
            "so, and the run neither charged its drag nor weighed it. "
            "Nothing here makes yaw stiffness, so Cn_beta is exactly zero "
            "and the Dutch roll and the spiral are what an aeroplane with no "
            "fin actually does. The fin switch in stage 5 cannot add one "
            "back: turn it on in stage 1 and run the design again.")
    if want_fin:
        # each of the three dimensions falls back independently: a user who
        # states only the chord keeps the DESIGN's height and station.
        h = (spec.fin_height_m if spec.fin_height_m
             else (abs(stated.height) if stated else 0.12 * b))
        c = (spec.fin_chord_m if spec.fin_chord_m
             else (stated.chord if stated else 0.65 * mac))
        # the LEADING EDGE is what is answered; the quarter chord is what
        # the lattice is placed by. One conversion, after ``c`` resolved.
        # A BLANK STATION PUTS THE QUARTER CHORD ON THE ARM, whatever chord
        # resolved above — the design's own arm where the report states a
        # fin, the tail station otherwise. Defaulting to the fin block's
        # LEADING EDGE instead looks equivalent and is not: the leading edge
        # is a fixed number, so a user who then changes only the CHORD moves
        # the quarter chord, and with it the yaw arm and the fin volume
        # coefficient they sized by. Measured: taking the Reynolds
        # recommendation shifted V_v 0.03964 -> 0.04019.
        x_le = (spec.fin_x_le_m if spec.fin_x_le_m is not None
                else ((stated.x_qc if stated
                       else (tail_surface.x if tail_surface else 0.45 * b))
                      - QUARTER * c))
        x_f = x_le + QUARTER * c
        # WHERE THE FIN'S FOOT IS — READ, not re-derived. A stated fin
        # carries its own root, and for a T-TAIL that root is the BODY
        # (0.0), because the tailplane sits on the fin's TIP. Taking the
        # tailplane's height for every layout put a T-tail's fin entirely
        # ABOVE the surface it carries — measured before this line: root
        # 1.04447 m, tip 2.08893 m, tailplane at 1.04447 m, i.e. "a
        # conventional tail just raised" with a fin bolted over the top.
        # ``api.design_report`` already fixed exactly this in the block it
        # WRITES; the rebuild was a second author of the same number and
        # re-introduced it. Now there is one: the report's.
        z_root_fin = (0.0 if spec.fin_ventral
                      else (float(stated.z_root) if stated is not None
                            else float(tail.get("dz_m") or 0.0)))
        vertical = VerticalSurface(
            height=(-h if spec.fin_ventral else h), chord=c, x=x_f,
            z_root=z_root_fin, N=12)
        if stated is None:
            notes.append(
                f"the fin is not in this report, so its size is assumed: "
                f"height {abs(h):.3g} m ({abs(h) / b:.0%} of span), chord "
                f"{c:.3g} m, leading edge at x = {x_le:.3g} m (quarter chord "
                f"{x_f:.3g} m, which is the yaw arm). Change it in stage 5 "
                f"and every yaw number moves with it.")
        elif str(getattr(stated, "kind", "fin")) == "mast":
            # A BOAT'S VERTICAL SURFACE IS NOT A TAIL, so it is not
            # described as one: quoting a volume coefficient of 0 against a
            # 0 m arm is what a fin-shaped read-out says about a strut.
            notes.append(
                f"the vertical surface is the DESIGN'S MAST: the strut that "
                f"carries the foil, {abs(h):.3g} m of submergence by a "
                f"{c:.3g} m chord ({stated.S:.3g} m2), which is exactly the "
                f"wetted area the run's own mast drag was charged on. Its "
                f"quarter chord stands at x = {x_f:.3g} m — on the fuselage "
                f"BETWEEN the two wings where a real mast is bolted, unless "
                f"this is a single foil, which has no fuselage and keeps "
                f"x = 0 — and the CG is at {x_cg:.3g} m, so the yaw arm is "
                f"{x_f - x_cg:+.3g} m: {'AFT of the CG and stabilising' if x_f > x_cg else 'AHEAD of the CG and DESTABILISING'}. "
                f"That sign is the craft's, not a modelling choice: a rider "
                f"stands over the mast and steers it. This craft's yaw "
                f"stiffness is the STRUT'S ALONE — a rudder and a hull are "
                f"what make a boat directionally stable and this package "
                f"models neither. Cn_beta below is that statement, not a "
                f"fin's.")
        else:
            notes.append(
                f"the fin is the DESIGN'S: area {stated.S:.3g} m2 at a "
                f"volume coefficient of {stated.V_v:g} against a "
                f"{stated.l_t:.3g} m arm, aspect ratio {stated.AR:g} — the "
                f"same surface the design's drag book sizes. Flown here as "
                f"height {abs(h):.3g} m, chord {c:.3g} m, leading edge "
                f"x = {x_le:.3g} m.")
            # NO "...but its drag was not charged" NOTE any more: it is,
            # unconditionally, so the L/D above knows the fin is there and
            # there is no allowance left to warn about.
    # the fin's own aspect ratio, for the induced drag its side force costs
    # (:meth:`sixdof.Aircraft.coefficients`). None when there is no fin, so
    # the term is absent rather than guessed.
    AR_v = (float(abs(h) ** 2 / max(abs(h) * c, 1e-12))
            if vertical is not None else None)

    # a TANDEM is referenced to its TOTAL area, which is what its own solver
    # does (tandem.py's Sref) — the pair's CL and every derivative built on it
    # would otherwise be quoted on the front wing alone and read ~2x high.
    S_ref = (float(bd["Sref"]) if tandem and bd.get("Sref") else None)
    # ---- THE TIP DEVICE, which this rebuild used to drop on the floor.
    # The report states it (``geometry["winglet"]``) and the VLM took no
    # winglet argument here, so every design came back with ZERO device
    # panels: measured on `tail + winglet`, Cl_beta was bit-identical at
    # -0.0147155577 for h_frac 0.00 / 0.05 / 0.10 / 0.15 while the SCORED
    # L/D moved 32.81 -> 34.49. A canted device is a nonplanar surface
    # carrying side force, so dropping it does not merely lose a drag count
    # — it removes most of the aeroplane's dihedral effect, which is exactly
    # the quantity stages 5 and 6 exist to report.
    wl = geom.get("winglet") or {}
    h_frac = float(wl.get("h_frac", 0.0) or 0.0)
    if h_frac > 0.0:
        notes.append(
            f"the tip device is IN the flown lattice: {h_frac:.1%} of the "
            f"semi-span at {float(wl.get('cant_deg', 90.0)):.0f} deg cant. "
            f"It carries side force, so it is a large part of Cl_beta — a "
            f"rebuild without it reports the dihedral effect of a different "
            f"aeroplane.")
    m = VLM(wing, N=60, V=V, tail=tail_surface, second=second,
            vertical=vertical, S_ref=S_ref,
            **({} if sec_a is None
               else {"a": sec_a, "alpha_L0": sec_aL0}),
            winglet_h_frac=h_frac,
            winglet_cant_deg=float(wl.get("cant_deg", 90.0) or 90.0),
            winglet_blend_frac=float(wl.get("blend_frac", 0.0) or 0.0),
            winglet_blend_shape=str(geom.get("winglet_blend_shape") or "arc"),
            winglet_wing_blend_frac=float(
                wl.get("wing_blend_frac", 0.0) or 0.0))

    controls = []
    if spec.aileron:
        controls.append(dyn.aileron(m, spec.aileron_span,
                                    spec.aileron_chord_frac))
    if spec.flap:
        controls.append(dyn.flap(m, spec.flap_span, spec.flap_chord_frac))
    # THE PITCH CONTROL GOES ON WHICHEVER SURFACE IS THE AFT ONE. A tandem
    # has no tail block and a rear wing instead, and gating this on
    # ``tail_surface`` alone is what made every registered tandem family
    # unflyable: no elevator column, so ``sixdof.trim_level`` had nothing to
    # solve for and stage 6 refused to arm. ``dyn.elevator`` picks the
    # surface (tail if there is one, else the second wing); this decides
    # only whether the aircraft HAS one at all.
    if spec.elevator and (tail_surface is not None or second is not None):
        controls.append(dyn.elevator(m, spec.elevator_chord_frac))
        if tail_surface is None and second is not None:
            notes.append(
                f"the pitch control is an ELEVON on the REAR WING: a tandem "
                f"has no tailplane to hinge, and the rear surface is the aft "
                f"one — {spec.elevator_chord_frac:.0%} of its local chord, "
                f"at the scored stagger. Roll is on the FRONT wing alone, so "
                f"the two hinges are not cut out of the same trailing edge.")
    if want_fin and spec.rudder:
        # a rudder is a hinge on the FIN; a V-tail's yaw control is a
        # differential ruddervator, which is the elevator channel run
        # antisymmetrically and is not this surface
        controls.append(dyn.rudder(m, spec.rudder_chord_frac))

    # LINEARISE AT THE STATE THE DESIGN WAS FLOWN AT, not at zero.
    # Cl_r and Cn_p are proportional to the base CL (dynamics.deck), so a deck
    # built at alpha = 0 reports both as exactly zero — a flight model that
    # silently has no adverse yaw and no roll-due-to-yaw. The report carries
    # the trim incidence the design was actually evaluated at; use it, and say
    # so when it is missing.
    #
    # AT THE SCORED LIFT, NOT THE SCORED ANGLE. Those are the same state only
    # if the two lattices agree exactly, and they do not: the design is
    # trimmed to a CL and its alpha is a CONSEQUENCE of that, through a
    # section convention and a solver core the rebuild does not share. Taking
    # the angle put the rebuilt aeroplane at the wrong lift — measured on
    # `tail + winglet`, CL 0.5857 where the design was scored at 0.5000, so
    # every derivative proportional to base lift was ~17 % out and Cl_r came
    # back at 1.36x the scored value.
    #
    # Trimming to the LIFT makes those derivatives right by construction and
    # turns the disagreement into a stated ANGLE difference, which is a
    # fidelity row rather than a silent error in the deck.
    i_t_trim = float(np.deg2rad(tail.get("i_t_deg", 0.0) or 0.0))
    alpha_trim = None
    CL_scored = bd.get("CL_total") or bd.get("CL_target")
    if CL_scored:
        lin = dyn.deck(m, x_cg=x_cg, mac=mac, b=b)
        if abs(lin.CL_alpha) > 1e-9:
            CL0 = float(m.solve(0.0, i_t=i_t_trim).CL)
            alpha_trim = (float(CL_scored) - CL0) / lin.CL_alpha
            stated = bd.get("alpha_rad")
            if stated is None and bd.get("alpha_deg") is not None:
                stated = np.deg2rad(float(bd["alpha_deg"]))
            if stated is not None:
                d_deg = np.rad2deg(alpha_trim - float(stated))
                notes.append(
                    f"the deck is linearised at the LIFT the design was "
                    f"scored at (CL {float(CL_scored):.4f}), which this "
                    f"lattice reaches at "
                    f"{np.rad2deg(alpha_trim):+.2f} deg where the solver "
                    f"recorded {np.rad2deg(float(stated)):+.2f} deg "
                    f"({d_deg:+.2f} deg apart). Cl_r and Cn_p are "
                    f"proportional to that lift, so matching the lift is what "
                    f"makes them the scored aeroplane's; the angle difference "
                    f"is the two cores disagreeing and is reported, not "
                    f"absorbed.")
    if alpha_trim is None:
        alpha_trim = bd.get("alpha_rad")
    if alpha_trim is None and bd.get("alpha_deg") is not None:
        alpha_trim = np.deg2rad(float(bd["alpha_deg"]))
    if alpha_trim is None:
        # last resort: the incidence that makes the report's own CL target
        lin = dyn.deck(m, x_cg=x_cg, mac=mac, b=b)
        CL_t = bd.get("CL_total") or bd.get("CL_target")
        alpha_trim = (float(CL_t) / lin.CL_alpha
                      if CL_t and abs(lin.CL_alpha) > 1e-9 else 0.0)
        notes.append(
            f"the report states no trim incidence, so the deck is linearised "
            f"at alpha = {np.rad2deg(alpha_trim):.2f} deg, back-solved from "
            f"its lift coefficient. Cl_r and Cn_p scale with that number.")
    alpha_trim = float(alpha_trim)

    deck = dyn.deck(m, x_cg=x_cg, mac=mac, b=b, controls=controls,
                    alpha=alpha_trim, i_t=i_t_trim)

    # ---- the self-check. The PLANFORM row is first because it is upstream
    # of the other three: a chord distribution the rebuild cannot reproduce
    # is why CL_alpha and the neutral point would disagree, and reporting
    # only the consequences leaves the cause invisible.
    rows = [("planform c(y)", 0.0, float(plan_rms), float(plan_rms))]
    for name, flown, got in (
            ("CL_alpha", bd.get("CL_alpha"), deck.CL_alpha),
            ("x_np", tail.get("x_np"), m.neutral_point()),
            ("SM", tail.get("SM", bd.get("SM")), deck.static_margin)):
        if flown is None:
            rows.append((name, None, float(got), None))
        else:
            fl = float(flown)
            err = (float(got) - fl) / fl if abs(fl) > 1e-12 else None
            rows.append((name, fl, float(got), err))
    fidelity = Fidelity(tuple(rows))

    # ---- mass, drag and inertia
    W = bd.get("W_N") or bd.get("W_total_N")
    mass = float(mass_kg if mass_kg is not None
                 else (float(W) / sd.G if W else 0.0))
    if not mass > 0.0:
        mass = float(bd.get("CL_target", 0.5)) * 0.5 * rho * V ** 2 * S / sd.G
        notes.append(
            f"the report states no weight, so mass was back-solved from the "
            f"trim lift coefficient at {V:.4g} m/s: {mass:.4g} kg. It scales "
            f"as V^2, so a rebuild at the wrong speed is a different "
            f"aeroplane, not the same one flown faster.")
    # the body length, from ONE place. It was a bare 1.6 here and is now
    # ``drag.BODY_LENGTH_FRAC``, because the fuselage DRAG term is built on
    # the same proxy — two copies of a constant that decides both a moment
    # of inertia and a drag count is how the fin came to have two sizes.
    length = float(body_length_m if body_length_m
                   else (_drag.body_length_for_arm(tail_surface.x)
                         if tail_surface else 0.8 * b))
    inert = inertia or sd.Inertia.from_layout(mass, b, length)

    cd0 = CD0
    if cd0 is None:
        cd0 = bd.get("CDp") or bd.get("CD0") or bd.get("cd0_extra")
        cd0 = float(cd0) if cd0 else 0.025
        if not (bd.get("CDp") or bd.get("CD0")):
            notes.append(f"no profile drag in the report; CD0 = {cd0:.4g} "
                         f"assumed.")
    e = float(oswald_e) if oswald_e else 0.85
    if oswald_e is None:
        notes.append(f"span efficiency e = {e:.2f} assumed for the drag "
                     f"polar the simulation flies.")

    stall, why = _stall_from_report(bd, deck)
    notes.append(why)
    ac = sd.Aircraft(deck=deck, inertia=inert, rho=rho, CD0=float(cd0),
                     oswald_e=e, AR_vertical=AR_v, stall=stall,
                     prop=sd.Propulsion(z_offset_m=spec.thrust_z_offset_m))
    if spec.thrust_z_offset_m:
        notes.append(
            f"the thrust line is {spec.thrust_z_offset_m:.3g} m BELOW the "
            f"CG, so opening the throttle pitches the nose UP. Answered in "
            f"stage 5; 'through the CG' makes no pitching moment at all.")
    return FlightModel(model=m, deck=deck, aircraft=ac, fidelity=fidelity,
                       assumptions=notes, controls=controls,
                       alpha_trim=alpha_trim, i_t_trim=i_t_trim)


#: the widest wing dihedral :func:`dihedral_for_spiral` will offer, in
#: degrees. Not a physical limit and not a refusal: it is the end of the
#: band the recommendation is searched over, and a design whose spiral does
#: not turn inside it is told so ("out of reach") instead of being handed
#: an angle nobody would build. ``geometry.DIHEDRAL_BOUNDS_DEG`` (-10 to
#: 15) is the box the SEARCH rides; its anhedral half is not a
#: recommendation's business, because the margin only grows with dihedral
#: and the smallest convergent cant is therefore never negative.
SPIRAL_DIHEDRAL_MAX_DEG = 15.0


@dataclass(frozen=True)
class SpiralFix:
    """How much WING DIHEDRAL turns this design's spiral, measured.

    ``status`` is the whole answer and the number is only meaningful under
    it:

    ``"converges"``   it already does, at the dihedral it flies now
    ``"found"``       ``gamma_deg`` is the SMALLEST dihedral in the band
                      whose margin is not negative
    ``"out_of_reach"``no dihedral up to the band's top converges it
    ``"refused"``     ``Cn_beta <= 0`` — the aeroplane does not weathercock,
                      and a positive spiral margin on it is arithmetic
                      rather than stability (``wing_score.spiral_refusal``)
    """

    status: str
    #: the dihedral the report's design flies today [deg]
    flown_deg: float
    #: the criterion at that dihedral (``dynamics.spiral_margin_of``)
    margin_now: float
    #: the yaw stiffness the answer is conditional on
    Cn_beta: float
    #: the recommendation [deg], None unless ``status`` is a number's
    gamma_deg: float | None = None
    #: the criterion AT the recommendation — the gate it has to clear
    margin_at: float | None = None
    #: the band searched, as (lo, hi) in degrees
    band: tuple = (0.0, SPIRAL_DIHEDRAL_MAX_DEG)

    @property
    def converges(self) -> bool:
        """Does the design as it stands have a convergent spiral?"""
        return self.status == "converges"

    @property
    def extra_deg(self) -> float | None:
        """How much MORE dihedral than it flies today, or None."""
        if self.gamma_deg is None:
            return None
        return float(self.gamma_deg) - float(self.flown_deg)


def _report_at_dihedral(report: dict, gamma_deg: float) -> dict:
    """``report`` with the WING's dihedral overridden, copied not mutated.

    ``build_flight_model`` reads the cant off ``breakdown["wing_dihedral_deg"]``
    (falling back to ``geometry``), so this is the one key a rebuild needs to
    fly a different wing — the same lever ``scripts/v5_spiral_reference.py``
    pulls, and the same one stage 5's fin scan pulls for the fin's height.
    """
    out = dict(report)
    out["breakdown"] = dict(report.get("breakdown") or {})
    out["breakdown"]["wing_dihedral_deg"] = float(gamma_deg)
    geom = report.get("geometry")
    if isinstance(geom, dict) and "wing_dihedral_deg" in geom:
        out["geometry"] = dict(geom)
        out["geometry"]["wing_dihedral_deg"] = float(gamma_deg)
    return out


def dihedral_for_spiral(report: dict, spec: ControlsSpec | None = None, *,
                        V: float | None = None, rho: float | None = None,
                        band: tuple = (0.0, SPIRAL_DIHEDRAL_MAX_DEG),
                        tol: float = 0.01, step: float = 1.0,
                        **kw) -> SpiralFix:
    """The dihedral that makes THIS design's spiral converge, or why none does.

    The package could already say a divergent spiral is a WING answer and
    not a fin one (stage 5 measures ten fin heights and none of them fixes
    it). It could not say HOW MUCH wing — the shell quoted one constant,
    5.7 deg, measured on ``tail [free cant]``'s box centre, at every
    configuration. Measured with this function over the families the shell
    actually derives, the crossing runs from 3.07 deg (``tail + winglet``,
    where the tip device is already carrying 88 % of ``Cl_beta``) to
    6.13 deg (``tail``, and the shell's DEFAULT ``tail + free chord law``
    with it) — so the one constant was 86 % too much on one family and not
    enough on the family a user lands on first. One constant offered as a
    step that leaves the aeroplane rolling off is the defect this exists to
    close: see ``RESULTS_SESSION69_SPIRAL_RECOMMENDATION.md``.

    THE INSTRUMENT is ``dynamics.spiral_margin_of`` on the deck of a
    REBUILD — the same number stage 5 prints, carrying the trim attitude,
    whose zero lands within a tenth of a degree of the 6-DOF spiral
    eigenvalue's (asserted in
    ``tests/test_the_spiral_criterion_tells_the_truth.py``, and the residual
    is the deck's attitude against the trim state's own). Each sample
    costs one lattice rebuild (~4 ms on a wing+tail), and the answer is a
    coarse walk up ``band`` at ``step`` degrees to find the first bracket
    plus a bisection to ``tol`` — about 20 rebuilds.

    WHAT IT DOES NOT DO is re-solve the design. The rebuild flies the
    stated cant on the planform the report already carries; it does not
    re-trim, re-search or re-price it, exactly as the fin scan beside it
    does not. On a family that can score a cant the answer is therefore a
    starting point for a run, not a substitute for one — and on a
    lifting-line family, which cannot score one at all, it is the only
    number there is.

    ``Cn_beta <= 0`` is REFUSED rather than answered, because the criterion
    is a difference of two products and driving the yaw stiffness through
    zero flips the second one: a "convergent" spiral on an aeroplane that
    yaws away from the airflow is arithmetic. Same guard, same reason, as
    ``wing_score.spiral_refusal``.
    """
    from . import dynamics as _dyn

    lo, hi = float(band[0]), float(band[1])

    def _margin(gamma: float):
        fm = build_flight_model(_report_at_dihedral(report, gamma),
                                spec, V=V, rho=rho, **kw)
        d = fm.deck
        return float(_dyn.spiral_margin_of(d)), float(d.Cn_beta)

    flown = float((report.get("breakdown") or {}).get(
        "wing_dihedral_deg",
        (report.get("geometry") or {}).get("wing_dihedral_deg", 0.0)) or 0.0)
    m_now, cnb = _margin(flown)
    base = dict(flown_deg=flown, margin_now=m_now, Cn_beta=cnb,
                band=(lo, hi))
    if cnb <= 0.0:
        return SpiralFix(status="refused", **base)
    if m_now > 0.0:
        return SpiralFix(status="converges", gamma_deg=flown,
                         margin_at=m_now, **base)

    # the recommendation is never LESS dihedral than the design already
    # flies: this answers "what turns it", and a wing at +3 deg is not
    # helped by being told about +1
    a = max(lo, flown)
    m_a = m_now if a == flown else _margin(a)[0]
    if m_a > 0.0:                       # the band's own floor already does
        return SpiralFix(status="found", gamma_deg=a, margin_at=m_a, **base)
    g = a
    while g < hi - 1e-9:
        b = min(hi, g + float(step))
        m_b, cnb_b = _margin(b)
        if m_b >= 0.0:
            if cnb_b <= 0.0:            # bought by not weathercocking
                return SpiralFix(status="refused", **base)
            while b - a > float(tol):
                mid = 0.5 * (a + b)
                if _margin(mid)[0] < 0.0:
                    a = mid
                else:
                    b = mid
            m_b, cnb_b = _margin(b)
            return SpiralFix(status="found", gamma_deg=float(b),
                             margin_at=float(m_b), **base)
        a, g = b, b
    return SpiralFix(status="out_of_reach", **base)


def _stall_from_report(bd: dict, deck) -> tuple["sd.Stall", str]:
    """The stall the DESIGN'S OWN SECTION has, not a hardcoded 1.4.

    ``sixdof.Stall`` ships with ``CL_max = 1.4`` at 14 degrees, and nothing
    used to overrule it — so a design whose section was measured to stall at
    1.1 flew to 1.4 here, and one measured at 1.8 was clipped at 1.4. The
    whole package exists to measure that number; not carrying it into the
    one place a pilot meets it is the "stated is not flown" defect on the
    stall.

    The section's ``cl_max`` is a 2-D number and the wing's ``CL_max`` is
    not: a finite wing reaches less of it, and the fraction is the
    lift-curve-slope ratio ``CL_alpha / cl_alpha`` — the same 3-D correction
    the lattice already applies to the slope. ``cl_alpha`` is taken as the
    thin-aerofoil ``2 pi``, which is what the polars are referenced to.

    The stall ANGLE then follows from the two numbers rather than being a
    second assumption: ``alpha_stall = CL_max / CL_alpha`` measured from the
    zero-lift line, plus the deck's own base incidence.
    """
    from . import sixdof as sd

    cl_max = bd.get("cl_max") or bd.get("clmax") or bd.get("cl_max_wing")
    pol = bd.get("polar") or {}
    if cl_max is None and isinstance(pol, dict):
        cl_max = pol.get("cl_max") or pol.get("clmax")
    if not cl_max:
        d = sd.Stall()
        return d, (f"the report states no section cl_max, so the stall is "
                   f"the shipped default: CL_max {d.CL_max:.2f} at "
                   f"{np.rad2deg(d.alpha_stall_rad):.1f} deg.")
    cl_max = float(cl_max)
    CLa = float(deck.CL_alpha)
    ratio = float(np.clip(CLa / (2.0 * np.pi), 0.2, 1.0))
    CL_max = cl_max * ratio
    a_st = CL_max / CLa if abs(CLa) > 1e-9 else np.deg2rad(14.0)
    return (sd.Stall(CL_max=float(CL_max), alpha_stall_rad=float(a_st)),
            f"the wing's CL_max is the section's measured cl_max "
            f"{cl_max:.3f} times the 3-D slope ratio CL_alpha/2pi = "
            f"{ratio:.3f}, giving {CL_max:.3f} at "
            f"{np.rad2deg(a_st):.1f} deg from the zero-lift line.")
