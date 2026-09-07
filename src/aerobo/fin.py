"""The vertical surface, sized ONCE.

Before this module the fin had two authors and neither of them was the design:

* ``tail.vtail_cd0`` sized one by volume coefficient (``V_V_DEFAULT`` 0.04,
  ``AR_VT_DEFAULT`` 1.5, t/c 0.10) purely to charge its parasite drag — and
  that charge was opt-in and OFF, so the shipped design paid
  ``cd0_fin = 0.0`` for a surface it flew. It is unconditional now, exactly
  as the tailplane's profile drag always was;
* ``flightmodel.build_flight_model`` sized a DIFFERENT one (height 12 % of
  span, chord 0.65 mac, quarter chord at the tail station) and that one
  produced one hundred per cent of ``Cn_beta`` and ``Cl_beta`` in flight.

So the surface that decided whether the aeroplane was directionally stable
was not the surface whose drag was (optionally) charged, and neither was
recorded in the report. This module is the one law. Everything that needs a
fin — the drag book, the flight rebuild, the CAD export, the shell — asks
:func:`size_fin` and gets the same object.

**A V-tail has no fin.** That is its whole raison d'etre and ``tail.py`` has
always said so for the drag; :func:`size_fin` returns ``None`` for one, so
the rule lives here rather than being re-stated at each call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FinGeometry", "size_fin", "mast", "fin_for_layout",
           "fin_from_report", "fin_shape_kwargs", "fin_law_kwargs",
           "fin_drag_kwargs", "has_fin", "states_no_fin",
           "V_V_DEFAULT", "AR_VT_DEFAULT", "FIN_TC_DEFAULT"]

#: vertical-tail volume coefficient (Raymer, *Aircraft Design: A Conceptual
#: Approach*, Table 6.4 — c_VT ~ 0.04 for single-engine GA). The VERTICAL
#: tail volume uses wing SPAN, not mac.
V_V_DEFAULT = 0.04
#: fin aspect ratio; the conventional range is 1.3-2.
AR_VT_DEFAULT = 1.5
#: fin thickness ratio, for the form factor its drag is built from.
FIN_TC_DEFAULT = 0.10

#: A fin this small is not a surface. Sized from a volume coefficient the
#: area collapses as the arm grows, and a canard-like layout (l_t -> 0)
#: sends it the other way; both ends are refused rather than drawn.
MIN_FIN_AREA_M2 = 1e-4

#: WHERE A TANDEM'S FIN STANDS, as a fraction of the pair's stagger.
#:
#: ON THE REAR WING. A conventional layout has a body to hang the fin off
#: and the answer is "at the tail"; a PAIR has no body, so the only honest
#: question is which structure the surface is bolted to — and a pair already
#: owns one at ``dx``. Landing "on top of the rear wing, inside that
#: surface's own chord footprint" is what a fin is SUPPOSED to do: it is the
#: mount. The midpoint has nothing there at all, so a fin at 0.5 is carried
#: by a boom the design neither draws, weighs nor charges for.
#:
#: This is the same rule the tailed layouts already obey by construction —
#: ``fin_arm`` puts the fin at the tailplane's own station, so one boom
#: reaches one place and is charged once (``tail._fuselage_charge``, called
#: once on ``cfg.dist``). A pair's rear wing IS that station. Nothing is
#: charged for being there, which is the whole point: the structure is
#: already paid for.
#:
#: WHAT IT IS WORTH, measured on the shipped pair at its box centre, moving
#: 0.5 -> 1.0. The volume-coefficient law is ``S_vt = V_v*b*S/l_t``, so
#: doubling the arm HALVES the area:
#:
#:     S_fin    3.2000  ->  1.6000 m2
#:     cd0_fin  0.0016952 -> 0.0009034   (-46.7 %)
#:     L/D      23.6293 -> 24.5478       (+3.89 %)
#:     Cn_beta  +0.081057 -> +0.117382   (+44.8 %)
#:
#: The yaw stiffness was expected to hold FLAT — ``Cn_beta`` goes as
#: ``S_vt*l_t`` and that product IS the volume coefficient — and instead it
#: rose 45 %, because the sizing law holds the fin's ASPECT RATIO at
#: :data:`AR_VT_DEFAULT` while shrinking its area, and the lattice's smaller
#: fin carries more side force per unit area than the law assumes. So the
#: move is not the volume coefficient's usual trade of wetted area against
#: arm; it is better on both axes at once.
#:
#: THE READING THIS REPLACES said the midpoint was "the only distinguished
#: point" of the interval between the wings. It is — of an interval with no
#: structure in it. The rear wing is distinguished by carrying something.
TANDEM_FIN_X_FRAC = 1.0

#: ...AND HOW FAR AFT OF THE REAR WING THE BOOM CARRIES IT, as a fraction of
#: the same stagger. The station above puts the fin ON the rear wing; this
#: puts it BEHIND one, which is where a reader of a three-view expects a
#: vertical tail and — measured — where the pair's spiral goes.
#:
#: A FRACTION and not a length, so it follows the layout: a pair staggered
#: 2 m does not get the 5 m boom a pair staggered 5 m wants. A stated
#: ``fin_boom_m`` overrides it in metres (``tandem_fin_station``), and
#: ``fin_boom_m = 0`` reproduces the on-the-wing station above bit-for-bit —
#: that answer is not taken away, it is made answerable.
#:
#: MEASURED on the shipped pair at its box centre (``cant_free``, lateral
#: armed), moving the foot from the rear wing's root to one stagger behind
#: it — x_qc 5 -> 10 m, z_root held at ``dz``:
#:
#:     S_fin           1.6000  ->  0.8000 m2
#:     L/D             24.5887 ->  25.1088   (+2.11 %)
#:     Cn_beta        +0.10311 -> +0.06653   (-35.5 %)
#:     Cn_r           -0.11955 -> -0.15572   (+30.3 % of damping)
#:     spiral margin  -0.009746 -> -0.001250
#:     least dihedral that turns the spiral   9.5 deg -> 3.5 deg
#:
#: THE TRADE, said out loud: the on-the-wing station buys yaw STIFFNESS (the
#: wing works as an end plate, which is the +44.8 % recorded above); the boom
#: buys yaw DAMPING, which is what the spiral criterion reads
#: (``Cl_beta*Cn_r - Cn_beta*Cl_r``). Stiffness alone does not turn a spiral
#: — it sits on the wrong side of that product — so a pair asked to be stable
#: wants the arm, and a pair asked for the smallest fin wants the wing.
TANDEM_FIN_BOOM_FRAC = 1.0


def tandem_fin_station(dx: float, dz: float = 0.0,
                       boom_m: float | None = None) -> tuple:
    '''``(x_qc, z_root)`` of a pair's fin, from its stagger and offset [m].

    THE ONE AUTHOR of that station. Three callers need it and they have to
    agree, or the surface is drawn in one place and flown in another:
    ``tandem.evaluate_tandem`` and ``tandemvlm.evaluate_tandem_vlm`` charge
    its drag on the arm, and ``api.design_report`` sizes, reports and lofts
    it at the same arm.

    The arm IS ``x_qc`` because the CG a tandem rebuild assumes is the front
    wing's quarter chord, x = 0 (``flightmodel.build_flight_model`` states
    that assumption in the report it writes) — so a fin at ``x_qc`` flies on
    exactly the arm it was sized against.

    ``z_root`` rides the same fraction up the pair's vertical offset, so at
    :data:`TANDEM_FIN_X_FRAC` = 1.0 the foot lands on the REAR WING'S ROOT —
    on the surface in both directions, not merely at its station in x and
    floating in z. That is the coupling that makes the fraction one number
    rather than two: a fin mounted on a wing is at that wing's x AND its z.

    ...AND THEN THE BOOM CARRIES IT AFT OF THAT WING (``boom_m``, defaulting
    to :data:`TANDEM_FIN_BOOM_FRAC` of the stagger). ``z_root`` does NOT
    follow it: a boom runs aft from the rear wing's root, so the foot stays
    at that wing's height however long the boom is. ``boom_m = 0`` is the
    on-the-wing station and reproduces it exactly — ``0.0 + x`` is ``x``.
    '''
    f = float(TANDEM_FIN_X_FRAC)
    boom = (float(TANDEM_FIN_BOOM_FRAC) * float(dx) if boom_m is None
            else float(boom_m))
    if boom < 0.0:
        raise ValueError(
            f"fin_boom_m {boom_m} is a length AFT of the rear wing and "
            f"cannot be negative — a fin ahead of the rear wing is the "
            f"station between the two wings this layout was moved off "
            f"(fin.TANDEM_FIN_BOOM_FRAC). Use 0 for a fin ON the rear "
            f"wing's root")
    return f * float(dx) + boom, f * float(dz)


@dataclass(frozen=True)
class FinGeometry:
    """One fin, in the lattice frame (x aft, y starboard, z up).

    ``height`` is SIGNED: negative is a ventral fin, which is also how a
    hydrofoil's strut is asked for (it hangs down). ``S`` is always positive.
    """

    S: float            # [m2] area
    height: float       # [m] span along z, signed (negative = ventral)
    chord: float        # [m] mean chord
    x_qc: float         # [m] quarter-chord station — THE YAW ARM
    z_root: float       # [m] where it meets the body
    AR: float           # geometric aspect ratio of the fin alone
    V_v: float          # the volume coefficient this realises
    l_t: float          # [m] the arm it was sized against
    tc: float = FIN_TC_DEFAULT
    #: WHICH VERTICAL SURFACE THIS IS, because a boat's is not an
    #: aeroplane's. ``"fin"`` is the volume-coefficient tail :func:`size_fin`
    #: builds. ``"mast"`` is a water craft's STRUT — the surface
    #: ``hydrofoil._mast_cd0`` has always charged, spanning the submergence
    #: depth from the foil up to the free surface. They are one class
    #: because every consumer (the report, the loft, the lattice, the
    #: section stage) treats them identically; they are two names because a
    #: volume coefficient and a yaw arm mean nothing for a strut, and a
    #: read-out that printed them anyway would be describing a tail the
    #: craft has not got.
    kind: str = "fin"
    # NO ``drag_charged`` FIELD. It recorded whether this surface's parasite
    # drag had been charged to the score, because the charge was a switch
    # that defaulted OFF — an allowance worth +5.5 % L/D that a design could
    # ride without saying so. The switch is gone: a fin is charged the way a
    # tailplane is, always, so there is nothing left to record and no state
    # in which the recorded answer could be "no".

    #: THE SECTION IS SYMMETRIC, always. A fin at zero sideslip must make no
    #: side force — ``vlm.VerticalSurface`` says so by pinning its zero-lift
    #: angle to 0 — and a cambered one flies a permanent side load the trim
    #: solve has nothing to balance it with. So there is no camber field
    #: here, only a thickness, and every consumer (the drag build-up, the CAD
    #: loft, the OpenVSP script) reads that one number.
    def reynolds(self, V: float, rho: float, mu: float) -> float:
        """Re on the FIN'S OWN mean chord, which is not the wing's.

        The fin is a short-chord surface: on the shipped tail design its
        chord is 0.696 m against a 1.021 m wing mac, so the wing's Reynolds
        number is 1.47x the fin's and a section chosen there is chosen 47 %
        too high. The drag book has always used this chord
        (``tail.vtail_cd0``); stating it here is what lets a section be
        CHOSEN at it too.
        """
        return float(rho) * float(V) * float(self.chord) / float(mu)

    @property
    def x_le(self) -> float:
        """Leading edge — what a drawing is dimensioned in, and what the
        shell asks for. The quarter chord is the internal coordinate."""
        return float(self.x_qc) - 0.25 * float(self.chord)

    def as_dict(self) -> dict:
        """The block a report carries, so a rebuild READS the fin."""
        return {"S": float(self.S), "height_m": float(self.height),
                "chord_m": float(self.chord), "x_qc_m": float(self.x_qc),
                "x_le_m": float(self.x_le), "z_root_m": float(self.z_root),
                "AR": float(self.AR), "V_v": float(self.V_v),
                "l_t_m": float(self.l_t), "tc": float(self.tc),
                "kind": str(self.kind)}


def size_fin(*, b: float, S: float, l_t: float, tail_type: str = "conventional",
             V_v: float = V_V_DEFAULT, AR: float = AR_VT_DEFAULT,
             tc: float = FIN_TC_DEFAULT, z_root: float = 0.0,
             ventral: bool = False,
             min_height: float | None = None) -> FinGeometry | None:
    """The fin a wing of span ``b`` and area ``S`` at arm ``l_t`` implies.

    ``S_vt = V_v * b * S / l_t`` — the volume-coefficient law the drag book
    has always used. Height and chord follow from the aspect ratio, and the
    quarter chord sits at the arm, which is what makes ``Cn_beta``.

    ``min_height`` is the span the fin MUST AT LEAST HAVE, and it exists for
    one layout: a T-tail's tailplane sits on the fin's tip, so the fin has to
    reach it. ``tail.tail_height`` floors that height at the kernel's
    regularisation clearance ``DZ_FRAC * b``, and where that floor BINDS the
    volume coefficient alone sizes a fin too short to touch the surface it
    carries — measured at b = 7 m, S = 1.25 m2, l_t = 8 m (aspect ratio 39.2,
    inside ``api.PLANFORM_AR_LIMITS``): a 0.256 m fin under a tailplane at
    0.350 m, a 37 % gap. So the AREA stays the volume coefficient's — that is
    what the yaw stiffness and the drag are built from — and the SPAN becomes
    the layout's, with the chord and the aspect ratio REALISED from the two
    rather than assumed. Where the floor is slack this is the identity map,
    bit-for-bit, because the floored height IS ``sqrt(AR * S_vt)``.

    Returns ``None`` for a V-tail: that layout carries no separate fin, and
    a caller that gets ``None`` must not draw one.
    """
    if str(tail_type) == "v_tail":
        return None
    arm = abs(float(l_t))
    if not arm > 0.0:
        raise ValueError(
            f"a fin is sized against an ARM and {l_t!r} is not one; a layout "
            f"with no separation has no volume coefficient")
    S_vt = float(V_v) * float(b) * float(S) / arm
    if not S_vt > MIN_FIN_AREA_M2:
        raise ValueError(
            f"the volume coefficient {V_v:g} at an arm of {arm:g} m sizes a "
            f"fin of {S_vt:.3g} m2, which is not a surface")
    # CHORD FIRST, as ``sqrt(S_vt / AR)`` — the form the drag book has always
    # used, so ``vtail_cd0`` keeps computing its Reynolds number from exactly
    # the double it always did. (MEASURED: the alternative ordering moves the
    # chord by one ulp and does NOT move ``cd0_fin`` at any arm tested, so
    # this is not what protects a frozen study. What it buys is below.)
    #
    # Height then follows as ``AR * c``, which makes ``height / chord`` come
    # back as the aspect ratio EXACTLY. Sizing the height first and dividing
    # for the chord does not: the fin is then built at an aspect ratio a
    # rounding away from the one asked for, at five of six arms tested.
    c = float(np.sqrt(S_vt / float(AR)))
    h = float(AR) * c
    ar = float(AR)
    # ...unless the layout states a span the fin has to span. The comparison
    # is STRICT, so a fin already tall enough takes the branch above and its
    # chord and aspect ratio are the ones every published run was sized with.
    if min_height is not None and float(min_height) > h:
        h = float(min_height)
        c = S_vt / h
        ar = h * h / S_vt
    return FinGeometry(
        S=S_vt, height=(-h if ventral else h), chord=c,
        x_qc=float(l_t), z_root=float(z_root), AR=ar,
        V_v=float(V_v), l_t=float(l_t), tc=float(tc))


def fin_arm(*, dist_m=None, l_t=None) -> float | None:
    """The arm a fin is sized and placed at, from whichever a record carries.

    THE FIN IS AFT ON EVERY LAYOUT, INCLUDING A CANARD. A canard's TRIMMING
    surface is forward, so its station ``l_t`` is negative — and its fin is
    not: ``wingtail`` puts the solver's own vertical at ``v["dist"]``, the
    unsigned separation, and the drag book charges it there
    (``cd0_fin`` is identical on a canard and its aft-tailed twin). A
    consumer that reached for the signed station instead would place the
    surface upstream of the wing, where it DEstabilises, and report a
    different fin from the one the run paid for.

    So the rule is one line and it lives here, beside :func:`size_fin`,
    rather than at each caller: prefer the separation the breakdown states,
    and where a record carries only the signed station take its magnitude.
    Returns ``None`` when neither is a number, which is a record this
    cannot answer for rather than an arm of zero.
    """
    for value in (dist_m, l_t):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = float(value)
        if v != v:                      # NaN is not an arm
            continue
        return abs(v)
    return None


def fin_from_report(geom: dict | None) -> FinGeometry | None:
    """The fin a report STATES, or ``None`` if it states none.

    STATED IS NOT FLOWN cuts both ways: a rebuild must fly the fin the design
    was scored with where there is one, and must say it is assuming one where
    there is not. This returns the first and leaves the caller to do the
    second — see :func:`states_no_fin` for the third case, which is a report
    that says there is none ON PURPOSE.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return None
    return FinGeometry(
        S=float(blk["S"]), height=float(blk["height_m"]),
        chord=float(blk["chord_m"]), x_qc=float(blk["x_qc_m"]),
        z_root=float(blk.get("z_root_m", 0.0)),
        AR=float(blk.get("AR", AR_VT_DEFAULT)),
        V_v=float(blk.get("V_v", V_V_DEFAULT)),
        l_t=float(blk.get("l_t_m", blk["x_qc_m"])),
        tc=float(blk.get("tc", FIN_TC_DEFAULT)),
        # a report written before the strut existed has no kind and is an
        # aeroplane's fin, which is what it always was
        kind=str(blk.get("kind") or "fin"))


def states_no_fin(geom: dict | None) -> bool:
    """Does this report say, ON PURPOSE, that the design has no fin?

    Three states, and the middle one is the whole reason this exists:

    * ``geometry["fin"]`` is a **block** — this design has that fin;
    * ``geometry["fin"]`` is present and **None** — this design was asked
      and has none. A rebuild must not draw one;
    * the key is **absent** — the family does not report a fin (a hydrofoil,
      a canard, any report written before there was a fin block). The
      rebuild's own assumed fin is the honest answer there, and it says so
      in ``FlightModel.assumptions``.

    Without the middle state, "no fin" and "nothing said" are the same
    reading, so a design with the fin switched off was rebuilt with an
    INVENTED one at 12 % of span — bigger than the fin it had been charged
    for, so switching the fin off RAISED ``Cn_beta`` from 0.1114 to 0.1351.
    """
    return geom is not None and "fin" in geom and geom["fin"] is None


def has_fin(prob) -> bool:
    """Does this aircraft carry a separate vertical surface AT ALL?

    Two answers say no and they are different kinds of no:

    * a **V-TAIL** has none by construction — its canted panels ARE the
      vertical surface, and avoiding a third surface is the whole point of
      the layout. That answer is a property of the design;
    * the **USER** says so. Stage 1 asks "add a vertical stabiliser (fin and
      rudder)", and until this function existed that answer reached nothing:
      the drag book charged a fin either way, the lattice flew one, the
      weight book weighed one, the report stated one, and the shell's own
      log line said "Cn_beta is exactly zero — not small" while the rebuilt
      deck read 0.1133. The only thing the switch changed was to drop the
      ``fin_tc`` flag, so turning the fin OFF silently replaced a section
      chosen on stage 2.7 with :data:`FIN_TC_DEFAULT`.

    Asked HERE rather than at each consumer for the reason
    :func:`fin_for_layout` is one function: the surface charged, flown,
    weighed, drawn and exported has to be the same surface, and "is there
    one" is exactly as much a part of that as how big it is.

    ``prob`` is a problem OBJECT or a flags MAPPING — the shells hold the
    same answer in a dict, and a second reader there is how a card comes to
    describe an aeroplane the solver is not flying.

    **Absent means yes.** Every published run predates the question, so a
    problem that states nothing keeps its fin and reproduces bit-for-bit.
    """
    get = (prob.get if hasattr(prob, "get")
           else lambda field: getattr(prob, field, None))
    if str(get("tail_type") or "conventional") == "v_tail":
        return False
    stated = get("fin")
    return True if stated is None else bool(stated)


def fin_for_layout(*, b: float, S: float, l_t: float, tail_type: str,
                   dz: float, fin: bool = True, **kw) -> FinGeometry | None:
    """The fin of a LAYOUT: :func:`size_fin` plus where the layout puts it.

    ``size_fin`` knows the sizing law. Where the surface's foot goes, and
    how tall it must be to reach what it carries, are properties of the
    ARRANGEMENT, and they were being worked out separately by every caller
    that wanted a fin — which is how a T-tail ended up with its fin above
    its own tailplane in one place and below it in another. One rule, here:

    * every layout but the T-tail hangs its tailplane off the body and
      stands the fin beside it, so the fin's root is the tailplane's height
      ``dz`` and its span is the sizing law's;
    * a T-TAIL is the other arrangement — the tailplane sits ON the fin's
      tip — so the fin's root is the BODY (0.0) and it must be at least
      ``dz`` tall to reach the surface it carries.

    Returns ``None`` for a V-tail, which has no separate fin at all — and
    for ``fin=False``, which is the same answer arrived at the other way
    (see :func:`has_fin`). Every caller already handles the V-tail's None,
    so routing the user's answer to the same place is what makes "no fin"
    mean no fin in the drag book, the lattice, the weight book and the
    report at once, instead of in none of them.
    """
    t = str(tail_type)
    if not fin:
        return None
    top = t == "t_tail"
    return size_fin(b=b, S=S, l_t=l_t, tail_type=t,
                    z_root=(0.0 if top else float(dz)),
                    min_height=(float(dz) if top else None), **kw)


def mast(*, depth: float, chord: float, tc: float = FIN_TC_DEFAULT,
         x_qc: float = 0.0) -> "FinGeometry | None":
    """A WATER craft's vertical surface: the STRUT that carries the foil.

    NOT a volume-coefficient tail, and this is the whole reason it is a
    separate constructor. ``size_fin`` sizes a surface from a wing span, a
    wing area and a tail arm; a strut has none of those. Its size is set by
    where the foil is: it spans the SUBMERGENCE DEPTH, from the foil up to
    the free surface, at the mast chord — which is exactly the wetted area
    ``hydrofoil._mast_cd0`` has always charged (``2 x depth x c_mast``).

    So this reports the surface the water run ALREADY PAYS FOR. Before it,
    the drag book charged a strut, the report stated no vertical surface at
    all, and ``flightmodel`` therefore invented a third one — 12 % of span
    (0.144 m against a real 0.3-0.6 m mast), placed at 0.45 b, carrying
    100 % of a hydrofoil's ``Cn_beta``. Three surfaces, one craft.

    The foot is the FOIL (``z_root = 0``) and the height is positive: the
    strut goes UP to the waterline and stops there, so no panel is drawn
    above the free surface the image model represents, and none hangs below
    the foil.

    ``x_qc`` IS THE STATION, and on the elevator family it is no longer
    zero: a real mast is bolted to the fuselage BETWEEN the front wing and
    the stabiliser (``hydrotail.X_MAST_FRAC``), not growing out of the front
    wing, so the surface that makes a foiling craft's yaw stiffness finally
    has somewhere to stand. A single foil has no fuselage and keeps 0.

    ``V_v`` and ``l_t`` are zero and mean it — and they are NOT the station.
    A strut is not sized by a volume coefficient and was not sized against
    an arm, so both are the honest zero; writing the station into ``l_t``
    would give one field two meanings and put a volume-coefficient reading
    on a surface that has none. What the yaw arm actually is depends on the
    CG, which is a property of the design and not of the surface: the
    consumer takes ``x_qc - x_cg`` (``hydrotail`` reports both).
    """
    h, c = float(depth), float(chord)
    if not (h > 0.0 and c > 0.0):
        return None
    return FinGeometry(S=h * c, height=h, chord=c, x_qc=float(x_qc),
                       z_root=0.0, AR=(h / c), V_v=0.0, l_t=0.0,
                       tc=float(tc), kind="mast")


def fin_shape_kwargs(prob, ar_key: str = "AR") -> dict:
    """``{V_v, AR, tc}`` a problem STATES, as kwargs for the fin law.

    Empty where nothing is stated, so the sizing law's own published
    defaults apply and every stored run is bit-for-bit. ONE reader, because
    the three numbers have to reach the drag book, the lateral deck, the
    report and the exporter identically — a fin drawn to one shape and
    charged at another is the defect this module exists to have fixed.

    ``ar_key`` is the aspect ratio's name in the callee: :func:`size_fin`
    spells it ``AR`` and ``tail.vtail_cd0`` spells it ``AR_vt``. The
    translation lives here, once, rather than at each call site.

    **DO NOT CALL THIS DIRECTLY** — call :func:`fin_law_kwargs` or
    :func:`fin_drag_kwargs`, which choose the spelling for you. A call site
    that picks the string itself is exactly how the two got SWAPPED: with a
    stated ``fin_ar``, ``wingtail`` passed ``AR`` to ``vtail_cd0`` (which
    has no such parameter) and ``AR_vt`` to ``size_fin`` (which has no such
    parameter either), so every lattice-backed wing+tail family raised
    ``TypeError`` on a number the shell offered a field for. The two
    wrappers below are what makes that unexpressible; this stays because it
    is the one reader of the three fields.

    ``prob`` is a problem OBJECT or a flags MAPPING — the shell holds the
    same three numbers in a dict, and a second reader there is how a card
    ends up describing a fin the solver is not flying.
    """
    get = (prob.get if hasattr(prob, "get")
           else lambda field: getattr(prob, field, None))
    out = {}
    for field, key in (("fin_volume_coeff", "V_v"), ("fin_ar", ar_key),
                       ("fin_tc", "tc")):
        value = get(field)
        if value is not None:
            out[key] = float(value)
    return out


def fin_law_kwargs(prob) -> dict:
    """The stated shape, spelled for :func:`size_fin` / :func:`fin_for_layout`."""
    return fin_shape_kwargs(prob, ar_key="AR")


def fin_drag_kwargs(prob) -> dict:
    """The stated shape, spelled for ``tail.vtail_cd0``."""
    return fin_shape_kwargs(prob, ar_key="AR_vt")
