"""The RIG's loads on a foiling craft — a couple, a lift relief, a side force.

Why this module exists
----------------------
``hydrofoil.py`` and ``hydrotail.py`` design an APPENDAGE: a foil, a mast and
(in the elevator family) a stabiliser, trimmed to carry a weight at a speed.
Nothing in that picture is what actually drives a windfoil, a wingfoil or a
kitefoil. The rig is — and it applies its force at a centre of effort one to
three metres ABOVE the water, while the resistance it balances acts a few
tenths of a metre BELOW it. In steady flight those two horizontal forces are
equal and opposite, so they form a COUPLE, and that couple is the dominant
longitudinal moment on the craft. Until this module existed the engine could
not see it at all: the only pitching moments in the balance were the two
surfaces' lift-times-arm and the sections' own ``cm_ac``.

HOW BIG, from measurements rather than from a feel for it. ~399 N.m for a
windfoil: 145 N of drive at a 2.75 m lever, from drake_blog's measured drive
force and zhang2025's measured rig centre-of-effort height Z_r = 2.0 m. The
project's own literature review (``LITERATURE_REVIEW_FOILINGBO.md``, row 2)
re-derived that a second and third way — as W z / (L/D) on martinez2026's
MEASURED whole-appendage L/D of 6.1-7.9 — and got 359-464 N.m, i.e. the same
number out of independent measurements. Referred to this engine's own CG,
that couple is worth a ~312 N stabiliser DOWNLOAD and a ~30 % overload of the
main foil (1342 N against a 1030 N craft). It decides the SIGN of the
stabiliser's load, which decides which way up its section is mounted, which
is what gets drawn and exported.

THE VERTICAL SHARE. A kite does not only drive: its line tension acts along
the tether at 14-33 deg above the horizontal, so a large part of it is
straight lift. vlugt2009 Table 6 gives 180-387 N of vertical relief against
a W_rider of 850 N across five measured cases — 21-46 %, mean 35 %. A foil
sized to the full weight is therefore sized to a load it does not carry. The
windsurf case is smaller but the same sign (drake_blog measures 73 N, ~7 %,
in a non-foiling run), and the wing-foil figure in circulation (12-23 %) is
derived rather than measured, so it is quoted here as what it is.

THE SIDE FORCE IS FLOWN BY THE MAST. It was carried as a value that nothing
resisted; ``hydrotail`` (``strut_model``, on by default) now hands it to the
surface that carries it on a real craft, and ``hydrofoil.strut_report``
returns the leeway angle the strut flies to make it and the induced drag
that costs. What is still true, and is the reason this module says so much
about the pitch plane below: a side force has no PITCHING moment in a
symmetric mirrored half-model, so it enters the drag book and nothing else.

It is also ANSWERED FOR in roll: :meth:`RigLoads.roll_moment_nm` states the
heeling couple it makes and :meth:`RigLoads.righting_lever_required_m` states
the lateral CG offset that would hold the craft up against it. Neither closes
the balance — the rider does, and this package has no rider — but they turn
"recorded and unused" into "recorded, priced, and handed to the one row that
can spend it":
``heel_deg``, which is the only quantity anywhere in these solvers that reads
how WIDE the foil is (``hydrofoil.heeled_z``, ``hydrofoil.immersion_margin``).
Before that row existed a searched span had no interior answer at all, because
nothing in the model paid for span.

Still true of the LONGITUDINAL solve: The horizontal
line-tension component is 506-723 N for the kite cases and 274 N (26 % of
weight) for the windsurf one, and it is the reason a real mast is a symmetric
NACA 0009 "so that it can generate lift equally in both directions"
[martinez2026, retrieved verbatim]. But every solver downstream of this
module is LONGITUDINAL: one symmetric mirrored half-model, a Trefftz plane
and a 2x2 trim in (alpha, i_t). A side force has no pitching moment in that
plane, so putting it into ``pitching_moment_nm`` would be inventing a term.
It is recorded because a value object that silently dropped a stated load
would be worse than one that says out loud where the load went. Where the
load goes is no longer a next step: it goes into the mast, and the mast pays
for it in drag.


THE SIGN CONVENTION, stated once and asserted in the tests
----------------------------------------------------------
The frame is the solvers' own: **x AFT positive** from the main foil's
quarter chord, **z UP positive** with the free surface at ``z = +depth``
above the foil plane, and the pitching moment **NOSE-UP positive**
(``vlm.VLM._cm``). For a force ``(F_x, F_z)`` applied at ``(x, z)``, the
nose-up moment about a CG at ``(x_cg, z_cg)`` is

    M = (z - z_cg) F_x - (x - x_cg) F_z

which reduces to ``(x_cg - x) F_z`` for a pure lift — exactly what
``vlm.VLM._cm`` sums, with lift AHEAD of the CG pitching nose-up.

The rig drives the craft FORWARD, i.e. in -x, so a positive ``thrust_n`` is
``F_x = -T`` applied at ``z = depth + z_ce_m``; the resistance it balances is
``F_x = +T`` at the appendage, ``z ~ 0``. The pair is a couple of magnitude
``T (depth + z_ce_m)`` and its sign is

    M_drive = - T (depth_m + z_ce_m)          [N.m, NOSE-UP POSITIVE]

**A rig driving the craft forward from a centre of effort above the water
pitches the craft NOSE-DOWN.** That is the statement every rider already
knows — sheet in hard and the bow goes down — and it is what the stabiliser
has to hold up, so the couple pushes ``i_t`` and the stabiliser's load
DOWNWARDS (test ``(g)``).

Being a couple is not incidental, it is the whole reason this term is
cheap: a couple is independent of the point it is taken about, so it does
not need a vertical CG station — which this engine does not have and would
have to invent. The lever is measured from the centre of effort down to the
FOIL, because the foil is where the force it balances acts.

The vertical share is NOT a couple: it is a force at a station, so it does
need a station, and its moment about the CG is ``(x_cg - x_ce) F_z``. It is
zero when the rig's centre of effort sits over the CG, which is roughly true
of a windsurf rig and roughly false of a kite.


THE DRIVE FORCE, and the loop this module refuses to close
----------------------------------------------------------
In steady flight the drive equals the total resistance, so it can be had
from the craft's lift-to-drag ratio:

    T = L_required / (L/D)_craft

and ``craft_lod`` is therefore a STATED input with a default of
:data:`CRAFT_LOD_DEFAULT` = 7.0.

**It must not be closed on this engine's own answer.** ``hydrotail`` reports
L/D 19-25 for its appendage; the only two MEASURED whole-appendage values in
the literature are 6.1 (alpha 0) and 7.9 (alpha 5) from a towing tank
[martinez2026] and 6.5 from CFD on a production kitefoil [backas2016]. The
gap is not a bug in either: the engine models a wing and a mast, and the
measurements are of a wing, a mast, a fuselage, the junctions, the spray and
the waves. Feeding the engine's own number back in would divide the drive by
~22 instead of ~7 and understate the dominant longitudinal moment by a
factor of 3-4 — an error that would grow as the optimiser improved the
appendage, i.e. exactly the direction that makes an optimum self-consistent
and wrong. So the loop stays open, the stated value is what divides, and
``evaluate_hydrofoil_tail`` reports the engine's own appendage L/D BESIDE
the stated craft L/D so the gap is on the page rather than in a docstring.

7.0 is a DEFAULT chosen inside the measured band, not itself a measurement:
it sits between martinez2026's two angles and just above backas2016's. Like
every calibration in this package it is widenable in both directions — a
tow-foil behind a boat has no rig at all, and a race kitefoil at 17 m/s is a
different craft from a tank model at 3.5 m/s.


What a caller does with this
----------------------------
:class:`RigLoads` is a value object and computes no aerodynamics. It answers
three questions — how much lift is left for the foil, how hard the rig
pushes, and what couple that makes — and ``hydrotail`` divides the couple by
``q S mac`` and hands it to the ONE channel that can carry it:
``vlm.VLM.solve_trim_moment(cm_ac=...)``, which enters the CONSTANT of the
affine trim map and therefore leaves the Jacobian, ``neutral_point()`` and
the static margin untouched AT FIXED GEOMETRY, BY CONSTRUCTION.

"At fixed geometry" is the whole of the claim, and it is narrower than it
looks. ``cm_ac`` never enters the influence matrix, so for one fixed
aeroplane the neutral point is bit-identical with the couple and without it
— pin the stabiliser's mounting and the two agree to the last bit
(``test_the_couple_leaves_the_neutral_point_and_the_margin_alone`` asserts
``==``, not ``approx``). END TO END it is FALSE, because the couple flips the
SIGN of the stabiliser's load and this engine has two rules that follow that
sign and are geometry: the section MOUNTING (``tail.orient_section`` mirrors
the section and bakes the mirrored ``alpha_L0`` into the panel normals) and,
on the families that ask for it, the stabiliser's TIP DEVICE
(``hydrotail``'s ``tail_winglet_follow``, which swaps the device's side and
re-flies before ``neutral_point()`` is read). Measured with the driven rig
at the published box centre: +0.11 % MAC from the mounting alone
(``hydrofoil + elevator``) and +0.88 % MAC once the device follows too
(``hydrofoil + elevator [designed elevator + tip device]``). Those are real
changes of shape, correctly reflected in x_np; what this module guarantees
is only that the MOMENT is not one of them. A claim that is true with a
named exception is worth more than a clean one that is wrong.

THE TRAP, measured and recorded here so nobody re-discovers it. Applying the
couple by SHIFTING the CG by ``Delta = M / L`` gives bit-identical alpha,
i_t and circulations (agreement 1.4e-17 to 2.2e-16 over Delta = +0.037 to
-0.43 m), which makes it a perfect test oracle and a disastrous
implementation. ``SM = (x_np - x_cg)/mac`` reads the SHIFTED station, so the
stability constraint becomes a margin about a CG the craft does not have:
x_cg 0.15 -> -0.28 m takes SM from -0.2276 to +3.2837 and the constraint
from -0.308 to +3.204, i.e. satisfied by every design in the box. A
constrained run would then optimise against one real constraint while
advertising two. The couple goes through ``cm_ac``; the CG shift lives in
``tests/test_rig_couple_is_a_moment_not_a_cg_move.py`` and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The whole-craft lift-to-drag ratio the DRIVE FORCE is derived from when
#: the caller states no thrust. A default inside the measured band, never a
#: measurement in its own right: the two whole-appendage values that exist
#: are 6.1 / 7.9 [martinez2026, towing tank, alpha 0 / alpha 5] and 6.5
#: [backas2016, CFD]. It is deliberately NOT this engine's own
#: reported L/D (19-25) — see the module docstring for why closing that loop
#: understates the couple by 3-4x and gets worse as the design improves.
CRAFT_LOD_DEFAULT = 7.0

#: NO RIG — the published state of every water family in this package, and
#: the value of ``HydrofoilTailProblem.rig`` unless a caller says otherwise.
#:
#: It is spelled ``None`` rather than ``RigLoads(0, 0, ...)`` on purpose. A
#: zero-valued rig would still be a rig: it would put a ``"rig"`` block in
#: every breakdown, a stated craft L/D on every card and an extra term in
#: ``cm_ac`` whose value happens to be 0.0 — and "happens to be zero" is the
#: shape of bug this repo has recorded twice (a criterion the box cannot
#: vary; a margin that is scalar on success). ``None`` means the question was
#: never asked, so every frozen water study reproduces bit-for-bit, and the
#: ``is None`` test is the one branch the whole feature hangs off.
NO_RIG = None


@dataclass(frozen=True)
class RigLoads:
    """What the rig does to the craft, as stated values.

    Frozen because it is a specification and not a state: the trim solve
    reads it many times per evaluation (once per candidate, twice when the
    stabiliser's tip device follows its load) and a value that could change
    underneath those reads would make an evaluation depend on its own
    history.

    Every field is in SI and in the solvers' frame (module docstring):

    ``z_ce_m``
        Centre-of-effort height ABOVE THE WATERLINE [m]. Required, and the
        only field with no default, because it is the lever: the couple is
        thrust times ``(depth + z_ce_m)``, and a rig height guessed on the
        user's behalf would silently set the size of the single largest
        pitching moment on the craft. Measured values: 2.0 m for a windsurf
        rig [zhang2025], a tether direction rather than a height for a kite,
        and 0.0 for a tow rope at water level — a factor of infinity apart,
        which is why it is not defaulted.
    ``thrust_n``
        The DRIVE force [N], positive FORWARD (in -x). ``None`` — the
        default — means "derive it", ``L_required / craft_lod``, which is
        the steady-flight closure. A stated value wins, because a rider with
        a load cell has better information than a ratio.
    ``side_n``
        The lateral force [N]. FLOWN BY THE STRUT where the family models one
        (``hydrotail.strut_model`` -> ``hydrofoil.strut_report``): it sets
        the mast's leeway angle and the induced drag of carrying it. It still
        has no pitching moment in a longitudinal solver and this module does
        not invent one (module docstring).
    ``vertical_n``
        The rig's VERTICAL share [N], positive UP, i.e. relieving the foil.
        21-46 % of rider weight for a kite [vlugt2009 Table 6, five measured
        cases]. Negative is allowed and means a rig pressing the craft down,
        which is a real state (a windsurf rig sheeted in hard downwind) and
        makes the foil work harder rather than less.
    ``x_ce_m``
        Where the vertical share acts, [m] AFT of the main foil's quarter
        chord — the same origin ``x_cg`` is measured from. Only the vertical
        share uses it; the drive is a couple and does not care.
    ``craft_lod``
        The whole-craft L/D that divides the couple. See
        :data:`CRAFT_LOD_DEFAULT`.
    """

    z_ce_m: float
    thrust_n: float | None = None
    side_n: float = 0.0
    vertical_n: float = 0.0
    x_ce_m: float = 0.0
    craft_lod: float = CRAFT_LOD_DEFAULT

    def __post_init__(self) -> None:
        for name in ("z_ce_m", "side_n", "vertical_n", "x_ce_m",
                     "craft_lod"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(
                    f"RigLoads.{name} = {getattr(self, name)!r} is not a "
                    f"finite number")
        if self.thrust_n is not None and not math.isfinite(
                float(self.thrust_n)):
            raise ValueError(
                f"RigLoads.thrust_n = {self.thrust_n!r} is not a finite "
                f"number (use None to derive it from craft_lod)")
        if float(self.z_ce_m) < 0.0:
            raise ValueError(
                f"RigLoads.z_ce_m = {self.z_ce_m!r} m is negative, and it is "
                f"a HEIGHT ABOVE THE WATERLINE: a rig whose centre of effort "
                f"is under water is not a rig. A tow rope at the surface is "
                f"0.0; the lever it makes is still (depth + z_ce_m), because "
                f"the foil is below the water. This is a frame statement, "
                f"not a ceiling — there is no upper bound.")
        if not (float(self.craft_lod) > 0.0):
            raise ValueError(
                f"RigLoads.craft_lod = {self.craft_lod!r} must be > 0: it "
                f"divides the lift to give the drive force, so zero is an "
                f"infinite rig and a negative value is a craft dragged "
                f"backwards by its own lift. No upper bound is imposed, but "
                f"see the module docstring before passing this engine's own "
                f"appendage L/D.")

    # ------------------------------------------------------------------
    # what the craft is left holding

    def lift_required_n(self, weight_n: float) -> float:
        """The lift the FOIL must make [N]: the weight less the rig's share.

        THE arithmetic of the relief, in one line and one place. Callers do
        not subtract ``vertical_n`` themselves — ``HydrofoilTailProblem
        .L_required`` is the single author on the problem side and it calls
        this, so ``CL_target`` and everything that reads it cannot disagree
        about how heavy the craft is.
        """
        return float(weight_n) - float(self.vertical_n)

    def validate_for_weight(self, weight_n: float) -> None:
        """Refuse a rig that leaves the foil nothing (or less) to lift.

        A vertical share at or above the craft's weight is not a hard design,
        it is a DIFFERENT design: the craft is hanging from its rig and the
        foil is a rudder. ``CL_target`` would be zero or negative, the trim
        solve would answer a question the objective does not score, and the
        cavitation margin — the constraint that is supposed to refuse an
        overloaded foil — would report a comfortable margin on a foil that
        is not lifting.

        This is a refusal of nonsense in the same register as
        ``api._weight_kwargs``'s, and NOT a cap: a rig carrying 95 % of the
        weight is accepted, priced and flown.
        """
        left = self.lift_required_n(weight_n)
        if not (left > 0.0):
            raise ValueError(
                f"RigLoads.vertical_n = {self.vertical_n!r} N is not less "
                f"than the craft's weight {float(weight_n)!r} N, so the foil "
                f"is asked to carry {left!r} N. The rig would be holding the "
                f"whole craft up and the foil would not be a lifting surface "
                f"at all. Anything strictly below the weight is accepted.")

    def drive_n(self, weight_n: float) -> float:
        """The DRIVE force [N], positive forward — stated, or closed.

        Stated wins. Otherwise the steady-flight closure: drive equals total
        resistance equals ``L_required / craft_lod``, with ``craft_lod`` the
        WHOLE CRAFT's ratio and never the engine's own appendage figure
        (module docstring — that substitution understates the couple 3-4x).
        """
        if self.thrust_n is not None:
            return float(self.thrust_n)
        return self.lift_required_n(weight_n) / float(self.craft_lod)

    # ------------------------------------------------------------------
    # the moment

    def lever_m(self, depth_m: float) -> float:
        """Vertical distance from the centre of effort down to the FOIL [m].

        ``depth_m`` (foil below the free surface) plus ``z_ce_m`` (centre of
        effort above it). It is a function of a DESIGN VARIABLE, which is the
        point: flying deeper lengthens the couple's lever, so the rig prices
        depth in a way the mast's drag alone never did.
        """
        return float(depth_m) + float(self.z_ce_m)

    def pitching_moment_nm(self, *, weight_n: float, depth_m: float,
                           x_cg_m: float) -> float:
        """The rig's pitching moment [N.m], NOSE-UP POSITIVE.

        Two terms, derived in the module docstring:

            M = - T (depth_m + z_ce_m)       the drive/resistance COUPLE
                + (x_cg_m - x_ce_m) Z        the vertical share's moment

        with ``T = drive_n(weight_n)`` and ``Z = vertical_n``. The first is
        reference-independent (it is a couple), the second is not and is
        therefore taken about the CG the trim solve is closed on — the same
        ``x_cg`` that appears in ``vlm.VLM._cm``.

        ``side_n`` does not appear: it has no component in the pitch plane
        (module docstring).

        Both terms are constant in ``(alpha, i_t)``, which is what lets the
        whole thing ride in on ``cm_ac`` and leave the neutral point and the
        static margin exactly where they were — for the SAME aeroplane. It
        does not stop the caller's own sign-following rules (the mounting,
        the tip device) from building a different one; the module docstring
        names both paths and prices them.
        """
        drive = self.drive_n(weight_n)
        return float(-drive * self.lever_m(depth_m)
                     + (float(x_cg_m) - float(self.x_ce_m))
                     * float(self.vertical_n))

    # ------------------------------------------------------------------
    # the ROLL the side force makes, and what would have to balance it

    def roll_moment_nm(self, depth_m: float) -> float:
        """The rig's HEELING moment [N.m] — a couple, like the drive's.

        ``side_n`` times the same lever the drive uses. In steady straight
        flight the appendage makes a side force equal and opposite to the
        rig's (that is what a symmetric mast section is FOR — martinez2026's
        NACA 0009 "so that it can generate lift equally in both
        directions"), so the pair is a couple and needs no reference point,
        exactly as the drive/resistance pair does.

        Positive means heeling to LEEWARD, the direction the rig pulls.
        """
        return float(self.side_n) * self.lever_m(depth_m)

    def righting_lever_required_m(self, *, weight_n: float,
                                  depth_m: float) -> float:
        """The lateral CG offset [m] that would hold the craft upright.

        ``roll_moment_nm / weight_n``, and it is REPORTED rather than
        applied. This is the closure the package deliberately does not
        make: what balances a foiling craft's roll couple is the RIDER, a
        mass on a lever they choose from one instant to the next, and a heel
        angle derived from the side force alone would be a rider model
        wearing a physics hat. It would also be the most consequential
        number in the run — heel is the only term that prices SPAN
        (``hydrofoil.heeled_z``), so a guessed heel is a guessed span limit.

        So the number is put on the page and the ANGLE is asked for
        (``heel_deg`` on all three water problems). A rider who cannot reach
        this lever is not sailing upright, and the difference is what they
        should type.
        """
        w = float(weight_n)
        if not (abs(w) > 0.0):
            return float("nan")
        return self.roll_moment_nm(depth_m) / w

    def equivalent_cg_shift_m(self, *, weight_n: float, depth_m: float,
                              x_cg_m: float) -> float:
        """``M / L`` [m] — the CG move that would produce the SAME trim.

        A DIAGNOSTIC, never a station. It is exactly the substitution the
        module docstring refuses: shifting the CG by this much reproduces
        alpha, i_t and every circulation to machine precision, and moves the
        static margin by ``-Delta/mac`` about a CG the craft does not have.
        It is reported because its SIZE is the honest measure of how large
        the couple is in this craft's own units — at the published centre it
        is a third of a metre against a 0.1225 m mean chord, i.e. ~3 MAC —
        and because a number on the card is harder to re-invent than a
        warning in a docstring.
        """
        lift = self.lift_required_n(weight_n)
        if not (abs(lift) > 0.0):
            return float("nan")
        return self.pitching_moment_nm(weight_n=weight_n, depth_m=depth_m,
                                       x_cg_m=x_cg_m) / lift
