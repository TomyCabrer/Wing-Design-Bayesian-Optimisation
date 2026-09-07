"""Two-element race-car rear wing: the same wing as ``carwing.py``, flying a
SLOTTED section instead of a single one.

What is new, and what deliberately is not
-----------------------------------------
Everything outside the section is ``carwing.py``, imported and reused rather
than restated: the mirrored frame (model +z IS the car's downforce), the
imaged nonplanar VLM with the track as a rigid wall above the wing, the
endplates as tip devices of free height in metres, the mount layouts and
their beam models (:func:`carwing.bending_moment`,
:func:`carwing.deflection_index`), the strut and junction charges, the
objective set, the budget switches and the failure contract. This module is
to ``carwing.py`` what ``tandemvlm.py`` is to ``tandem.py``: a sibling that
adds ONE freedom and disturbs nothing else.

The freedom is the SECTION. Four rows join the design vector::

    x = [taper, twist_root_deg, twist_tip_deg, alpha_deg,
         endplate_h_m, ride_height_m,
         flap_chord_frac, flap_deflection_deg,
         slot_gap_frac, slot_overlap_frac,
         (S_m2,) b_m]                       (+ chord coefficients)

and they sit in the FAMILY block, AHEAD of the size block, so
:func:`carwing.span_from_x` and :func:`carwing.s_from_x` — which count
BACKWARDS from the chord coefficients — read exactly the same slots they
read for the single-element family. Nothing about the size block moved.
(Gated by ``test_the_size_block_is_still_read_from_the_end``, on a vector
whose every entry is different, because an index read from the wrong end is
a mistake this repository has made before and it hides behind a symmetric
test vector.)

The section is resolved in 2-D, not in the lattice
--------------------------------------------------
A slot is a CHORDWISE interaction between two bodies at one spanwise
station: the flap sits in the main element's downwash at a gap of a few per
cent of chord, and its own upwash feeds back. The lattice carries one
chordwise panel per strip and cannot represent that at all. So the two
elements are solved in ``panel2d.py`` (Hess & Smith, two bodies, one dense
system) and enter the lattice the way every section in this package enters
it: as a POLAR (:class:`CascadePolar`). The wing is ONE lifting surface
whose section happens to have two parts.

That architecture answers the endplate question by construction.

ONE ENDPLATE, NOT TWO — and why
-------------------------------
A car wing's endplate is one physical sheet that spans BOTH elements, so
modelling it once is the geometry, not an approximation. The plate is
carried by the lifting surface, which is the whole assembly; the flap is
not a lattice surface and has no tip device of its own.

That is also the only thing this solver could honestly do. MEASURED here
(``vlm.VLM`` with a ``SecondWing`` at the same span, both carrying a
90-degree tip device, over the track image at ride 0.2 m): at a streamwise
stagger of 0.30 m the pair solves (CL 1.1910, CDi 0.028761); brought to
ZERO stagger — two vertical sheets at the same spanwise station, which is
what "an endplate on the flap as well" would draw — the influence matrix is
EXACTLY SINGULAR (scipy reports "Diagonal number 64 is exactly zero") and
CL and CDi come back NaN. The second device's control points sit on the
first's own bound and trailing filaments. This is the same double-counted
sheet ``tandemvlm._device_clearance`` exists to refuse, and it is why a
second plate is not offered as an option to be used carefully: at the
station it would occupy, there is no answer to use.

The section model
-----------------
Each element has its OWN section, or the same one if the caller says nothing.
``section_coords`` (or ``tc``) is the MAIN element's; ``flap_coords`` or
``flap_tc`` gives the flap a shape of its own, and with both unset the flap is
the main element's shape scaled to the flap chord — which is the only thing
this family could say before, and that path is unchanged to the bit.

Everything the flap can now differ in, it differs in ALL THE WAY: its own
measured thickness (``tc_flap``), its own Reynolds number (it always had one),
its own isolated Cp_min -> alpha map for the drag bridge, its own wide-alpha
drag table, and its own suction-peak ceiling. Judging a designed flap against
the main element's ceiling would refuse or admit it for a shape it does not
have, so ``stall_margin`` divides each element's peak by its own.

THE SUBSTITUTION THAT GETS HARDER, AND WHAT WAS DECIDED. The ceiling is built
from the measured stall of a NACA 24XX at the element's thickness. With one
NACA section that is a Reynolds-number substitution; with a DESIGNED flap it
is also a shape substitution, and the shape is the thing being designed.
Three responses were open — refuse a designed section, bound the thickness
range, or flag it harder. This family does the last two and declines the
first:

  * REFUSING would ban a whole class of design because of a table's limits,
    which is what `a-calibration-is-a-default-not-a-ban` exists to prevent.
  * THE THICKNESS RANGE IS BOUNDED, and now per element:
    :func:`suction_peak_ceiling` refuses outside the range the shipped
    wide-alpha bank RESOLVES a stall over (t/c in [0.09, 0.18]; 0.06 is
    censored), and each element is checked at its OWN measured thickness.
  * THE FLAG IS LOUDER: ``ceiling_source`` says which of the two paths a
    number came from, and ``ceiling_section`` names both elements and marks
    the substituted one ("main NACA2412 / flap NACA2409 SUBSTITUTED for a
    designed section").

What CLOSES the substitution rather than labelling it is a stall measured on
the shape being flown, at the Reynolds number it is flown at:
``section_stall.measured_ceilings`` runs XFOIL on each element's own
coordinates and hands the pair back through ``section_ceilings``. It
reproduces the substitution EXACTLY where the two should agree — a NACA 2412
at Re 1e6 on this module's own 120-node loop gives Cp_min -13.880825 either
way, from a stall angle of 16.0 deg either way — and it is expensive (one
XFOIL process per element), so it belongs on the designs a study VERIFIES and
not in an optimiser's inner loop. It also puts a number on the Reynolds-number
direction this docstring used to state without quantifying: the same NACA 2412
ceiling is -13.881 at Re 1e6, -12.323 at 3e5 and -10.133 at 1.5e5, so at the
~2e5 a small car wing's flap actually reaches, the Re-1e6 ceiling is about
30 % too deep — optimistic, and not by a little.

THE SLOT HAS THREE LENGTHS AND THEY ARE THREE QUANTITIES.
``slot_gap_frac`` is the PLACEMENT row (an input); ``slot_width_frac`` is the
minimum surface-to-surface separation anywhere (the RESOLUTION question);
``slot_gap_te_frac`` (:func:`slot_gap_te`) is the distance from the main
element's trailing edge to the flap's surface, which is what a wind-tunnel
report means by *gap*. MEASURED at the box centre: 0.031000, 0.023963 and
0.024177 — all different. A bound taken from a paper is stated against the
third, which is also why the non-merging floor
(:data:`SLOT_GAP_TE_NON_MERGING_FRAC`) is stated against it and not against
the row an optimiser moves.

Both elements are the SAME base section (NACA 24XX at ``tc``, or coordinates
the caller supplies) scaled to their own chords WHEN THE FLAP HAS NONE OF ITS
OWN:

    c_main = (1 - flap_chord_frac) c_ref,   c_flap = flap_chord_frac c_ref

so the element chords SUM to the reference chord: ``c_ref`` is the stowed
chord, and it is the chord the wing's planform area is built from. Every
coefficient below is on ``c_ref`` unless it is explicitly called an element's
OWN (``cl_main`` / ``cl_flap``), and the per-element shares reconcile the two
(``lift_share_main`` + ``lift_share_flap`` = 1) — the convention
``tandemvlm``/``wingtail``/``hydrotail`` already use for two surfaces.

The flap's leading edge is placed at

    x = x_TE(main) - slot_overlap_frac * c_ref
    y = y_TE(main) - slot_gap_frac     * c_ref

and the flap is then rotated about that placed leading edge by
``-flap_deflection_deg`` (``panel2d``'s counter-clockwise-positive frame, so
a negative rotation puts the trailing edge DOWN, which in the mirrored frame
is MORE downforce). Gap and overlap therefore POSITION the flap; the slot
that results is a consequence of the rotation as well, so the true minimum
surface-to-surface distance is MEASURED from the built geometry and reported
as ``slot_width_frac``. It is smaller than the placement gap — MEASURED at
the box centre, the 0.031 c gap row leaves a 0.0240 c slot — and it is the
width the refusals below are stated against, because it is the width the
flow sees. It also OPENS as the flap deflects (0.0152 / 0.0216 / 0.0251 c at
0 / 15 / 30 deg on a 0.030 c placement gap), because the flap turns about its
own leading edge and its upper surface swings away from the main element's
trailing edge — the opposite of what "more flap, tighter slot" suggests,
which is the whole reason it is measured.

Two solves, every incidence — DERIVED
--------------------------------------
The panel system is LINEAR in the freestream vector: the tangency and Kutta
right-hand sides are both ``-(something) . Vinf`` and the matrix does not
depend on alpha at all. So with ``S_0`` the solution for ``Vinf = V(1, 0)``
and ``S_90`` that for ``Vinf = V(0, 1)``,

    v_t(alpha) = cos(alpha) v_t,0 + sin(alpha) v_t,90        EXACTLY,

and every surface pressure, force and suction peak at every incidence
follows from two solves and some arithmetic on vectors. (Cp is QUADRATIC in
v_t, so the section's cl is not linear in alpha and this is not a linear-
polar assumption — it is an exact reconstruction. MEASURED against
``panel2d.solve`` run independently at -4, 0, 4, 8 and 12 degrees: the
largest disagreement in the assembly cl is 4.4e-16, in cm 1.1e-16 and in an
element's Cp_min 1.1e-14, i.e. round-off. Gated by
``test_two_basis_solves_reproduce_a_direct_solve``.) The whole polar costs
two solves instead of one per incidence, which is the difference between a
5.8 ms section and a 350 ms one.

Where the profile drag comes from — the one real modelling choice
-----------------------------------------------------------------
``panel2d`` is inviscid, so it cannot price profile drag, and the obvious
bridge does not work: the main element of a two-element section routinely
carries cl ~ 2.8 ON ITS OWN CHORD, which is far outside any single-element
polar's alpha range (NACA 2412 at Re 1e6 peaks at cl 1.53). Looking its drag
up "at the angle that would give the same lift" asks the table a question it
has no data for, and clamping the answer would understate the drag of
exactly the designs this family exists to find.

So the bridge is the SUCTION PEAK, not the lift:

    each element pays the profile drag of the ISOLATED section at the
    incidence that gives the isolated section the SAME inviscid Cp_min,
    at the element's own Reynolds number, on its own chord;
    cd_section = sum_i (c_i / c_ref) cd_i.

Why that variable. The peak suction sets the adverse pressure gradient the
boundary layer then has to survive, which is what the viscous drag of an
attached section is a function of; the lift is not, once a slot is feeding
the flap. It also has the property a reduced-order model needs most: it is
EXACT in the single-element limit. Take the flap 500 chords away and the map
Cp_min -> alpha becomes the identity on the remaining element (MEASURED: the
equivalent incidence comes back as the element's own incidence to within
2e-3 deg over alpha in [-4, +12]), so the MAIN ELEMENT is charged exactly
the isolated section's cd, at its own Reynolds number, on its own chord.
That is what the limit says, and it is what
``test_a_lonely_element_is_priced_exactly_like_the_single_element_family``
gates.

WHAT THE LIMIT DOES NOT SAY — stated because the opposite was once claimed
here, and it is false. The ASSEMBLY's cd in that limit is NOT
``carwing.py``'s cd and cannot be. MEASURED at t/c 0.12, a 0.30 c flap 500
chords away, Re_ref 1e6, against ``polar.default_polar_family().at(0.12)``
— the table ``carwing.py`` reads — this family's section CD runs

    alpha       -4      -2       0       2       4       6       8      10      12
    this        .00988  .00776  .00674  .00726  .00836  .01029  .01396  .01849  .02481
    carwing     .00770  .00659  .00564  .00578  .00693  .00905  .01234  .01567  .02001
    delta      +28.3 % +17.8 % +19.5 % +25.7 % +20.6 % +13.7 % +13.2 % +18.0 % +24.0 %

i.e. 13-28 % HIGH at every incidence, smallest at alpha 8 and largest at
alpha -4, for two structural reasons and neither of them a bug:

  * THE FLAP IS STILL BEING CHARGED. ``cd_section`` sums both elements, and
    a flap 500 chords away is still 0.30 chords of wetted section paying at
    its own Reynolds number: +0.00297 at alpha 4, which is +42.8 % of
    ``carwing``'s 0.00693 on its own.
  * THE MAIN ELEMENT IS NOT THE WING'S SECTION. It is 0.70 chords long, so
    it flies at 0.70 Re_ref (7e5 against 1e6 here) where the same table
    charges 0.00770 rather than 0.00693, and it is then weighted by its own
    0.70 chord fraction: 0.00539, i.e. -22.2 % against ``carwing``.

+42.8 % and -22.2 % leave +20.6 %, which is the alpha-4 column above.

WHAT IS NOT A REASON: the bank. The wide-alpha bank and the cruise bank are
BIT-FOR-BIT equal in cd at Re 1e6 over their shared alpha range (measured at
every even degree from -6 to +14, difference identically 0.0), so which bank
is read contributes nothing to the gap at a matched Reynolds number. Only
the chord split and the element Reynolds numbers do. The bank matters for
the RANGE — alpha to +22 against +14 — which is the separate control
:func:`single_element_problem` exists for.

The inverse is well posed because the isolated section's Cp_min(alpha) is
strictly monotone on each side of its least-suction incidence — MEASURED on
NACA 2412 over alpha in [-14, +24] deg, strictly decreasing above -1.0 deg
and strictly increasing below it — so the branch is chosen by whether the
element is loaded with or against the section's own camber, and each branch
is a monotone interpolation.

The drag tables are the shipped WIDE-ALPHA XFOIL bank
(``data/airfoils/naca24??_re*_stall.pol``, alpha to +22 deg), read at each
element's own Reynolds number through ``polar.polar_at_re``. The cruise bank
(alpha to +14) would not reach the angles this map produces.

THIS MODULE IS A DECLARED PRODUCTION CONSUMER OF ``polar.polar_at_re``, and
what it is entitled to read off one is written down where the decision lives:
``tests/test_api_config_promises.py::test_polar_at_re_consumers_are_declared_
and_read_only_what_is_argued`` names this file and enforces the entitlement
by reading the source. The premise that test protects is the
``blend_slope=True`` default — whether a Reynolds-blended polar STATES its
linear pair or re-extracts it from the blended table — and it is NOT
load-bearing here, for a reason that is a property of this module rather than
a hope: nothing in this file ever reads ``a_lin`` or ``alpha_L0`` off a
``polar_at_re`` result. The section's linear pair is fitted from the PANEL
solve (``cascade_polar``), the endplate's comes from
``polar.default_polar_family()`` (which is a file-backed member, not a
blend), and what is read off the Reynolds-blended table is ``cd`` and
``alpha_valid`` and nothing else. So the choice of default cannot move a
number this family produces.

(This paragraph once said that this module was the FIRST such consumer and
that the tripwire "has still fired" and was waiting for its owner to record
the argument. Both have since stopped being true — the argument was recorded,
and ``carwing.py``'s own flown-Reynolds opt-in is the consumer for which the
default IS load-bearing — so the claim is restated rather than left standing
against a test name that no longer exists.)

BIAS DIRECTION. OPTIMISTIC, on balance, and for a reason that is structural
rather than incidental: NOTHING here charges the slot itself. There is no
mixing loss for the jet, no cove separation on the main element's lower
surface, and no wake from the main element passing over the flap (panel2d
sheds no wake panels). Against that, the model also ignores the one effect
that would make the flap CHEAPER — the slot re-energising its boundary
layer, A. M. O. Smith's fresh-boundary-layer effect — so the flap's own
number is conservative while the assembly's is not. The net is optimistic:
the omissions on the loss side are several and the one on the credit side is
one.

Stall: the suction-peak criterion, DERIVED from data already in the tree
------------------------------------------------------------------------
An element is refused when its inviscid suction peak goes deeper than the
peak the SAME section is measured to sustain alone. The ceiling is built,
never pasted:

  1. ``polar.stall_point`` on the shipped wide-alpha family gives the
     RESOLVED stall incidence of NACA 24XX at Re 1e6 — resolved meaning the
     table continues past the peak, so it is a stall and not the point where
     XFOIL's viscous march gave up. (t/c = 0.06 is censored and is therefore
     not an anchor, exactly as ``stall.section_clmax`` excludes it; the
     usable range here is t/c in [0.09, 0.18].)
  2. ``panel2d`` solves the ISOLATED section at that incidence, at this
     module's own panel count, and its Cp_min is the ceiling.

MEASURED at the shipped anchors, at this module's default 120-node
panelling: t/c 0.09 -> Cp_min -16.78 at 14.0 deg, 0.12 -> -13.88 at 16.0,
0.15 -> -11.15 at 17.0, 0.18 -> -9.34 at 17.5. (At 160 nodes: -16.92,
-13.93, -11.28, -9.37 at the same incidences — the ceiling is a panel-count
quantity and moves by under 1 %, which is why it is rebuilt at whatever
count the section is flown on rather than stored.) Between anchors the stall
INCIDENCE is interpolated in t/c and the section is then solved exactly, so
the ceiling is never an interpolated pressure.

The ceiling is a statement about an ISOLATED section, applied to an element
in a slot, and the direction of that is known: a slot exists precisely to let
an element hold a peak it could not hold alone (Smith's dumping effect). So
this refusal is CONSERVATIVE — it refuses two-element designs a real slot
would carry — and it is applied anyway, because the alternative is a stall
criterion with nothing behind it. It is also a Re-1e6 measurement used at
whatever Reynolds number the wing flies; a section stalls EARLIER at lower
Re, so at the element Reynolds numbers a small car wing actually reaches
(~2e5 for a flap) the ceiling is OPTIMISTIC. The two directions do not
cancel and neither is quantified here.

THE NON-MERGING FLOOR, and why an inviscid method needs one imposed on it
--------------------------------------------------------------------------
Smith's own bound on the regime in which a slot works is a bound on the GAP
(p. 515-516): *"gaps between airfoil elements should be so large that wakes
and boundary layers do not merge for, if they do, early separation will set
in."* Immediately after it comes the sentence that licenses an inviscid
multi-body solve INSIDE that regime — *"there is no merging within the slot.
Topologically, the process of boundary-layer development is no different from
that on a biplane."*

``panel2d`` has no boundary layer, so it has no merging mechanism, so it
cannot find that floor for itself: it will close the slot and report the lift
of a flow that would already have separated. MEASURED — left free, a search on
this family rides the tightest gap its box offers. So the floor is a stated
option, :attr:`CarWingMultiProblem.slot_gap_te_min_frac`, whose value comes
from the only OPEN two-element rigging measurement the literature review could
verify in full (:data:`SLOT_GAP_TE_NON_MERGING_FRAC` carries the derivation).

It is OFF by default. That is a deliberate asymmetry with ``carsection.py``,
which turns it ON by default: switching it on changes which designs this
family admits, and every number this family has published was measured with it
off. A SEARCH needs the bound because the search is what exploits the model's
blind spot; an already-published comparison must not move.

What is NOT modelled
--------------------
* VISCOSITY in the slot, entirely (see above): no boundary layer, no wake,
  no separation, no slot mixing loss, no cove. The non-merging floor above is
  a BOUND ON WHERE THE MODEL IS USED, not a model of any of them.
* The main element's COVE. A real slotted wing's main element has a shaped,
  usually cusped, lower-surface cove; here the main element is simply the
  base section scaled down, with its own sharp trailing edge. The flap is
  likewise the same section rather than the more highly cambered one a real
  design would use.
* Any spanwise variation of the SLOT. One cascade polar is built per
  candidate, at the mean aerodynamic chord's Reynolds number, and flown at
  every strip — the same economy ``carwing.py`` uses for its single section.
  The elements' own Reynolds numbers therefore vary across a tapered span
  and only their MAC values are reported.
* DRS, the flap as a moving part, unsteadiness, compressibility.
* Everything ``carwing.py`` already lists as not modelled: the swan-neck
  suction-side loss, the diffuser and floor, wheel wakes, and the track as a
  moving belt.

MEASURED — what the four extra rows cost
-----------------------------------------
On the development machine (Darwin 24.5.0, .venv numpy), at the default
120-node panelling (238 panels over the two elements), as the mean over 400
uniform random points of each family's own box, refusals included, with
``_CASCADE_CACHE`` cleared between arms (leaving it warm turns the
repeats measurement into a cold one and the cold one into repeats):

    carwing.evaluate_car_wing  (7 rows, dCZ/dh OFF — the default)   1.5 ms
    carwing.evaluate_car_wing  (7 rows, dczdh_report=True)          4.5 ms
    evaluate_car_wing_multi    (11 rows, a slot row moved each time) 10.5 ms
    evaluate_car_wing_multi    (11 rows, the section repeats)        3.2 ms

WHAT THE SLOT COSTS IS 10.5 ms A CANDIDATE. That absolute is the number to
plan a budget with, and it is quoted first because the DENOMINATOR of any
multiple is not a fixed quantity: ``carwing.evaluate_car_wing`` grew a
ride-height sensitivity (``dCZ/dh``: two extra lattice solves per
candidate), so the same slot costs

    6.5-6.8x a carwing candidate with dCZ/dh OFF
    2.2-2.4x a carwing candidate with dCZ/dh ON

over four repeats of the 400-point sweep. The FIRST of those is the
comparison a caller actually meets, because ``dczdh_report`` is False by
default and the ceiling that implies it (``dczdh_max_per_m``) is None. A
bare multiple quoted without saying which of the two it is has already gone
stale once in this docstring — it read 6.2x against a denominator that then
moved twice, first gaining an unconditional dCZ/dh and then having it made
opt-in — and would go stale again the next time either family gains a
switch. That is why the absolute is the headline here and the multiple is
the footnote.

Restricted to 40 FEASIBLE candidates with a slot row moving every time and
the cache cleared per candidate, the figure is 11.5 ms, so the refusals are
not what makes it slow: they are, if anything, the cheap ones.

The cache key is the section geometry alone — thickness, the four slot rows
and the panel count — because that is what the expensive part (the two panel
solves) depends on; the Reynolds-dependent drag tables are rebuilt per
candidate and are table interpolation, not a solve. The repeats that DO
happen are worth having: a mount comparison, a restart replaying a point, a
pinned slot row, and :func:`compare_split`, whose deflection walk reuses each
section across the whole incidence walk (measured on the same run as the
table above: a full 25 x 15 comparison takes 1.1 s, against 3.7 s with
``_CACHE_MAX`` set to zero — 3.4x).

MEASURED — how much of the box is flyable
------------------------------------------
Over 400 uniform random points of the default box drawn with
``numpy.random.default_rng(0)`` — the seed is stated because without one
this census is not a number anybody can check — 271 (67.8 %) are feasible,
against 376/400 (94.0 %) for the single-element family on its own box. The
32 % is physics, not a band:

     53  the wing's effective incidence leaves the section's flyable window
     47  slots that stall at every incidence
     22  the endplate reaching the track (the single-element family refuses
         24 of the same 400 for the same reason — this is carwing's refusal,
         not the slot's)
      4  an intersecting flap
      1  a flyable window too narrow to state a lift-curve slope on
      1  a slot below the resolution floor
      1  the endplate's own incidence leaving its own polar

Repeated at seeds 11 and 63 the feasible count is 257 and 273, and the
ordering of the first three classes does not change: the wing's effective
incidence leaving the section window is the largest refusal class at every
seed (53 / 77 / 62), which is why deleting the gate that produces it
(:func:`evaluate_car_wing_multi`) is the single most damaging edit that can
be made to this file, and why it has a test of its own.

THE CLAIM, MEASURED — does splitting the wing buy downforce?
------------------------------------------------------------
At the SAME reference area (0.4 m^2) and the SAME drag budget, walking each
family's incidence — and the two-element family's flap deflection as well —
to the most downforce it can make inside the budget (:func:`compare_split`,
box centre, both families reading the WIDE-ALPHA drag table so that the
comparison is a slot and not two XFOIL tables):

    CD budget    0.015   0.018   0.020   0.025   0.030   0.11   0.15   0.20
    CZ, 1 elem   0.3865  0.4763  0.5212  0.6559  0.7456  1.6434 1.8230 2.0923
    CZ, 2 elems  0.3386  0.4754  0.5219  0.6596  0.7736  1.7119 2.0282 2.3439
    delta       -12.4 %  -0.2 %  +0.1 %  +0.6 %  +3.8 %  +4.2 % +11.3 % +12.0 %

(the first five columns on a 49 x 29 incidence/deflection walk, the last
three on 25 x 15)

and at a budget of 0.012 the two-element family has NO feasible design at all
while the single-element one makes CZ 0.252. So:

  * WHERE IT DOES NOT PAY. Below a drag budget of about 0.018 — a sixth of
    the published allowance — splitting the wing LOSES. There the wing is
    barely loaded, the slot's second element is wetted area buying lift
    nobody asked for, and its minimum drag eventually exceeds the whole
    budget. This is a real region and it is the first thing this section
    reports.
  * AT THE PUBLISHED BUDGET it buys +4.2 %. That is not nothing, but it is
    not what a two-element wing is usually sold as either.
  * WHAT IT REALLY BUYS IS A CEILING. Turn the question round and ask what
    each family must SPEND for a given downforce, and the answer is monotone
    and much larger:

        CZ demanded   0.80   1.00   1.20   1.40   1.60   1.80   2.00   2.20
        CD, 1 elem   .03317 .04542 .06373 .08106 .10730 .13457 .17198 .23347
        CD, 2 elems  .03158 .04200 .05649 .07437 .09553 .11706 .14426 .17575
        ratio         0.95   0.93   0.89   0.92   0.89   0.87   0.84   0.75

    The slot's value grows with how much downforce is asked for, because the
    single-element wing runs out of SECTION before it runs out of budget.
  * AND IT MATTERS MOST NEAR THE TRACK AND ON A LONG SPAN. At the published
    budget the gain is +17.1 % at a 0.15 m ride height (against +4.2 % at
    0.375 m) and +11.5 % at a 2.0 m span (against +4.4 % at 1.2 m) — both
    cases where the single-element wing is section-limited rather than
    drag-limited.

A CONTROL THIS MEASUREMENT NEEDS, and the reason it is stated separately.
``carwing.py`` as shipped reads the CRUISE polar, which stops at alpha 14
deg, while this family prices its elements off the wide-alpha bank. Compare
them as they ship and the slot is credited +23.4 % at a 0.15 budget and
+48.2 % at 0.30 — two to seven times the controlled number — because above a
0.11 budget the single-element wing is not stalling, it is running out of
TABLE. :func:`single_element_problem` takes ``wide_alpha=True`` for exactly
this, and :func:`compare_split` reports both.

A SECOND CONTROL, on the incidence band. Walked over the family's own
0-12 deg alpha row instead of a widened one, the single-element wing tops out
at alpha 12 having spent only 0.0598 of a 0.11 drag budget: it is
INCIDENCE-limited, and the slot is then credited +43.4 % that is mostly the
box's. Every answer :func:`compare_split` returns says which limit it rides.

Panel count is a knob with a measured price. At alpha 4 deg on the box
centre's own slot (flap 0.275 c at 17.5 deg, gap 0.031 c, overlap 0.020 c):

    nodes per element      100      120      160      240
    assembly cl        1.86546  1.86822  1.87158  1.87282
    Cp_min, main        -7.215   -7.172   -7.264   -7.263
    ceiling, t/c 0.12  -13.647  -13.881  -13.926  -13.994
    two basis solves     3.7 ms   5.2 ms  10.1 ms  22.6 ms

so the 120-node default is 0.25 % below the 240-node cl and 1.25 % SHALLOW
in the main element's peak. Everything quoted here and in
:data:`N_SECTION_NODES` is against the 240-node column; quoting the peak
against 240 and the ceiling against 160, as this docstring once did, makes a
ratio out of two different references and understates the error in it.

The peak error is partly self-cancelling and it is worth saying why: the
suction-peak CEILING is rebuilt at the same count and is shallow in the same
direction, so the stall MARGIN — 1 - Cp_min/ceiling, the ratio of the two —
is more converged than either number in it. MEASURED on the row above: at
120 nodes the peak is 1.25 % shallow and the ceiling 0.81 % shallow, and the
margin they make is 0.4833 against 0.4810 at 240 nodes. It does not cancel
exactly, and the residual direction is OPTIMISTIC — 0.48 % more margin
claimed than the 240-node grid finds — but it is a quarter of the error in
the peak alone.

References
----------
Only one, and only for the two NAMED effects this module says it does not
model. Everything numerical above is measured here or read off tables already
in the tree.

Smith, A. M. O., "High-Lift Aerodynamics," Journal of Aircraft, Vol. 12,
    No. 6, 1975, pp. 501-530. Section 5 sets out the five primary effects of
    a slot; the two this module names are the DUMPING effect (the forward
    element's trailing edge discharges into a region of higher velocity, so
    it can carry more circulation than it could alone) and the
    FRESH-BOUNDARY-LAYER effect (each downstream element starts a new
    boundary layer at its own leading edge). ``panel2d.py`` cites the same
    paper for the same reason.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import airfoil, carmount, geometry, junction, panel2d
# carwing.py is the parent: its frame, its beam, its mounts, its budgets and
# its box rows are IMPORTED, never restated. Three of these names are not
# called anywhere below and are here on purpose — ``AREA_BOUNDS_M2``,
# ``SPAN_BOUNDS_M`` and ``published_drag_budget_n`` are the numbers a caller
# of THIS family needs to state its own bands and its own newton budget, and
# a sibling family that made a reader import them from two modules would be
# inviting the two to drift apart.
from .carwing import (
    AREA_BOUNDS_M2, CAR_OBJECTIVES, G_FAIL, MOUNTS, MU_AIR, PENALTY, RHO_AIR,
    SPAN_BOUNDS_M, _strut_cd0, area_row, bending_moment, deflection_index,
    published_drag_budget_n, s_from_x, span_from_x, span_row,
)
from .vlm import VLM, ImagePlane

__all__ = [
    # re-exported from carwing.py so this family is stated in one place
    "AREA_BOUNDS_M2", "SPAN_BOUNDS_M", "published_drag_budget_n",
    "area_row", "span_row", "s_from_x", "span_from_x",
    "bending_moment", "deflection_index", "MOUNTS", "CAR_OBJECTIVES",
    # this family's own
    # ...this family's own. Every name here must EXIST: two of them did not
    # (``SHARE_MIN_CL`` and ``check_section_coords``, both from an earlier
    # draft of the section model), so ``from carwing_multi import *`` raised
    # AttributeError and nothing in the suite could see it, because nothing
    # in the package does a star import. Gated now by
    # ``test_every_exported_name_exists``, which sweeps all 47 modules.
    "ALPHA_TABLE_DEG", "CAR_MULTI_OBJECTIVES", "CEILING_SHAPES",
    "CarWingMultiProblem",
    "CascadePolar", "FLAP_CHORD_FRAC_BOUNDS", "FLAP_DEFLECTION_BOUNDS_DEG",
    "N_SECTION_NODES", "SLOT_GAP_FRAC_BOUNDS",
    "SLOT_GAP_TE_NON_MERGING_FRAC", "SLOT_OVERLAP_FRAC_BOUNDS",
    "SLOT_ROW_OFFSET", "SLOT_WIDTH_PANELS",
    "cascade_coords", "cascade_polar", "compare_split",
    "element_base_coords", "evaluate_car_wing_multi", "fg_car_wing_multi",
    "single_element_problem", "single_element_x", "slot_from_x",
    "slot_gap_te", "slot_rows", "suction_peak_ceiling",
]

#: the objectives THIS family can score, with ``carwing``'s own labels.
#:
#: Stated here as an explicit selection rather than read off
#: ``carwing.CAR_OBJECTIVES``, because the two sets are NOT the same set and
#: reading the sibling's was a real defect: ``carwing`` grew a fourth entry,
#: ``laptime``, which needs a circuit (``CarWingProblem.track_spec``) and a
#: car; this family carries neither field, so a lap time is not a question it
#: can answer. Validating against the sibling's dict let
#: ``CarWingMultiProblem(objective='laptime')`` CONSTRUCT and then raise
#: ``KeyError('laptime')`` out of :func:`evaluate_car_wing_multi` and out of
#: :func:`fg_car_wing_multi` — an out-of-contract raise from inside an
#: optimiser, from a switch the constructor had already accepted.
#:
#: The selection is written as a lookup INTO ``CAR_OBJECTIVES``, so the two
#: cannot drift silently in either direction: a name this family scores that
#: the sibling drops or renames fails at IMPORT with a KeyError naming it,
#: and a name the sibling adds is refused here until someone decides what it
#: means for a two-element wing.
#: ``test_the_objective_set_is_the_one_this_family_can_score`` is the tripwire
#: that keeps the scorer and this dict the same set.
#:
#: ``laptime`` joined the set when this family gained a circuit
#: (:attr:`CarWingMultiProblem.track_spec`), and the paragraph above is left
#: standing because the defect it describes is real and the fix for it is not
#: "add the name": the name is here AND the family now carries the fields that
#: make it answerable, AND it is still refused — by
#: :meth:`CarWingMultiProblem.__post_init__`, in contract, with the reason —
#: when no circuit is stated. Adding the entry without the fields would have
#: reproduced exactly the KeyError this comment exists to remember.
#:
#: WHY THE LAP IS HERE AT ALL, given that this family exists to measure a
#: section. Because the section's own objective is a lap: what a two-element
#: rear wing trades is CL_max in the corner against L/D on the straight, and
#: `a-weight-is-not-an-exchange-rate` in this repository's memory records what
#: happens when that trade is settled with weights instead — the composite
#: sells whatever it weights highest and the weights become the answer. The
#: circuit's corner radii and straight lengths ARE the exchange rate, measured
#: rather than chosen. A section optimised on a lap can only be COMPARED with
#: the wing it is transferred to if the wing can be scored on the same lap, so
#: this family had to be able to time one.
CAR_MULTI_OBJECTIVES: dict[str, str] = {
    k: CAR_OBJECTIVES[k]
    for k in ("cz", "downforce", "efficiency", "cd", "drag", "laptime",
              # a two-element wing is the high-load configuration, so the
              # total load it puts into the car is exactly the question this
              # family gets asked. Needs no field this family lacks: both
              # terms are already in its own breakdown.
              "downforce_plus_drag")}

# ------------------------------------------------------------------ the box

#: flap chord as a fraction of the STOWED chord. The band is the family's
#: own statement of what counts as a flap rather than a tab or a second wing:
#: below the floor the flap's own Reynolds number falls off the shipped
#: viscous bank on a small wing, and above the ceiling the flap is no longer
#: a flap. MEASURED at the box centre, the main element's lift share falls
#: 0.861 / 0.842 / 0.813 / 0.781 / 0.757 across flap fractions 0.15 / 0.20 /
#: 0.275 / 0.35 / 0.40 — the main element still carries three quarters of the
#: load at the top of the band, which is what says the band ends in the right
#: place rather than in a second wing.
FLAP_CHORD_FRAC_BOUNDS = (0.15, 0.40)

#: flap deflection, degrees, trailing edge towards the car's downforce
#: direction. The floor is zero — an undeflected slotted section is a real
#: design and it is also the control this family needs, because it is the
#: nearest thing to "no flap at all" the vector can say.
#:
#: THE CEILING IS THE BOX CENTRE'S OWN LIMIT AT THIS FAMILY'S PUBLISHED
#: THICKNESS, and not the family's. That is a correction to what this comment
#: used to claim, and the numbers it claimed it from are all still here and
#: all still reproduce.
#:
#: MEASURED in 2-D on a 0.30 c flap at a 0.02 c gap and overlap, at t/c 0.12,
#: the flyable incidence window's upper end runs +9.75 deg at a 10 deg flap,
#: +6.5 at 20 and +3.5 at 30, and at 40 deg there is NO flyable incidence at
#: all. Every one of those is re-derived, to the digit, by
#: ``scripts/carsection_deflection_ceiling.py`` (stage 1, arm ``naca``).
#:
#: WHAT DOES NOT FOLLOW is "the section is past its ceiling everywhere", which
#: is what this comment inferred from them. That measurement fixes THREE of
#: the four slot rows and the thickness at one point each, and all four move:
#: over the slot box's own corners (same script, stage 1b, 48-point deflection
#: grid, 12 designed draws per corner) the highest deflection with a flyable
#: window is
#:
#:     t/c 0.12   box centre 35 deg   |  most permissive corner 48 deg (NACA),
#:                                       47 deg (designed)
#:     t/c 0.18   box centre 40 deg   |  most permissive corner 52 deg (NACA),
#:                                       56 deg (designed)
#:
#: where the permissive corner is flap 0.40 c, placement gap 0.012 c, overlap
#: -0.02 c, and 8 of 12 designed sections still fly there at 40 deg (t/c 0.12)
#: and 10 of 12 at 45 deg (t/c 0.18). ``carsection.py`` searches t/c to 0.18
#: and searches all four slot rows, so its answer rides this ceiling on every
#: seed while designs above it fly.
#:
#: THE BAND IS NOT MOVED ALL THE SAME, and the two reasons are downstream of
#: this module rather than in it. Both are recorded in
#: RESULTS_SESSION65_DEFLECTION_CEILING.md with the recipe for lifting them:
#:
#:   1. ``carsection``'s 2-D -> 3-D bridge is FITTED over
#:      ``BRIDGE_CALIBRATION_SLOTS`` (deflections 2, 17.5 and 32 deg) and
#:      VALIDATED over ``BRIDGE_HOLDOUT_SLOTS`` (top 30 deg), and its published
#:      0.258 % held-out CL error is a statement about that range. A section's
#:      zero-lift angle runs to about -21 deg at 35 deg of flap, so a band
#:      reaching 50 would put a third of the row outside the range the
#:      surrogate was ever measured on. Widening the band means re-calibrating
#:      the bridge FIRST.
#:   2. :func:`compare_split` walks this row (``n_deflection`` points across
#:      ``p2.bounds[SLOT_ROW_OFFSET + 1]``), so the band is an input to a
#:      published comparison. Moving it moves those numbers.
#:
#: AND THE BAND IS CURRENTLY A BAN RATHER THAN A DEFAULT, which is the defect
#: the measurement above exposed and the one worth fixing first. Widening this
#: row through ``bounds_overrides`` moves the SAMPLER's row and not the
#: problem's, so every draw above 35 deg comes back ``feasible=False,
#: reason='bounds violation'`` without any physics running — MEASURED: widen
#: the row to (0, 50) and 19 of 64 Sobol draws are refused that way; widen
#: ``flap_chord_frac`` to (0.10, 0.55) and 29 of 64 are. All four slot rows
#: behave this way at both ends. It is the same defect ``api._CAR_SIZE_ROWS``
#: records and fixed for ``b_m`` / ``S_m2``, and the cure is the same: the row
#: has to reach the problem's own band, which needs a field here and a
#: bounds-override map in ``api``.
FLAP_DEFLECTION_BOUNDS_DEG = (0.0, 35.0)

#: slot gap, as a fraction of the stowed chord, POSITIONING the flap's
#: leading edge below the main element's trailing edge. Not the slot width —
#: see the module docstring and ``slot_width_frac``. The floor is where the
#: solver stops resolving the slot (:data:`SLOT_WIDTH_PANELS`) with the
#: rotation at the top of the deflection band taken into account; the ceiling
#: is where the two elements stop being a cascade and become two wings in
#: tandem, which is a different family.
SLOT_GAP_FRAC_BOUNDS = (0.012, 0.050)

#: slot overlap, fraction of the stowed chord, POSITIVE when the flap's
#: leading edge is UPSTREAM of the main element's trailing edge (the sense
#: every wind-tunnel report uses). Negative is a gap in x — a flap set back,
#: which is the Fowler direction and a real design.
SLOT_OVERLAP_FRAC_BOUNDS = (-0.02, 0.06)

#: THE NON-MERGING FLOOR: the smallest trailing-edge gap (:func:`slot_gap_te`,
#: stowed-chord normalised) at which an INVISCID cascade solve is licensed.
#: It is OFF by default (``CarWingMultiProblem.slot_gap_te_min_frac = None``)
#: and it is a bound on the METHOD's blind spot, not a claim about where this
#: wing's optimum is.
#:
#: Why a bound is owed at all. Smith, *High-Lift Aerodynamics*, p. 515-516:
#:
#:   "gaps between airfoil elements should be so large that wakes and boundary
#:   layers do not merge for, if they do, early separation will set in."
#:
#: and, on p. 518, the sentence that licenses an inviscid multi-body solve
#: INSIDE that regime:
#:
#:   "...that each component develops its own boundary layer under the
#:   influence of the main stream, and there is no merging within the slot.
#:   Topologically, the process of boundary-layer development is no different
#:   from that on a biplane."
#:
#: ``panel2d`` has no boundary layer, so it has no merging mechanism, so it
#: cannot find the floor Smith names — it will happily close the slot and
#: report the lift a merged, separated flow would never make. MEASURED here:
#: left free, the two-element optimum rides the tightest gap the box offers,
#: at every objective tried. That is the model's blind spot answering, not
#: the physics.
#:
#: Where 0.020 comes from, and what it is NOT. It is the bottom of the only
#: OPEN two-element rigging measurement this repository's literature review
#: could verify in full: Storms & Ross, *Experimental study of lift-enhancing
#: tabs on a two-element airfoil*, J. Aircraft 32(5), 1995, Table 1 (read from
#: the page image) — four configurations the paper itself calls "four
#: near-optimum configurations", selected from a 35-configuration parametric
#: sweep of gap, overlap and flap deflection:
#:
#:     flap deflection   22     32     43     42    deg
#:     gap / c         0.020  0.031  0.031  0.052
#:     overlap / c     0.066  0.043  0.042  0.035
#:
#: The chain from Smith to that table is: at the optimum rigging the layers do
#: NOT merge (Smith, quoting Foster et al., p. 516 — "when the flap is in the
#: optimum position ... the interaction between the wing wake and the flap
#: boundary layer is comparatively mild, with the two layers retaining their
#: separate identity almost to the flap trailing edge"), so a measured
#: near-optimum gap is evidence of the non-merging regime. 0.020 is the
#: smallest gap in that evidence.
#:
#: It is NOT an optimum for a race-car rear wing and is not used as one: those
#: riggings are a transport section at Re 3.7e6 at 22-43 deg of flap, and
#: LITERATURE_REVIEW_S63_CARWING.md section 7.2 lists importing them as an
#: optimum among the four things this session was warned not to do. What is
#: taken is one number's ROLE — the lower edge of the regime in which the
#: method being used is licensed — which is the kind of thing a bound may be
#: taken for and a target may not.
SLOT_GAP_TE_NON_MERGING_FRAC = 0.020

#: nodes per element handed to ``panel2d``. 120 nodes (119 panels) is the
#: default because it is the cheapest count at which the LIFT has stopped
#: moving: 0.25 % in the assembly cl against 240 nodes, for a quarter of the
#: cost (the panel-count table in the module docstring, alpha 4 on the box
#: centre's own slot).
#:
#: The main element's Cp_min has NOT stopped moving there, and this comment
#: once claimed it had ("0.0 % in Cp_min"). MEASURED against the same 240
#: nodes, 120 is 1.25 % SHALLOW in the peak. That is not free and the module
#: docstring prices it: the ceiling is 0.81 % shallow on the same reference,
#: so the stall MARGIN the two make is 0.48 % optimistic rather than 1.25 %.
#: Raise it (``CarWingMultiProblem.n_section_nodes``) and the SUCTION-PEAK
#: CEILING is rebuilt at the same count, so the criterion and the quantity it
#: judges are always measured on the same grid.
N_SECTION_NODES = 120

#: how many trailing-edge panel lengths of slot the method needs before its
#: answer is a solution rather than two bodies sharing a control point.
#:
#: MEASURED, and separated from the INTERSECTION refusal above it so that
#: each number means one thing. Sweeping the placement gap on a t/c 0.12
#: section with a 0.30 c flap at 25 deg and a 0.02 c overlap, over the
#: geometries that do NOT intersect, and comparing the 160-node assembly cl
#: at alpha 4 with the 960-node one:
#:
#:     width / L_TE     0.47     1.18     2.96     8.30    51.0    158
#:     err(160 nodes)  +1.07 %  -0.18 %  -0.33 %  -0.33 %  -0.33 % -0.33 %
#:
#: From about one trailing-edge panel of slot the discretisation error is
#: FLAT at a third of a per cent and has settled on one sign; at half a panel
#: it is three times larger and the OTHER sign, i.e. it is no longer the same
#: error being refined away. So the floor is one main-element trailing-edge
#: panel.
#:
#: It is a RESOLUTION statement, not a physical minimum: it scales with the
#: panelling, and refining the section lowers it. The geometries that produce
#: the truly wild numbers — an assembly cl swinging -4.7 %, -1.9 %, +5.0 %
#: over three neighbouring gaps — are the ones where the two loops actually
#: INTERSECT, and those are refused by their own test before this one is
#: reached.
SLOT_WIDTH_PANELS = 1.0

#: incidence grid the cascade polar is tabulated on [deg]. Wide enough that
#: the lattice's effective angles land inside it at every corner of the box,
#: fine enough that linear interpolation of Cp_min — the steepest quantity
#: here — is not the error. Costs nothing: the two basis solves are already
#: paid, and a station is a few vector operations.
ALPHA_TABLE_DEG = np.arange(-20.0, 20.0 + 1e-9, 0.25)

#: t/c range this family can be flown at: the range over which the wide-alpha
#: XFOIL family RESOLVES a stall, which is what the suction-peak ceiling is
#: built from. Narrower than ``geometry.TC_BOUNDS`` for the same reason
#: ``stall.section_clmax`` is: a censored peak is a lower bound, not a peak,
#: and a ceiling built on one would be a number with nothing behind it.
#: Derived at import-free call time from the shipped tables, never pasted.


def slot_rows() -> np.ndarray:
    """(4, 2) design-box rows for the slot — the family's new freedom.

    In the order they occupy the design vector: flap chord fraction, flap
    deflection [deg], slot gap fraction, slot overlap fraction. Stated as one
    function for the same reason :func:`carwing.span_row` is: the four rows
    are one question ("what slot?") and they are validated in one place.
    """
    return np.array([
        FLAP_CHORD_FRAC_BOUNDS,
        FLAP_DEFLECTION_BOUNDS_DEG,
        SLOT_GAP_FRAC_BOUNDS,
        SLOT_OVERLAP_FRAC_BOUNDS,
    ], dtype=float)


#: index of the first slot row in the design vector. The family block ahead
#: of it is ``carwing``'s, unchanged and of fixed length, so the slot rows are
#: read FORWARDS from a constant offset while the size block keeps being read
#: BACKWARDS from the chord coefficients (:func:`carwing.span_from_x`). Two
#: directions, no collision, and that is the whole reason the slot rows go
#: here rather than next to the span.
SLOT_ROW_OFFSET = 6


def slot_from_x(x) -> tuple[float, float, float, float]:
    """``(flap_chord_frac, flap_deflection_deg, slot_gap, slot_overlap)``.

    Read FORWARDS from :data:`SLOT_ROW_OFFSET`, because the block ahead of the
    slot is fixed-length whatever else the problem switches on. The size and
    chord blocks behind it are read backwards by ``carwing``'s own readers and
    are untouched by this one.
    """
    xs = np.asarray(x, dtype=float)
    n = SLOT_ROW_OFFSET
    if xs.size < n + 4:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for four slot "
            f"rows at offset {n}")
    return (float(xs[n]), float(xs[n + 1]), float(xs[n + 2]),
            float(xs[n + 3]))


# ------------------------------------------------------------- section data


def _base_coords(tc: float, n_nodes: int,
                 coords: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Unit-chord element coordinates, CLOSED, in panel2d's loop order.

    ``panel2d`` refuses a blunt trailing edge and says so, and a section from
    almost any source has one, so the closure is applied here rather than
    left to the caller to remember (module docstring of
    :func:`panel2d.close_trailing_edge` for what the closure changes). A
    NACA 24XX from ``airfoil.naca4_coords`` already closes exactly, and the
    closure returns it unchanged bit-for-bit.

    THE CLOSURE IS LOAD-BEARING, and the reason it does not look it is that
    every section this package generates is already closed. MEASURED on the
    TEXTBOOK NACA 4-digit — the open-trailing-edge thickness polynomial with
    -0.1015 as its last coefficient, which is what published NACA tables and
    most coordinate files carry, and which ``airfoil.naca4_coords`` names in
    its own docstring as the variant it does NOT use — a 2412 at 120 nodes
    lands its two trailing nodes 0.00252 chords apart. That is 2.5x
    ``panel2d.TE_GAP_MAX_FRAC``, so without this line ``check_body`` refuses
    it outright ("trailing edge is blunt") and every caller-supplied section
    with a real trailing edge comes back as an infeasible design. Gated by
    ``test_a_blunt_section_is_closed_before_it_is_flown``.
    """
    if coords is None:
        code = f"24{int(round(float(tc) * 100)):02d}"
        c = airfoil.naca4_coords(code, int(n_nodes))
    else:
        c = np.asarray(coords, dtype=float)
    return panel2d.close_trailing_edge(c[:, 0], c[:, 1])


def element_base_coords(tc: float, n_nodes: int = N_SECTION_NODES,
                        coords: np.ndarray | None = None):
    """``(x, y)`` — ONE element's unit-chord loop, exactly as it will be flown.

    The public name of :func:`_base_coords`, and it is public because a
    caller that wants to MEASURE something about an element (its stall, its
    boundary layer, its Cp at some incidence) has to be able to get the
    identical array the cascade will solve, node for node. Handing such a
    caller a denser sampling of the same shape is the mistake
    :func:`aerobo.section_stall.measured_ceiling` documents and measures:
    Cp_min is a panel-count quantity, so a ceiling read on one grid and a
    peak read on another do not cancel their shared discretisation error.

    Note what ``n_nodes`` does and does not do. With ``coords=None`` it is the
    NACA 24XX generator's node count. With coordinates GIVEN it is ignored —
    a supplied section is flown at its own sampling, because resampling a
    caller's shape would change it — so a caller measuring a given section
    passes that same array to both.
    """
    return _base_coords(tc, n_nodes, coords)


def _thickness_of(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum thickness/chord of a closed loop in the package's order.

    The loop runs trailing edge -> upper -> leading edge -> lower -> trailing
    edge, so the leading edge is its minimum-x node and the two surfaces are
    the two sides of it. Both are resampled onto a common x grid and
    subtracted; the maximum of the difference is the thickness. Used only for
    a caller-supplied section, whose thickness is a property of the shape
    rather than something to be asked for twice.
    """
    i_le = int(np.argmin(x))
    xu, yu = x[:i_le + 1][::-1], y[:i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    chord = float(np.ptp(x))
    grid = np.linspace(float(x.min()), float(x.max()), 401)
    tu = np.interp(grid, xu, yu)
    tl = np.interp(grid, xl, yl)
    return float(np.max(tu - tl) / chord)


def cascade_coords(tc: float, flap_chord_frac: float,
                   flap_deflection_deg: float, slot_gap_frac: float,
                   slot_overlap_frac: float,
                   n_nodes: int = N_SECTION_NODES,
                   coords: np.ndarray | None = None,
                   flap_coords: np.ndarray | None = None,
                   flap_tc: float | None = None):
    """Both elements' node loops, on a stowed reference chord of 1.

    Returns ``((x_main, y_main), (x_flap, y_flap))``. The main element is the
    base section scaled to ``1 - flap_chord_frac``; the flap is ITS OWN
    section scaled to ``flap_chord_frac``, placed with its LEADING EDGE at
    ``(x_TE(main) - overlap, y_TE(main) - gap)`` and then rotated about that
    placed leading edge by ``-flap_deflection_deg``.

    THE FLAP'S SECTION IS ITS OWN, and the default is that it is not.
    ``flap_coords`` (an (n, 2) closed loop in the package's order) or
    ``flap_tc`` (a NACA 24XX thickness) give the flap a shape of its own;
    with both ``None`` the flap is the MAIN element's shape scaled to the
    flap chord, which is what this function did when it could say nothing
    else, and that path is bit-for-bit unchanged (gated by
    ``test_a_flap_with_no_section_of_its_own_is_the_old_answer_bit_for_bit``).

    Why the default is the old behaviour rather than, say, a more cambered
    flap: a real slotted wing's flap IS usually a different and more highly
    cambered section, but which one is a DESIGN decision, and inventing one
    here would move every published number of this family for a shape nobody
    chose. The freedom is offered; the default is not a choice made on the
    caller's behalf.

    Transformations are applied to the COORDINATE ARRAYS, not through
    ``panel2d.Body.translated`` / ``.rotated``, and that is deliberate: those
    two methods rebuild a Body from the DE-DUPLICATED node loop, whose closing
    segment is then the last surface panel rather than a base, so they refuse
    a body that constructed perfectly well as soon as that panel is longer
    than ``panel2d.TE_GAP_MAX_FRAC``. MEASURED on the closed NACA 2412 of
    ``airfoil.naca4_coords``: ``Body(...).translated(1, 0)`` raises at 60 and
    80 nodes and passes from 100 (where the gap it measures is 9.99e-4
    against a 1e-3 refusal — one part in a thousand from refusing there too).
    Reported to the owner of ``panel2d.py``; worked around here.
    """
    xb, yb = _base_coords(tc, n_nodes, coords)
    if flap_coords is None and flap_tc is None:
        xbf, ybf = xb, yb
    else:
        xbf, ybf = _base_coords(
            tc if flap_tc is None else flap_tc, n_nodes, flap_coords)
    c_main = 1.0 - float(flap_chord_frac)
    c_flap = float(flap_chord_frac)
    xm, ym = xb * c_main, yb * c_main
    # the loop STARTS at the trailing edge (panel2d's required order)
    x_te, y_te = float(xm[0]), float(ym[0])

    xf, yf = xbf * c_flap, ybf * c_flap
    i_le = int(np.argmin(xf))
    x_hinge = x_te - float(slot_overlap_frac)
    y_hinge = y_te - float(slot_gap_frac)
    xf = xf + (x_hinge - float(xf[i_le]))
    yf = yf + (y_hinge - float(yf[i_le]))
    ang = -np.deg2rad(float(flap_deflection_deg))     # TE down = more lift
    ca, sa = np.cos(ang), np.sin(ang)
    dx, dy = xf - x_hinge, yf - y_hinge
    xf = x_hinge + ca * dx - sa * dy
    yf = y_hinge + sa * dx + ca * dy
    return (xm, ym), (xf, yf)


def _loops_cross(A: np.ndarray, B: np.ndarray) -> bool:
    """Do two closed polylines (n, 2) intersect?

    Segment-segment orientation test, vectorised over all pairs. ``panel2d``
    checks a body against ITSELF; nothing checks one body against another,
    and two overlapping elements are a design an optimiser will propose the
    moment the overlap row is free.
    """
    p, q = A, np.roll(A, -1, axis=0)
    r, s = B, np.roll(B, -1, axis=0)
    d1 = q - p
    d2 = s - r
    # cross products, broadcast to (nA, nB)
    def cr(ux, uy, vx, vy):
        return ux * vy - uy * vx
    o1 = cr(d1[:, None, 0], d1[:, None, 1],
            r[None, :, 0] - p[:, None, 0], r[None, :, 1] - p[:, None, 1])
    o2 = cr(d1[:, None, 0], d1[:, None, 1],
            s[None, :, 0] - p[:, None, 0], s[None, :, 1] - p[:, None, 1])
    o3 = cr(d2[None, :, 0], d2[None, :, 1],
            p[:, None, 0] - r[None, :, 0], p[:, None, 1] - r[None, :, 1])
    o4 = cr(d2[None, :, 0], d2[None, :, 1],
            q[:, None, 0] - r[None, :, 0], q[:, None, 1] - r[None, :, 1])
    return bool(np.any((o1 * o2 < 0.0) & (o3 * o4 < 0.0)))


def _min_gap(A: np.ndarray, B: np.ndarray) -> float:
    """Minimum distance between two closed polylines, point-to-SEGMENT.

    Node-to-node would over-report the slot wherever the two surfaces cross
    at an angle, which near a rotated flap's leading edge is everywhere.
    """
    def pt_seg(P, S0, S1):
        d = S1 - S0
        L2 = (d ** 2).sum(axis=1)
        L2 = np.where(L2 > 0.0, L2, 1.0)
        t = ((P[:, None, :] - S0[None, :, :]) * d[None, :, :]).sum(-1) / L2
        t = np.clip(t, 0.0, 1.0)
        Q = S0[None, :, :] + t[..., None] * d[None, :, :]
        return np.hypot(P[:, None, 0] - Q[..., 0], P[:, None, 1] - Q[..., 1])

    A0, A1 = A, np.roll(A, -1, axis=0)
    B0, B1 = B, np.roll(B, -1, axis=0)
    return float(min(pt_seg(A, B0, B1).min(), pt_seg(B, A0, A1).min()))


def _point_to_loop(P: np.ndarray, B: np.ndarray) -> float:
    """Shortest distance from ONE point to a closed polyline, point-to-SEGMENT."""
    S0, S1 = B, np.roll(B, -1, axis=0)
    d = S1 - S0
    L2 = (d ** 2).sum(axis=1)
    L2 = np.where(L2 > 0.0, L2, 1.0)
    t = np.clip(((P[None, :] - S0) * d).sum(-1) / L2, 0.0, 1.0)
    Q = S0 + t[:, None] * d
    return float(np.min(np.hypot(P[0] - Q[:, 0], P[1] - Q[:, 1])))


def slot_gap_te(main_nodes: np.ndarray, flap_nodes: np.ndarray) -> float:
    """THE WIND TUNNEL'S "gap": main trailing edge to the flap's surface.

    Distance from the main element's TRAILING-EDGE POINT to the nearest point
    of the flap's surface, on a stowed reference chord of 1. This is the
    quantity every high-lift report tabulates under the name *gap*, and it is
    NOT any of the other three lengths this module carries:

      * ``slot_gap_frac`` is a PLACEMENT row — how far below the main element's
        trailing edge the flap's leading edge is put, before the flap is
        rotated. It is an input, not a measurement of the built geometry.
      * ``slot_width_frac`` (:func:`_min_gap`) is the minimum surface-to-surface
        distance between the two loops ANYWHERE. It is the resolution question
        — is there enough slot for the panelling to resolve — and its minimum
        need not be at the main element's trailing edge.
      * the overlap is the streamwise offset and is a different axis entirely.

    Three lengths, three jobs, and the reason they are three functions is that
    a bound taken from a paper is stated against ONE of them. Storms & Ross's
    four near-optimum riggings (gap/c 0.020-0.052, overlap/c 0.035-0.066) are
    on this one, and on the flap-STOWED chord — which is also this module's
    reference chord, so the two normalisations agree and no conversion is
    owed. LITERATURE_REVIEW_S63_CARWING.md section 1.4 records the trap: our
    plan originally proposed the deployed streamwise extent as the reference
    chord, which would NOT have agreed, and a gap fraction crossing that
    boundary unconverted is a silent factor-of-order-one error.

    MEASURED at the box centre (flap 0.275 c at 17.5 deg, placement gap
    0.031 c, overlap 0.020 c, 120 nodes): placement gap 0.031000,
    trailing-edge gap 0.024177, minimum surface separation 0.023963. All three
    differ, which is why quoting one against a bound stated on another is not
    a rounding error — the placement row is 28 % above the gap a tunnel would
    have reported for the same rigging.

    The last two are ordered, always and by construction: the minimum over ALL
    of the main element's surface cannot exceed the distance from one of its
    points, so ``slot_width_frac <= slot_gap_te`` identically. That is the
    cheap check that these two are not the same quantity under two names
    (gated by ``test_the_three_slot_lengths_are_three_quantities``).
    """
    return _point_to_loop(np.asarray(main_nodes, dtype=float)[0],
                          np.asarray(flap_nodes, dtype=float))


# ------------------------------------------------- the isolated-section maps

_ISO_CACHE: dict = {}
_CEIL_CACHE: dict = {}
_CASCADE_CACHE: dict = {}
_STALL_BANK_CACHE: dict = {}

#: how many cascade sections to keep. A search moves a slot row every
#: candidate, so the cache is there for the repeats that DO happen — a mount
#: comparison, a restart replaying a point, a pinned slot row, the split
#: comparison below — not to pretend a search is cheaper than it is.
#:
#: When it is full it EVICTS the least recently used entry.
#:
#: It used to stop taking new entries instead, on the argument that "a search
#: whose sections never repeat gets nothing from either policy" and that an
#: eviction rule would be a second mechanism able to hand back the wrong
#: section. The second half of that was never true — every entry is keyed on
#: the geometry it was built from, so an eviction can only cost a recompute,
#: never a wrong answer — and the first half stopped being true the moment ONE
#: EVALUATION started needing repeats.
#:
#: A lap flies the same section at several speeds (``track_points``), and the
#: two panel solves are keyed on geometry alone precisely so the second speed
#: is free. With a frozen full cache it is not free: every speed re-solves.
#: MEASURED on the 2-D section problem (``carsection.py``, four track points,
#: five polars a candidate), evaluating 700 random box points in three
#: blocks, with the freezing policy:
#:
#:     evaluations       0-100    100-300   300-700
#:     cache entries    117/256   256/256   256/256   (full from ~evaluation 250)
#:     ms / evaluation    20.4      36.8      51.2
#:
#: — the cost per candidate grew 2.5x as the cache filled and then stopped
#: taking entries, which is exactly the 5-solves-instead-of-1 the split was
#: designed to avoid. Under least-recently-used eviction the same sweep stays
#: flat, because the entry a candidate needs five times in a row is the most
#: recently used one there is.
_CACHE_MAX = 256

#: how many caller-supplied coordinate arrays to keep addressable. Bounded
#: for a reason a search finds and a test does not: a 20 000-candidate section
#: search hands this module 40 000 distinct loops, and an unbounded store of
#: them is tens of megabytes of arrays nothing will ask for again. The bound
#: is generous against what one evaluation needs — at most four distinct keys
#: are inserted between a store and the lookup that resolves it — so an entry
#: can never be evicted while its own call is still running.
_COORDS_MAX = 64


def _cache_get(cache: dict, key):
    """Least-recently-used read: a hit moves its entry to the fresh end."""
    hit = cache.get(key)
    if hit is not None:
        cache[key] = cache.pop(key)      # dicts keep insertion order
    return hit


def _cache_put(cache: dict, key, val, cap: int = _CACHE_MAX, keep=()):
    """Least-recently-used write. ``keep`` names entries never evicted.

    THE ``pop`` IS LOAD-BEARING. A plain ``cache[key] = val`` on a key that is
    already present leaves its INSERTION POSITION untouched — Python dicts
    only order by first insertion — so an entry written again after a long
    time stays at the oldest end and is the next thing evicted. That is not a
    stale-cache nuisance, it is a KeyError: ``_COORDS_BY_KEY`` is written by
    :func:`_coords_key` and read by :func:`_cascade_inviscid` and
    :func:`_isolated_maps` a few statements later, so an entry evicted between
    the write and the read takes a live lookup with it.

    MEASURED, and it is the shape of thing only a real search finds: with a GA
    re-evaluating its elite, ``cascade_polar`` re-writes the main element's
    key (old position, no refresh), then writes the FLAP's key (new), which
    tips the store over its cap and evicts the main element's — and the next
    line looks it up. Random and Sobol arms never see it, because their
    sections never repeat and their keys are always fresh; the two GA arms
    that carry a designed section died on it at every seed.
    """
    cache.pop(key, None)
    cache[key] = val
    while len(cache) > cap:
        for k in cache:
            if k not in keep:
                del cache[k]
                break
        else:                             # everything is protected
            break
    return val


def _basis(bodies) -> tuple:
    """The two freestream basis solves (module docstring: exact in alpha)."""
    return (panel2d.solve(bodies, 0.0, V=1.0, c_ref=1.0, ref_point=(0.25, 0.0)),
            panel2d.solve(bodies, 0.5 * np.pi, V=1.0, c_ref=1.0,
                          ref_point=(0.25, 0.0)))


def _sweep(s0, s90, alpha_deg: np.ndarray):
    """Per-element ``(cl, cd_dalembert, cm, cp_min)`` over an incidence grid.

    Reconstructed from the two basis solves by the exact superposition of the
    module docstring, then integrated with the SAME pressure-integral
    formulae ``panel2d.solve`` uses (its :class:`~panel2d.BodySolution`
    carries the panel geometry precisely so that a caller can). Every
    coefficient is on the assembly reference chord handed to :func:`_basis`,
    which here is the stowed chord 1, so the elements' values ADD to the
    assembly's.

    The three conventions below are :func:`_basis`'s and are written in here
    rather than passed, because they are not free: V = 1 (so Cp is
    1 - v_t^2), c_ref = 1 (the stowed chord, which is what makes the elements
    add) and the moment reference at the stowed quarter chord. Changing one
    of them means changing both functions.
    """
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
    out = []
    for b0, b90 in zip(s0.bodies, s90.bodies):
        vt = ca * b0.vt[None, :] + sa * b90.vt[None, :]
        cp = 1.0 - vt ** 2
        n = np.column_stack([b0.nx, b0.ny])
        w = cp * b0.panel_len[None, :]
        Fx = -(w @ n[:, 0])
        Fy = -(w @ n[:, 1])
        Mz = -(w @ ((b0.x_cp - 0.25) * n[:, 1] - b0.y_cp * n[:, 0]))
        cl = -np.sin(a) * Fx + np.cos(a) * Fy
        cd = np.cos(a) * Fx + np.sin(a) * Fy
        out.append({"cl": cl, "cd": cd, "cm": -Mz, "cp_min": cp.min(axis=1)})
    return out


def _isolated_maps(tc: float, n_nodes: int, coords_key):
    """``(alpha_deg, cl, cp_min, i_ideal)`` for the ISOLATED base section.

    The two curves the drag bridge and the stall ceiling both need, on the
    same panelling as the cascade. ``i_ideal`` indexes the least-suction
    incidence, which is where the Cp_min curve turns and therefore where its
    two monotone branches meet.
    """
    key = (float(tc), int(n_nodes), coords_key)
    hit = _cache_get(_ISO_CACHE, key)
    if hit is not None:
        return hit
    coords = None if coords_key is None else _coords_of(coords_key)
    x, y = _base_coords(tc, n_nodes, coords)
    body = panel2d.Body(x, y, name="isolated")
    s0, s90 = _basis([body])
    grid = np.arange(-26.0, 26.0 + 1e-9, 0.25)
    (d,) = _sweep(s0, s90, grid)
    return _cache_put(_ISO_CACHE, key,
                      (grid, d["cl"], d["cp_min"],
                       int(np.argmax(d["cp_min"]))))


#: caller-supplied sections, keyed by a digest of their own bytes.
#:
#: A numpy array is not hashable, and the two obvious shortcuts are both
#: wrong in the same way: ``id(arr)`` is reused by CPython the moment the
#: original is collected, so a later, DIFFERENT section can inherit a cached
#: answer, and rounding the coordinates into a key silently merges two
#: sections that differ below the rounding. A content digest does neither.
_COORDS_BY_KEY: dict = {}


def _coords_of(key):
    """The array a content key names, refreshing its place in the store.

    Read through :func:`_cache_get` rather than by plain indexing so that a
    coordinate loop being USED is the freshest entry, not merely a
    not-yet-oldest one. A missing key is a programming error — the key was
    minted by :func:`_coords_key` moments earlier — so it raises.
    """
    got = _cache_get(_COORDS_BY_KEY, key)
    if got is None:
        raise KeyError(
            f"coordinate key {key} is not in the store: it was evicted "
            f"between being minted and being read, which is a bug in the "
            f"cache and not a design failure")
    return got


def _coords_key(coords):
    """Content-addressed key for a caller-supplied section, or None."""
    import hashlib

    if coords is None:
        return None
    arr = np.ascontiguousarray(np.asarray(coords, dtype=float))
    key = ("coords", arr.shape,
           hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
    _cache_put(_COORDS_BY_KEY, key, arr, cap=_COORDS_MAX)
    return key


#: WHOSE SHAPE the suction-peak ceiling is read on, for an element whose own
#: stall has not been measured.
#:
#: ``"naca"`` (the default) solves a NACA 24XX of the element's thickness at
#: that member's own measured stall angle. ``"element"`` solves the element's
#: OWN coordinates at the same angle. For a NACA element the two are the same
#: solve and the same number; they differ only for a caller-supplied section.
#:
#: THE DEFAULT IS ``"naca"`` BECAUSE ``"element"`` IS VACUOUS INSIDE A SEARCH,
#: and that is a measurement rather than a preference.
#:
#: The criterion is "an element may hold a suction peak no deeper than the one
#: this section holds ALONE at its own stall". With ``"element"`` the ceiling
#: is the element's OWN Cp_min at a fixed incidence, so the test degenerates
#: into ``Cp_min(alpha_flown) >= Cp_min(alpha_stall_of_a_NACA_of_this
#: thickness)`` — a bound on the ANGLE, not on the peak, and one a section can
#: raise for itself simply by having a deeper peak at that angle. It rewards a
#: SHARP nose, which is the opposite of what a high-lift section wants.
#:
#: MEASURED, and the average does not show it. Over 24 uniformly drawn
#: designed sections (``carsection``'s own box, rng 7, Re 6e5), against the
#: stall XFOIL measures on the shape itself, the median |ceiling/measured - 1|
#: is 0.164 for ``"element"`` and 0.399 for ``"naca"`` — so on a RANDOM draw
#: the element's own shape is the better substitute, by more than a factor of
#: two. At the ARGMAX it is the opposite, because an optimiser does not draw
#: at random: on this study's own 2-D winners the ``"element"`` ceilings came
#: out at -15.6 and -23.1 against measured values of -10.2 and -9.7, i.e. 1.5x
#: and 2.4x too permissive, and BOTH winners were infeasible once the stall
#: was measured on them. The worst ``"element"`` case in the random draw is
#: 5.5x too permissive against 2.2x for ``"naca"``.
#:
#: A criterion used inside a search has to be judged at its argmax and not at
#: a random draw. ``"naca"`` is worse on average and cannot be gamed by shape
#: — the only lever left is the thickness, which is bounded to the range the
#: bank resolves — so it is the default. ``"element"`` stays reachable because
#: it is the more accurate estimator when nobody is optimising against it, and
#: because the measurement above needs both.
CEILING_SHAPES = ("naca", "element")


def suction_peak_ceiling(tc: float, n_nodes: int = N_SECTION_NODES,
                         coords: np.ndarray | None = None,
                         shape: str = "naca") -> tuple[float, float]:
    """``(Cp_min_ceiling, alpha_stall_deg)`` for the base section — DERIVED.

    ``shape`` selects whose loop the ceiling is read on
    (:data:`CEILING_SHAPES`); it changes nothing when ``coords`` is ``None``,
    because then the element IS the NACA.

    The peak inviscid suction the ISOLATED section is measured to sustain,
    and the incidence at which it does. Built in two steps, both from data
    already in the tree (module docstring for the argument):

      1. ``polar.stall_point`` on ``polar.stall_polar_family()`` gives the
         RESOLVED stall incidence at each anchor thickness. Censored members
         are not anchors — a march that gave up is a lower bound, not a peak
         — which is the rule ``stall.section_clmax`` already applies, and at
         Re 1e6 it excludes t/c = 0.06.
      2. ``panel2d`` solves the isolated section at that incidence, on THIS
         module's panelling, and reports its Cp_min.

    Between anchors the stall INCIDENCE is interpolated in t/c and the
    section is then solved exactly, so no pressure is ever interpolated.

    Raises ValueError outside the resolved anchor range: a section with no
    measured stall gets no ceiling rather than an extrapolated one. Callers
    inside an optimiser reach this through
    :func:`evaluate_car_wing_multi`, which turns it into a refusal reason.
    """
    from .polar import stall_point, stall_polar_family

    if str(shape) not in CEILING_SHAPES:
        raise ValueError(
            f"unknown ceiling shape {shape!r}; choose from "
            f"{list(CEILING_SHAPES)}")
    # "naca" reads the ceiling on the NACA 24XX of this thickness, so the
    # caller's coordinates are used for nothing here and must not enter the
    # key either — otherwise one ceiling would be cached under as many keys as
    # there are sections of that thickness
    ckey = _coords_key(coords) if str(shape) == "element" else None
    key = (float(tc), int(n_nodes), ckey)
    hit = _cache_get(_CEIL_CACHE, key)
    if hit is not None:
        return hit

    fam = stall_polar_family()
    pts = {t: stall_point(p, t) for t, p in fam.members.items()}
    anchors = np.array(sorted(t for t, sp in pts.items() if not sp.censored),
                       dtype=float)
    if anchors.size == 0:
        raise ValueError(
            "no member of the wide-alpha NACA 24XX family resolved a cl peak "
            "(all censored): there is no suction-peak ceiling to build")
    lo, hi = float(anchors[0]), float(anchors[-1])
    tcf = float(tc)
    if tcf < lo - 1e-12 or tcf > hi + 1e-12:
        raise ValueError(
            f"t/c = {tcf:.4f} is outside the RESOLVED stall range "
            f"[{lo:g}, {hi:g}] of the wide-alpha NACA 24XX family; the "
            "suction-peak ceiling is built from a measured stall and "
            "censored members are not anchors (see stall.section_clmax)")
    a_stall = float(np.interp(
        tcf, anchors, [pts[float(t)].alpha_stall_deg for t in anchors]))

    grid, _cl, cpm, _i = _isolated_maps(tcf, n_nodes, ckey)
    ceiling = float(np.interp(a_stall, grid, cpm))
    return _cache_put(_CEIL_CACHE, key, (ceiling, a_stall))


def _equivalent_alpha(cp_min: np.ndarray, cl_own: np.ndarray,
                      grid: np.ndarray, cl_iso: np.ndarray,
                      cpm_iso: np.ndarray, i_ideal: int) -> np.ndarray:
    """Incidence at which the ISOLATED section has this suction peak [deg].

    The drag bridge of the module docstring. ``cp_min`` is the element's own
    inviscid peak and ``cl_own`` only picks the BRANCH: the isolated Cp_min
    curve is strictly monotone either side of its least-suction incidence
    (measured; see the module docstring), so an element loaded with the
    section's camber is matched above that incidence and one loaded against
    it below.
    """
    a_up, cp_up = grid[i_ideal:], cpm_iso[i_ideal:]
    a_dn, cp_dn = grid[:i_ideal + 1][::-1], cpm_iso[:i_ideal + 1][::-1]
    # np.interp needs an increasing x: Cp_min DECREASES along both branches
    # as one moves away from the ideal incidence, so both are reversed.
    up = np.interp(cp_min, cp_up[::-1], a_up[::-1])
    dn = np.interp(cp_min, cp_dn[::-1], a_dn[::-1])
    return np.where(cl_own >= float(cl_iso[i_ideal]), up, dn)


def _wide_polar(tc: float, re: float):
    """Wide-alpha NACA 24XX polar at ``tc`` and Reynolds number ``re``.

    The shipped ``*_stall.pol`` bank, read at the requested Reynolds number
    through ``polar.polar_at_re`` (log cd against log Re — that function's own
    derivation) and blended in t/c by ``polar.PolarFamily``, which is exact at
    a member and refuses outside the anchors.

    ``re`` is CLAMPED to the bank rather than extrapolated or refused: a flap
    of 0.15 chords on a 0.10 m^2 wing reaches Re ~ 3e4, below the bank's
    1e5 floor, and refusing a design because a real table stops is a ban
    dressed as physics. The clamp is reported (``re_clamped_*``) and its
    direction is known: cd RISES as Re falls (measured in ``polar.py``'s own
    bank note — NACA 2412 at alpha 2 runs 0.00578 at Re 1e6 and 0.01551 at
    1e5), so a clamped-up Reynolds number UNDER-charges the element and the
    model is OPTIMISTIC there.

    Returns ``(polar, re_used, clamped)``.
    """
    from pathlib import Path

    from .polar import (RE_BANK_TAGS, PolarFamily, load_xfoil_polar,
                        polar_at_re)

    bank = _cache_get(_STALL_BANK_CACHE, "bank")
    if bank is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
        bank = {}
        for re_tag, tag in RE_BANK_TAGS:
            for pol in sorted(data_dir.glob(f"naca24??_re{tag}_stall.pol")):
                code = pol.stem[4:8]
                t = int(code[2:]) / 100.0
                bank.setdefault(t, {})[float(re_tag)] = load_xfoil_polar(
                    pol, name=f"NACA{code} Re{tag} stall (XFOIL)")
        if len(bank) < 2:
            raise FileNotFoundError(
                f"need >= 2 naca24??_re*_stall.pol files in {data_dir}; run "
                "scripts/gen_stall_polar_family.py")
        _STALL_BANK_CACHE["bank"] = bank

    res = sorted({r for cell in bank.values() for r in cell})
    re_used = float(min(max(float(re), res[0]), res[-1]))
    clamped = abs(re_used - float(re)) > 1e-9 * max(float(re), 1.0)
    key = ("wide", round(float(tc), 12), round(re_used, 6))
    hit = _cache_get(_STALL_BANK_CACHE, key)
    if hit is None:
        members = {}
        for t, cell in bank.items():
            rr = sorted(cell)
            if re_used < rr[0] - 1e-9 or re_used > rr[-1] + 1e-9:
                continue                      # ragged cell: this t/c cannot
            members[t] = polar_at_re(t, re_used, bank=bank)
        if len(members) < 2:
            raise FileNotFoundError(
                f"the wide-alpha bank has fewer than two thicknesses at "
                f"Re = {re_used:.3g}")
        hit = _cache_put(_STALL_BANK_CACHE, key, PolarFamily(members),
                         keep=("bank",))
    return hit.at(float(tc)), re_used, clamped


# --------------------------------------------------------------- the polar


@dataclass(frozen=True, eq=False)
class CascadePolar:
    """The two-element section as a polar the lattice can fly.

    Same interface as ``polar.TablePolar`` where it matters — ``a_lin``,
    ``alpha_L0``, ``cl``/``cd``/``cm``/``cp_min`` interpolants and
    ``alpha_valid`` — so every consumer of a section in this package works
    unchanged. What it carries BEYOND a section polar is the per-element
    split, because a two-element section that reported only its total would
    hide the one thing the design vector is moving.

    Arrays are tabulated on :data:`ALPHA_TABLE_DEG`. ``cl_*`` and ``cd_*``
    fields named for an element are on the ASSEMBLY reference chord (so they
    add to the total); ``cl_own_*`` are on that element's OWN chord, and
    ``share_*`` reconciles the two and sums to 1.

    ``alpha_valid`` is NOT the tabulated range: it is the widest contiguous
    run of incidences on which BOTH elements stay above their own measured
    suction-peak ceiling. Outside it the section is refused, which is the
    same contract ``carwing.py`` has always had with its polar's validity
    window.
    """

    alpha_deg: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CM: np.ndarray
    CPMIN: np.ndarray
    cl_main: np.ndarray
    cl_flap: np.ndarray
    cl_own_main: np.ndarray
    cl_own_flap: np.ndarray
    share_main: np.ndarray
    cd_main: np.ndarray
    cd_flap: np.ndarray
    cpmin_main: np.ndarray
    cpmin_flap: np.ndarray
    alpha_eq_main: np.ndarray
    alpha_eq_flap: np.ndarray
    a_lin: float                 # [1/rad]
    alpha_L0: float              # [rad] — polar.py's unit, not degrees
    alpha_valid: tuple           # [deg]
    tc: float                    # the MAIN element's thickness/chord
    tc_flap: float               # the FLAP's own, measured on its own shape
    c_main: float
    c_flap: float
    slot_width_frac: float
    slot_gap_te_frac: float
    cp_ceiling_main: float
    cp_ceiling_flap: float
    alpha_stall_iso_deg: float
    alpha_stall_iso_flap_deg: float
    #: where the two ceilings came from: ``"naca24xx-substitution"`` (the
    #: shipped path — a NACA 24XX's measured stall at the element's own
    #: thickness, standing in for the element's own shape) or ``"measured"``
    #: (a stall measured on the shape actually being flown). It is a field and
    #: not a comment because the substitution is the model's largest single
    #: assumption and a reader of a result must be able to see which one it is.
    ceiling_source: str
    ceiling_scale: float
    #: whose loop the substituted ceiling was read on
    #: (:data:`CEILING_SHAPES`). Meaningless when ``ceiling_source`` is
    #: ``"measured"``, and recorded anyway so a stored row says which of the
    #: two substitutions it would have used.
    ceiling_shape: str
    #: the flap flies a shape of its own (given coordinates, or its own NACA
    #: 24XX thickness) rather than the main element's scaled down.
    flap_is_own_section: bool
    #: each element's shape came from CALLER COORDINATES rather than from the
    #: NACA 24XX generator — i.e. a NACA-24XX-derived ceiling is a
    #: SUBSTITUTION for that element and not a measurement of it. Two flags
    #: and not one, because the two elements can differ in this
    #: independently, and a label that said "designed" of an assembly with one
    #: designed element would be true of the assembly and false of an element.
    main_is_designed: bool
    flap_is_designed: bool
    re_main: float
    re_flap: float
    re_clamped_main: bool
    re_clamped_flap: bool
    name: str = "cascade"
    has_cp_min: bool = True

    @property
    def cp_ceiling(self) -> float:
        """The MAIN element's suction-peak ceiling.

        It kept the unqualified name through the change that gave the flap a
        section of its own, and that is a decision rather than an oversight.
        Every consumer that reads ``cp_ceiling`` — this family's own refusal
        message, ``api.design_report``, the tests — was written when there was
        exactly ONE ceiling because there was exactly one shape, and on that
        path :attr:`cp_ceiling_flap` is equal to it by construction. So the
        name still means what those readers meant by it, and a reader that
        needs to know the flap is different asks :attr:`cp_ceiling_flap` or
        :attr:`flap_is_own_section` rather than being silently handed the
        main element's number under a name that suggests it is the section's.
        """
        return float(self.cp_ceiling_main)

    def _at(self, table, alpha_deg):
        return np.interp(np.asarray(alpha_deg, dtype=float),
                         self.alpha_deg, table)

    def cl(self, alpha_deg):
        return self._at(self.CL, alpha_deg)

    def cd(self, alpha_deg):
        return self._at(self.CD, alpha_deg)

    def cm(self, alpha_deg):
        return self._at(self.CM, alpha_deg)

    def cp_min(self, alpha_deg):
        return self._at(self.CPMIN, alpha_deg)

    def share(self, alpha_deg):
        """Main element's share of the section's lift at an incidence."""
        return self._at(self.share_main, alpha_deg)

    def stall_margin(self, alpha_deg):
        """``(main, flap)`` suction-peak margins: >= 0 is under the ceiling.

        ``1 - Cp_min / Cp_ceiling``, both negative, so the margin is the
        fraction of the measured ceiling still unspent — the same normalised
        form every gate in this package reports (``airfoil.norm_margin``).

        EACH ELEMENT AGAINST ITS OWN CEILING. When the two elements are the
        same shape the two ceilings are the same number and this is what it
        always was; when they are not, dividing the flap's peak by the main
        element's ceiling would report a margin against a shape the flap is
        not.
        """
        return (1.0 - self._at(self.cpmin_main, alpha_deg) / self.cp_ceiling_main,
                1.0 - self._at(self.cpmin_flap, alpha_deg) / self.cp_ceiling_flap)


def _cascade_inviscid(tc, ff, defl, gap, ovl, n_nodes, coords_key,
                      flap_coords_key=None, flap_tc=None):
    """The geometry-only half of a cascade section: bodies + basis sweep.

    Cached on exactly the arguments the two panel solves depend on — BOTH
    sections, the four slot rows and the panelling — and NOT on the Reynolds
    number, which only enters the viscous tables built on top of this. That
    split is the whole point of the cache: the 5.8 ms is here.

    Returns ``(data, None)`` or ``(None, reason)``; a geometry that cannot be
    solved is an infeasible DESIGN, not a programming error.
    """
    key = (round(float(tc), 12), round(float(ff), 12), round(float(defl), 12),
           round(float(gap), 12), round(float(ovl), 12), int(n_nodes),
           coords_key, flap_coords_key,
           None if flap_tc is None else round(float(flap_tc), 12))
    hit = _cache_get(_CASCADE_CACHE, key)
    if hit is not None:
        return hit

    coords = None if coords_key is None else _coords_of(coords_key)
    fcoords = (None if flap_coords_key is None
               else _coords_of(flap_coords_key))
    try:
        (xm, ym), (xf, yf) = cascade_coords(tc, ff, defl, gap, ovl,
                                            n_nodes, coords,
                                            flap_coords=fcoords,
                                            flap_tc=flap_tc)
    except (ValueError, IndexError) as exc:
        return None, f"section geometry: {exc}"

    reason = panel2d.check_body(xm, ym) or panel2d.check_body(xf, yf)
    if reason is not None:
        return None, f"element geometry: {reason}"
    main = panel2d.Body(xm, ym, name="main")
    flap = panel2d.Body(xf, yf, name="flap")

    if _loops_cross(main.nodes, flap.nodes):
        return None, (
            f"the flap intersects the main element (chord fraction "
            f"{ff:.3f}, deflection {defl:.2f} deg, gap {gap:.4f}, overlap "
            f"{ovl:.4f} of the stowed chord): that is not a slot, it is one "
            f"body drawn through another")
    width = _min_gap(main.nodes, flap.nodes)
    floor = SLOT_WIDTH_PANELS * float(main.geom.L[0])
    if width < floor:
        return None, (
            f"the slot is {width:.3e} chords wide against a resolution floor "
            f"of {floor:.3e} ({SLOT_WIDTH_PANELS:g} main-element "
            f"trailing-edge panels at {n_nodes} nodes). Below that the panel "
            f"solution's discretisation error stops shrinking and changes "
            f"sign (measured: +1.07 % at width/L_TE = 0.47 against a flat "
            f"-0.33 % from one panel up). Open the gap, cut the overlap, "
            f"reduce the deflection, or refine the section")

    s0, s90 = _basis([main, flap])
    dm, df = _sweep(s0, s90, ALPHA_TABLE_DEG)
    val = ({"main": dm, "flap": df,
            "c_main": 1.0 - float(ff), "c_flap": float(ff),
            "slot_width_frac": width,
            "slot_gap_te_frac": slot_gap_te(main.nodes, flap.nodes),
            "kutta": max(s0.kutta_residual, s90.kutta_residual),
            "tangency": max(s0.tangency_residual, s90.tangency_residual)},
           None)
    return _cache_put(_CASCADE_CACHE, key, val)


def cascade_polar(tc: float, flap_chord_frac: float,
                  flap_deflection_deg: float, slot_gap_frac: float,
                  slot_overlap_frac: float, re_ref: float,
                  n_nodes: int = N_SECTION_NODES,
                  coords: np.ndarray | None = None,
                  flap_coords: np.ndarray | None = None,
                  flap_tc: float | None = None,
                  ceiling_scale: float = 1.0,
                  gap_te_min_frac: float | None = None,
                  ceilings: tuple | None = None,
                  ceiling_shape: str = "naca"):
    """The two-element section polar, or a refusal reason.

    ``re_ref`` is the Reynolds number on the STOWED chord; each element's own
    Reynolds number is that scaled by its own chord fraction, which is the
    number its viscous table is read at.

    ``flap_coords`` / ``flap_tc`` give the flap a section of its own; with
    both ``None`` it is the main element's shape scaled to the flap chord and
    every number below is bit-for-bit what it was before the flap could have
    one (:func:`cascade_coords`).

    ``ceiling_scale`` multiplies BOTH elements' suction-peak ceilings. It is
    the criterion-sensitivity knob and nothing else: 1.0 is the criterion,
    values above 1 make it MORE permissive (the ceilings are negative, so
    scaling up deepens them) and below 1 stricter. It exists because the
    ceiling is a MODEL and a result that moves under it is a result about the
    model. Default 1.0 changes nothing.

    ``gap_te_min_frac`` refuses a rigging whose trailing-edge gap
    (:func:`slot_gap_te`) is below it — the non-merging floor an inviscid
    method cannot find for itself (:data:`SLOT_GAP_TE_NON_MERGING_FRAC`).
    ``None`` is off and is the default, because switching it on changes which
    designs this family admits and every published number of this family was
    measured with it off.

    ``ceiling_shape`` selects whose loop each element's substituted ceiling is
    read on (:data:`CEILING_SHAPES`). It changes nothing unless an element
    flies caller-supplied coordinates, and the default is ``"naca"`` because
    the alternative is vacuous inside a search — that constant's own comment
    carries the measurement.

    ``ceilings`` replaces the NACA-24XX substitution with a MEASURED pair per
    element, ``((cp_main, a_stall_main), (cp_flap, a_stall_flap))``. It is
    what :mod:`aerobo.section_stall` hands back after running XFOIL on the
    element's actual shape, and it is the only way to get a ceiling here that
    is about the shape being flown rather than about a NACA 24XX of the same
    thickness. ``None`` uses the substitution and says so
    (``ceiling_source``).

    Returns ``(CascadePolar, None)`` or ``(None, reason)``. Every failure in
    here is a design that cannot be flown — an intersecting flap, an
    unresolvable slot, a thickness with no measured stall, an incidence
    window that is empty — so all of them come back as reasons, never as
    exceptions (the ``objective._fail`` contract).
    """
    ckey = _coords_key(coords)
    fkey = _coords_key(flap_coords)
    tcf = float(tc) if coords is None else _thickness_of(
        *_base_coords(tc, n_nodes, coords))
    # THE FLAP'S OWN THICKNESS, measured from the shape it will actually fly.
    # With no section of its own the flap IS the main element scaled, so its
    # thickness/chord is the main element's — the same number, not a second
    # measurement of it, so the legacy path stays bit-for-bit.
    if flap_coords is None and flap_tc is None:
        tcf_flap = tcf
    elif flap_coords is None:
        tcf_flap = float(flap_tc)
    else:
        tcf_flap = _thickness_of(*_base_coords(
            tcf if flap_tc is None else flap_tc, n_nodes, flap_coords))

    scale = float(ceiling_scale)
    if not scale > 0.0:
        raise ValueError(
            f"ceiling_scale must be > 0 (it multiplies a negative pressure "
            f"ceiling), got {ceiling_scale!r}")
    if ceilings is None:
        ceiling_source = "naca24xx-substitution"
        try:
            ceil_m, a_stall_iso = suction_peak_ceiling(
                tcf, n_nodes, coords, ceiling_shape)
        except ValueError as exc:
            return None, f"suction-peak ceiling (main element): {exc}"
        if tcf_flap == tcf and flap_coords is None:
            ceil_f, a_stall_f = ceil_m, a_stall_iso
        else:
            try:
                ceil_f, a_stall_f = suction_peak_ceiling(
                    tcf_flap, n_nodes, flap_coords, ceiling_shape)
            except ValueError as exc:
                return None, f"suction-peak ceiling (flap): {exc}"
    else:
        ceiling_source = "measured"
        (ceil_m, a_stall_iso), (ceil_f, a_stall_f) = (
            (float(a), float(b)) for a, b in ceilings)
    ceiling = ceil_m * scale
    ceiling_flap = ceil_f * scale

    data, reason = _cascade_inviscid(tcf, flap_chord_frac, flap_deflection_deg,
                                     slot_gap_frac, slot_overlap_frac,
                                     int(n_nodes), ckey,
                                     flap_coords_key=fkey,
                                     flap_tc=flap_tc)
    if reason is not None:
        return None, reason

    if gap_te_min_frac is not None:
        te_gap = float(data["slot_gap_te_frac"])
        if te_gap < float(gap_te_min_frac):
            return None, (
                f"the trailing-edge gap is {te_gap:.4f} chords against a "
                f"non-merging floor of {float(gap_te_min_frac):.4f}. An "
                f"inviscid solve has no mechanism by which a wake and a "
                f"boundary layer can merge, so below the floor it reports the "
                f"lift of a flow that would already have separated (Smith, "
                f"High-Lift Aerodynamics, pp. 515-516; see "
                f"SLOT_GAP_TE_NON_MERGING_FRAC). Open the gap, cut the "
                f"overlap, or reduce the deflection")

    grid = ALPHA_TABLE_DEG
    dm, df = data["main"], data["flap"]
    c_m, c_f = data["c_main"], data["c_flap"]
    CL = dm["cl"] + df["cl"]
    CM = dm["cm"] + df["cm"]
    CPMIN = np.minimum(dm["cp_min"], df["cp_min"])

    # --- the isolated maps, then each element's equivalent incidence. ONE
    # PER ELEMENT, because the map Cp_min -> alpha is a property of the SHAPE
    # and the two elements need not be the same shape any more. With no flap
    # section of its own the second call is the first one's cache hit and the
    # two maps are the same object, so the legacy path is unchanged.
    iso_grid, iso_cl, iso_cpm, i_ideal = _isolated_maps(tcf, int(n_nodes), ckey)
    if fkey is None and tcf_flap == tcf:
        f_grid, f_cl, f_cpm, f_ideal = iso_grid, iso_cl, iso_cpm, i_ideal
    else:
        f_grid, f_cl, f_cpm, f_ideal = _isolated_maps(
            tcf_flap, int(n_nodes), fkey)
    cl_own_m = dm["cl"] / c_m
    cl_own_f = df["cl"] / c_f
    aeq_m = _equivalent_alpha(dm["cp_min"], cl_own_m, iso_grid, iso_cl,
                              iso_cpm, i_ideal)
    aeq_f = _equivalent_alpha(df["cp_min"], cl_own_f, f_grid, f_cl,
                              f_cpm, f_ideal)

    try:
        pol_m, re_m, clip_m = _wide_polar(tcf, float(re_ref) * c_m)
        pol_f, re_f, clip_f = _wide_polar(tcf_flap, float(re_ref) * c_f)
    except (ValueError, FileNotFoundError) as exc:
        return None, f"element drag table: {exc}"

    # --- the flyable incidence window. TWO conditions, and they are both
    # part of "can this section be flown here", so they belong in the SAME
    # window rather than one being a window and the other a separate refusal:
    #
    #   * both elements under the measured suction-peak ceiling (stall), and
    #   * both elements' equivalent incidences inside the wide-alpha table
    #     that prices their drag (data).
    #
    # The second bites on the LOW side, and for a reason worth stating: a
    # two-element section's zero-lift angle is far negative, so at strongly
    # negative incidence the MAIN element is loaded against its own camber
    # and its equivalent incidence walks off the bottom of a table that stops
    # at -6 deg. That is not a stall, it is the end of the data, and the
    # window says so rather than the module inventing a number there.
    lo_m, hi_m = pol_m.alpha_valid
    lo_f, hi_f = pol_f.alpha_valid
    # EACH ELEMENT AGAINST ITS OWN CEILING. The two ceilings are equal when
    # the two elements are the same shape, which is what makes the
    # single-shape path bit-for-bit; they are not equal otherwise, and
    # judging a designed flap against the main element's ceiling would refuse
    # or admit it for a shape it does not have.
    stall_ok = (dm["cp_min"] >= ceiling) & (df["cp_min"] >= ceiling_flap)
    data_ok = ((aeq_m >= lo_m - 1e-9) & (aeq_m <= hi_m + 1e-9)
               & (aeq_f >= lo_f - 1e-9) & (aeq_f <= hi_f + 1e-9))
    lo_i, hi_i = _longest_run(stall_ok & data_ok)
    if lo_i is None:
        why = ("every incidence puts an element past its ceiling"
               if not np.any(stall_ok) else
               "no incidence that clears the ceiling also has both elements "
               "inside the wide-alpha drag table")
        same = (ceiling == ceiling_flap)
        ceil_txt = (f"Cp_min = {ceiling:.3f} (the isolated section's own "
                    f"measured stall peak, at {a_stall_iso:.2f} deg)" if same
                    else (f"Cp_min = {ceiling:.3f} on the main element (its "
                          f"own measured stall peak, at {a_stall_iso:.2f} "
                          f"deg) and {ceiling_flap:.3f} on the flap (at "
                          f"{a_stall_f:.2f} deg)"))
        return None, (
            f"no incidence in [{grid[0]:g}, {grid[-1]:g}] deg can fly this "
            f"slot: {why}. The suction-peak ceiling is {ceil_txt} and the "
            f"deepest peaks over the sweep are {dm['cp_min'].min():.2f} on "
            f"the main element and {df['cp_min'].min():.2f} on the flap")
    a_lo, a_hi = float(grid[lo_i]), float(grid[hi_i])

    # The table is CLIPPED at its own ends only so that the tabulated arrays
    # are finite everywhere; every clipped station is outside the window
    # computed just above and is therefore refused before anything reads it
    # (gated by ``test_no_flyable_incidence_reads_a_clipped_drag``).
    cd_m = c_m * np.asarray(pol_m.cd(np.clip(aeq_m, lo_m, hi_m)), dtype=float)
    cd_f = c_f * np.asarray(pol_f.cd(np.clip(aeq_f, lo_f, hi_f)), dtype=float)
    CD = cd_m + cd_f

    # --- the linear pair, fitted on the FLYABLE window only. Outside it the
    # section is refused, so a slope fitted through incidences nobody can fly
    # would set the lattice's section slope from stalled data.
    sl = slice(lo_i, hi_i + 1)
    if hi_i - lo_i < 2:
        return None, (
            f"the flyable incidence window is {grid[hi_i] - grid[lo_i]:.2f} "
            f"deg wide — too narrow to state a lift-curve slope on")
    coef = np.polyfit(np.deg2rad(grid[sl]), CL[sl], 1)
    a_lin = float(coef[0])                     # [1/rad], the package's unit
    if not a_lin > 0.0:
        return None, (
            f"the section's fitted lift-curve slope is {a_lin:.4g} /rad over "
            f"its flyable window: not a lifting section")
    alpha_L0 = float(-coef[1] / coef[0])       # [rad], polar.py's convention

    share_m = np.where(np.abs(CL) > 1e-12, dm["cl"] / np.where(
        np.abs(CL) > 1e-12, CL, 1.0), np.nan)

    return CascadePolar(
        alpha_deg=grid, CL=CL, CD=CD, CM=CM, CPMIN=CPMIN,
        cl_main=dm["cl"], cl_flap=df["cl"],
        cl_own_main=cl_own_m, cl_own_flap=cl_own_f,
        share_main=share_m, cd_main=cd_m, cd_flap=cd_f,
        cpmin_main=dm["cp_min"], cpmin_flap=df["cp_min"],
        alpha_eq_main=aeq_m, alpha_eq_flap=aeq_f,
        a_lin=a_lin, alpha_L0=alpha_L0, alpha_valid=(a_lo, a_hi),
        tc=tcf, tc_flap=float(tcf_flap), c_main=c_m, c_flap=c_f,
        slot_width_frac=data["slot_width_frac"],
        slot_gap_te_frac=float(data["slot_gap_te_frac"]),
        cp_ceiling_main=float(ceiling),
        cp_ceiling_flap=float(ceiling_flap),
        alpha_stall_iso_deg=float(a_stall_iso),
        alpha_stall_iso_flap_deg=float(a_stall_f),
        ceiling_source=str(ceiling_source), ceiling_scale=scale,
        ceiling_shape=str(ceiling_shape),
        flap_is_own_section=bool(flap_coords is not None or flap_tc is not None),
        main_is_designed=bool(coords is not None),
        flap_is_designed=bool(flap_coords is not None
                              or (coords is not None and flap_coords is None
                                  and flap_tc is None)),
        re_main=float(re_m), re_flap=float(re_f),
        re_clamped_main=bool(clip_m), re_clamped_flap=bool(clip_f),
        name=(f"cascade NACA24{int(round(tcf * 100)):02d}{_flap_name(flap_coords, flap_tc, tcf_flap)} "
              f"flap {flap_chord_frac:.3f}c at {flap_deflection_deg:.1f} deg, "
              f"gap {slot_gap_frac:.3f}c overlap {slot_overlap_frac:.3f}c"),
    ), None


def _flap_name(flap_coords, flap_tc, tcf_flap) -> str:
    """The FLAP's share of a section's name, or "" when it has none.

    Empty for the single-shape path, so that path's name is bit-for-bit what
    it was; otherwise it names the flap's own section, because a polar called
    ``cascade NACA2412`` whose flap is a designed shape is a label that lies
    about the thing it is on.
    """
    if flap_coords is not None:
        return f" / flap {_thickness_label(tcf_flap)} (given coordinates)"
    if flap_tc is not None:
        return f" / flap NACA24{int(round(float(tcf_flap) * 100)):02d}"
    return ""


def _thickness_label(tc: float) -> str:
    return f"t/c {float(tc):.4f}"


def _ceiling_section_label(pol: "CascadePolar") -> str:
    """WHAT the suction-peak ceiling was actually measured on — per element.

    This string is the family's flag on its largest single substitution, and
    it got louder when the flap gained a section of its own, on purpose. With
    one shape it reads exactly as it always did:

        NACA2412 (wide-alpha XFOIL, Re 1e6)

    With two, one name cannot be true of both elements, so it names both and
    says which side of each is the substitute:

        main NACA2412 / flap NACA2409 (wide-alpha XFOIL, Re 1e6)
        main NACA2412 / flap NACA2409 SUBSTITUTED for a designed section
          (wide-alpha XFOIL, Re 1e6)

    THE DECISION THIS LABEL RECORDS. A designed flap is judged against the
    stall of a NACA 24XX of the same thickness, which is a shape it is not.
    Three responses were open — refuse a designed section outright, bound the
    thickness range, or flag it harder — and the family does the second and
    third and declines the first. Refusing would make the freedom
    unreachable, which is a ban on a whole class of design in the name of a
    model's limits; the thickness range is bounded already and per-element now
    (``suction_peak_ceiling`` refuses outside the RESOLVED stall anchors,
    t/c in [0.09, 0.18], and each element is checked at its OWN thickness);
    and the flag is this string plus ``ceiling_source``. What closes the
    substitution rather than labelling it is a stall MEASURED on the shape
    being flown — ``aerobo.section_stall.measured_ceilings`` — which is why
    ``ceiling_source`` exists to say which of the two a number came from.
    """
    substituted = str(pol.ceiling_source) == "naca24xx-substitution"
    tail = ("(wide-alpha XFOIL, Re 1e6)" if substituted else
            f"(measured on the section flown; source {pol.ceiling_source})")

    def one(tc, designed):
        s = f"NACA24{int(round(float(tc) * 100)):02d}"
        return s + (" SUBSTITUTED for a designed section"
                    if designed and substituted else "")

    if not pol.flap_is_own_section:
        return f"{one(pol.tc, pol.main_is_designed)} {tail}"
    return (f"main {one(pol.tc, pol.main_is_designed)} / "
            f"flap {one(pol.tc_flap, pol.flap_is_designed)} {tail}")


def _longest_run(mask: np.ndarray):
    """``(lo, hi)`` index bounds of the longest True run, or ``(None, None)``."""
    m = np.asarray(mask, dtype=bool)
    best = (None, None, 0)
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < m.size and m[j + 1]:
            j += 1
        if j - i + 1 > best[2]:
            best = (i, j, j - i + 1)
        i = j + 1
    return best[0], best[1]


# --------------------------------------------------------------- problem


@dataclass
class CarWingMultiProblem:
    """Downforce maximisation for a TWO-ELEMENT rear wing.

    ``carwing.CarWingProblem`` with four rows added and nothing removed: the
    mount layouts, the beam, the budgets, the objective set and the failure
    contract are that class's, imported rather than restated. Read its
    docstring for all of those; what follows is only what is different.

    Design vector::

        x = [taper, twist_root_deg, twist_tip_deg, alpha_deg,
             endplate_h_m, ride_height_m,
             flap_chord_frac, flap_deflection_deg,
             slot_gap_frac, slot_overlap_frac,
             (S_m2,) b_m]                     (+ chord coefficients)

    THE FREE-AREA TWIN follows ``carwing.py``'s own choice, and for its own
    reason rather than for consistency's sake: with the reference area free,
    ``objective='cz'`` is maximised by shrinking the area it is referenced
    to, so it is refused here as well. A two-element section does not change
    that argument — it makes it worse, because the slot buys CZ and the
    shrink would then be credited to the slot.

    ``tc`` IS NOT A DESIGN VARIABLE, exactly as in the single-element family,
    but it now has a narrower legal range: the suction-peak ceiling is built
    from a MEASURED stall, and the wide-alpha family resolves one only for
    t/c in [0.09, 0.18] (t/c 0.06 is censored). Outside that the family
    refuses in contract with the reason rather than guessing a ceiling.

    Objective (MAXIMISE) and constraints: identical to
    ``carwing.CarWingProblem`` — see :attr:`constraint_labels`.
    """

    b: float = 1.6
    S: float = 0.4
    N_vlm: int = 40
    n_endplate: int = 8
    V: float = 55.0
    rho: float = RHO_AIR
    mu: float = MU_AIR
    tc: float = 0.12
    #: which plate-borne layout (carwing.MOUNTS) — where along the span the
    #: sheets grip. Both layouts take the load out through the plates; the
    #: single endplate this family carries is at the tip, so that is the
    #: default here too.
    mount: str = "tips"
    strut_chord: float = 0.12
    strut_cf: float = 0.005
    ei_nm2: float = 4.0e4
    CD_budget: float | None = 0.11
    drag_budget_n: float | None = None
    downforce_min_n: float | None = None
    deflection_limit_m: float = 0.010
    blend_frac: float = 0.0
    blend_shape: str = "arc"
    #: the plate's cant and chord law, exactly as carwing.CarWingProblem
    #: documents them. This family carries ONE plate per side rather than a
    #: pair (a second sheet at the same station makes the influence matrix
    #: singular), which changes what a plate costs and not what it IS.
    endplate_cant_deg: float = 90.0
    #:
    #: PRICED since the span row became the OVERALL width here too:
    #: ``evaluate_car_wing_multi`` takes the plate's outboard projection out
    #: of the wing's span before anything is flown, so a lean costs. It was
    #: withheld from the api while b_m meant the wing's span (cant 60 scored
    #: 39.79 against 90's 38.28 by flying 1.725 m inside a 1.6 m band), which
    #: never closed the hole — ``blend_frac`` was declared and used it.
    endplate_chord_follows: bool = False
    junction_drag: bool = True
    chord_order: int = 0
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND
    chord_law: str = geometry.DEFAULT_CHORD_LAW
    chord_limits: "geometry.ChordLimits | None" = None
    #: nodes per element for the 2-D solve. See :data:`N_SECTION_NODES` for
    #: the measured price of raising it; the suction-peak ceiling is rebuilt
    #: at the same count, so the criterion and the quantity it judges never
    #: sit on different grids.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None
    n_section_nodes: int = N_SECTION_NODES
    #: fly a CHOSEN base section (an (n, 2) closed loop in the package's
    #: order) instead of the NACA 24XX at :attr:`tc`. This is the MAIN
    #: element's shape, and it is the flap's too unless :attr:`flap_coords` or
    #: :attr:`flap_tc` gives the flap one of its own; its thickness is
    #: MEASURED from the coordinates rather than asked for again. The
    #: suction-peak ceiling is still built from the NACA 24XX stall data at
    #: the measured thickness, which is a substitution and is reported as one
    #: (``ceiling_section``, ``ceiling_source``).
    section_coords: Any = None
    #: the FLAP's own section: an (n, 2) closed loop in the package's order,
    #: on a unit chord, scaled to the flap's chord. ``None`` = the flap is the
    #: main element's shape scaled, which is what this family could only say
    #: before and is bit-for-bit unchanged.
    flap_coords: Any = None
    #: the flap's own NACA 24XX thickness, when it should differ from the main
    #: element's without being a designed shape. ``None`` = the main
    #: element's. Ignored when :attr:`flap_coords` is given (the coordinates
    #: ARE the shape, and its thickness is measured off them).
    flap_tc: float | None = None
    #: whose loop each element's substituted suction-peak ceiling is read on
    #: (:data:`CEILING_SHAPES`). Changes nothing unless an element flies
    #: caller-supplied coordinates.
    ceiling_shape: str = "naca"
    #: multiplies BOTH suction-peak ceilings — the criterion-sensitivity knob.
    #: 1.0 is the criterion; above 1 is more permissive, below 1 stricter.
    #: A RESULT THAT MOVES UNDER THIS IS A RESULT ABOUT THE CRITERION, which
    #: is why it is a field on the problem and not a private constant: the
    #: sweep is part of any answer this family gives, not a footnote to one.
    ceiling_scale: float = 1.0
    #: refuse a rigging whose TRAILING-EDGE gap (``carwing_multi.slot_gap_te``,
    #: the quantity a wind-tunnel report calls *gap*) is below this fraction
    #: of the stowed chord. ``None`` is off and is the default; the
    #: literature-derived value is :data:`SLOT_GAP_TE_NON_MERGING_FRAC`, whose
    #: comment carries the derivation and the reason a bound is owed at all.
    #: Default None because switching it on changes which designs this family
    #: admits, and every number this family has published was measured with it
    #: off.
    slot_gap_te_min_frac: float | None = None
    #: MEASURED suction-peak ceilings, ``((cp, a_stall), (cp, a_stall))`` for
    #: (main, flap), replacing the NACA 24XX substitution. ``None`` uses the
    #: substitution. :func:`aerobo.section_stall.measured_ceilings` builds the
    #: pair by running XFOIL on the shapes actually being flown.
    section_ceilings: Any = None
    #: the section the ENDPLATE flies, or None for the NACA 24XX family
    #: member at :attr:`tc` — which is exactly what ``carwing.py`` gives it.
    #: See :meth:`plate_section` for why it is not the cascade.
    plate_polar: Any = None
    #: the car the wing is bolted to (:class:`cartrack.CarSpec`), and the
    #: circuit (:class:`cartrack.TrackSpec`). Both ``None`` is the published
    #: family, which has no mission and whose drag budget stands in for one.
    #: A track without a car uses ``cartrack``'s reference car; a car without
    #: a track states a frame but times no lap. Same fields, same meanings and
    #: same defaults as ``carwing.CarWingProblem``, so a lap timed on one
    #: family is comparable with a lap timed on the other — which is why this
    #: family needed them at all.
    car_spec: Any = None
    track_spec: Any = None
    #: how many representative speeds the lap samples the wing at
    #: (``cartrack.representative_points``). Each is a re-solve of the section
    #: AND the lattice, so it is a cost knob; 4 is ``carwing``'s.
    track_points: int = 4
    span_bounds_m: tuple | None = None
    objective: str = "cz"
    area_bounds_m2: tuple | None = None

    TAPER_BOUNDS = (0.4, 1.0)
    TWIST_ROOT_BOUNDS_DEG = (-4.0, 4.0)
    TWIST_TIP_BOUNDS_DEG = (-6.0, 2.0)
    #: the WING's incidence band, and it is deliberately the single-element
    #: family's. The slot's camber does not belong in this row — that is what
    #: the deflection row is for — and moving it would make the two families
    #: incomparable at the same nominal design point, which is the comparison
    #: this module exists to make ("one question, one place").
    ALPHA_BOUNDS_DEG = (0.0, 12.0)
    ENDPLATE_H_M_BOUNDS = (0.0, 0.25)
    RIDE_HEIGHT_BOUNDS_M = (0.15, 0.60)

    def __post_init__(self):
        if self.mount not in MOUNTS:
            raise ValueError(
                f"unknown mount {self.mount!r}; choose from {sorted(MOUNTS)}")
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        if self.objective not in CAR_MULTI_OBJECTIVES:
            extra = ""
            if self.objective in CAR_OBJECTIVES:
                extra = (
                    f" — {self.objective!r} is declared by "
                    f"carwing.CAR_OBJECTIVES but this family has no scorer "
                    f"for it (CAR_MULTI_OBJECTIVES is the set it can score, "
                    f"and it is deliberately not the sibling's)")
            raise ValueError(
                f"unknown objective {self.objective!r}; "
                f"choose from {sorted(CAR_MULTI_OBJECTIVES)}{extra}")
        # A LAP IS A TASK AND NEEDS A CIRCUIT. Same refusal, same words, as
        # carwing.CarWingProblem's: the objective is scorable by this family
        # now, but only when there is a lap to time, and a family that
        # accepted the name without a track would raise KeyError at the first
        # evaluation — which is the defect CAR_MULTI_OBJECTIVES' own comment
        # exists to remember.
        if self.objective == "laptime" and self.track_spec is None:
            raise ValueError(
                "objective 'laptime' needs a circuit: set track_spec (a "
                "cartrack.TrackSpec, e.g. cartrack.synthetic_lap()) and, if "
                "the car is not cartrack's reference car, car_spec. A lap "
                "time is a property of a wing ON A CIRCUIT")
        if self.track_spec is not None and int(self.track_points) < 1:
            raise ValueError(
                f"track_points must be >= 1, got {self.track_points!r}")
        if self.area_free and self.objective == "cd":
            raise ValueError(
                "objective 'cd' is meaningless with a free reference area: "
                "the drag COEFFICIENT is referenced to the very area being "
                "searched, so it is minimised by growing the wing. Use "
                "objective='drag' for the force (with a downforce_min_n "
                "floor) — or fix the area (area_bounds_m2=None). Same "
                "refusal, same reason, as carwing.CarWingProblem")
        if self.area_free and self.objective == "cz":
            raise ValueError(
                "objective 'cz' is meaningless with a free reference area: "
                "the downforce COEFFICIENT is referenced to the very area "
                "being searched, so it is maximised by shrinking the wing. "
                "Use objective='downforce' for the force, or 'efficiency' "
                "with a downforce_min_n floor — or fix the area "
                "(area_bounds_m2=None). Same refusal, same reason, as "
                "carwing.CarWingProblem")
        if self.area_free:
            area_row(self.area_bounds_m2)
        if self.CD_budget is not None and not float(self.CD_budget) > 0.0:
            raise ValueError(
                f"CD_budget must be > 0 or None (no coefficient drag "
                f"margin), got {self.CD_budget!r}")
        for name in ("drag_budget_n", "downforce_min_n"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0.0:
                raise ValueError(f"{name} must be > 0 or None, got {v!r}")
        if self.ceiling_shape not in CEILING_SHAPES:
            raise ValueError(
                f"unknown ceiling_shape {self.ceiling_shape!r}; choose from "
                f"{list(CEILING_SHAPES)}")
        if not float(self.ceiling_scale) > 0.0:
            raise ValueError(
                f"ceiling_scale must be > 0 (it multiplies a negative "
                f"pressure ceiling), got {self.ceiling_scale!r}")
        if (self.slot_gap_te_min_frac is not None
                and not float(self.slot_gap_te_min_frac) > 0.0):
            raise ValueError(
                f"slot_gap_te_min_frac must be > 0 or None (no non-merging "
                f"floor), got {self.slot_gap_te_min_frac!r}")
        if self.flap_tc is not None and self.flap_coords is not None:
            raise ValueError(
                "flap_tc and flap_coords both set: the coordinates ARE the "
                "flap's shape and its thickness is measured off them, so a "
                "second thickness would be read by nothing. Give one")
        if int(self.n_section_nodes) < 4 * panel2d.MIN_PANELS:
            raise ValueError(
                f"n_section_nodes = {self.n_section_nodes} is too coarse for "
                f"a slotted section; panel2d refuses below "
                f"{panel2d.MIN_PANELS} panels a body and a slot needs the "
                f"trailing-edge panels short enough to resolve it "
                f"(SLOT_WIDTH_PANELS)")

    @property
    def area_free(self) -> bool:
        return self.area_bounds_m2 is not None

    @property
    def car_used(self):
        """The :class:`cartrack.CarSpec` in force; cartrack's reference car
        when a circuit was stated without one. Same rule, same words, as
        ``carwing.CarWingProblem.car_used`` — a lap needs a car, and a caller
        who names a circuit without naming a car gets the module's own
        reference car rather than a refusal."""
        if self.car_spec is not None:
            return self.car_spec
        from . import cartrack
        return cartrack.CarSpec()

    def plate_section(self):
        """The section the ENDPLATE flies — and it is NOT the cascade.

        ``carwing.py`` gives its tip device the wing's own polar, which is
        fair while the two surfaces carry the same aerofoil. Here they do not.
        The wing's section is a DEPLOYED SLOTTED CASCADE: at the box centre
        its zero-lift angle is -12.86 deg and its lift-curve slope 6.35 /rad,
        so handing it to a flat vertical sheet would give that sheet the
        camber of a deployed flap and make it produce a large side force at
        zero incidence. A car wing's endplate is a plate.

        So the plate flies the single-element section instead — the same
        ``polar.default_polar_family().at(tc)`` object ``carwing.py`` uses,
        so the two families charge their plates identically and a difference
        between them is the WING's section and nothing else. The VLM carries
        it through ``winglet_a`` / ``winglet_alpha_L0``, which exist for
        exactly this.

        What is still not modelled: a plate is a flat sheet and this gives it
        a 12 %-thick cambered aerofoil's drag polar. That is ``carwing.py``'s
        own approximation, kept rather than changed, because changing it here
        would make the two families' plate charges different and quietly move
        the comparison this module exists to make.
        """
        if self.plate_polar is not None:
            return self.plate_polar
        from .polar import default_polar_family
        return default_polar_family().at(float(self.tc))

    @property
    def bounds(self) -> np.ndarray:
        # [family][size][chord]. The slot rows are the LAST four of the
        # family block, so the size block's own readers (carwing.span_from_x,
        # carwing.s_from_x) count back from the chord coefficients exactly as
        # they always did.
        family = np.vstack([
            np.array([
                self.TAPER_BOUNDS,
                self.TWIST_ROOT_BOUNDS_DEG,
                self.TWIST_TIP_BOUNDS_DEG,
                self.ALPHA_BOUNDS_DEG,
                self.ENDPLATE_H_M_BOUNDS,
                self.RIDE_HEIGHT_BOUNDS_M,
            ], dtype=float),
            slot_rows(),
        ])
        size = ([area_row(self.area_bounds_m2)] if self.area_free else [])
        box = np.vstack([family, *size, span_row(self.span_bounds_m)])
        return geometry.with_chord_bounds(box, self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def param_labels(self) -> tuple:
        """Row names, in the vector's own order."""
        out = ["taper", "twist_root_deg", "twist_tip_deg", "alpha_deg",
               "endplate_h_m", "ride_height_m", "flap_chord_frac",
               "flap_deflection_deg", "slot_gap_frac", "slot_overlap_frac"]
        if self.area_free:
            out.append("S_m2")
        out.append("b_m")
        out += [f"chord_k{i + 1}" for i in range(int(self.chord_order))]
        return tuple(out)

    @property
    def constraint_labels(self) -> tuple:
        """Display names for the margins this problem actually returns.

        Identical to ``carwing.CarWingProblem.constraint_labels`` and for the
        identical reason: a run with a budget switched off returns a SHORTER
        margin vector, and ``api.design_report`` reads these rather than the
        static family labels. The slot adds no constraint — its limits are
        REFUSALS (an intersecting flap, an unresolvable slot, a stalled
        element), which is the right shape for a limit that has no meaningful
        "how far past it are we".
        """
        out = []
        if self.CD_budget is not None:
            out.append("drag budget margin")
        if self.drag_budget_n is not None:
            out.append("drag force margin")
        out.append("deflection margin")
        if self.downforce_min_n is not None:
            out.append("downforce floor margin")
        return tuple(out)

    @property
    def n_constraints(self) -> int:
        return len(self.constraint_labels)

    @property
    def mount_spec(self) -> dict:
        return MOUNTS[self.mount]


# -------------------------------------------------------------- evaluation


def evaluate_car_wing_multi(x: np.ndarray,
                            prob: CarWingMultiProblem | None = None) -> dict:
    """Full evaluation with breakdown; never raises for in-contract failures.

    The lattice half is ``carwing.evaluate_car_wing``'s, step for step, so
    that a difference between the two families is the SECTION and nothing
    else. What is added is the section build ahead of the solve and the
    per-element reporting after it.
    """
    from .objective import _fail
    from .sizing import check_ar as _sizing_check_ar

    prob = prob or CarWingMultiProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    taper, tw_root, tw_tip, alpha_deg, ep_h, ride = (float(v) for v in x[:6])
    ff, defl, gap, ovl = slot_from_x(x)
    # THE SPAN ROW IS THE OVERALL WIDTH, PLATE INCLUDED — the accounting
    # carwing.py documents at length and endplate.py has always done. A
    # vertical unblended plate projects nothing and never enters the
    # arithmetic (the guard keeps it bit-for-bit); a leaning or blended one
    # reaches outboard and the wing pays for that reach out of its own span,
    # rather than flying wider than the row the user stated.
    b_overall = span_from_x(x, prob.chord_order)
    if prob.blend_frac > 0.0 or prob.endplate_cant_deg != 90.0:
        semi = geometry.developed_semispan(0.5 * b_overall, ep_h,
                                           prob.endplate_cant_deg,
                                           prob.blend_frac, prob.blend_shape)
        if semi <= 0.0:
            return _fail(
                f"the endplates' outboard projection "
                f"({0.5 * b_overall - semi:.3g} m a side) uses up the whole "
                f"{b_overall:.3g} m width: there is no wing left to carry "
                f"them. Blend less, lean less, or shorten the plates.")
    else:
        semi = 0.5 * b_overall
    b = 2.0 * semi                      # the WING's span — beam, VLM, AR
    s_ref = (s_from_x(x, prob.chord_order) if prob.area_free
             else float(prob.S))
    # both AR gates are asked of the WING's span, never of the width
    if prob.area_free:
        why = _sizing_check_ar(b, s_ref)
        if why is not None:
            return _fail(why)
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(b, s_ref, prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}")
    ep_frac = ep_h / semi     # the plate keeps its metres as b shrinks

    try:
        wing = geometry.Wing(
            b=b, S=s_ref, taper=taper, twist_root_deg=tw_root,
            twist_tip_deg=tw_tip, tc=prob.tc,
            chord_limits=prob.chord_limits,
            chord_coeffs=geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                      prob.chord_law))
    except ValueError as exc:
        return _fail(f"planform: {exc}")

    spec = prob.mount_spec
    solved: dict = {}

    def _point(V: float, ride_h: float) -> dict:
        """This wing at ONE speed and ride height: the section, then the
        lattice, then the drag build-up.

        Extracted from the straight line it used to be because a LAP flies
        the same wing at several speeds, and each of them has its own
        Reynolds number and therefore its own section drag. Referring one
        point to another by a q ratio would be exact only if the coefficients
        did not move with speed, which is the approximation
        MISSION_DESIGN_WATER_TRACK.md section 3 measured to be false by up to
        4.8 % of CD on the endplate family — so the wing is RE-FLOWN instead,
        exactly as ``carwing.evaluate_car_wing`` re-flies its own.

        Memoised on ``(V, ride)``: the base point and the lap point at the
        family's own speed are then the SAME dict, not two solves that agree
        to a tolerance. The section's own cache does the rest of the work —
        the two panel solves are keyed on geometry alone, so a second speed
        costs the Reynolds-dependent drag tables and one lattice solve.
        """
        key = (round(float(V), 9), round(float(ride_h), 9))
        hit = solved.get(key)
        if hit is not None:
            return hit
        out = _fly(float(V), float(ride_h))
        solved[key] = out
        return out

    def _fly(V: float, ride_h: float) -> dict:
        # the section is flown at the MAC's Reynolds number, and each
        # element's own is that scaled by its own chord fraction
        # (cascade_polar does the scaling, so "which chord" is stated once)
        re_ref = float(prob.rho * V * wing.mac / prob.mu)
        pol, reason = cascade_polar(prob.tc, ff, defl, gap, ovl, re_ref,
                                    n_nodes=int(prob.n_section_nodes),
                                    coords=prob.section_coords,
                                    flap_coords=prob.flap_coords,
                                    flap_tc=prob.flap_tc,
                                    ceiling_scale=float(prob.ceiling_scale),
                                    gap_te_min_frac=prob.slot_gap_te_min_frac,
                                    ceilings=prob.section_ceilings,
                                    ceiling_shape=prob.ceiling_shape)
        if reason is not None:
            return _fail(f"section: {reason}")
        try:
            plate = prob.plate_section()
        except ValueError as exc:
            return _fail(f"endplate section: {exc}")

        try:
            model = VLM(
                wing, N=prob.N_vlm, winglet_h_frac=ep_frac,
                winglet_cant_deg=prob.endplate_cant_deg,   # the ONE
                #      endplate; 90 = at the track, less leans it outboard
                n_winglet=prob.n_endplate, winglet_blend_frac=prob.blend_frac,
                winglet_chord_follows=prob.endplate_chord_follows,
                winglet_blend_shape=prob.blend_shape,
                a=pol.a_lin, alpha_L0=pol.alpha_L0,
                # THE PLATE IS NOT SLOTTED — CarWingMultiProblem.plate_section
                winglet_a=plate.a_lin, winglet_alpha_L0=plate.alpha_L0,
                V=V, image=ImagePlane(z=ride_h, kind="ground"),
            )
            res = model.solve(np.deg2rad(alpha_deg))
        except (ValueError, np.linalg.LinAlgError) as exc:
            return _fail(f"solver failure: {exc}")

        alpha_eff_deg = np.rad2deg(res.alpha_eff)
        is_plate = res.is_winglet
        on_wing = ~is_plate
        a_wing, a_plate = alpha_eff_deg[on_wing], alpha_eff_deg[is_plate]
        lo, hi = pol.alpha_valid
        if a_wing.min() < lo or a_wing.max() > hi:
            # (main, flap) margins, each evaluated at BOTH ends of what the
            # wing flies — which end broke the window is the first thing a
            # reader asks
            m_main, m_flap = pol.stall_margin([a_wing.min(), a_wing.max()])
            return _fail(
                "effective AoA outside the section's flyable window: one or "
                "both elements pass their measured suction-peak ceiling "
                f"(Cp_min = {pol.cp_ceiling:.3f}), or leave the wide-alpha "
                f"drag table. The window is [{lo:.2f}, {hi:.2f}] deg and the "
                f"wing flies [{a_wing.min():.2f}, {a_wing.max():.2f}]",
                alpha_eff_range=(float(a_wing.min()), float(a_wing.max())),
                alpha_valid=(float(lo), float(hi)),
                stall_margin_main=(float(m_main[0]), float(m_main[1])),
                stall_margin_flap=(float(m_flap[0]), float(m_flap[1])),
            )
        if a_plate.size:
            lo_p, hi_p = plate.alpha_valid
            if a_plate.min() < lo_p or a_plate.max() > hi_p:
                return _fail(
                    "endplate effective AoA outside its own section's "
                    f"validity [{lo_p:g}, {hi_p:g}] deg (it flies "
                    f"[{a_plate.min():.2f}, {a_plate.max():.2f}]) — the plate "
                    "is a plate and is flown on the single-element polar, not "
                    "the cascade",
                    alpha_eff_range=(float(a_plate.min()),
                                     float(a_plate.max())),
                )

        # profile drag: the CASCADE over the wing strips, the PLATE's own
        # section over the plate strips. carwing.py integrates one polar over
        # both, which is defensible while the two surfaces carry the same
        # aerofoil; here they do not, and the difference is not small (the
        # cascade's zero-lift angle is near -13 deg).
        cdp_w = float(np.sum(pol.cd(a_wing)
                             * res.c[on_wing] * res.width[on_wing]))
        cdp_p = float(np.sum(np.asarray(plate.cd(a_plate), dtype=float)
                             * res.c[is_plate] * res.width[is_plate])
                      ) if a_plate.size else 0.0
        CDp = (cdp_w + cdp_p) / res.S
        CDp_main = float(np.sum(np.interp(a_wing, pol.alpha_deg, pol.cd_main)
                                * res.c[on_wing] * res.width[on_wing]) / res.S)
        CDp_flap = cdp_w / res.S - CDp_main
        CDp_plate = cdp_p / res.S
        # no pylon on either layout (carwing.MOUNTS); the inboard layout's
        # two EXTRA sheets are what it pays instead, at the wing's chord
        # where they grip it and standing the endplate's own height
        c_grip = float(wing.chord(np.array(
            [min(float(spec["station_frac"]) * 0.5 * b, 0.5 * b)]))[0])
        cd0_struts = _strut_cd0(spec["n_sheets"], ep_h, c_grip,
                                s_ref, prob.strut_cf)
        cd_junction = 0.0
        if prob.junction_drag and spec["n_junctions"]:
            cd_junction = junction.junction_cd(
                prob.tc, float(wing.chord(np.array([0.0]))[0]), s_ref,
                radius=0.0, n_junctions=spec["n_junctions"])
        CD = res.CDi + CDp + cd0_struts + cd_junction
        CZ = res.CL
        if not np.isfinite(CD) or CD <= 0.0:
            return _fail("non-finite drag")
        return {
            "feasible": True, "reason": "",
            "V": float(V), "ride_height_m": float(ride_h),
            "CZ": float(CZ), "CD": float(CD),
            "CDi": float(res.CDi), "CDp": CDp,
            "CDp_main": CDp_main, "CDp_flap": CDp_flap,
            "CDp_endplate": CDp_plate,
            "cd0_struts": cd0_struts, "CD_junction": cd_junction,
            "Re_mac": re_ref, "res": res, "pol": pol, "plate": plate,
            "a_wing": a_wing, "a_plate": a_plate, "on_wing": on_wing,
            "is_plate": is_plate, "alpha_valid": (float(lo), float(hi)),
        }

    def _ride_at(V_req: float) -> float:
        """The ride height at a stated speed: the heave law's, or the stated
        one. With no car (or a rigid one) this is ``ride`` bit-for-bit."""
        if prob.car_spec is None and prob.track_spec is None:
            return ride
        return float(prob.car_used.ride_height_at(ride, float(V_req)))

    base = _point(prob.V, ride)
    if not base["feasible"]:
        return base                      # the published failure dict, verbatim

    # Only what the REST of the evaluation reads. The plate's own strips,
    # section and effective angles are all consumed inside ``_fly`` — the
    # profile-drag split is built there — and unpacking them here as well
    # would be three names a reader has to check for a second use that does
    # not exist.
    res, pol = base["res"], base["pol"]
    a_wing, on_wing = base["a_wing"], base["on_wing"]
    lo, hi = base["alpha_valid"]
    re_ref = base["Re_mac"]
    CZ, CD = base["CZ"], base["CD"]
    CDp, CDp_main = base["CDp"], base["CDp_main"]
    CDp_flap, CDp_plate = base["CDp_flap"], base["CDp_endplate"]
    cd0_struts, cd_junction = base["cd0_struts"], base["CD_junction"]

    # --- per-element lift. The SHARE is a 2-D property of the section at the
    # strip's own effective incidence; the MAGNITUDE is the lattice's. Split
    # the lattice's sectional lift by that share and integrate, so the two
    # element loads add EXACTLY to the surface's own integrated load — which
    # is the reconciliation ``tandemvlm``/``wingtail`` report as a share key.
    share = np.interp(a_wing, pol.alpha_deg, pol.share_main)
    lift_w = res.cl[on_wing] * res.c[on_wing] * res.width[on_wing]
    lift_tot = float(np.sum(lift_w))
    lift_share_main = (float(np.sum(share * lift_w) / lift_tot)
                       if abs(lift_tot) > 1e-12 else float("nan"))
    CZ_wing = float(lift_tot / res.S)

    # --- the critical strip, and each element's margin there
    i_crit = int(np.argmax(a_wing))
    a_crit = float(a_wing[i_crit])
    g_main, g_flap = pol.stall_margin(a_crit)

    # --- structure (carmount's beam at the grip station)
    y_m, gam_m = res.y[on_wing], res.Gamma[on_wing]
    load = prob.rho * prob.V * np.abs(gam_m)
    # one signed beam at the grip station, both layouts — see carwing's own
    # call site for why a magnitude cannot serve an interior support
    y_station_b = float(spec["station_frac"]) * 0.5 * b
    m_signed = carmount.beam_moment(y_m, load, y_station_b)
    moment = np.abs(m_signed)
    defl_m = (carmount.beam_deflection_index(y_m, m_signed, y_station_b)
              / prob.ei_nm2)

    q = 0.5 * prob.rho * prob.V ** 2
    downforce_n = float(q * s_ref * CZ)
    drag_n = float(q * s_ref * CD)

    # --- the lap, when a circuit was stated. The wing is RE-FLOWN at the
    # speeds the lap spends its time at (cartrack's E5 interface) and the
    # coefficients travel back as laws of speed. Step for step
    # ``carwing.evaluate_car_wing``'s, including the clamped interpolation,
    # because a lap timed differently on the two families would make the
    # transfer measurement this family exists to support a comparison of two
    # lap models rather than of two wings.
    lap = track_points_rows = None
    if prob.track_spec is not None:
        from . import cartrack
        car = prob.car_used
        seed = cartrack.representative_points(
            CZ * s_ref, CD * s_ref, car, prob.track_spec,
            n_points=int(prob.track_points), ride_height_m=ride)
        if not seed["feasible"]:
            return _fail(f"track: {seed['reason']}")
        rows = []
        for pt in sorted(seed["points"], key=lambda p: p["V"]):
            h = float(pt.get("ride_height_m", ride))
            got = _point(float(pt["V"]), h)
            if not got["feasible"]:
                return _fail(
                    f"the lap flies this wing at {pt['V']:.4g} m/s and "
                    f"{h:.4g} m: {got['reason']}")
            rows.append({"V": float(pt["V"]), "weight": float(pt["weight"]),
                         "ride_height_m": h,
                         "CZ": got["CZ"], "CD": got["CD"],
                         "cz_a_m2": got["CZ"] * s_ref,
                         "cd_a_m2": got["CD"] * s_ref,
                         "Re_mac": got["Re_mac"]})
        Vs = np.array([r["V"] for r in rows], dtype=float)
        cza = np.array([r["cz_a_m2"] for r in rows], dtype=float)
        cda = np.array([r["cd_a_m2"] for r in rows], dtype=float)
        # linear in V between the evaluated points and CLAMPED outside them:
        # np.interp holds the end values, so the lap never extrapolates a
        # coefficient into a speed the wing was not flown at
        lap = cartrack.lap_time(lambda V: float(np.interp(V, Vs, cza)),
                                lambda V: float(np.interp(V, Vs, cda)),
                                car, prob.track_spec)
        if not lap["feasible"]:
            return _fail(f"lap: {lap['reason']}")
        track_points_rows = rows

    margins, g_drag, g_drag_n, g_down = [], None, None, None
    if prob.CD_budget is not None:
        g_drag = float(prob.CD_budget - CD)
        margins.append(g_drag)
    if prob.drag_budget_n is not None:
        g_drag_n = float(1.0 - drag_n / prob.drag_budget_n)
        margins.append(g_drag_n)
    g_defl = float(1.0 - defl_m / prob.deflection_limit_m)
    margins.append(g_defl)
    if prob.downforce_min_n is not None:
        g_down = float(downforce_n / prob.downforce_min_n - 1.0)
        margins.append(g_down)

    # the scorer and :data:`CAR_MULTI_OBJECTIVES` are the same set, and the
    # constructor validates against the latter, so this lookup cannot see a
    # name it has no entry for (gated by
    # ``test_the_objective_set_is_the_one_this_family_can_score``)
    scores = {"cz": float(CZ), "downforce": downforce_n,
              "efficiency": float(CZ / CD),
              # MINUS the drag, for the reason the lap is minus the lap:
              # everything here MAXIMISES. carwing.CAR_OBJECTIVES carries the
              # measurement that put these two in the set, and the PENALTY
              # sentinel trap 'drag' shares with 'laptime'.
              "cd": -float(CD), "drag": -float(drag_n),
              # MINUS the lap: everything in this package MAXIMISES, and a lap
              # is won by being small. None when there is no circuit, which
              # the constructor has already made unreachable for this
              # objective (a lap needs a track), so the lookup below cannot
              # hand back a None score.
              "laptime": (None if lap is None
                          else -float(lap["lap_time_s"])),
              # ADDED, not traded — carwing.CAR_OBJECTIVES carries the
              # measurement and the ratchet warning.
              "downforce_plus_drag": float(downforce_n) + float(drag_n)}
    score = scores[prob.objective]

    return {
        "feasible": True, "reason": "",
        "score": float(score), "CZ": float(CZ),
        "objective": prob.objective,
        "objective_label": CAR_MULTI_OBJECTIVES[prob.objective],
        "g": margins,
        "g_drag": g_drag, "g_deflection": g_defl,
        "g_drag_force": g_drag_n, "g_downforce": g_down,
        "constraint_labels": list(prob.constraint_labels),
        "CL_model": float(res.CL), "CD": float(CD),
        "CDi": float(res.CDi), "CDp": CDp,
        "CDp_main": CDp_main, "CDp_flap": CDp_flap,
        "CDp_endplate": CDp_plate,
        "cd0_struts": cd0_struts, "CD_junction": cd_junction,
        "efficiency": float(CZ / CD),
        "e": float(res.e), "AR": float(res.AR),
        "alpha_deg": alpha_deg,
        "ride_height_m": ride, "ride_height_over_b": float(ride / b),
        "b_m": float(b), "S_m2": float(s_ref),
        # the width the row stated, and what the plate took out of it
        "overall_width_m": float(b_overall),
        "endplate_projection_m": float(0.5 * b_overall - semi),
        "area_free": bool(prob.area_free),
        "endplate_h_m": float(ep_h), "endplate_h_frac": float(ep_frac),
        "mount": prob.mount, "mount_label": spec["label"],
        "supports": spec["supports"],
        "frame": geometry.MIRRORED_FRAME,
        "M_max_Nm": float(np.max(moment)),
        "M_root_Nm": float(moment[np.argmin(np.abs(y_m))]),
        "deflection_m": float(defl_m),
        "downforce_N": downforce_n, "drag_N": drag_n,
        "q_Pa": float(q),
        "Re_mac": re_ref,
        "polar": pol.name,
        # ---- the slot: what was asked for, and what it built
        "flap_chord_frac": float(ff),
        "flap_deflection_deg": float(defl),
        "slot_gap_frac": float(gap),
        "slot_overlap_frac": float(ovl),
        "slot_width_frac": float(pol.slot_width_frac),
        "slot_width_m": float(pol.slot_width_frac * wing.mac),
        # the wind tunnel's "gap": main TE to the flap's surface. Reported
        # ALWAYS, bound only when the caller sets slot_gap_te_min_frac — a
        # measurement a reader can see is not the same thing as a limit
        "slot_gap_te_frac": float(pol.slot_gap_te_frac),
        "slot_gap_te_m": float(pol.slot_gap_te_frac * wing.mac),
        "slot_gap_te_min_frac": (None if prob.slot_gap_te_min_frac is None
                                 else float(prob.slot_gap_te_min_frac)),
        "c_main_frac": float(pol.c_main), "c_flap_frac": float(pol.c_flap),
        "n_elements": 2,
        # ---- per element, each on its OWN reference, plus the share key
        "cl_own_main": float(np.interp(a_crit, pol.alpha_deg,
                                       pol.cl_own_main)),
        "cl_own_flap": float(np.interp(a_crit, pol.alpha_deg,
                                       pol.cl_own_flap)),
        "lift_share_main": lift_share_main,
        "lift_share_flap": (1.0 - lift_share_main
                            if np.isfinite(lift_share_main) else float("nan")),
        "CZ_wing_panels": CZ_wing,
        "Re_main": float(pol.re_main), "Re_flap": float(pol.re_flap),
        "re_clamped_main": bool(pol.re_clamped_main),
        "re_clamped_flap": bool(pol.re_clamped_flap),
        "alpha_eq_main_deg": float(np.interp(a_crit, pol.alpha_deg,
                                             pol.alpha_eq_main)),
        "alpha_eq_flap_deg": float(np.interp(a_crit, pol.alpha_deg,
                                             pol.alpha_eq_flap)),
        "cp_min_main": float(np.interp(a_crit, pol.alpha_deg, pol.cpmin_main)),
        "cp_min_flap": float(np.interp(a_crit, pol.alpha_deg, pol.cpmin_flap)),
        "cp_ceiling": float(pol.cp_ceiling),
        "cp_ceiling_main": float(pol.cp_ceiling_main),
        "cp_ceiling_flap": float(pol.cp_ceiling_flap),
        "alpha_stall_iso_deg": float(pol.alpha_stall_iso_deg),
        "alpha_stall_iso_flap_deg": float(pol.alpha_stall_iso_flap_deg),
        "stall_margin_main": float(g_main),
        "stall_margin_flap": float(g_flap),
        "alpha_eff_crit_deg": a_crit,
        "alpha_valid_deg": (float(lo), float(hi)),
        "section_a_lin": float(pol.a_lin),
        "section_alpha_L0_deg": float(np.rad2deg(pol.alpha_L0)),
        "tc_main": float(pol.tc), "tc_flap": float(pol.tc_flap),
        "flap_is_own_section": bool(pol.flap_is_own_section),
        "ceiling_source": str(pol.ceiling_source),
        "ceiling_scale": float(pol.ceiling_scale),
        "ceiling_shape": str(pol.ceiling_shape),
        "ceiling_section": _ceiling_section_label(pol),
        # ---- the lap, when a circuit was stated. THE SAME THREE KEYS
        # ``carwing.evaluate_car_wing`` reports and no others: the circuit
        # name, the mean speed and the warnings all live inside ``lap``, and a
        # second spelling of them here would be a second place to read one
        # question from.
        "lap_time_s": (None if lap is None else float(lap["lap_time_s"])),
        "lap": lap, "track_points": track_points_rows,
        "y": y_m, "load_Npm": load, "moment_Nm": moment,
        "wing": wing, "vlm": res, "section": pol,
    }


def fg_car_wing_multi(x: np.ndarray, prob: CarWingMultiProblem | None = None
                      ) -> tuple[float, list]:
    """Constrained-harness callable: ``(score, margins)``.

    The failure vector's WIDTH follows the problem's own declaration, exactly
    as ``carwing.fg_car_wing``'s does: a penalty returned at the wrong width
    is a shape error inside the optimiser rather than a bad score.
    """
    prob = prob or CarWingMultiProblem()
    out = evaluate_car_wing_multi(x, prob)
    if not out["feasible"]:
        return PENALTY, [G_FAIL] * prob.n_constraints
    return float(out["score"]), [float(g) for g in out["g"]]


# ---------------------------------------------------------------- the claim


def single_element_x(x: np.ndarray, prob: CarWingMultiProblem):
    """The matching ``carwing`` design vector: the same wing, one element.

    The slot rows are dropped and everything else is carried across
    unchanged, so the two vectors describe the same planform at the same
    incidence, ride height, span and reference area. This is the ONLY honest
    way to ask "what did the slot buy" — a comparison at different planforms
    would credit the slot with the planform.
    """
    x = np.asarray(x, dtype=float)
    keep = np.ones(x.size, dtype=bool)
    keep[SLOT_ROW_OFFSET:SLOT_ROW_OFFSET + 4] = False
    return x[keep]


def single_element_problem(prob: CarWingMultiProblem, wide_alpha: bool = False):
    """The matching ``carwing.CarWingProblem``: same everything but the slot.

    ``wide_alpha=True`` gives it the WIDE-ALPHA XFOIL table (alpha to +22 deg)
    instead of the cruise one (to +14), which is the CONTROL any comparison
    against this module needs. The two-element family prices its elements off
    the wide-alpha bank, because the suction-peak bridge produces equivalent
    incidences past 14 deg; ``carwing.py`` as shipped reads the cruise table
    and therefore refuses at 14 deg. MEASURED at the box centre, that alone
    caps the single-element wing at CZ 1.643 with CD 0.107 — so above a 0.11
    drag budget an uncontrolled comparison would be measuring the two tables
    against each other and calling the difference a slot.
    """
    from .carwing import CarWingProblem

    section = None
    if wide_alpha:
        from .polar import stall_polar_family
        section = stall_polar_family().at(float(prob.tc))

    return CarWingProblem(
        section_polar=section,
        b=prob.b, S=prob.S, N_vlm=prob.N_vlm, n_endplate=prob.n_endplate,
        V=prob.V, rho=prob.rho, mu=prob.mu, tc=prob.tc, mount=prob.mount,
        strut_chord=prob.strut_chord, strut_cf=prob.strut_cf,
        ei_nm2=prob.ei_nm2, CD_budget=prob.CD_budget,
        drag_budget_n=prob.drag_budget_n,
        downforce_min_n=prob.downforce_min_n,
        deflection_limit_m=prob.deflection_limit_m,
        blend_frac=prob.blend_frac, blend_shape=prob.blend_shape,
        # ...and the plate's CANT with them. Both families charge the
        # plate's outboard projection against the span row now, so a
        # control built without the cant would fly a VERTICAL plate
        # (a wider wing) against a canted candidate — an unmatched
        # comparison in the one function whose whole job is matching.
        endplate_cant_deg=prob.endplate_cant_deg,
        junction_drag=prob.junction_drag, chord_order=prob.chord_order,
        chord_max_frac=prob.chord_max_frac, chord_law=prob.chord_law,
        chord_limits=prob.chord_limits, span_bounds_m=prob.span_bounds_m,
        objective=prob.objective, area_bounds_m2=prob.area_bounds_m2,
    )


def compare_split(x: np.ndarray, prob: CarWingMultiProblem | None = None,
                  alpha_walk_deg: tuple | None = (0.0, 24.0),
                  n_alpha: int = 25, n_deflection: int = 15) -> dict:
    """One element or two, at the SAME area and the SAME drag budget.

    The direct answer to the family's own question, and it is answered the
    way a race engineer would: for each family, walk the wing's incidence
    (and, for the two-element family, its flap deflection as well) and report
    the most downforce it can make with its drag inside the budget. Comparing
    the two at a COMMON incidence would answer a different question, because
    the drag budget is what makes downforce scarce and each family reaches it
    at a different angle.

    ``alpha_walk_deg`` widens the incidence walked BEYOND the design box, and
    defaults to doing so, because otherwise this function measures the box
    instead of the physics. MEASURED: with the walk left at the family's own
    row (0-12 deg) the single-element wing tops out at alpha 12 with
    CD = 0.0598 against a 0.11 budget — it is INCIDENCE-limited, has not
    spent half its drag allowance, and the slot is then credited +43.4 %
    that is partly the alpha row's. Walking both to 24 deg lets each meet
    whichever limit is really its own, and every answer reports which one
    that is (``limit_single`` / ``limit_multi``). Pass ``None`` to walk the
    box exactly as declared.

    ``x`` sets everything the two share (planform, ride height, span, area,
    endplate); the incidence entry — and, for the two-element family, the
    deflection entry — are overwritten by the walk. The returned dict carries
    both winners, both walks, the delta and the limit each answer rides.

    WHICH LIMIT AN ANSWER RIDES IS DECIDED BY MECHANISM, NOT BY PROXIMITY.
    An answer rides the drag budget when the walk contains a design that made
    MORE downforce, was aerodynamically feasible, and was thrown out because
    its drag was over the budget — i.e. when the budget is demonstrably what
    stopped the walk. It does NOT ride the drag budget merely because its own
    CD came close to it.

    That is a correction, and the thing it corrects is worth recording. This
    function used to ask ``CD >= CD_budget * (1 - 1/(n_alpha - 1))``, which
    ties the verdict to the WALK RESOLUTION: refining the same measurement
    flips the mechanism it reports. MEASURED at the box centre with the
    published 0.11 budget, the single-element arm's answer is CD 0.10730 at
    every resolution, and the old rule called it

        n_alpha        13      25      49      97
        tolerance   0.9167  0.9583  0.9792  0.9896
        reported    interior  DRAG    interior  interior
                    (stall)  BUDGET   (stall)  (stall)

    — "drag budget" at 25 stations and "interior (section stall)" at 13, 49
    and 97, for one unchanged answer. The mechanism rule reports "drag
    budget" at all four, and it is right to: the next station up (alpha 18)
    is feasible at CZ 1.7332 and CD 0.12022, over the budget. The old rule's
    "section stall" was not merely resolution-dependent, it was false — that
    arm does not stall anywhere in the walk.

    The remaining entries are ceilings of the walk itself (exact equality, no
    tolerance), a fall-through, and one report that is deliberately NOT a
    mechanism. The fall-through fires when nothing the walk flew beat the
    answer AND LIVED: what lies above the answer was REFUSED, and that is
    said ("section refusal above the answer's incidence") rather than
    guessed. It is a fall-through and not a parallel hit because a budget
    that bit is the tighter limit of the two. And the walk does not enforce
    the structural deflection limit — ``budget_ok`` checks the drag budgets
    only — so "structural deflection limit" says the WINNER sits on that
    limit, not that the limit bound the answer.

    MEASURED over 13 / 25 / 49 / 97 stations at the box centre: both
    wide-alpha arms report "drag budget" at every resolution. The
    cruise-table arm (``limit_single``) moves from the fall-through to "drag
    budget" between 49 and 97 stations, and that is the boundary being
    resolved rather than the verdict flipping: at 0.25-deg steps the walk
    finally holds a feasible design above the answer that the budget throws
    out, where at 0.5 deg the next station up was already refused.
    """
    from dataclasses import replace as _replace

    from .carwing import evaluate_car_wing

    prob = prob or CarWingMultiProblem()
    x = np.asarray(x, dtype=float).copy()

    p2 = _replace(prob)
    p1 = single_element_problem(prob)
    p1w = single_element_problem(prob, wide_alpha=True)
    if alpha_walk_deg is not None:
        # The incidence band is a CLASS attribute on all three problems
        # (there is no field for it), so the walk widens it on its own copies
        # by shadowing. Nothing else about any problem moves, and the box the
        # families ship with is untouched.
        band = (float(alpha_walk_deg[0]), float(alpha_walk_deg[1]))
        p2.ALPHA_BOUNDS_DEG = band
        p1.ALPHA_BOUNDS_DEG = band
        p1w.ALPHA_BOUNDS_DEG = band
    a_lo, a_hi = p2.bounds[3]
    alphas = np.linspace(a_lo, a_hi, int(n_alpha))
    x1 = single_element_x(x, prob)

    def budget_ok(out):
        if not out["feasible"]:
            return False
        if prob.CD_budget is not None and out["CD"] > prob.CD_budget:
            return False
        if prob.drag_budget_n is not None and out["drag_N"] > prob.drag_budget_n:
            return False
        return True

    best1, walk1 = None, []
    best1w, walk1w = None, []
    for a in alphas:
        xx = x1.copy()
        xx[3] = a
        out = evaluate_car_wing(xx, p1)
        walk1.append((float(a), out))
        if budget_ok(out) and (best1 is None or out["CZ"] > best1["CZ"]):
            best1 = out
        outw = evaluate_car_wing(xx, p1w)
        walk1w.append((float(a), outw))
        if budget_ok(outw) and (best1w is None or outw["CZ"] > best1w["CZ"]):
            best1w = outw

    d_lo, d_hi = p2.bounds[SLOT_ROW_OFFSET + 1]
    defls = np.linspace(d_lo, d_hi, int(n_deflection))
    best2, walk2 = None, []
    for a in alphas:
        for d in defls:
            xx = x.copy()
            xx[3] = a
            xx[SLOT_ROW_OFFSET + 1] = d
            out = evaluate_car_wing_multi(xx, p2)
            walk2.append((float(a), float(d), out))
            if budget_ok(out) and (best2 is None or out["CZ"] > best2["CZ"]):
                best2 = out

    def rides(out, walk, top_alpha, top_defl=None):
        """WHICH limit this answer sits on — decided by MECHANISM.

        ``walk`` is ``[(alpha_deg, out), ...]``: every point this arm flew.
        A budget is named as the limit only when the walk holds a FEASIBLE
        design that made more downforce and was rejected by that budget, so
        the verdict is a statement about what stopped the walk and not about
        how near a number came to a line. See this function's own docstring
        for the resolution-tied rule this replaces and what it got wrong.
        """
        if out is None:
            return "no feasible design"
        cz = float(out["CZ"])
        # designs that made MORE downforce and were thrown out by a budget
        better = [o for _a, o in walk
                  if o["feasible"] and float(o["CZ"]) > cz]
        hits = []
        if prob.CD_budget is not None and any(
                float(o["CD"]) > prob.CD_budget for o in better):
            hits.append("drag budget")
        if prob.drag_budget_n is not None and any(
                float(o["drag_N"]) > prob.drag_budget_n
                for o in better):
            hits.append("drag force budget")
        if abs(out["alpha_deg"] - top_alpha) <= 1e-9:
            hits.append("incidence ceiling of the walk")
        if top_defl is not None and abs(
                out["flap_deflection_deg"] - top_defl) <= 1e-9:
            hits.append("flap deflection ceiling")
        if not hits:
            # nothing the walk flew beat this and lived: what lies above it
            # was REFUSED, and by what is the answer here
            above = [o for a, o in walk
                     if not o["feasible"] and a > float(out["alpha_deg"])]
            hits.append("section refusal above the answer's incidence"
                        if above else "interior")
        if out["g_deflection"] <= 1e-3:
            # NOT a mechanism: the walk does not enforce this limit (see the
            # function docstring). It is a warning about the winner.
            hits.append("structural deflection limit")
        return " + ".join(hits)

    cz1 = best1["CZ"] if best1 else None
    cz1w = best1w["CZ"] if best1w else None
    cz2 = best2["CZ"] if best2 else None

    def frac(a, bb):
        return None if (a is None or bb is None or a == 0.0) else (bb - a) / a

    return {
        "single": best1, "single_wide": best1w, "multi": best2,
        "walk_single": walk1, "walk_single_wide": walk1w, "walk_multi": walk2,
        "CZ_single": cz1, "CZ_single_wide": cz1w, "CZ_multi": cz2,
        "limit_single": rides(best1, walk1, float(alphas[-1])),
        "limit_single_wide": rides(best1w, walk1w, float(alphas[-1])),
        "limit_multi": rides(best2, [(a, o) for a, _d, o in walk2],
                             float(alphas[-1]), float(defls[-1])),
        "alpha_walk_deg": (float(alphas[0]), float(alphas[-1])),
        "delta_CZ": (None if (cz1 is None or cz2 is None) else cz2 - cz1),
        "delta_frac": frac(cz1, cz2),
        # THE CONTROLLED NUMBER: both families on the same wide-alpha table.
        "delta_frac_wide": frac(cz1w, cz2),
        "CD_budget": prob.CD_budget, "drag_budget_n": prob.drag_budget_n,
        "S_m2": (float(prob.S) if not prob.area_free
                 else float(s_from_x(x, prob.chord_order))),
    }
