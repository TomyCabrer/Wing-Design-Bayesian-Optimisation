"""What the wing is BUILT OF — the one question ``weights.py`` never asked.

``weights.py`` closes the aero-structural loop with two models and exactly
one material assumption between them, spelled once, as a constant:

    SIGMA_ALLOW_PA = 345e6 / 1.5      # aluminium 2024-T3, FAR 23.303

That number reaches the STRESS CONSTRAINT and nothing else. The WEIGHT —
``wing_weight_raymer``, the thing that actually decides how much span a
design can afford (see aircraft.py's objective note: D_i ~ W_total^2 / b^2)
— is a statistical regression over general-aviation aeroplanes, and a
regression has no material argument at all. So until this module existed a
carbon wing and an aluminium wing weighed the same, and the only thing the
user could change was a limit the answer usually did not ride.

Measured before writing it (2-D (b, S) grid, chord law and twist at box
centre, "trim wing + free planform + free chord law"):

    sigma_allow 230 -> 500 MPa      b* 13.00 -> 13.50 m, score +1.0 %
    sigma_allow 500 -> infinity     no change whatever
    k_w         1.00 -> 0.87        score +12.7 %

    d ln f / d ln k_w   ~ -1.0
    d ln f / d ln sigma ~ +0.01

Two orders of magnitude apart. A material menu wired only to the allowable
would be a control that moves nothing; the WEIGHT is where a material has to
enter, and this module is how it gets there.

THE MODEL (and why it is shaped like this)
------------------------------------------
Wing weight is split in two, because the two halves answer to different
properties:

    naive(m) = phi * (rho_cap  / rho_ref) * (sigma_ref / sigma_cap)
             + (1 - phi) * (rho_skin / rho_ref)

* the first term is the SPAR CAPS, which are strength-sized: hold the load
  and the allowable, and cap area goes as 1/sigma, so cap mass goes as
  rho/sigma. This is the same two-cap idealisation ``root_bending_stress``
  already uses for the constraint, so the two halves of the structural model
  finally agree about what a spar is;
* the second is SKIN, RIBS, JOINTS and fittings, which are thickness-driven
  rather than strength-driven, so only density moves them;
* ``phi`` is the strength-sized share of a wing's structural mass.

``rho_skin`` is a SEPARATE input, not a copy of ``rho_cap``, because the
combination that made this module necessary — a hot-wire foam core with a
carbon spar — is two materials and is the normal way a model or an ultralight
wing is built. For a homogeneous material the two densities are equal and the
expression collapses to the usual one.

WHY THE SECOND PARAMETER (eta) IS NOT OPTIONAL
----------------------------------------------
``naive`` alone cannot be the answer, and it is worth showing why rather than
asserting it. For carbon/epoxy against 2024-T3:

    phi = 1 (all caps, full credit for strength)   naive = 0.26
    phi = 0 (no strength credit at all, density)   naive = 0.575
    Raymer's empirical composite wing factor       ~ 0.85 - 0.90

**No phi in [0, 1] reaches 0.87.** Even pure density scaling — carbon given
zero credit for being stronger — is more optimistic than what composite wings
actually weigh. So the missing physics is not the split: it is that a
composite part is not thickness-equivalent to an aluminium one. Minimum
gauge, ply drop-offs, buckling-driven skins, bonded joints, fittings and
barely-visible-impact damage tolerance all push a real laminate thicker than
strength alone would ask. Every one of those is outside this model, so they
are carried by ONE fitted number:

    k_w = 1 - eta * (1 - naive)

``eta`` is the fraction of the theoretical material advantage a real
structure realises. Fitted at the one point where an empirical answer exists
(carbon prepreg -> 0.87) it comes out at ~0.22: **about a fifth**. That is
the whole reason a density typed into a naive rho/sigma formula produces a
fantasy aeroplane (k_w 0.42 -> span +23 %, drag -46 %) and this one does not.

HONESTY LABELS (read before quoting anything from here)
-------------------------------------------------------
* ONE free parameter fitted to ONE literature number. ``eta`` is anchored on
  Raymer's composite weight-savings factor for a wing, recalled and NOT
  verified against a primary source in this repo. :data:`ETA_ANCHOR_K_W` is
  where it enters; a test asserts the round trip, which is a tripwire on the
  arithmetic and not a citation.
* ``phi`` is ASSUMED, not measured. With a single anchor, phi and eta trade
  off against each other: several pairs reproduce carbon equally well and
  disagree about materials far from it. Foam, PLA, balsa and steel are
  therefore EXTRAPOLATION of a two-parameter fit, and are labelled as such in
  the table.
* Raymer's correlation is itself a GA-aluminium regression, calibrated for
  W_dg ~ 1-8 klb. Multiplying it by k_w does not make it a composite-wing (or
  a model-aeroplane) regression: its exponents — above all (A/cos^2 L)^0.6,
  which is exactly the term the span answer rides — were fitted to aluminium
  structure. A carbon answer at AR 20 is an aluminium fit times a constant.
* There is no buckling criterion, no minimum gauge and no stiffness or
  aeroelastic model anywhere on the aircraft side. All of it is inside eta.
* An allowable for a laminate is a LAYUP property, not a material property.
  The composite entries below quote a strain-limited design number
  (sigma ~ E * 0.4 % in compression), which is how composite primary
  structure is actually sized; a different layup is a different number, and
  that is what :func:`custom` is for.

The aluminium entry is the reference by construction, so ``k_w`` is EXACTLY
1.0 there and every run this repo has ever stored is reproduced bit for bit
by the default.
"""

from __future__ import annotations

from dataclasses import dataclass

from .weights import SIGMA_ALLOW_PA

__all__ = [
    "Material",
    "MATERIALS",
    "MATERIAL_GROUPS",
    "REFERENCE_KEY",
    "PHI",
    "ETA",
    "ETA_ANCHOR_KEY",
    "ETA_ANCHOR_K_W",
    "RHO_REF_KGM3",
    "SIGMA_REF_PA",
    "custom",
    "get",
    "naive_ratio",
    "weight_factor",
    "resolve",
]

#: Density of the reference material [kg/m^3] — 2024-T3 aluminium sheet, the
#: alloy ``weights.SIGMA_ALLOW_PA`` is already the allowable of. Paired with
#: it here so the two halves of "what the wing is made of" cannot drift.
RHO_REF_KGM3 = 2780.0

#: ...and its allowable, imported rather than restated: one number, one home.
SIGMA_REF_PA = SIGMA_ALLOW_PA

#: Strength-sized share of a wing's structural mass (spar caps against skin,
#: ribs, joints and fittings). ASSUMED — see the module docstring's honesty
#: labels. 0.5 is the round number a GA wing box is usually quoted near.
PHI = 0.50

#: The key whose empirical weight factor fits :data:`ETA`.
ETA_ANCHOR_KEY = "cfrp_prepreg"

#: Raymer's composite weight-savings factor for a WING, mid-band of the
#: 0.85-0.90 usually quoted. This is the single empirical number the whole
#: extrapolation hangs on; ``tests/test_the_wing_is_built_of_something.py`` pins the round trip.
ETA_ANCHOR_K_W = 0.87


@dataclass(frozen=True)
class Material:
    """What a wing is built of, as the two numbers the models can use.

    ``sigma_allow_Pa`` is a DESIGN ALLOWABLE — the stress the structure is
    sized to, safety factor already applied — not a handbook ultimate. It is
    the number ``weights.root_bending_stress`` is compared against, so the
    two must mean the same thing.

    ``rho_skin_kgm3`` defaults to the cap density: a homogeneous material is
    the common case, and a core-plus-spar build is the one that has to say so.
    """

    key: str
    label: str
    group: str
    rho_cap_kgm3: float
    sigma_allow_Pa: float
    rho_skin_kgm3: float | None = None
    note: str = ""
    #: False where no empirical weight factor exists for this material class,
    #: i.e. where k_w is the (phi, eta) fit EXTRAPOLATED. True only for the
    #: aluminium reference and the carbon anchor.
    anchored: bool = False

    def __post_init__(self):
        for name in ("rho_cap_kgm3", "sigma_allow_Pa"):
            v = float(getattr(self, name))
            if not (v > 0.0):
                raise ValueError(
                    f"{self.key}: {name} must be > 0 (got {v})")
        if self.rho_skin_kgm3 is not None and not (
                float(self.rho_skin_kgm3) > 0.0):
            raise ValueError(
                f"{self.key}: rho_skin_kgm3 must be > 0 when given "
                f"(got {self.rho_skin_kgm3})")

    @property
    def rho_skin(self) -> float:
        """Skin/core density [kg/m^3] — the cap's own where none was given."""
        return float(self.rho_skin_kgm3 if self.rho_skin_kgm3 is not None
                     else self.rho_cap_kgm3)

    @property
    def naive(self) -> float:
        """The (phi-split) material ratio BEFORE the realisation factor.

        Exposed because it is the number a reader is most likely to want to
        argue with, and because the gap between it and :attr:`k_w` is the
        whole content of ``eta``.
        """
        return naive_ratio(self.rho_cap_kgm3, self.sigma_allow_Pa,
                           self.rho_skin)

    @property
    def k_w(self) -> float:
        """Weight factor multiplying ``weights.wing_weight_raymer``."""
        return weight_factor(self)

    def report(self) -> dict:
        """The material as breakdown keys, for a stored run to be read by.

        A design whose weight came from a material and whose record does not
        say which material is a design nobody can reproduce, so every sized
        result carries these.
        """
        return {
            "material": self.key,
            "material_label": self.label,
            "material_rho_kgm3": float(self.rho_cap_kgm3),
            "material_rho_skin_kgm3": self.rho_skin,
            "material_k_w": self.k_w,
            # NOT "sigma_allow_Pa": that key belongs to the allowable the
            # constraint was actually evaluated against, and both sizing and
            # aircraft accept an explicit override of it. A material stating
            # its own under the same name would overwrite the number the
            # margin was computed from — reported and used disagreeing, which
            # is the one thing a breakdown may never do.
            "material_sigma_allow_Pa": float(self.sigma_allow_Pa),
        }


def naive_ratio(rho_cap: float, sigma_allow_Pa: float,
                rho_skin: float | None = None, phi: float = PHI) -> float:
    """Strength-sized caps + thickness-driven skin, against the reference.

    Both terms are ratios to 2024-T3, so the absolute calibration of the cap
    model cancels: this never claims to know a cap's mass, only how one
    material's cap compares with another's at the same load and the same
    allowable stress.
    """
    if not (sigma_allow_Pa > 0.0 and rho_cap > 0.0):
        raise ValueError("rho_cap and sigma_allow_Pa must both be > 0 "
                         f"(got {rho_cap}, {sigma_allow_Pa})")
    skin = float(rho_cap if rho_skin is None else rho_skin)
    if not (skin > 0.0):
        raise ValueError(f"rho_skin must be > 0 (got {rho_skin})")
    cap_term = (float(rho_cap) / RHO_REF_KGM3) * (SIGMA_REF_PA
                                                  / float(sigma_allow_Pa))
    skin_term = skin / RHO_REF_KGM3
    return float(phi * cap_term + (1.0 - phi) * skin_term)


def _fit_eta() -> float:
    """Solve ``eta`` from the one anchor: k_w(carbon prepreg) = 0.87.

    Written as a solve rather than a literal so the anchor and the constant
    cannot disagree — change the anchor material's properties and eta follows,
    which is what keeps ``k_w`` on 0.87 there by construction rather than by
    coincidence.
    """
    m = _ANCHOR
    gap = 1.0 - naive_ratio(m.rho_cap_kgm3, m.sigma_allow_Pa, m.rho_skin)
    if abs(gap) < 1e-12:                       # pragma: no cover - guard
        raise ValueError("eta anchor has no material advantage to scale")
    return (1.0 - ETA_ANCHOR_K_W) / gap


def weight_factor(m: Material) -> float:
    """``k_w``: what ``wing_weight_raymer`` is multiplied by for ``m``.

    Exactly 1.0 for the reference material — not approximately, and not by a
    tolerance: the reference's own ratio is 1.0 by construction, so
    ``1 - eta * (1 - 1)`` is 1.0 in floating point too, and every published
    sized run is reproduced bit for bit.
    """
    return float(1.0 - ETA * (1.0 - m.naive))


# --------------------------------------------------------------- the table
# Ordered as the shell offers them: what a model or a homebuilt is actually
# made of first, certified structure last. GROUP is the menu's heading.

_G_MODEL = "model / RC / UAV"
_G_HOME = "homebuilt / recreational"
_G_CERT = "certified GA / transport"

REFERENCE_KEY = "al_2024_t3"

_TABLE = (
    Material(
        key="eps_foam", label="EPS foam, unreinforced", group=_G_MODEL,
        rho_cap_kgm3=25.0, sigma_allow_Pa=0.15e6,
        note="A hot-wire core with nothing in it. Included because it is the "
             "honest answer to 'can I just cut a wing out of foam': at this "
             "allowable the spar constraint binds long before the drag "
             "optimum does, so the design that comes back is short, fat and "
             "riding sigma = sigma_allow. Foam is a CORE, and the model says "
             "so rather than refusing the question."),
    Material(
        key="foam_cf_spar", label="EPS core + carbon spar", group=_G_MODEL,
        rho_cap_kgm3=1600.0, sigma_allow_Pa=500e6, rho_skin_kgm3=25.0,
        note="How a model, a UAV or an ultralight wing is really built: the "
             "caps carry the bending, the core carries the shape. The two "
             "densities are what the phi-split exists for."),
    Material(
        key="pla_printed", label="PLA, 3-D printed", group=_G_MODEL,
        rho_cap_kgm3=500.0, sigma_allow_Pa=12e6,
        note="Sparse infill (~40 %) and a layer-adhesion knockdown, so both "
             "numbers are EFFECTIVE properties of a printed part, not of the "
             "polymer: solid PLA is ~1240 kg/m^3 and ~50 MPa. A different "
             "infill or print orientation is a different material — use "
             "custom."),
    Material(
        key="balsa_film", label="balsa + film", group=_G_MODEL,
        rho_cap_kgm3=160.0, sigma_allow_Pa=12e6,
        note="Built-up balsa under heat-shrink film. Grain direction matters "
             "more than the mean density; this is the along-grain case."),
    Material(
        key="spruce", label="Sitka spruce + ply", group=_G_HOME,
        rho_cap_kgm3=450.0, sigma_allow_Pa=28e6,
        note="The classic wooden wing. The allowable is compression parallel "
             "to grain with the static factor already applied. Wood is "
             "STRENGTH-limited here, which is why the answer comes back as a "
             "low-aspect-ratio wing — the shape wooden aeroplanes have."),
    Material(
        key="al_6061_t6", label="aluminium 6061-T6", group=_G_HOME,
        rho_cap_kgm3=2700.0, sigma_allow_Pa=276e6 / 1.5,
        note="The homebuilder's alloy: weaker than 2024 and easier to work. "
             "Yield 276 MPa with the FAR 23.303 factor of 1.5."),
    Material(
        key="gfrp", label="E-glass / epoxy", group=_G_HOME,
        rho_cap_kgm3=1900.0, sigma_allow_Pa=250e6,
        note="Motorgliders and the first generation of composite homebuilts. "
             "Strain-limited allowable, not a fibre strength."),
    Material(
        key=REFERENCE_KEY, label="aluminium 2024-T3", group=_G_CERT,
        rho_cap_kgm3=RHO_REF_KGM3, sigma_allow_Pa=SIGMA_REF_PA, anchored=True,
        note="THE REFERENCE. Raymer's correlation was regressed over "
             "aeroplanes built of this, so k_w is exactly 1.0 and this is "
             "the only entry that needs no extrapolation at all."),
    Material(
        key="al_7075_t6", label="aluminium 7075-T6", group=_G_CERT,
        rho_cap_kgm3=2810.0, sigma_allow_Pa=503e6 / 1.5,
        note="The high-strength cap alloy. Note what the model does with it: "
             "the extra strength is only worth weight where the stress "
             "constraint was binding, so it buys less than its yield "
             "suggests."),
    Material(
        key="steel_4130", label="steel 4130 (normalised)", group=_G_CERT,
        rho_cap_kgm3=7850.0, sigma_allow_Pa=460e6 / 1.5,
        note="Chromoly. Strong, and heavy enough that k_w goes ABOVE 1: the "
             "map is not a discount, it is a ratio, and it is allowed to "
             "make an aeroplane worse."),
    Material(
        key="cfrp_wet", label="carbon / epoxy, wet layup", group=_G_CERT,
        rho_cap_kgm3=1600.0, sigma_allow_Pa=400e6,
        note="Hand layup: lower fibre volume fraction and more scatter than "
             "prepreg, so a lower allowable for the same fibre."),
    Material(
        key=ETA_ANCHOR_KEY, label="carbon / epoxy, prepreg", group=_G_CERT,
        rho_cap_kgm3=1580.0, sigma_allow_Pa=550e6, anchored=True,
        note="Autoclave-cured aerospace prepreg — the material the empirical "
             "0.87 weight factor describes, and therefore the point eta is "
             "fitted at. Allowable is strain-limited (~0.4 % compression on "
             "a ~130 GPa laminate), which is how composite primary structure "
             "is sized in practice."),
)

MATERIALS: dict[str, Material] = {m.key: m for m in _TABLE}

_ANCHOR = MATERIALS[ETA_ANCHOR_KEY]

#: Realisation factor — the share of the theoretical material advantage a
#: real structure delivers. Solved from the anchor; ~0.22, i.e. about a
#: fifth. See the module docstring for why it cannot be 1.
ETA = _fit_eta()

#: Menu order: {group: (key, ...)}, built from the table so a new entry
#: appears in the shell by being added ONCE, here.
MATERIAL_GROUPS: dict[str, tuple] = {}
for _m in _TABLE:
    MATERIAL_GROUPS.setdefault(_m.group, ())
    MATERIAL_GROUPS[_m.group] += (_m.key,)
del _m


def custom(rho_kgm3: float, sigma_allow_Pa: float,
           rho_skin_kgm3: float | None = None,
           label: str = "custom") -> Material:
    """A material the table does not have, from its two (or three) numbers.

    The point of the whole (phi, eta) shape: a user who types a density and
    an allowable gets an answer on the same footing as the presets, rather
    than the naive rho/sigma answer — which for carbon is optimistic by a
    factor of two and would hand back an aeroplane that cannot be built.
    """
    return Material(
        key="custom", label=label, group="custom",
        rho_cap_kgm3=float(rho_kgm3), sigma_allow_Pa=float(sigma_allow_Pa),
        rho_skin_kgm3=(None if rho_skin_kgm3 is None
                       else float(rho_skin_kgm3)),
        note="Typed by the user. k_w comes from the same fit as the presets, "
             "so it is an EXTRAPOLATION of a two-parameter model anchored on "
             "aluminium and carbon.")


def get(key: str) -> Material:
    """A table entry by key. Raises KeyError naming what exists."""
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(
            f"unknown material {key!r}; choose from {sorted(MATERIALS)} "
            f"or build one with materials.custom(rho, sigma_allow)") from None


def resolve(material) -> Material:
    """Whatever a caller has -> a Material. ``None`` is the reference.

    ``None`` meaning aluminium is load-bearing: every sized problem in this
    package defaults its ``material`` field to None, so the default IS the
    published calibration and no stored run moves.
    """
    if material is None:
        return MATERIALS[REFERENCE_KEY]
    if isinstance(material, Material):
        return material
    return get(str(material))
