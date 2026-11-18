"""Map ASL Citizen glosses onto the 250-sign asl-signs vocabulary.

Pure Python without third-party imports: scripts/build_kaggle_notebooks.py inlines this file into
the Kaggle extraction notebook, so the notebook and the repository apply the same mapping.

A gloss maps to a label by the first rule that applies:

1. ``exact``: the gloss and the label are equal once lower-cased with everything except letters
   and digits removed, so THANKYOU, "thank you" and Thank-You all match the label "thankyou".
2. ``variant``: the gloss is the label plus a numeric suffix (CAT1, CAT2 and CAT3 for "cat").
   ASL Citizen numbers the ASL-LEX 2.0 variants of a word. Every variant is mapped unless
   VARIANT_EXCLUDED lists it, which it does where ASL-LEX gives that variant another meaning.
3. ``synonym``: SYNONYMS lists other glosses per label, chosen where ASL-LEX 2.0's English
   translations of the sign name the label's word (mom: MOTHER).

A gloss maps to at most one label (build_mapping raises on a conflict), and the labels in
UNRESOLVED stay unmapped until someone decides; their candidate glosses are in REVIEW_GLOSSES,
which the extraction notebook processes too, so a later decision needs no second extraction.

REVIEW_STATUS applies to VARIANT_EXCLUDED, SYNONYMS and UNRESOLVED: they were chosen with AI
assistance from ASL-LEX 2.0's translations and have not been confirmed by a signer.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

REVIEW_STATUS = "ai-assisted, unconfirmed"
RULES = ("exact", "variant", "synonym")

# Variants whose ASL-LEX 2.0 entry gives them a meaning other than the label's.
VARIANT_EXCLUDED: dict[str, str] = {
    "DRINK2": "ASL-LEX translations: alcohol, beverage, drink (the alcohol sign)",
    "EMPTY2": "ASL-LEX dominant translation: dump",
    "HEAR2": "ASL-LEX dominant translation: listen, which is a label of its own",
    "TALK2": "ASL-LEX dominant translation: duck (the blah-blah gesture); duck is a label",
    "TOY2": "ASL-LEX dominant translation: shave",
}

# label -> (glosses, basis). The basis quotes the ASL-LEX 2.0 English translations.
SYNONYMS: dict[str, tuple[tuple[str, ...], str]] = {
    "alligator": (("CROCODILE",), "translations: alligator, animal, crocodile"),
    "another": (("OTHER",), "translations: another, else, other, otherwise"),
    "beside": (("NEXTTO",), "translations: besides, by, near, next to"),
    "callonphone": (("PHONE", "CALL1"), "translations: call, phone, telephone (both)"),
    "dad": (("FATHER",), "translations: dad, daddy, father, papa"),
    "every": (("EACH",), "translations: apiece, each, every"),
    "eye": (("EYES",), "translations: body part, eyes"),
    "food": (("EAT1", "EAT2"), "translations: consume, food, eat (both)"),
    "garbage": (("TRASH",), "translations: bin, garbage, trash, trash can"),
    "grandma": (("GRANDMOTHER",), "translations: grandma, grandmother, nana"),
    "grandpa": (("GRANDFATHER",), "translations: baba, grandfather, grandpa"),
    "haveto": (("MUST",), "translations: have to, imperative, must, need, should"),
    "hesheit": (("HE",), "dominant translation: he (the pointing pronoun)"),
    "into": (("IN",), "ASL-LEX entry 'in' (no translations listed)"),
    "kitty": (("KITTEN",), "translations: baby cat, kitten, kitty, kitty cat"),
    "look": (("LOOKAT",), "translations: look, look at, see, to watch"),
    "minemy": (("MY",), "ASL-LEX entry 'my' (MINE is the digging sign)"),
    "mitten": (("GLOVE", "GLOVES1", "GLOVES2"), "translations: glove, gloves, mittens"),
    "mom": (("MOTHER",), "translations: mama, mom, mommy, mother"),
    "noisy": (("NOISE",), "translations: noise, sound (LOUD is the label loud)"),
    "nuts": (("NUT1", "NUT2"), "translations: almond, cashew, nut, nuts, peanut"),
    "owie": (("HURT", "PAIN"), "translations: ache, hurt, injury, pain (both)"),
    "police": (("POLICEMAN1", "POLICEMAN2"), "translations: cop, police, police officer"),
    "potty": (("TOILET", "BATHROOM"), "translations: bathroom, restroom, toilet (both)"),
    "puzzle": (("JIGSAWPUZZLE",), "translations: game, puzzle, puzzle piece"),
    "shhh": (("SSHH",), "translations: be quiet, shh, shush (QUIET is the label quiet)"),
    "shoe": (("SHOES",), "translations: footwear, shoe, shoes, sneakers"),
    "store": (("SHOP1",), "translations: sell, shop, shopping, store"),
    "tooth": (("TEETH",), "translations: teeth, tooth"),
    "weus": (("WE", "US"), "translations: pronoun, us, we (both)"),
}

# Labels left unmapped until a reviewer decides between senses.
UNRESOLVED: dict[str, str] = {
    "can": "CAN1 means able to / possible and CAN2 has no translation; the asl-signs sense "
    "(ability or container) is unknown",
    "fall": "FALL1 is the season and FALL2 is falling down; the asl-signs sense is unknown",
    "glasswindow": "GLASS1-3 (cup, material) and WINDOW are different signs; which one "
    "asl-signs records is unknown",
}

# Why the remaining labels have no mapping (from the ASL-LEX search).
UNMAPPED_NOTES: dict[str, str] = {
    "feet": "no ASL Citizen gloss",
    "goose": "no ASL Citizen gloss",
    "pajamas": "no ASL Citizen gloss",
    "jeans": "only PANTS1/PANTS2, likely a different sign from JEANS",
    "nap": "only SLEEP, which is the label sleep",
    "pool": "only POOLGAME (billiards)",
    "puppy": "only DOG1-4, which are the label dog",
    "say": "only MENTION, a different sign",
    "sleepy": "only SLEEP (the label sleep) and TIRED, a different sign",
    "snack": "only COOKIE, DESSERT and SWEET",
    "wake": "only AWAKE, which is the label awake",
}

# Extracted but unmapped, so a reviewer can map them later without re-extracting.
REVIEW_GLOSSES: tuple[str, ...] = (
    "CAN1",
    "CAN2",
    "FALL1",
    "FALL2",
    "GLASS1",
    "GLASS2",
    "GLASS3",
    "WINDOW",
    "DRINK2",
    "EMPTY2",
    "HEAR2",
    "TALK2",
    "TOY2",
    "PANTS1",
    "PANTS2",
    "SHOP2",
    "ENTER",
    "NEED",
    "SHOULD",
    "THIS/IT",
    "GLOVES3",
    "GLOVES4",
    "TIRED",
)


def normalize(text: str) -> str:
    """Lower-case and keep only a-z and 0-9: "Thank-You" -> "thankyou"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def variant_base(gloss: str) -> str | None:
    """The normalized word before a numeric variant suffix (CAT2 -> "cat"), or None when the
    gloss has no suffix or is a number with letters first (1DOLLAR stays as it is)."""
    match = re.fullmatch(r"([a-z][a-z0-9]*?[a-z])(\d+)", normalize(gloss))
    return match.group(1) if match else None


@dataclass(frozen=True)
class MappingRow:
    label: str
    gloss: str
    rule: str
    basis: str


@dataclass
class MappingResult:
    """Rows sorted by label then gloss; ``unmapped`` maps each label without rows to a reason."""

    rows: list[MappingRow]
    unmapped: dict[str, str]
    review: list[str]
    labels: list[str]
    by_gloss: dict[str, MappingRow] = field(default_factory=dict)

    def label_rule(self, label: str) -> str | None:
        """The strongest rule among a label's rows (exact, then variant, then synonym)."""
        rules = {row.rule for row in self.rows if row.label == label}
        return next((rule for rule in RULES if rule in rules), None)

    def coverage(self) -> dict[str, int]:
        """Label counts by strongest rule, plus mapped, unmapped and the vocabulary size."""
        counts = dict.fromkeys(RULES, 0)
        for label in self.labels:
            rule = self.label_rule(label)
            if rule is not None:
                counts[rule] += 1
        mapped = sum(counts.values())
        return {
            **counts,
            "mapped": mapped,
            "unmapped": len(self.labels) - mapped,
            "vocabulary": len(self.labels),
        }

    def candidate_glosses(self) -> list[str]:
        """Every gloss to extract: the mapped ones and the review list."""
        return sorted(set(self.by_gloss) | set(self.review))


def build_mapping(
    labels: Sequence[str], glosses: Iterable[str], *, strict: bool = True
) -> MappingResult:
    """Apply the rules to the dataset's gloss list.

    With ``strict`` (the default), a gloss named in SYNONYMS or VARIANT_EXCLUDED that the dataset
    lacks raises ValueError, so a changed dataset release is noticed rather than mapped silently.
    """
    present = sorted({gloss.strip() for gloss in glosses if gloss.strip()})
    present_set = set(present)
    if strict:
        named = {gloss for glosses_, _ in SYNONYMS.values() for gloss in glosses_}
        missing = sorted((named | set(VARIANT_EXCLUDED)) - present_set)
        if missing:
            raise ValueError(
                f"glosses named in the mapping are missing from the dataset: {missing}"
            )
    by_norm: dict[str, str] = {}
    for label in labels:
        key = normalize(label)
        if key in by_norm:
            raise ValueError(f"labels {by_norm[key]!r} and {label!r} normalize to {key!r}")
        by_norm[key] = label

    candidates: dict[str, list[MappingRow]] = {}

    def add(row: MappingRow) -> None:
        candidates.setdefault(row.gloss, []).append(row)

    for gloss in present:
        exact = by_norm.get(normalize(gloss))
        if exact is not None:
            add(MappingRow(exact, gloss, "exact", "normalized gloss equals the label"))
            continue
        base = variant_base(gloss)
        variant = by_norm.get(base) if base is not None else None
        if base is not None and variant is not None and gloss not in VARIANT_EXCLUDED:
            add(MappingRow(variant, gloss, "variant", f"numbered variant of {base.upper()}"))
    for label, (synonyms, basis) in SYNONYMS.items():
        if label not in by_norm.values():
            continue
        for gloss in synonyms:
            if gloss in present_set:
                add(MappingRow(label, gloss, "synonym", basis))

    rows: list[MappingRow] = []
    for gloss, options in sorted(candidates.items()):
        targets = sorted({row.label for row in options})
        if len(targets) > 1:
            raise ValueError(f"gloss {gloss!r} maps to more than one label: {targets}")
        if options[0].label in UNRESOLVED:
            continue
        rows.append(options[0])
    rows.sort(key=lambda row: (row.label, row.gloss))

    mapped = {row.label for row in rows}
    unmapped = {
        label: UNRESOLVED.get(label) or UNMAPPED_NOTES.get(label) or "no matching gloss"
        for label in labels
        if label not in mapped
    }
    review = sorted(set(REVIEW_GLOSSES) & present_set)
    return MappingResult(
        rows=rows,
        unmapped=unmapped,
        review=review,
        labels=list(labels),
        by_gloss={row.gloss: row for row in rows},
    )


def mapping_csv(result: MappingResult) -> str:
    """label,gloss,rule,basis,status rows (status is REVIEW_STATUS for non-exact rules)."""
    lines = ["label,gloss,rule,basis,status"]
    for row in result.rows:
        status = "automatic" if row.rule == "exact" else REVIEW_STATUS
        basis = row.basis.replace('"', "'")
        lines.append(f'{row.label},{row.gloss},{row.rule},"{basis}",{status}')
    return "\n".join(lines) + "\n"


def unmapped_text(result: MappingResult) -> str:
    """One "label: reason" line per unmapped label."""
    return "".join(f"{label}: {reason}\n" for label, reason in sorted(result.unmapped.items()))
