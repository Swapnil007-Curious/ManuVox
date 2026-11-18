"""The ASL Citizen -> asl-signs label mapping (edgevision.data.citizen_labels), on gloss lists
written out here: the dataset itself is never part of the repository."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from edgevision.data import citizen_labels as cl

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "web/public/model/meta.json").read_text("utf-8"))["labels"]
# Every gloss the mapping names, so strict mode is satisfied, plus a few others.
NAMED = sorted(
    {gloss for glosses, _ in cl.SYNONYMS.values() for gloss in glosses} | set(cl.VARIANT_EXCLUDED)
)


def mapping(*glosses: str, labels: list[str] | None = None) -> cl.MappingResult:
    return cl.build_mapping(labels or LABELS, [*NAMED, *glosses])


def rule_of(result: cl.MappingResult, gloss: str) -> tuple[str, str] | None:
    row = result.by_gloss.get(gloss)
    return (row.label, row.rule) if row else None


@pytest.mark.parametrize("text", ["THANKYOU", "thank you", "Thank-You", " thank_you "])
def test_normalize_handles_case_spaces_and_punctuation(text):
    assert cl.normalize(text) == "thankyou"


def test_compound_glosses_map_exactly():
    result = mapping("THANKYOU", "FRENCHFRIES", "ICECREAM", "TV")
    assert rule_of(result, "THANKYOU") == ("thankyou", "exact")
    assert rule_of(result, "FRENCHFRIES") == ("frenchfries", "exact")
    assert rule_of(result, "TV") == ("TV", "exact")


def test_numbered_variants_map_unless_excluded():
    result = mapping("CAT1", "CAT2", "CAT3", "DRINK1")
    assert {rule_of(result, g) for g in ("CAT1", "CAT2", "CAT3")} == {("cat", "variant")}
    assert rule_of(result, "DRINK1") == ("drink", "variant")
    for excluded in cl.VARIANT_EXCLUDED:
        assert excluded not in result.by_gloss
        assert excluded in result.review


def test_variant_base_leaves_numbers_and_number_led_glosses_alone():
    assert cl.variant_base("CAT2") == "cat"
    assert cl.variant_base("HELICOPTER3") == "helicopter"
    assert cl.variant_base("1DOLLAR") is None
    assert cl.variant_base("APPLE") is None


def test_synonyms_map_with_their_basis():
    result = mapping("MOTHER", "FATHER")
    assert rule_of(result, "MOTHER") == ("mom", "synonym")
    assert "mom" in result.by_gloss["MOTHER"].basis


def test_unresolved_labels_stay_unmapped_but_their_glosses_are_extracted():
    result = mapping("CAN1", "CAN2", "FALL1", "FALL2", "WINDOW")
    for label in cl.UNRESOLVED:
        assert label in result.unmapped
        assert result.unmapped[label] == cl.UNRESOLVED[label]
    assert {"CAN1", "CAN2", "FALL1", "FALL2", "WINDOW"} <= set(result.candidate_glosses())
    assert not {"CAN1", "FALL2"} & set(result.by_gloss)


def test_a_gloss_mapping_to_two_labels_is_an_error(monkeypatch):
    monkeypatch.setitem(cl.SYNONYMS, "dad", (("MOTHER",), "a deliberate conflict"))
    with pytest.raises(ValueError, match="more than one label"):
        mapping("MOTHER")


def test_strict_mode_names_missing_glosses():
    with pytest.raises(ValueError, match="missing from the dataset"):
        cl.build_mapping(LABELS, ["APPLE"])
    assert cl.build_mapping(LABELS, ["APPLE"], strict=False).by_gloss["APPLE"].label == "apple"


def test_labels_that_normalize_alike_are_rejected():
    with pytest.raises(ValueError, match="normalize"):
        cl.build_mapping(["thank-you", "thankyou"], [], strict=False)


def test_every_named_label_is_in_the_vocabulary():
    vocabulary = set(LABELS)
    for table in (cl.SYNONYMS, cl.UNRESOLVED, cl.UNMAPPED_NOTES):
        assert set(table) <= vocabulary
    assert not set(cl.SYNONYMS) & set(cl.UNRESOLVED)
    # A synonym never points at a gloss that is another label's exact match.
    for glosses, _ in cl.SYNONYMS.values():
        assert not {cl.normalize(g) for g in glosses} & {cl.normalize(label) for label in LABELS}


def test_coverage_and_csv():
    result = mapping("APPLE", "CAT1", "MOTHER")
    coverage = result.coverage()
    assert coverage["vocabulary"] == 250
    assert coverage["mapped"] + coverage["unmapped"] == 250
    assert coverage["exact"] >= 1 and coverage["variant"] >= 1 and coverage["synonym"] >= 1
    lines = cl.mapping_csv(result).splitlines()
    assert lines[0] == "label,gloss,rule,basis,status"
    assert any(
        line.startswith("apple,APPLE,exact,") and line.endswith(",automatic") for line in lines
    )
    assert any(
        line.startswith("mom,MOTHER,synonym,") and cl.REVIEW_STATUS in line for line in lines
    )
    assert "can: " in cl.unmapped_text(result)


def test_the_kaggle_notebooks_are_up_to_date():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_kaggle_notebooks.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_the_extraction_notebook_holds_the_mapping_and_labels():
    notebook = json.loads((ROOT / "kaggle/asl_citizen_extract.ipynb").read_text("utf-8"))
    (cell,) = notebook["cells"]
    source = "".join(cell["source"])
    assert source.startswith("%%bash")
    assert "def build_mapping(" in source and json.dumps(LABELS) in source
    assert "exit 0" in source.splitlines()[-1]
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is True
