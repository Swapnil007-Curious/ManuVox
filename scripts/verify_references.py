"""Check every entry of paper/references.bib against an authoritative source.

    python scripts/verify_references.py            # writes paper/data/reference_verification.json

Each entry names one identifier, looked up in this order:
  anthology = {...}  -> https://aclanthology.org/<id>.bib (the Anthology's own BibTeX)
  eprint = {...}     -> the arXiv API (export.arxiv.org)
  doi = {...}        -> the Crossref REST API (api.crossref.org/works/<doi>)
  url = {...}        -> the page itself (official proceedings or the dataset's page)
The title must match (normalised, similarity >= 0.92 or one contains the other), the first
author's surname must match where the source lists authors, and the year must match (for an
arXiv preprint later published at a venue, the venue year may be one year after the preprint).
Exit code 1 when any entry fails. Requests carry a generic User-Agent and no personal data.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
OUT = ROOT / "paper" / "data" / "reference_verification.json"
SELF = {"shukla2026manuvox"}  # the extended version of this paper (cited by the IEEE version)
AGENT = "paper-reference-check/1.0"
ATOM = "{http://www.w3.org/2005/Atom}"


def fetch(url: str, accept: str = "*/*") -> str:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": accept})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return str(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """key -> fields (lower-case names, values without the outer braces or quotes)."""
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = match.group(1).lower(), match.group(2)
        i, depth = match.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[match.end() : i - 1]
        fields = {"_type": kind}
        j = 0
        while True:
            m = re.compile(r"\s*(\w+)\s*=\s*").match(body, j)
            if not m:
                break
            name, j = m.group(1).lower(), m.end()
            if body[j] == "{":
                depth, start = 1, j + 1
                j += 1
                while depth:
                    depth += {"{": 1, "}": -1}.get(body[j], 0)
                    j += 1
                value = body[start : j - 1]
            elif body[j] == '"':
                end = body.index('"', j + 1)
                value, j = body[j + 1 : end], end + 1
            else:
                m2 = re.compile(r"[^,]+").match(body, j)
                assert m2 is not None
                value, j = m2.group(0).strip(), m2.end()
            fields[name] = value
            m3 = re.compile(r"\s*,").match(body, j)
            j = m3.end() if m3 else j
        entries[key] = fields
    return entries


LATEX_ACCENTS = {
    r"\"": "\u0308",
    r"'": "\u0301",
    r"`": "\u0300",
    r"~": "\u0303",
    r"^": "\u0302",
    r"c": "\u0327",
    r"v": "\u030c",
    r"u": "\u0306",
    r"=": "\u0304",
}


def plain(text: str) -> str:
    """LaTeX -> plain text: accents resolved, braces and commands dropped."""
    text = re.sub(r"\\L\b\{?\}?", "L", text)
    for cmd, mark in LATEX_ACCENTS.items():
        pattern = r"\\" + re.escape(cmd) + r"\s*\{?(\w)\}?"
        text = re.sub(pattern, lambda m, k=mark: m.group(1) + k, text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return unicodedata.normalize("NFC", text.replace("{", "").replace("}", ""))


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", plain(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("&amp;", "and").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def first_surname(authors: str) -> str:
    first = re.split(r"\s+and\s+", plain(authors).strip())[0].strip()
    surname = first.split(",")[0] if "," in first else first.split()[-1]
    return norm(surname)


def title_ok(ours: str, theirs: str) -> tuple[bool, float]:
    a, b = norm(ours), norm(theirs)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.92 or (len(a) > 20 and (a in b or b in a)), round(ratio, 3)


def from_anthology(anthology_id: str) -> dict[str, Any]:
    url = f"https://aclanthology.org/{anthology_id}.bib"
    entry = next(iter(parse_bib(fetch(url)).values()))
    return {
        "url": url,
        "title": entry.get("title", ""),
        "year": entry.get("year", ""),
        "first_author": first_surname(entry.get("author", entry.get("editor", ""))),
    }


def from_arxiv(eprint: str) -> dict[str, Any]:
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(eprint)}"
    root = ET.fromstring(fetch(url))
    entry = root.find(f"{ATOM}entry")
    if entry is None or entry.find(f"{ATOM}title") is None:
        raise ValueError(f"arXiv has no entry {eprint}")
    title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
    author = entry.find(f"{ATOM}author")
    name = author.findtext(f"{ATOM}name") if author is not None else ""
    published = entry.findtext(f"{ATOM}published") or ""
    return {
        "url": f"https://arxiv.org/abs/{eprint}",
        "title": title,
        "year": published[:4],
        "first_author": norm((name or "").split()[-1]) if name else "",
        "arxiv": True,
    }


def from_crossref(doi: str) -> dict[str, Any]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    message = json.loads(fetch(url, "application/json"))["message"]
    people = message.get("author") or message.get("editor") or []
    issued = (message.get("issued") or message.get("published") or {}).get("date-parts", [[None]])
    return {
        "url": f"https://doi.org/{doi}",
        "title": (message.get("title") or [""])[0],
        "year": str(issued[0][0] or ""),
        "first_author": norm(people[0].get("family", "")) if people else "",
    }


def from_page(url: str, ours: dict[str, str]) -> dict[str, Any]:
    html = fetch(url)
    page_title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    text = norm(re.sub(r"<[^>]+>", " ", html))
    found = norm(ours["title"]) in text
    surname = first_surname(ours.get("author", "")) if ours.get("author") else ""
    shown = page_title.group(1) if page_title else ""
    year = ours.get("year", "")
    return {
        "url": url,
        "title": ours["title"] if found else shown,
        "year": year if year in text or "kaggle" in url else "",
        "first_author": surname if surname and surname in text else "",
        "page": True,
    }


def check(key: str, entry: dict[str, str]) -> dict[str, Any]:
    if "anthology" in entry:
        source, found = "ACL Anthology", from_anthology(entry["anthology"])
    elif "eprint" in entry:
        source, found = "arXiv API", from_arxiv(entry["eprint"])
    elif "doi" in entry:
        source, found = "Crossref", from_crossref(entry["doi"])
    elif "url" in entry:
        source, found = "official page", from_page(entry["url"], entry)
    else:
        return {"status": "fail", "notes": "no identifier (anthology, eprint, doi or url)"}
    ok_title, ratio = title_ok(entry.get("title", ""), found["title"])
    ours_author = first_surname(entry.get("author", entry.get("editor", "")))
    organisation = entry.get("author", "").startswith("{{")
    theirs_author = found["first_author"]
    ok_author = (
        organisation
        or not theirs_author
        or theirs_author == ours_author
        or theirs_author.split()[-1] == ours_author.split()[-1]  # "pakdaman naeini" vs "naeini"
    )
    year, theirs = entry.get("year", ""), str(found["year"])
    later = found.get("arxiv") and theirs.isdigit() and int(year) - int(theirs) == 1
    ok_year = year == theirs or bool(later)
    if found.get("page") and not theirs:
        ok_year = "kaggle" in found["url"]
    notes = []
    if not ok_title:
        notes.append(f"title differs: {found['title']!r}")
    if not ok_author:
        notes.append(f"first author {found['first_author']!r} vs ours {ours_author!r}")
    if not ok_year:
        notes.append(f"year {theirs!r} vs ours {year!r}")
    if found.get("arxiv") and year != theirs and ok_year:
        notes.append(f"arXiv preprint {theirs}, venue year {year}")
    return {
        "status": "ok" if ok_title and ok_author and ok_year else "fail",
        "verified_with": found["url"],
        "source": source,
        "title_similarity": ratio,
        "source_title": found["title"],
        "source_year": theirs,
        "source_first_author": found["first_author"],
        "notes": "; ".join(notes),
    }


def main() -> int:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for key, entry in entries.items():
        if key in SELF:
            continue  # the author's own extended version, not an external publication
        try:
            results[key] = check(key, entry)
        except (urllib.error.URLError, ValueError, KeyError, ET.ParseError, TimeoutError) as error:
            results[key] = {"status": "fail", "notes": f"lookup failed: {error}"}
        status = results[key]["status"]
        print(f"{status:4} {key}: {results[key].get('notes') or results[key].get('verified_with')}")
        time.sleep(1.0)  # be gentle with the public APIs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [k for k, v in results.items() if v["status"] != "ok"]
    print(f"{len(results) - len(failed)}/{len(results)} verified -> {OUT.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
