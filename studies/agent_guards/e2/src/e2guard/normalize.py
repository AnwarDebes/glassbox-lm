"""Value matching for provenance: does an argument value occur in a source text (the user's query or a tool output)?

Plain code, no model. Every rule here is a choice frozen with the code hash; tests in tests/test_facts.py.

- Text: case-folded, every run of white space collapsed to one space. A string value matches when it occurs in the
  source with no letter or digit directly before or after it ("13" does not match inside "113" or "x13").
- Numbers (int and float values, and strings made only of digits): matched against the number tokens of the source.
  A number token is not glued to a letter, digit, '-', '/', ':' or '.digit' (so no number is read out of a date
  "2024-05-13", a time "10:20", an IBAN or "1j1l-2k3j"); "30,000" reads as 30000; a currency sign or '%' may touch it.
- Dates (YYYY-MM-DD) and date-times (YYYY-MM-DD HH:MM): the source's date mentions are extracted (ISO dates, "May 13",
  "May 13th, 2024", "13 May", "13th of May 2024", and ranges "13th to the 17th of May", "May 13-17"); a value matches when
  its month and day occur with the same year or with no year. A date-time also needs its time among the source's times
  ("10:20", "10:20 am", "2 pm", "noon"; 12-hour times are read as 24-hour).
- URLs: also tried without "http(s)://" and without a trailing "/".
- Tool outputs are AgentDojo's YAML text. The source text is the raw text plus the scalars parsed with PyYAML's
  BaseLoader (every scalar stays a string, escapes are undone), so a value is found whether or not YAML quoted, escaped or
  folded it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:  # PyYAML is in both venvs (AgentDojo needs it); without it only the raw text is used
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

WS = re.compile(r"\s+")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MONTH_OF = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_OF.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
MONTH_OF["sept"] = 9
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
ORD = r"(?:st|nd|rd|th)?"
NUM = re.compile(r"(?<![\w\-/:.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w\-/:]|\.\d)")
ISO_D = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
ISO_DT = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[ t](\d{1,2}):(\d{2})")
MD = re.compile(MON + r"\.? (\d{1,2})" + ORD + r"(?![\d:])(?:,? (\d{4}))?")
DM = re.compile(r"(?<![\d:])(\d{1,2})" + ORD + r" (?:of )?" + MON + r"\.?(?:,? (\d{4}))?")
RANGE_DM = re.compile(r"(?<![\d:])(\d{1,2})" + ORD + r" ?(?:to|until|till|through|and|-|–) ?(?:the )?(\d{1,2})" + ORD
                      + r" (?:of )?" + MON + r"\.?(?:,? (\d{4}))?")
RANGE_MD = re.compile(MON + r"\.? (\d{1,2})" + ORD + r" ?(?:to|until|till|through|and|-|–) ?(?:the )?(\d{1,2})" + ORD
                      + r"(?![\d:])(?:,? (\d{4}))?")
TIME_HM = re.compile(r"(?<![\d.])(\d{1,2})[:.](\d{2})(?!\d)(?: ?(a\.?m\.?|p\.?m\.?))?")
TIME_H = re.compile(r"(?<![\d:.])(\d{1,2}) ?(a\.?m\.?|p\.?m\.?)(?![a-z])")
VAL_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
VAL_DT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::\d{2})?$")
DIGITS = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
URLISH = re.compile(r"^(https?://)?(www\.)?[a-z0-9\-]+(\.[a-z0-9\-]+)+(/\S*)?$")


def norm_text(s: str) -> str:
    return WS.sub(" ", s.casefold()).strip()


def find_bounded(hay: str, needle: str) -> bool:
    """needle occurs in hay with no letter or digit glued to either end (checked only where needle's own end is one)."""
    if not needle:
        return False
    start = 0
    n0, n1 = needle[0].isalnum(), needle[-1].isalnum()
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return False
        j = i + len(needle)
        if (i == 0 or not n0 or not hay[i - 1].isalnum()) and (j == len(hay) or not n1 or not hay[j].isalnum()):
            return True
        start = i + 1


def _hour24(h: int, ampm: str | None) -> int:
    if not ampm:
        return h
    pm = ampm.startswith("p")
    if h == 12:
        return 12 if pm else 0
    return h + 12 if pm else h


@dataclass
class Source:
    """One searchable source: its normalized text and the numbers, dates and times found in it."""

    label: str
    text: str
    numbers: set = field(default_factory=set)
    dates: set = field(default_factory=set)  # (year or None, month, day)
    times: set = field(default_factory=set)  # (hour24, minute)


def _yaml_scalars(raw: str) -> list[str]:
    if yaml is None or not raw:
        return []
    try:
        obj = yaml.load(raw, Loader=yaml.BaseLoader)
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):  # not YAML: the raw text is used alone
        return []
    out: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                out.append(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif o is not None:
            out.append(str(o))

    walk(obj)
    return out


def make_source(label: str, raw: str, parse_yaml: bool = True) -> Source:
    parts = [raw or ""]
    if parse_yaml:
        parts += _yaml_scalars(raw or "")
    text = norm_text("\n".join(parts))
    src = Source(label=label, text=text)
    for m in NUM.finditer(text):
        tok = m.group(0).replace(",", "")
        try:
            src.numbers.add(float(tok))
        except ValueError:
            pass
    for m in ISO_D.finditer(text):
        src.dates.add((int(m[1]), int(m[2]), int(m[3])))
    for m in ISO_DT.finditer(text):
        src.times.add((int(m[4]), int(m[5])))
    for m in MD.finditer(text):
        src.dates.add((int(m[3]) if m[3] else None, MONTH_OF[m[1]], int(m[2])))
    for m in DM.finditer(text):
        src.dates.add((int(m[3]) if m[3] else None, MONTH_OF[m[2]], int(m[1])))
    for m in RANGE_DM.finditer(text):
        y = int(m[4]) if m[4] else None
        src.dates.add((y, MONTH_OF[m[3]], int(m[1])))
        src.dates.add((y, MONTH_OF[m[3]], int(m[2])))
    for m in RANGE_MD.finditer(text):
        y = int(m[4]) if m[4] else None
        src.dates.add((y, MONTH_OF[m[1]], int(m[2])))
        src.dates.add((y, MONTH_OF[m[1]], int(m[3])))
    for m in TIME_HM.finditer(text):
        h, mi = int(m[1]), int(m[2])
        if h <= 24 and mi < 60:
            src.times.add((_hour24(h, m[3]), mi))
    for m in TIME_H.finditer(text):
        h = int(m[1])
        if 1 <= h <= 12:
            src.times.add((_hour24(h, m[2]), 0))
    if "noon" in text:
        src.times.add((12, 0))
    if "midnight" in text:
        src.times.add((0, 0))
    return src


@dataclass(frozen=True)
class Atom:
    """One matchable piece of an argument value."""

    kind: str  # num | date | datetime | str
    value: object
    shown: str  # the value as written in the call, for logs and clause explanations


def atoms_of(value) -> list[Atom]:
    """The atoms of an argument value. None, booleans, empty strings and empty lists carry no data and give no atom.
    Lists give one atom per element; dicts (e-mail attachments) give the atoms of their values except the key 'type'."""
    out: list[Atom] = []
    if value is None or isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(Atom("num", float(value), repr(value)))
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return out
        m = VAL_DT.match(s)
        if m:
            out.append(Atom("datetime", (int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])), s))
            return out
        m = VAL_DATE.match(s)
        if m:
            out.append(Atom("date", (int(m[1]), int(m[2]), int(m[3])), s))
            return out
        if DIGITS.match(s):
            out.append(Atom("num", float(s), s))
            return out
        out.append(Atom("str", norm_text(s), s))
        return out
    if isinstance(value, (list, tuple)):
        for v in value:
            out.extend(atoms_of(v))
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            if k != "type":
                out.extend(atoms_of(v))
        return out
    out.append(Atom("str", norm_text(str(value)), str(value)))
    return out


def url_variants(s: str) -> list[str]:
    out = [s]
    if URLISH.match(s):
        t = re.sub(r"^https?://", "", s)
        out += [t, t.rstrip("/"), s.rstrip("/")]
    return list(dict.fromkeys(v for v in out if v))


def matches(atom: Atom, src: Source) -> bool:
    if atom.kind == "num":
        v = float(atom.value)  # type: ignore[arg-type]
        return any(abs(x - v) <= 1e-9 * max(1.0, abs(v)) for x in src.numbers)
    if atom.kind == "date":
        y, m, d = atom.value  # type: ignore[misc]
        return (y, m, d) in src.dates or (None, m, d) in src.dates
    if atom.kind == "datetime":
        y, m, d, hh, mi = atom.value  # type: ignore[misc]
        return ((y, m, d) in src.dates or (None, m, d) in src.dates) and (hh, mi) in src.times
    return any(find_bounded(src.text, v) for v in url_variants(str(atom.value)))
