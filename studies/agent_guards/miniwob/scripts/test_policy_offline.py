"""Offline check of scripted_policy.py WITHOUT a browser.

What this tests: the policy logic plus MiniWoB's own Python parsing path
(FIELD_EXTRACTORS -> DOMElement -> create_observation -> serialize_dom_element)
on hand-made DOM trees shaped like core.getDOMInfo() output (core.js lines 337-470).
What this does NOT test: the real task pages, the browser, Selenium, timing, rewards.
"""
import sys

from miniwob.dom import DOMElement
from miniwob.fields import FIELD_EXTRACTORS
from miniwob.observation import create_empty_screenshot, create_observation

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scripted_policy import decide  # noqa: E402

_ref = [0]
_tref = [0]


def el(tag, children=(), **kw):
    _ref[0] += 1
    d = {"tag": tag, "left": 10.0, "top": 60.0 + _ref[0], "width": 40.0, "height": 12.0,
         "ref": _ref[0], "children": list(children), "id": kw.pop("id", ""),
         "classes": kw.pop("classes", ""), "bgColor": "rgba(0, 0, 0, 0)", "fgColor": "rgb(0, 0, 0)"}
    d.update(kw)
    if not d["children"] and "text" not in d:
        d["text"] = ""  # element without element children -> textOnly -> text = joined text ("")
    return d


def t(text):
    _tref[0] -= 1
    return {"tag": "t", "left": 0.0, "top": 0.0, "width": 20.0, "height": 10.0, "ref": _tref[0],
            "children": [], "text": text}


def page(*area_children, area_id="area"):
    return el("BODY", [el("DIV", [el("DIV", list(area_children), id=area_id)], id="wrap")])


def observe(task, utterance, raw_root):
    fields = FIELD_EXTRACTORS[task](utterance)
    return create_observation(utterance, DOMElement(raw_root), create_empty_screenshot(160, 210), fields)


def find(raw, pred):
    if pred(raw):
        return raw
    for c in raw["children"]:
        r = find(c, pred)
        if r:
            return r
    return None


results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + f": got {got} want {want}")


# click-button
raw = page(el("DIV", text="lorem ipsum dolor"), el("BUTTON", text="submit"), el("BUTTON", text="Okay"),
           el("SPAN", text="sit amet: "), el("INPUT_text", value=""), el("BUTTON", text="no"))
okay = find(raw, lambda d: d.get("text") == "Okay")
obs = observe("click-button", 'Click on the "Okay" button.', raw)
check("click-button picks Okay", decide("click-button", obs), {"action_type": "CLICK_ELEMENT", "ref": okay["ref"]})

# click-link
raw = page(t("lorem ipsum"), el("SPAN", text="dolor", classes="alink"), t("dolor sit"),
           el("SPAN", text="amet", classes="alink"))
link = find(raw, lambda d: d.get("classes") == "alink" and d.get("text") == "dolor")
obs = observe("click-link", 'Click on the link "dolor".', raw)
check("click-link picks span.alink 'dolor' (not the plain text)", decide("click-link", obs),
      {"action_type": "CLICK_ELEMENT", "ref": link["ref"]})

# focus-text
raw = page(el("INPUT_text", id="tt", value=""), area_id="area")
box = find(raw, lambda d: d.get("id") == "tt")
obs = observe("focus-text", "Focus into the textbox.", raw)
check("focus-text clicks #tt", decide("focus-text", obs), {"action_type": "CLICK_ELEMENT", "ref": box["ref"]})

# enter-text (two steps: type field 0, then submit)
raw = page(el("DIV", [el("INPUT_text", id="tt", value="")], id="form"), el("BUTTON", text="Submit", id="subbtn"))
box = find(raw, lambda d: d.get("id") == "tt")
sub = find(raw, lambda d: d.get("id") == "subbtn")
utt = 'Enter "Kasey" into the text field and press Submit.'
obs = observe("enter-text", utt, raw)
check("enter-text step 1 types field 0 into #tt", decide("enter-text", obs),
      {"action_type": "FOCUS_ELEMENT_AND_TYPE_FIELD", "ref": box["ref"], "field": 0})
box["value"] = "Kasey"
obs = observe("enter-text", utt, raw)
check("enter-text step 2 clicks Submit", decide("enter-text", obs), {"action_type": "CLICK_ELEMENT", "ref": sub["ref"]})

# click-checkboxes (labels hold <input type=checkbox> + a text node)
def cb(name, checked=False):
    return el("LABEL", [el("INPUT_checkbox", value=checked), t(name)])


raw = page(cb("abc"), cb("de", checked=True), cb("fgh"), el("BUTTON", text="Submit", id="subbtn"), area_id="boxes")
boxes = {}
for lab in find(raw, lambda d: d.get("id") == "boxes")["children"]:
    if lab["tag"] == "LABEL":
        boxes[lab["children"][1]["text"]] = lab["children"][0]
sub = find(raw, lambda d: d.get("id") == "subbtn")
utt = "Select abc, fgh and click Submit."
obs = observe("click-checkboxes", utt, raw)
print("   serialized checkbox values:", [(e["tag"], repr(e["value"])) for e in obs["dom_elements"] if e["tag"] == "input_checkbox"])
check("checkboxes step 1 ticks abc", decide("click-checkboxes", obs), {"action_type": "CLICK_ELEMENT", "ref": boxes["abc"]["ref"]})
boxes["abc"]["value"] = True
obs = observe("click-checkboxes", utt, raw)
check("checkboxes step 2 unticks de (not a target)", decide("click-checkboxes", obs),
      {"action_type": "CLICK_ELEMENT", "ref": boxes["de"]["ref"]})
boxes["de"]["value"] = False
obs = observe("click-checkboxes", utt, raw)
check("checkboxes step 3 ticks fgh", decide("click-checkboxes", obs), {"action_type": "CLICK_ELEMENT", "ref": boxes["fgh"]["ref"]})
boxes["fgh"]["value"] = True
obs = observe("click-checkboxes", utt, raw)
check("checkboxes step 4 clicks Submit", decide("click-checkboxes", obs), {"action_type": "CLICK_ELEMENT", "ref": sub["ref"]})
raw2 = page(cb("xy"), el("BUTTON", text="Submit", id="subbtn"), area_id="boxes")
sub2 = find(raw2, lambda d: d.get("id") == "subbtn")
obs = observe("click-checkboxes", "Select nothing and click Submit.", raw2)
check("checkboxes 'nothing' clicks Submit at once", decide("click-checkboxes", obs), {"action_type": "CLICK_ELEMENT", "ref": sub2["ref"]})

print("tags seen after real serialisation:", sorted({e["tag"] for e in obs["dom_elements"]}))
print(f"OFFLINE RESULT: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
