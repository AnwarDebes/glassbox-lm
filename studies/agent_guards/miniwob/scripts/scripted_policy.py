"""Scripted (no-LLM, no-learning) DOM policy for five MiniWoB++ tasks.

It reads only the MiniWoB observation dict (utterance, fields, dom_elements)
and returns one action spec per step:
    {"action_type": "CLICK_ELEMENT", "ref": int}
    {"action_type": "FOCUS_ELEMENT_AND_TYPE_FIELD", "ref": int, "field": int}
    {"action_type": "NONE"}
The spec is turned into a gym action with env.unwrapped.create_action(...).

Facts used (all from the observation; same facts a TM encoder would get):
- tag (lowercased; inputs get the type appended, e.g. "input_checkbox"),
  text, value, id, classes, parent ref, flags.
- Serialised checkbox value is str(value or ""): checked -> "True", unchecked -> "".
"""
from __future__ import annotations

from typing import Any

TASKS = ("click-button", "click-checkboxes", "enter-text", "click-link", "focus-text")


def field_value(obs: dict[str, Any], key: str) -> str | None:
    for k, v in obs.get("fields", ()):
        if k == key:
            return v
    return None


def field_index(obs: dict[str, Any], key: str) -> int | None:
    for i, (k, _v) in enumerate(obs.get("fields", ())):
        if k == key:
            return i
    return None


def elements(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return list(obs.get("dom_elements", ()))


def is_checked(el: dict[str, Any]) -> bool:
    return str(el.get("value", "")) == "True"


def label_text(obs: dict[str, Any], el: dict[str, Any]) -> str:
    """Text of the 't' pseudo-elements that share the element's parent (e.g. <label>)."""
    parts = [
        str(e["text"]).strip()
        for e in elements(obs)
        if e["parent"] == el["parent"] and e["tag"] == "t" and e["ref"] != el["ref"]
    ]
    return " ".join(p for p in parts if p)


def find_submit(obs: dict[str, Any]) -> dict[str, Any] | None:
    for e in elements(obs):
        if e["id"] == "subbtn":
            return e
    for e in elements(obs):
        if e["tag"] == "button" and str(e["text"]).strip().lower() == "submit":
            return e
    return None


def click(el: dict[str, Any]) -> dict[str, Any]:
    return {"action_type": "CLICK_ELEMENT", "ref": int(el["ref"])}


NONE = {"action_type": "NONE"}


def decide(task: str, obs: dict[str, Any]) -> dict[str, Any]:
    els = elements(obs)
    if task == "click-button":
        target = field_value(obs, "target")
        for e in els:
            if e["tag"] == "button" and str(e["text"]).strip() == target:
                return click(e)
        return NONE

    if task == "click-link":
        target = field_value(obs, "target")
        for e in els:
            if "alink" in str(e["classes"]).split() and str(e["text"]).strip() == target:
                return click(e)
        return NONE

    if task == "focus-text":
        for e in els:
            if e["tag"] in ("input_text", "textarea") and e["id"] == "tt":
                return click(e)
        for e in els:
            if e["tag"] in ("input_text", "textarea"):
                return click(e)
        return NONE

    if task == "enter-text":
        target = field_value(obs, "target")
        idx = field_index(obs, "target")
        box = next((e for e in els if e["tag"] == "input_text"), None)
        if box is None or target is None or idx is None:
            return NONE
        if str(box["value"]) != target:
            return {"action_type": "FOCUS_ELEMENT_AND_TYPE_FIELD", "ref": int(box["ref"]), "field": idx}
        sub = find_submit(obs)
        return click(sub) if sub else NONE

    if task == "click-checkboxes":
        targets = {v for k, v in obs.get("fields", ()) if k.startswith("target ")}
        for e in els:
            if e["tag"] != "input_checkbox":
                continue
            want = label_text(obs, e) in targets
            if want != is_checked(e):
                return click(e)  # toggle one wrong box per step
        sub = find_submit(obs)
        return click(sub) if sub else NONE

    raise ValueError(f"no scripted policy for task {task!r}")
