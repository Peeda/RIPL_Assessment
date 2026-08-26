#!/usr/bin/env python
"""Why did a generation come back as stubs? Bisect the request, one knob at a time.

    python3 t3/bisect.py --run t3/artifacts/gap            # free: read what we have
    python3 t3/bisect.py --run t3/artifacts/gap --probe    # paid: try variants

TEMPORARY. Delete once the cause is found and the fix is in generate.py.

Step 0 costs nothing: it reads response_failed.json and reports whether the
model thought at all, what it stopped on, and how long each field came back.
`thinking` blocks absent means extra_body never reached the API, which is a
different bug from the model choosing to stub.

The probes replay the SAME prompt (blocks.json + system.txt, so this is not a
new prompt being tested) with one knob changed each time, in decreasing order
of suspicion, and STOP at the first variant that produces real content. Each
one costs roughly one generation, so the ordering matters.
"""
import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import EXTRA_BODY, MAX_TOKENS, TOOL, _degenerate  # noqa: E402

FLOOR = 200          # chars of reward_py that count as "the model engaged"


def report(msg, label):
    kinds = {}
    for b in msg.content:
        kinds[b.type] = kinds.get(b.type, 0) + 1
    u = msg.usage
    print(f"  {label}")
    print(f"    blocks      {kinds}")
    print(f"    stop        {msg.stop_reason}")
    print(f"    usage       in {u.input_tokens} / out {u.output_tokens}"
          f"  (cache r{getattr(u,'cache_read_input_tokens',0)} "
          f"w{getattr(u,'cache_creation_input_tokens',0)})")
    think = [b for b in msg.content if b.type == "thinking"]
    print(f"    thinking    {'ABSENT - extra_body did not apply' if not think else str(sum(len(b.thinking) for b in think)) + ' chars'}")
    tu = [b for b in msg.content if b.type == "tool_use"]
    if not tu:
        txt = "".join(b.text for b in msg.content if b.type == "text")
        print(f"    no tool_use. text: {txt[:300]!r}")
        return False
    inp = tu[0].input
    print(f"    fields      " + ", ".join(f"{k}={len(v) if isinstance(v,str) else v}"
                                          for k, v in inp.items()))
    return not _degenerate(inp)


def offline(run):
    p = os.path.join(run, "response_failed.json")
    if not os.path.exists(p):
        p = os.path.join(run, "response.json")
    if not os.path.exists(p):
        sys.exit(f"no response.json or response_failed.json in {run}")
    raw = json.load(open(p))
    print(f"\n=== step 0: {os.path.basename(p)} (free) ===")
    kinds = {}
    for b in raw["content"]:
        kinds[b["type"]] = kinds.get(b["type"], 0) + 1
    print(f"  blocks        {kinds}")
    print(f"  stop_reason   {raw.get('stop_reason')}")
    print(f"  usage         {raw.get('usage')}")
    think = [b for b in raw["content"] if b["type"] == "thinking"]
    print(f"  thinking      {'ABSENT - the request never enabled it' if not think else str(sum(len(b.get('thinking','')) for b in think)) + ' chars'}")
    for b in raw["content"]:
        if b["type"] == "tool_use":
            print("  fields        " + ", ".join(
                f"{k}={len(v) if isinstance(v,str) else v}" for k, v in b["input"].items()))
    return bool(think)


def content(blocks, run):
    out = []
    for b in blocks:
        if b["type"] == "image":
            path = b["path"] if os.path.isabs(b["path"]) else os.path.join(run, os.path.basename(os.path.dirname(b["path"])), os.path.basename(b["path"]))
            if not os.path.exists(path):
                path = b["path"]
            with open(path, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            out.append({"type": "image", "source": {"type": "base64",
                        "media_type": "image/jpeg", "data": data}})
        else:
            blk = {"type": "text", "text": b["text"]}
            if b.get("cache"):
                blk["cache_control"] = {"type": "ephemeral"}
            out.append(blk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", action="store_true", help="make paid API calls")
    ap.add_argument("--model", default=os.environ.get("T3_MODEL", "claude-opus-5"))
    a = ap.parse_args()

    had_thinking = offline(a.run)
    if not a.probe:
        print("\n  --probe to try request variants (each costs ~one generation).")
        return

    import anthropic
    client = anthropic.Anthropic()
    blocks = json.load(open(os.path.join(a.run, "blocks.json")))
    system = open(os.path.join(a.run, "system.txt")).read()
    full = content(blocks, a.run)
    text_only = [b for b in full if b["type"] == "text"]

    tool_nostrict = {k: v for k, v in TOOL.items() if k != "strict"}
    forced = {"type": "tool", "name": TOOL["name"]}

    variants = [
        ("A no-strict     ", dict(tools=[tool_nostrict], tool_choice=forced,
                                  extra_body=EXTRA_BODY, content=full)),
        ("B tool_choice=auto", dict(tools=[TOOL], tool_choice={"type": "auto"},
                                    extra_body=EXTRA_BODY, content=full)),
        ("C no extra_body ", dict(tools=[TOOL], tool_choice=forced,
                                  extra_body=None, content=full)),
        ("D no images     ", dict(tools=[TOOL], tool_choice=forced,
                                  extra_body=EXTRA_BODY, content=text_only)),
    ]

    for label, v in variants:
        print(f"\n=== probe {label.strip()} ===")
        kw = dict(model=a.model, max_tokens=MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": v["content"]}],
                  tools=v["tools"], tool_choice=v["tool_choice"])
        if v["extra_body"]:
            kw["extra_body"] = v["extra_body"]
        try:
            with client.messages.stream(**kw) as s:
                msg = s.get_final_message()
        except Exception as e:
            print(f"    ERROR  {type(e).__name__}: {str(e)[:400]}")
            continue
        good = report(msg, label.strip())
        with open(os.path.join(a.run, f"bisect_{label.split()[0]}.json"), "w") as f:
            f.write(msg.to_json())
        if good:
            print(f"\n  >> {label.strip()} PRODUCED REAL CONTENT. That knob is the cause.")
            print(f"     saved: {a.run}/bisect_{label.split()[0]}.json")
            return
    print("\n  no variant produced real content - the cause is not in these four.")


if __name__ == "__main__":
    main()
