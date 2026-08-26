#!/usr/bin/env python
"""Why did a generation come back as stubs? Isolate the knob that kills thinking.

    python3 t3/request_probe.py --run t3/artifacts/gap          # free
    python3 t3/request_probe.py --run t3/artifacts/gap --micro  # ~$0.02
    python3 t3/request_probe.py --run t3/artifacts/gap --full   # ~one generation

TEMPORARY. Delete once the cause is in generate.py.

NOT NAMED bisect.py. Running `python3 t3/<name>.py` puts t3/ first on sys.path
for the whole process, so the file shadows any stdlib module of the same name -
and `random` imports `bisect`, so `import anthropic` died inside email.utils.
Nothing in t3/ may take a stdlib name.

WHY MICRO PROBES RATHER THAN REPLAYING THE PROMPT. The question is no longer
"what does the model write" but "why are there zero thinking tokens", and that
is answerable with a two-line prompt. Four tiny requests separate the knobs -
tools at all, tool_choice forced, `strict` - for cents, where four replays of a
25k-char prompt with ten images cost a generation each and confound the answer
with what the model happened to sample.
"""
import argparse
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import (EXTRA_BODY, MAX_TOKENS, THINKING, TOOL,  # noqa: E402
                      TOOL_CHOICE, _degenerate)

# Small, but genuinely needs a moment's arithmetic - a prompt with nothing to
# think about produces zero thinking tokens for honest reasons and proves
# nothing.
MICRO = ("Two 40 mm cubes sit with their centres 62 mm apart. Both are rotated "
         "45 degrees about the vertical axis. What is the clearance between "
         "their nearest faces along the line joining the centres? Reason it "
         "through, then give the number in mm.")


def load_key():
    """run.sh sources $RIPL_ROOT/anthropic.env; running this file directly does
    not, which is how four probes died on auth after paying nothing."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    p = os.path.join(os.environ.get("RIPL_ROOT", "/workspace/ripl"), "anthropic.env")
    if not os.path.exists(p):
        sys.exit(f"\n!! no API key, and {p} does not exist.\n"
                 f"   source it, or export ANTHROPIC_API_KEY.\n")
    for line in open(p):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(f"\n!! {p} did not define ANTHROPIC_API_KEY\n")
    return p


def thinking_tokens(msg):
    d = getattr(msg.usage, "output_tokens_details", None)
    n = getattr(d, "thinking_tokens", None) if d is not None else None
    if n is None and isinstance(d, dict):
        n = d.get("thinking_tokens")
    return n if n is not None else sum(
        len(getattr(b, "thinking", "")) for b in msg.content if b.type == "thinking")


def offline(run, pick=None):
    cands = [pick] if pick else ["response_failed.json", "response.json"]
    found = [(c, os.path.getmtime(os.path.join(run, c)))
             for c in cands if os.path.exists(os.path.join(run, c))]
    if not found:
        print(f"  (no response file in {run})")
        return
    for c, m in sorted(found, key=lambda t: -t[1]):
        print(f"  present       {c}  ({time.strftime('%H:%M:%S', time.localtime(m))}, "
              f"{(time.time()-m)/60:.0f} min ago)")
    raw = json.load(open(os.path.join(run, found[0][0])))
    print(f"\n=== step 0: {found[0][0]} (free) ===")
    kinds = {}
    for b in raw["content"]:
        kinds[b["type"]] = kinds.get(b["type"], 0) + 1
    print(f"  blocks        {kinds}")
    print(f"  stop_reason   {raw.get('stop_reason')}")
    det = (raw.get("usage") or {}).get("output_tokens_details") or {}
    print(f"  thinking      {det.get('thinking_tokens', 'not reported')} tokens")
    for b in raw["content"]:
        if b["type"] == "tool_use":
            print("  fields        " + ", ".join(
                f"{k}={len(v) if isinstance(v,str) else v}" for k, v in b["input"].items()))


def micro(client):
    """Four tiny requests. The only number that matters is thinking_tokens."""
    nostrict = {k: v for k, v in TOOL.items() if k != "strict"}
    forced = {"type": "tool", "name": TOOL["name"]}
    cases = [
        ("1  no tools at all       ", {}),
        ("2  tools, choice=auto    ", dict(tools=[TOOL], tool_choice={"type": "auto"})),
        ("3  tools, FORCED         ", dict(tools=[TOOL], tool_choice=forced)),
        ("4  tools, FORCED, nostrict", dict(tools=[nostrict], tool_choice=forced)),
    ]
    print("\n=== micro probes: does `thinking` survive each knob? ===")
    results = {}
    for label, extra in cases:
        kw = dict(model=os.environ.get("T3_MODEL", "claude-opus-5"),
                  max_tokens=4000, thinking=THINKING,
                  messages=[{"role": "user", "content": MICRO}], **extra)
        try:
            with client.messages.stream(**kw) as s:
                msg = s.get_final_message()
        except Exception as e:
            print(f"  {label}  ERROR {type(e).__name__}: {str(e)[:200]}")
            continue
        t = thinking_tokens(msg)
        results[label.strip()] = t
        print(f"  {label}  thinking {t:>6}   out {msg.usage.output_tokens:>5}   "
              f"stop {msg.stop_reason}")
    if results:
        print("\n  A row at 0 with rows above it non-zero names the knob that "
              "suppresses thinking.")
    return results


def content(blocks, run):
    out = []
    for b in blocks:
        if b["type"] == "image":
            path = b["path"]
            if not os.path.exists(path):
                path = os.path.join(run, "frames", os.path.basename(path))
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


def full(client, run, model):
    blocks = json.load(open(os.path.join(run, "blocks.json")))
    system = open(os.path.join(run, "system.txt")).read()
    body = content(blocks, run)
    nostrict = {k: v for k, v in TOOL.items() if k != "strict"}
    forced = {"type": "tool", "name": TOOL["name"]}
    variants = [
        ("auto", dict(tools=[TOOL], tool_choice={"type": "auto"})),
        ("nostrict", dict(tools=[nostrict], tool_choice=forced)),
        ("noextra", dict(tools=[TOOL], tool_choice=forced, _noextra=True)),
    ]
    for name, v in variants:
        print(f"\n=== full probe: {name} ===")
        kw = dict(model=model, max_tokens=MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": body}],
                  thinking=THINKING,
                  tools=v["tools"], tool_choice=v["tool_choice"])
        if not v.get("_noextra"):
            kw["extra_body"] = EXTRA_BODY
        try:
            with client.messages.stream(**kw) as s:
                msg = s.get_final_message()
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {str(e)[:300]}")
            continue
        tu = [b for b in msg.content if b.type == "tool_use"]
        print(f"  thinking {thinking_tokens(msg)}   out {msg.usage.output_tokens}   "
              f"stop {msg.stop_reason}")
        if tu:
            print("  fields   " + ", ".join(
                f"{k}={len(x) if isinstance(x,str) else x}" for k, x in tu[0].input.items()))
        with open(os.path.join(run, f"probe_{name}.json"), "w") as f:
            f.write(msg.to_json())
        if tu and not _degenerate(tu[0].input):
            print(f"\n  >> '{name}' PRODUCED REAL CONTENT - that knob is the cause.")
            print(f"     saved {run}/probe_{name}.json")
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--file")
    ap.add_argument("--micro", action="store_true", help="four tiny paid requests")
    ap.add_argument("--full", action="store_true", help="replay the real prompt")
    ap.add_argument("--model", default=os.environ.get("T3_MODEL", "claude-opus-5"))
    a = ap.parse_args()

    offline(a.run, a.file)
    if not (a.micro or a.full):
        print("\n  --micro to isolate the knob (~$0.02), --full to replay the prompt.")
        return

    src = load_key()
    print(f"\n  api key from {src} (...{os.environ['ANTHROPIC_API_KEY'][-4:]})")
    import anthropic
    print(f"  anthropic    {anthropic.__version__}")
    client = anthropic.Anthropic()
    if a.micro:
        micro(client)
    if a.full:
        full(client, a.run, a.model)


if __name__ == "__main__":
    main()
