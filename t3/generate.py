#!/usr/bin/env python
"""The one API call: prompt + frames -> reward.py, sampler.py. Needs `anthropic`.

    export ANTHROPIC_API_KEY=...
    python3 t3/generate.py --run t3/artifacts/gap [--model claude-opus-5]

Reads the run directory t3/assemble.py wrote and writes the generated artifacts
back into it. Nothing else in this repo imports `anthropic`, so every other
stage runs without a key.

WHY TOOL USE RATHER THAN PARSING A FENCED BLOCK. Two Python files and two prose
fields have to come back separately and intact. Fence-parsing a markdown reply
gets that right most of the time, which is the worst possible failure rate - a
mis-split writes half a reward function to disk and surfaces two stages later as
a syntax error. A tool call with `strict: true` returns four named fields the
API itself validates.

tool_choice is NOT FORCED, and the comment that used to sit here saying it was
safe to force was wrong. It claimed the `thinking: disabled` restriction
alongside a forced tool_choice was Amazon Bedrock only. It is not: measured on
this API, with an identical two-line prompt and thinking set to adaptive,

    no tools at all              thinking  321   out   818   end_turn
    tools, tool_choice auto      thinking 4000   out  4000   max_tokens
    tools, tool_choice FORCED    thinking    0   out   149   tool_use
    tools, FORCED, no strict     thinking    0   out  4000   max_tokens

Forcing zeroes the thinking. `strict: true` then turns a model that has not
reasoned into one that emits a minimal schema-valid object - row three is 149
output tokens, which is this repo's "x" bug reproduced on arithmetic.

So the model chooses to call the tool, having reasoned first, and the retry for
a turn with no tool_use block is now load-bearing rather than belt-and-braces.
Its firing is logged rather than silently smoothed over.

STREAMING because two source files plus two rationales is comfortably 8-16k
output tokens and adaptive thinking on a hard prompt can run for minutes.

NOTHING IS OVERWRITTEN AND NOTHING IS THROWN AWAY. The call is sampled and costs
money, so anything clobbered is gone; and a generation that fails review is
evidence too - "the model wrote this and the checker caught it" is the report's
strongest paragraph on reward hacking.

The one thing NOT kept as a generation is a degenerate one. A reward that
violates the contract is a finding worth reporting; a reward whose entire text
is the word "placeholder" is an accident of ours, and writing it leaves the run
directory needing --force to retry a call that never really happened. It goes to
response_failed.json instead.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loader import check_static  # noqa: E402
from spec import REWARD_FILE, SAMPLER_FILE  # noqa: E402

MODEL = os.environ.get("T3_MODEL", "claude-opus-5")

# THIS IS A TOTAL OUTPUT BUDGET AND IT IS SHARED WITH THINKING. The one
# generation that came out whole before adaptive thinking was turned on already
# spent 12,119 output tokens on the four fields alone; adaptive thinking on a
# prompt carrying an environment's source and ten frames is added on top of
# that, not carved out of it. 32000 left no margin. Raise this before lowering
# effort - the thinking is what makes the reward worth checking at all.
MAX_TOKENS = int(os.environ.get("T3_MAX_TOKENS", 64000))

# Below these the field did not survive the generation, whatever the API said.
# The shortest defensible reward.py is a REWARD_MAX line and a def; anything
# under a few hundred characters is a stub or a truncation, not brevity.
MIN_CODE_CHARS = 200
MIN_PROSE_CHARS = 80
# Observed verbatim in three of the first four generations' fields.
STUB_WORDS = frozenset({"x", "placeholder", "todo", "tbd", "n/a", "...", "..."})

# Parameters that not every SDK version knows as a keyword argument. Routed
# through extra_body so the request body stays minimal and this file works
# against both the 1.x SDK on the pod and the 0.x one nixpkgs ships. Edit here
# to change effort or thinking display; nothing else needs to move.
EXTRA_BODY = {
    "output_config": {"effort": "high"},
}

# ADAPTIVE THINKING IS NOT OPTIONAL HERE, and leaving it out is what produced
# this repo's first generations. `strict: true` constrained-decodes the tool
# input in schema order, so with no thinking the model is asked for reward_py -
# the hardest field - cold, before it has reasoned about the mechanism at all.
# Measured: the literal string "placeholder" in three fields with one real
# sampler, then "x" in all four. The forced tool_choice above is only defensible
# BECAUSE the model thinks first; the docstring asserted that and the request
# never sent it.
#
# PASSED AS A REAL KEYWORD, not through EXTRA_BODY. `thinking` is a first-class
# parameter the 1.x SDK knows; EXTRA_BODY is the hatch for parameters it does
# NOT know, and putting a known one there buys nothing while hiding whether it
# applied. `budget_tokens` is not the spelling any more - Opus 5 rejects it.
THINKING = {"type": "adaptive"}

# `auto`, NOT {"type": "tool", ...} and not "any". Forcing a tool call zeroes the
# thinking tokens - measured, see the docstring - and "any" is forcing with one
# tool defined. The tool description and the system prompt both say the tool is
# the only way to deliver the answer, and RETRY_NUDGE covers a turn that answers
# in prose anyway.
TOOL_CHOICE = {"type": "auto"}

TOOL = {
    "name": "emit_artifacts",
    "description": (
        "Return the dense reward function and the episode-configuration "
        "sampler, as two complete standalone Python modules, plus your "
        "reasoning. This is the only way to deliver the answer."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reward_py": {
                "type": "string",
                "description": (
                    "The complete contents of reward.py: module-level "
                    "REWARD_MAX and a compute_reward(env, obs, action, info) "
                    "function. Raw Python source, no markdown fences."),
            },
            "sampler_py": {
                "type": "string",
                "description": (
                    "The complete contents of sampler.py: a "
                    "sample_cube_poses(b, device) function. Raw Python "
                    "source, no markdown fences."),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "Markdown. Why each term is there, which mechanism from "
                    "the video and the description it addresses, and the "
                    "explicit stage ladder with the numeric band each stage "
                    "occupies."),
            },
            "uncertainties": {
                "type": "string",
                "description": (
                    "Markdown. What you are least sure of, which term you "
                    "expect an optimiser to exploit first, and what you "
                    "predict validation will find."),
            },
        },
        "required": ["reward_py", "sampler_py", "rationale", "uncertainties"],
        "additionalProperties": False,
    },
}

RETRY_NUDGE = (
    "You did not call the emit_artifacts tool. Call it now with the four "
    "required fields. Do not answer in prose.")


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("\n!! the `anthropic` package is not importable.\n\n"
                 "   pod:     uv pip install --python $RIPL_ROOT/venv/bin/python anthropic\n"
                 "   laptop:  nix-shell -p \"python3.withPackages(ps: [ps.anthropic])\" \\\n"
                 "              --run \"python3 t3/generate.py ...\"\n")
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("\n!! ANTHROPIC_API_KEY is not set.\n\n"
                 "   export ANTHROPIC_API_KEY=sk-ant-...\n"
                 "   Keep it out of the repo: .gitignore already covers .env, "
                 "*.key and credentials*.\n"
                 "   On a pod put it in $RIPL_ROOT/anthropic.env (chmod 600) - "
                 "NOT in env.sh,\n   which setup_runpod.sh rewrites and which "
                 "gets echoed into build logs.\n")
    return anthropic.Anthropic(), anthropic


def _content(blocks):
    """blocks.json -> API content, base64-ing the images at the last moment."""
    out = []
    for b in blocks:
        if b["type"] == "image":
            with open(b["path"], "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            out.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg", "data": data}})
        else:
            blk = {"type": "text", "text": b["text"]}
            # One breakpoint, at the end of the block that is byte-identical for
            # every mode and every run (the env source, the API surface and the
            # hacking guidance). Everything volatile is after it, so re-running
            # a second mode, or a second generation of the same one, reads that
            # prefix from cache.
            if b.get("cache"):
                blk["cache_control"] = {"type": "ephemeral"}
            out.append(blk)
    return out


def _tool_input(msg):
    for blk in msg.content:
        if blk.type == "tool_use" and blk.name == TOOL["name"]:
            return blk.input
    return None


def _text(msg):
    return "\n".join(b.text for b in msg.content if b.type == "text")


def _thinking_tokens(msg):
    d = getattr(msg.usage, "output_tokens_details", None)
    n = getattr(d, "thinking_tokens", None) if d is not None else None
    if n is None and isinstance(d, dict):
        n = d.get("thinking_tokens")
    return n if n is not None else sum(
        len(getattr(b, "thinking", "")) for b in msg.content if b.type == "thinking")


def _degenerate(args_in):
    """-> list of complaints about a tool call that came back unusable.

    A forced tool_choice plus `strict: true` guarantees a schema-valid object;
    it does NOT guarantee the fields contain anything. When the output budget
    runs out mid-call the decoder closes the JSON with minimal fills, and the
    result validates perfectly while being worthless. Checking the fields is the
    only thing that catches that, and not writing them is what keeps the run
    directory retryable without --force.
    """
    bad = []
    for field, floor in (("reward_py", MIN_CODE_CHARS),
                         ("sampler_py", MIN_CODE_CHARS),
                         ("rationale", MIN_PROSE_CHARS),
                         ("uncertainties", MIN_PROSE_CHARS)):
        v = (args_in.get(field) or "").strip()
        why = ("a stub" if v.lower().strip(".") in STUB_WORDS
               else f"{len(v)} chars" if len(v) < floor else None)
        if why:
            bad.append(f"{field}: {why}, {v[:60]!r}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the directory assemble.py wrote")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    reward_path = os.path.join(a.run, REWARD_FILE)
    if os.path.exists(reward_path) and not a.force:
        sys.exit(f"\n!! {reward_path} already exists.\n\n"
                 f"   Refusing to overwrite: the call costs money and the output "
                 f"is sampled,\n   so anything clobbered is gone. Same rule as "
                 f"t2/eval_modes.py.\n\n"
                 f"   For an independent second generation, use a new directory:\n"
                 f"     GEN=2 bash t3/run.sh generate\n"
                 f"   --force overwrites, deliberately.\n")

    with open(os.path.join(a.run, "blocks.json")) as f:
        blocks = json.load(f)
    with open(os.path.join(a.run, "system.txt")) as f:
        system = f.read()

    client, anthropic = _client()
    content = _content(blocks)
    n_img = sum(1 for b in content if b["type"] == "image")
    print(f"  model       {a.model}")
    print(f"  content     {len(content)} blocks, {n_img} images")

    messages = [{"role": "user", "content": content}]
    t0 = time.time()
    msg, args_in, attempts = None, None, 0

    for attempts in (1, 2):
        try:
            with client.messages.stream(
                model=a.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
                tools=[TOOL],
                tool_choice=TOOL_CHOICE,
                thinking=THINKING,
                extra_body=EXTRA_BODY,
            ) as stream:
                msg = stream.get_final_message()
        except anthropic.APIStatusError as e:
            sys.exit(f"\n!! API error {e.status_code}: "
                     f"{getattr(e, 'message', e)}\n")
        except anthropic.APIConnectionError as e:
            sys.exit(f"\n!! could not reach the API: {e}\n")

        if getattr(msg, "stop_reason", None) == "refusal":
            det = getattr(msg, "stop_details", None)
            sys.exit(f"\n!! the model declined this request "
                     f"(category: {getattr(det, 'category', '?')}).\n"
                     f"   {getattr(det, 'explanation', '')}\n")

        args_in = _tool_input(msg)
        if args_in is not None:
            bad = _degenerate(args_in)
            if not bad:
                break
            # The API said yes and the schema validated. The content did not
            # survive. Do not write it: a run directory holding a one-character
            # reward.py needs --force to retry, which is the wrong prompt to be
            # given at the moment the generation failed.
            with open(os.path.join(a.run, "response_failed.json"), "w") as f:
                f.write(msg.to_json())
            print(f"\n!! the tool call came back schema-valid and empty "
                  f"(stop_reason={msg.stop_reason}, "
                  f"out {msg.usage.output_tokens}/{MAX_TOKENS}):")
            for b in bad:
                print(f"     {b}")
            if msg.stop_reason == "max_tokens":
                sys.exit(
                    f"\n   The output budget ran out mid-tool-call, so "
                    f"`strict: true` closed the JSON\n   with minimal valid "
                    f"strings. Nothing was written except response_failed.json,"
                    f"\n   so this directory is still clean.\n\n"
                    f"   MAX_TOKENS is shared with thinking. Raise it:\n"
                    f"     T3_MAX_TOKENS={MAX_TOKENS * 2} bash t3/run.sh generate\n"
                    f"   or lower the effort in generate.py's EXTRA_BODY if the "
                    f"model has no room\n   left to raise it into.\n")
            sys.exit(f"\n   Raw reply saved to {a.run}/response_failed.json\n")
        # Forced tool_choice makes this very unlikely, which is exactly why it
        # is worth recording when it happens rather than retrying invisibly.
        print(f"  !! attempt {attempts}: no tool_use block "
              f"(stop_reason={msg.stop_reason}). Nudging.")
        messages += [{"role": "assistant", "content": msg.content},
                     {"role": "user", "content": RETRY_NUDGE}]

    if args_in is None:
        with open(os.path.join(a.run, "response_failed.json"), "w") as f:
            f.write(msg.to_json())
        sys.exit(f"\n!! two attempts, no tool call. Raw reply saved to "
                 f"{a.run}/response_failed.json\n")

    wall = time.time() - t0
    u = msg.usage
    print(f"  usage       in {u.input_tokens} "
          f"(cache write {getattr(u, 'cache_creation_input_tokens', 0)}, "
          f"read {getattr(u, 'cache_read_input_tokens', 0)}) "
          f"/ out {u.output_tokens}   {wall:.0f}s")
    # A request that silently dropped `thinking` is otherwise invisible, and
    # that is exactly the failure this file already shipped once.
    print(f"  thinking    {_thinking_tokens(msg)} tokens")

    # --- write everything -------------------------------------------------
    files = {REWARD_FILE: args_in["reward_py"],
             SAMPLER_FILE: args_in["sampler_py"],
             "rationale.md": args_in["rationale"],
             "uncertainties.md": args_in["uncertainties"]}
    for name, text in files.items():
        with open(os.path.join(a.run, name), "w") as f:
            f.write(text.rstrip() + "\n")

    with open(os.path.join(a.run, "response.json"), "w") as f:
        f.write(msg.to_json())
    with open(os.path.join(a.run, "request.json"), "w") as f:
        json.dump(dict(
            model=a.model, max_tokens=MAX_TOKENS, thinking=THINKING,
            tool_choice=TOOL_CHOICE, extra_body=EXTRA_BODY, tool=TOOL,
            # sha256, not hash(): the builtin is PYTHONHASHSEED-salted, so two
            # runs over a byte-identical system prompt recorded two different
            # values. A provenance field that changes when nothing changed is
            # worse than no field.
            system_sha256=hashlib.sha256(system.encode()).hexdigest()[:16],
            # image bytes elided on purpose: a request.json carrying ten
            # base64 jpegs is 2 MB of noise that no diff can read, and the
            # frames themselves are committed next to it.
            content=[{"type": b["type"],
                      **({"path": os.path.relpath(b["path"], a.run)}
                         if b["type"] == "image" else {"chars": len(b["text"])})}
                     for b in blocks],
        ), f, indent=2)

    # --- the static check, immediately ------------------------------------
    # The cheapest possible feedback loop: if the generation violates the
    # contract, say so now, in the same command, rather than on the pod.
    print()
    ok = True
    for name, kind in ((REWARD_FILE, "reward"), (SAMPLER_FILE, "sampler")):
        errors, warnings = check_static(files[name], kind, name)
        ok &= not errors
        print(f"  {'FAIL' if errors else 'PASS'}  {name}")
        for e in errors:
            print(f"          ERROR    {e}")
        for w in warnings:
            print(f"          warning  {w}")

    with open(os.path.join(a.run, "manifest.json"), "w") as f:
        json.dump(dict(model=a.model, stop_reason=msg.stop_reason,
                       request_id=getattr(msg, "_request_id", None),
                       attempts=attempts, wall_seconds=round(wall, 1),
                       input_tokens=u.input_tokens,
                       output_tokens=u.output_tokens,
                       cache_read=getattr(u, "cache_read_input_tokens", 0),
                       cache_write=getattr(u, "cache_creation_input_tokens", 0),
                       thinking_tokens=_thinking_tokens(msg),
                       loadable=ok,
                       anthropic_version=getattr(anthropic, "__version__", "?")),
                  f, indent=2)

    print(f"\n  wrote       {a.run}/{{{REWARD_FILE},{SAMPLER_FILE},rationale.md,"
          f"uncertainties.md}}")
    if not ok:
        print("\n  This generation will not load. It is kept as evidence - which "
              "check caught\n  it is the report's account of how LLM-written "
              "rewards fail. Tune\n  t3/prompts/hacking.md and regenerate into a "
              "new directory rather than\n  editing the generated file by hand.")
        sys.exit(1)
    print("\n  Read rationale.md and uncertainties.md, then:\n"
          f"    T3_RUN={a.run} bash t3/run.sh check summary")


if __name__ == "__main__":
    main()
