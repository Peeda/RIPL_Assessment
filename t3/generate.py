#!/usr/bin/env python
"""The one API call: prompt + frames -> reward.py, sampler.py. Needs `anthropic`.

    export ANTHROPIC_API_KEY=...
    python3 t3/generate.py --run t3/artifacts/gap [--model claude-opus-5]

Reads the run directory t3/assemble.py wrote (blocks.json, system.txt, frames/)
and writes the generated artifacts back into it. Nothing else in this repo
imports `anthropic`, so every other stage runs without a key.

WHY TOOL USE RATHER THAN PARSING A FENCED BLOCK
    Two Python files and two prose fields have to come back separately and
    intact. Fence-parsing a markdown reply gets that right most of the time,
    which is the worst possible failure rate - a mis-split writes half a reward
    function to disk and the error surfaces three stages later as a syntax
    violation. A tool call with `strict: true` returns four named fields that
    the API itself validates.

    tool_choice is FORCED. That is safe here: the restriction requiring
    `thinking: {"type": "disabled"}` alongside a forced tool_choice is specific
    to Amazon Bedrock, and this is the first-party API - so the model still
    thinks before it answers. There is a retry for a turn that comes back
    without a tool_use block anyway, and the retry firing is itself worth
    knowing about, so it is logged rather than silently smoothed over.

WHY STREAMING
    Two source files plus two rationales is comfortably 8-16k output tokens and
    adaptive thinking on a hard prompt can run for minutes. A non-streaming
    request risks the SDK's ten-minute timeout for no benefit.

WHAT IS SAVED, AND WHY ALL OF IT
    request.json (image data elided, paths kept - so it stays diffable),
    response.json (raw, with usage and the request id), reward.py, sampler.py,
    rationale.md, uncertainties.md, manifest.json. A generation that FAILS
    validation is evidence too - "the model wrote this and the gate caught it"
    is the report's strongest paragraph on reward hacking - so nothing is
    thrown away and nothing is silently overwritten.
"""
import argparse
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loader import check_static  # noqa: E402
from spec import REWARD_FILE, SAMPLER_FILE  # noqa: E402

MODEL = os.environ.get("T3_MODEL", "claude-opus-5")
MAX_TOKENS = 32000

# Parameters that not every SDK version knows as a keyword argument. Routed
# through extra_body so the request body stays minimal and this file works
# against both the 1.x SDK on the pod and the 0.x one nixpkgs ships. Edit here
# to change effort or thinking display; nothing else needs to move.
EXTRA_BODY = {
    "output_config": {"effort": "high"},
}

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
                tool_choice={"type": "tool", "name": TOOL["name"]},
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
            break
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
            model=a.model, max_tokens=MAX_TOKENS, extra_body=EXTRA_BODY,
            tool=TOOL, system_sha=hash(system) & 0xFFFFFFFF,
            # image bytes elided on purpose: a request.json carrying ten
            # base64 jpegs is 2 MB of noise that no diff can read, and the
            # frames themselves are committed next to it.
            content=[{"type": b["type"],
                      **({"path": os.path.relpath(b["path"], a.run)}
                         if b["type"] == "image" else {"chars": len(b["text"])})}
                     for b in blocks],
        ), f, indent=2)

    # --- layer A, immediately ---------------------------------------------
    # The cheapest possible feedback loop: if the generation violates the
    # contract, say so now, in the same command, rather than on the pod.
    print()
    ok = True
    for name, kind in ((REWARD_FILE, "reward"), (SAMPLER_FILE, "sampler")):
        bad = check_static(files[name], kind, name)
        if bad:
            ok = False
            print(f"  FAIL  {name} violates the contract:")
            for b in bad:
                print(f"          {b}")
        else:
            print(f"  PASS  {name}  (layer A)")

    with open(os.path.join(a.run, "manifest.json"), "w") as f:
        json.dump(dict(model=a.model, stop_reason=msg.stop_reason,
                       request_id=getattr(msg, "_request_id", None),
                       attempts=attempts, wall_seconds=round(wall, 1),
                       input_tokens=u.input_tokens,
                       output_tokens=u.output_tokens,
                       cache_read=getattr(u, "cache_read_input_tokens", 0),
                       cache_write=getattr(u, "cache_creation_input_tokens", 0),
                       layer_a_passed=ok,
                       anthropic_version=getattr(anthropic, "__version__", "?")),
                  f, indent=2)

    print(f"\n  wrote       {a.run}/{{{REWARD_FILE},{SAMPLER_FILE},rationale.md,"
          f"uncertainties.md}}")
    if not ok:
        print("\n  Layer A rejected this generation. It is kept as evidence - a "
              "rejected\n  generation is a report paragraph. Tune "
              "t3/prompts/hacking.md and regenerate\n  into a new directory "
              "rather than editing the generated file by hand.")
        sys.exit(1)
    print("\n  Read rationale.md and uncertainties.md before validating. Then:\n"
          f"    T3_RUN={a.run} bash t3/run.sh probes sampler align verify")


if __name__ == "__main__":
    main()
