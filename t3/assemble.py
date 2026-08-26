#!/usr/bin/env python
"""Turn an mp4 and a mode tag into the exact prompt the LLM is sent. Stdlib.

    python3 t3/assemble.py --mode gap --video clip.mp4 --out t3/artifacts/gap
                           [--frames 10] [--select uniform|diverse] [--with-stats]

Two jobs, both deliberately kept out of generate.py so that neither needs an API
key and both can be inspected before a penny is spent:

    1. FRAMES. Shell out to the ffmpeg binary rather than importing anything.
       ffmpeg is already installed on the pod (setup/setup_runpod.sh) and is one
       `nix-shell -p ffmpeg` away on the laptop, so frame extraction adds no
       Python dependency to a repo whose analysis half runs on a bare
       interpreter.

    2. ASSEMBLY. The prompt is concatenated from t3/prompts/*.md with two
       sections spliced in from t3/spec.py - the contract and the allowed API
       surface. Splicing rather than duplicating is the point: the model is told
       exactly what t3/loader.py checks, and a rule cannot be tightened in the
       checker without the prompt changing too.

Outputs into the run directory:

    frames/frame_00.jpg ...   what the model saw, committed as evidence
    blocks.json               the content blocks, images by path (generate.py
                              base64s them) - so this file stays diffable
    prompt.txt                the whole thing rendered readable, with the images
                              marked in place. THIS is what the report quotes.
    prompt_manifest.json      provenance: mode, video, seed, frame indices,
                              which prompt files went in, the env-source hash.

THE ENV-SOURCE HASH GATE. The environment's source is handed to the model
Eureka-style, from a committed snapshot under t3/env_source/ rather than read
live out of $MANISKILL_REPO. A live read cannot drift from the installed
version but makes an old run's prompt unreproducible once the pod is gone, and
invisible in `git diff`. A snapshot is reproducible and reviewable but rots. So:
snapshot, plus a re-hash of the installed copy on every assembly, which refuses
to build a prompt if the two differ. That buys both properties for the price of
one comparison - the same move t2/harness.py:219 makes with ckpt_sha256.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spec import api_surface_markdown, contract_markdown  # noqa: E402

PROMPTS = os.path.join(HERE, "prompts")
ENV_SOURCE = os.path.join(HERE, "env_source", "stack_cube.py")
PROVENANCE = os.path.join(HERE, "env_source", "PROVENANCE.json")

# Assembly order. The first group is IDENTICAL for every mode and every run, so
# it is one cacheable prefix; everything volatile comes after it. See
# generate.py, which puts the cache breakpoint at the end of the stable block.
STABLE = ["env_source.md", "api_surface.md", "hacking.md"]
VOLATILE_HEAD = ["failure_{mode}.md"]          # + stats.md when --with-stats
VOLATILE_TAIL = ["frames.md", "task.md"]       # after the images


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def _ffmpeg():
    exe = os.environ.get("FFMPEG", "ffmpeg")
    if shutil.which(exe):
        return exe
    sys.exit(
        f"\n!! '{exe}' is not on PATH, and frame extraction needs it.\n\n"
        f"   On the pod it is installed already (setup/setup_runpod.sh).\n"
        f"   On this laptop:   nix-shell -p ffmpeg --run 'bash t3/run.sh frames'\n"
        f"   Or point at one:  FFMPEG=/path/to/ffmpeg ...\n")


def _decode_all(video, tmp, exe):
    """Every frame as a 512-wide jpeg. -> sorted list of paths.

    Decoding the lot and then choosing is simpler and more robust than asking
    ffmpeg for specific timestamps: it needs no ffprobe, no duration parsing and
    no frame-rate assumption, and a 200-step episode is only a few hundred small
    files.
    """
    subprocess.run(
        [exe, "-nostdin", "-loglevel", "error", "-i", video,
         "-vf", "scale=512:-2", "-q:v", "3", os.path.join(tmp, "%05d.jpg")],
        check=True)
    return sorted(glob.glob(os.path.join(tmp, "*.jpg")))


def _thumbs(video, exe, n_frames, size=32):
    """Every frame as `size`x`size` grayscale raw bytes, for `diverse`.

    Pure bytes, compared in pure Python - no PIL, no numpy. One frame is 1 kB.
    """
    out = subprocess.run(
        [exe, "-nostdin", "-loglevel", "error", "-i", video,
         "-vf", f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    step = size * size
    frames = [out[i:i + step] for i in range(0, len(out), step)]
    return [f for f in frames if len(f) == step][:n_frames]


def _pick_uniform(total, n):
    """Evenly spaced, with the FIRST and LAST frames pinned.

    The first frame is the initial configuration - which is what a T-II failure
    mode actually IS, a region of the initial-state distribution - and the last
    frame is the outcome. Neither is negotiable; the rest fill in between.
    """
    if total <= n:
        return list(range(total))
    inner = n - 2
    mid = [round(1 + i * (total - 2 - 1) / max(inner - 1, 1)) for i in range(inner)]
    return sorted(set([0] + mid + [total - 1]))


def _pick_diverse(thumbs, n):
    """Greedy farthest-point selection over the grayscale thumbnails.

    Every episode runs the full 200 steps because evaluation sets
    ignore_terminations=True, so a uniform sample spends most of its frames on a
    motionless tail. This spends them on the moments that differ from each
    other: seed with the first and last frames, then repeatedly add whichever
    frame is furthest (in mean absolute pixel difference) from everything picked
    so far.
    """
    total = len(thumbs)
    if total <= n:
        return list(range(total))

    def dist(i, j):
        a, b = thumbs[i], thumbs[j]
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    picked = [0, total - 1]
    best = [min(dist(i, p) for p in picked) for i in range(total)]
    while len(picked) < n:
        nxt = max(range(total), key=lambda i: best[i] if i not in picked else -1)
        picked.append(nxt)
        for i in range(total):
            best[i] = min(best[i], dist(i, nxt))
    return sorted(picked)


def extract_frames(video, out_dir, n, select):
    """-> (frame_paths, frame_indices, total_frames)."""
    exe = _ffmpeg()
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(frames_dir, "*.jpg")):
        os.remove(stale)

    with tempfile.TemporaryDirectory() as tmp:
        allf = _decode_all(video, tmp, exe)
        if not allf:
            sys.exit(f"!! ffmpeg decoded 0 frames from {video} - is it a video?")
        if select == "diverse":
            idx = _pick_diverse(_thumbs(video, exe, len(allf)), n)
        else:
            idx = _pick_uniform(len(allf), n)
        kept = []
        for k, i in enumerate(idx):
            dst = os.path.join(frames_dir, f"frame_{k:02d}.jpg")
            shutil.copyfile(allf[i], dst)
            kept.append(dst)
    return kept, idx, len(allf)


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def _read(name):
    with open(os.path.join(PROMPTS, name)) as f:
        return f.read()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def env_source_text(allow_drift=False):
    """The committed snapshot, after checking it still matches what is installed."""
    with open(ENV_SOURCE) as f:
        src = f.read()
    prov = json.load(open(PROVENANCE))
    drift = False

    repo = os.environ.get("MANISKILL_REPO")
    live = os.path.join(repo, prov["source"]) if repo else None
    if live and os.path.exists(live):
        if _sha(live) != prov["sha256"]:
            drift = True
            msg = (f"\n!! t3/env_source/stack_cube.py no longer matches the "
                   f"installed ManiSkill.\n"
                   f"   snapshot: {prov['sha256'][:16]}  ({prov['maniskill_commit'][:8]})\n"
                   f"   installed:{_sha(live)[:16]}\n\n"
                   f"   The prompt would describe an environment that is not the one\n"
                   f"   the reward will run in. Re-snapshot and re-generate:\n"
                   f"     cp {live} {ENV_SOURCE}   # then update PROVENANCE.json\n"
                   f"   or pass --allow-source-drift to proceed knowingly (it is\n"
                   f"   stamped into the manifest, so the report cannot hide it).\n")
            if not allow_drift:
                sys.exit(msg)
            print(msg)
    return src, prov, drift


def assemble(mode, frame_paths, seed, outcome, with_stats, allow_drift):
    """-> (blocks, prompt_text, provenance dict).

    `blocks` is the content list in API order, with images carried as paths.
    generate.py base64s them at the last moment; keeping them as paths here is
    what lets blocks.json be read and diffed.
    """
    src, prov, drift = env_source_text(allow_drift)

    stable = []
    used = []
    for name in STABLE:
        text = _read(name)
        if name == "env_source.md":
            text = text.replace("{{ENV_SOURCE}}", src)
        elif name == "api_surface.md":
            text = text.replace("{{API_SURFACE}}", api_surface_markdown())
        stable.append(text)
        used.append(name)

    head = []
    for name in [n.format(mode=mode) for n in VOLATILE_HEAD]:
        head.append(_read(name))
        used.append(name)
    if with_stats:
        head.append(_read("stats.md"))
        used.append("stats.md")

    tail = []
    for name in VOLATILE_TAIL:
        text = _read(name)
        if name == "frames.md":
            text = (text.replace("{{N_FRAMES}}", str(len(frame_paths)))
                        .replace("{{SEED}}", str(seed))
                        .replace("{{OUTCOME}}", outcome))
        elif name == "task.md":
            text = text.replace("{{CONTRACT}}", contract_markdown())
        tail.append(text)
        used.append(name)

    for i, part in enumerate(stable + head + tail):
        if "{{" in part:
            sys.exit(f"!! unresolved placeholder in prompt part {i}: "
                     f"{part[part.index('{{'):part.index('{{') + 40]!r}")

    blocks = [{"type": "text", "text": "\n\n".join(stable), "cache": True},
              {"type": "text", "text": "\n\n".join(head)}]
    blocks += [{"type": "image", "path": p} for p in frame_paths]
    blocks.append({"type": "text", "text": "\n\n".join(tail)})

    rendered = []
    for b in blocks:
        if b["type"] == "text":
            rendered.append(b["text"])
        else:
            rendered.append(f"[IMAGE: {os.path.relpath(b['path'])}]")
    return blocks, "\n\n".join(rendered), dict(prov=prov, used=used, drift=drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--select", default="uniform", choices=["uniform", "diverse"])
    ap.add_argument("--seed", default="unknown")
    ap.add_argument("--outcome", default="failure")
    ap.add_argument("--with-stats", action="store_true")
    ap.add_argument("--allow-source-drift", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(PROMPTS, f"failure_{a.mode}.md")):
        sys.exit(f"!! no failure description for mode '{a.mode}'.\n"
                 f"   Expected {PROMPTS}/failure_{a.mode}.md - it is written by "
                 f"hand,\n   from the T-II measurement, and it is a deliverable.")
    os.makedirs(a.out, exist_ok=True)

    # record_seeds.py names its clips rgb_seed<N>_<outcome>_sep<M>mm.mp4, so the
    # seed and the outcome are already in the path. Read them from there when
    # they were not passed explicitly.
    #
    # This is not convenience. prompts/frames.md tells the model "the episode's
    # outcome was: <outcome>", and asserting `failure` over a clip that actually
    # succeeded actively misleads it about what it is looking at - a worse
    # error than saying nothing. Forgetting one env var should not be able to
    # cause that.
    if a.video:
        m = re.search(r"_seed(\d+)_", os.path.basename(a.video))
        if m and a.seed == "unknown":
            a.seed = m.group(1)
            print(f"  seed        {a.seed}  (read from the filename)")
        m = re.search(r"_seed\d+_(fail|success)_", os.path.basename(a.video))
        if m:
            got = "failure" if m.group(1) == "fail" else "success"
            if got != a.outcome:
                print(f"  outcome     {got}  (from the filename; "
                      f"--outcome said '{a.outcome}')")
                a.outcome = got

    existing = sorted(glob.glob(os.path.join(a.out, "frames", "*.jpg")))
    if a.video:
        paths, idx, total = extract_frames(a.video, a.out, a.frames, a.select)
        print(f"  frames      {len(paths)} of {total} by '{a.select}': {idx}")
    elif existing:
        paths, idx, total = existing, [], 0
        print(f"  frames      reusing {len(paths)} already in {a.out}/frames")
    else:
        sys.exit(f"\n!! no --video and no frames in {a.out}/frames.\n\n"
                 f"   T-III's input is a clip of a real failure, produced by T-II:\n"
                 f"     SEEDS=<pick from mode_{a.mode}_seed1.csv> bash t2/run.sh videos\n"
                 f"   mp4s are gitignored, so there is never one in a fresh clone.\n")

    blocks, text, prov = assemble(a.mode, paths, a.seed, a.outcome,
                                  a.with_stats, a.allow_source_drift)

    with open(os.path.join(a.out, "blocks.json"), "w") as f:
        json.dump(blocks, f, indent=2)
    with open(os.path.join(a.out, "prompt.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "system.txt"), "w") as f:
        f.write(_read("system.md"))
    with open(os.path.join(a.out, "prompt_manifest.json"), "w") as f:
        json.dump(dict(mode=a.mode, video=a.video, video_seed=a.seed,
                       outcome=a.outcome, n_frames=len(paths), frame_indices=idx,
                       total_frames=total, select=a.select,
                       with_stats=a.with_stats,
                       prompt_files=prov["used"],
                       env_source=prov["prov"],
                       env_source_drift=prov["drift"],
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest()[:16],
                       chars=len(text)), f, indent=2)

    print(f"  prompt      {len(text):,} chars, ~{len(text) // 4:,} tokens, "
          f"{len(blocks)} blocks")
    print(f"  wrote       {a.out}/prompt.txt, blocks.json, prompt_manifest.json")
    print(f"\n  Read prompt.txt before generating. It is a deliverable in its own "
          f"right,\n  and it is cheaper to fix a prompt than to validate a "
          f"generation from a bad one.")


if __name__ == "__main__":
    main()
