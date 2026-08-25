#!/usr/bin/env python
"""Check t2/geometry.py against cases worked out by hand. No deps, no pytest.

    python3 t2/test_geometry.py

Runs in about a second on a laptop with nothing installed, because geometry.py
imports only `math`. That is the point: the definition of a failure mode is
checkable without a GPU, a pod, or a ManiSkill install, so it can be checked
before an hour of rollouts is spent on it rather than after.

The face_gap cases are the ones that matter. face_gap is the axis both modes
are defined on, it is not a quantity anyone can eyeball, and getting the
bearing convention wrong would leave every number in the report subtly off in a
way no downstream check would catch.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (DISCOVERY, MODES, PANDA_BASE_XY, cube_features,  # noqa: E402
                      geom_features, geom_from_row, half_extent, wilson,
                      wrap, wrap90, yaw)

PASS = FAIL = 0


def ok(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"   {detail}" if detail else ""))


def close(got, want, tol, label):
    ok(abs(got - want) < tol, label, f"got {got:.6f}, want {want:.6f}")


def quat_z(theta):
    """wxyz quaternion for a rotation of `theta` about z."""
    return [math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]


# ---------------------------------------------------------------------------
print("angles")
# ---------------------------------------------------------------------------
close(wrap(math.pi), math.pi, 1e-12, "wrap(pi) stays at +pi, not -pi")
close(wrap(-math.pi), math.pi, 1e-12, "wrap(-pi) folds to +pi")
close(wrap(3 * math.pi / 2), -math.pi / 2, 1e-12, "wrap(3pi/2)")
close(wrap(0.3), 0.3, 1e-12, "wrap is identity inside the range")

# 4-fold symmetry: these four describe the SAME cube geometry, so wrap90 must
# collapse them onto one value. This is why relative_yaw is the wrong axis to
# regress against and relative_yaw_mod90 is the right one.
base = wrap90(math.radians(5))
for k in (1, 2, 3):
    close(wrap90(math.radians(5) + k * math.pi / 2), base, 1e-12,
          f"wrap90 is invariant under a {90 * k} deg cube rotation")
close(wrap90(math.radians(85)), math.radians(-5), 1e-9,
      "wrap90(+85 deg) == -5 deg, the same geometry")

close(yaw(quat_z(0.0)), 0.0, 1e-12, "yaw of identity quaternion")
close(yaw(quat_z(math.radians(30))), math.radians(30), 1e-9, "yaw of 30 deg")
close(yaw(quat_z(math.radians(200))), math.radians(200 - 360), 1e-9,
      "yaw wraps past pi")

# ---------------------------------------------------------------------------
print("half_extent - a 40 mm cube presents 56.6 mm across its diagonal")
# ---------------------------------------------------------------------------
close(half_extent(0.0), 0.02, 1e-12, "0 deg: half the cube width")
close(half_extent(math.pi / 2), 0.02, 1e-12, "90 deg: same, by symmetry")
close(half_extent(math.radians(45)), 0.02 * math.sqrt(2), 1e-12,
      "45 deg: half the diagonal")
ok(all(0.02 - 1e-12 <= half_extent(math.radians(d)) <= 0.02 * math.sqrt(2) + 1e-12
       for d in range(0, 360, 3)),
   "half_extent stays within [h, h*sqrt(2)] for every angle")

# ---------------------------------------------------------------------------
print("face_gap - the axis both modes are defined on")
# ---------------------------------------------------------------------------
# Both cubes axis-aligned, 60 mm apart along +x. Each presents a 40 mm face, so
# 20 mm of the 60 is cube and 40 mm... no: each contributes HALF a width to the
# gap, 20 + 20 = 40, leaving 20 mm of clear air.
g = geom_features(0.0, 0.0, 0.0, 0.060, 0.0, 0.0)
close(g["face_gap"], 0.020, 1e-12,
      "axis-aligned, 60 mm apart -> 60 - 20 - 20 = 20 mm of clear space")

# Same centres, both cubes turned 45 deg to the bearing. Each now presents its
# diagonal, 28.28 mm of half-extent, so the gap nearly closes: 60 - 56.6 = 3.4.
g = geom_features(0.0, 0.0, math.radians(45), 0.060, 0.0, math.radians(45))
close(g["face_gap"], 0.060 - 2 * 0.02 * math.sqrt(2), 1e-12,
      "both at 45 deg -> 60 - 28.3 - 28.3 = 3.4 mm")

# NEGATIVE face_gap is real, not a bug: the sampler's 58.6 mm centre floor does
# not stop two diagonally-presented cubes from overlapping along the bearing.
g = geom_features(0.0, 0.0, math.radians(45), 0.0545, 0.0, math.radians(45))
ok(g["face_gap"] < 0, "face_gap goes negative at the sampler's floor",
   f"got {g['face_gap'] * 1000:.1f} mm")

# The bearing is what resolves each yaw, so face_gap must not depend on the
# direction the PAIR is laid out in - only on each cube's yaw RELATIVE to it.
for deg in range(0, 360, 15):
    psi = math.radians(deg)
    a = (0.0, 0.0)
    b = (0.150 * math.cos(psi), 0.150 * math.sin(psi))
    # both cubes rotated with the bearing: the presented faces are identical
    g = geom_features(a[0], a[1], psi, b[0], b[1], psi)
    close(g["face_gap"], 0.150 - 0.04, 1e-9,
          f"face_gap is bearing-invariant when yaws follow it ({deg} deg)")

# 4-fold symmetry again, this time through the whole feature: rotating either
# cube by 90 deg cannot change the geometry.
ref = geom_features(0.0, 0.0, 0.3, 0.09, 0.02, -0.4)["face_gap"]
for da in (math.pi / 2, math.pi, 3 * math.pi / 2):
    for db in (0, math.pi / 2, math.pi):
        got = geom_features(0.0, 0.0, 0.3 + da, 0.09, 0.02, -0.4 + db)["face_gap"]
        close(got, ref, 1e-9, "face_gap is invariant under 90 deg cube rotations")

# face_gap <= separation - 2h always, since each half-extent is at least h.
for i in range(200):
    t = i / 200 * 2 * math.pi
    g = geom_features(0.0, 0.0, t, 0.13, 0.05, 2 * t)
    sep = math.dist((0.0, 0.0), (0.13, 0.05))
    ok(g["face_gap"] <= sep - 0.04 + 1e-12, "face_gap never exceeds sep - 2h")

# ---------------------------------------------------------------------------
print("reach distances - measured from the Panda's base, not the origin")
# ---------------------------------------------------------------------------
g = geom_features(PANDA_BASE_XY[0], PANDA_BASE_XY[1], 0.0, 0.0, 0.0, 0.0)
close(g["dist_A"], 0.0, 1e-12, "a cube at the base is 0 m away")
close(g["dist_B"], 0.615, 1e-12, "a cube at the origin is 615 mm from the base")
ok(g["dist_max"] == g["dist_B"] and g["dist_min"] == g["dist_A"],
   "dist_max / dist_min order the two")

# The sign error this constant exists to prevent: from the world origin these
# two would be equidistant, from the base they are not.
gg = geom_features(0.1, 0.0, 0.0, -0.1, 0.0, 0.0)
ok(gg["dist_A"] > gg["dist_B"],
   "+x is FARTHER from the base than -x (the constant is doing its job)")

# ---------------------------------------------------------------------------
print("cube_features round-trip")
# ---------------------------------------------------------------------------
a_pose = [0.05, -0.02, 0.02] + quat_z(math.radians(20))
b_pose = [0.14, 0.03, 0.02] + quat_z(math.radians(55))
f = cube_features(a_pose, b_pose)
close(f["cubeA_theta"], math.radians(20), 1e-9, "cubeA_theta from the quaternion")
close(f["cubeB_theta"], math.radians(55), 1e-9, "cubeB_theta from the quaternion")
close(f["relative_yaw"], math.radians(35), 1e-9, "relative_yaw is B - A")
close(f["separation"], math.dist((0.05, -0.02), (0.14, 0.03)), 1e-12, "separation")
close(f["relative_yaw_mod90"], math.radians(35), 1e-9, "mod90 column")

# 45 deg is the ONE ambiguous input: it sits exactly on the boundary of
# (-45, 45], so a value a single ULP either side of it comes back as +45 or
# -45. Those describe the same geometry, which is the whole point of the mod-90
# axis - but it means never asserting an exact mod90 value near the boundary,
# and never binning on it without folding |mod90| first.
ok(abs(abs(wrap90(math.radians(45))) - math.pi / 4) < 1e-9
   and abs(abs(wrap90(math.radians(45) + 1e-15)) - math.pi / 4) < 1e-9,
   "+-45 deg are the same geometry; only the sign is unstable at the boundary")
# geom_from_row on a row that already carries the columns must return them
# unchanged, and on one that does not must recompute the same values.
row = {k: str(v) for k, v in f.items()}
close(geom_from_row(row)["face_gap"], f["face_gap"], 1e-12,
      "geom_from_row reads stored columns")
bare = {k: str(f[k]) for k in ("cubeA_x", "cubeA_y", "cubeA_theta",
                               "cubeB_x", "cubeB_y", "cubeB_theta")}
close(geom_from_row(bare)["face_gap"], f["face_gap"], 1e-12,
      "geom_from_row recomputes for a row that predates the columns")

# ---------------------------------------------------------------------------
print("wilson")
# ---------------------------------------------------------------------------
lo, hi = wilson(0, 20)
ok(lo == 0.0 and 0.15 < hi < 0.17, "0/20 -> [0, 0.161], not +-0.000",
   f"got [{lo:.3f}, {hi:.3f}]")
lo, hi = wilson(20, 20)
ok(hi == 1.0 and 0.8 < lo < 0.85, "20/20 -> upper bound clamps at 1")
for k, n in ((0, 5), (1, 3), (50, 100), (99, 100)):
    lo, hi = wilson(k, n)
    ok(0.0 <= lo <= k / n <= hi <= 1.0, f"wilson({k},{n}) brackets the estimate")
ok(math.isnan(wilson(0, 0)[0]), "wilson(0, 0) is nan rather than a crash")
# The bounds at the extremes must be EXACTLY 0 and 1, not a few ULP short.
# Computing c and h separately lands at 0.9999999999999999 for k = n, which
# prints as 1.000 and produces a negative error bar that matplotlib refuses.
for n in (1, 3, 20, 100, 300, 1000):
    ok(wilson(n, n)[1] == 1.0, f"wilson({n},{n}) upper bound is exactly 1.0",
       f"got {wilson(n, n)[1]!r}")
    ok(wilson(0, n)[0] == 0.0, f"wilson(0,{n}) lower bound is exactly 0.0",
       f"got {wilson(0, n)[0]!r}")
    ok(wilson(n, n)[1] - n / n >= 0 and n / n - wilson(n, n)[0] >= 0,
       f"wilson({n},{n}) brackets 1.0 with non-negative error bars")
ok(wilson(5, 10)[1] - wilson(5, 10)[0] > wilson(50, 100)[1] - wilson(50, 100)[0],
   "the interval narrows as n grows")

# ---------------------------------------------------------------------------
print("the two modes are disjoint, on the real initial-state distribution")
# ---------------------------------------------------------------------------
# Disjointness is what makes each mode's rate an independent measurement of its
# own mechanism. Asserted analytically AND against the committed evidence.
ok(not any(MODES["gap"](g) and MODES["farb"](g)
           for g in ({"face_gap": fg, "dist_A": da, "dist_B": db,
                      "dist_max": max(da, db), "dist_min": min(da, db)}
                     for fg in (-0.01, 0.0, 0.019, 0.02, 0.05, 0.1)
                     for da in (0.4, 0.6, 0.71, 0.76, 0.9)
                     for db in (0.4, 0.6, 0.75, 0.76, 0.9))),
   "gap and farb cannot both hold (dist_max < 0.76 vs dist_B >= 0.76)")

here = os.path.dirname(os.path.abspath(__file__))
nominal = os.path.join(here, "results", "nominal.csv")
if os.path.exists(nominal):
    rows = list(csv.DictReader(open(nominal)))
    counts = {"gap": 0, "farb": 0, "both": 0}
    succ = {"gap": 0, "farb": 0, "nominal": 0}
    for r in rows:
        g = geom_from_row(r)
        hits = [t for t in ("gap", "farb") if MODES[t](g)]
        for t in hits:
            counts[t] += 1
            succ[t] += r["success_once"] == "1"
        if len(hits) > 1:
            counts["both"] += 1
        succ["nominal"] += r["success_once"] == "1"
    ok(counts["both"] == 0,
       f"0 of {len(rows)} discovery episodes satisfy both modes",
       f"{counts['both']} do")
    # The DISCOVERY table is quoted in the report and pre-registers what the
    # confirmation passes have to reproduce. It must match the CSV it came from.
    for tag in ("gap", "farb", "nominal"):
        n = counts[tag] if tag != "nominal" else len(rows)
        want_rate, want_lo, want_hi, want_n = DISCOVERY[tag]
        ok(n == want_n, f"DISCOVERY['{tag}'] n matches nominal.csv",
           f"table says {want_n}, csv has {n}")
        close(succ[tag] / n, want_rate, 5e-4, f"DISCOVERY['{tag}'] rate")
        lo, hi = wilson(succ[tag], n)
        close(lo, want_lo, 5e-4, f"DISCOVERY['{tag}'] wilson lo")
        close(hi, want_hi, 5e-4, f"DISCOVERY['{tag}'] wilson hi")
else:
    print(f"  (skipped: {nominal} not present)")

# ---------------------------------------------------------------------------
print("")
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
