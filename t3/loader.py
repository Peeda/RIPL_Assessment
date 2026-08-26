#!/usr/bin/env python
"""Layer A: check generated code against the contract, then install it. Stdlib.

    python3 t3/loader.py t3/artifacts/gap        # lint an artifact directory
    python3 t3/loader.py t3/fixtures/hack_grasp_only.py --kind reward

The cheapest gate in T-III and the first one that runs. It walks the syntax tree
BEFORE the module is ever imported, so a generation that reaches for `os` or
writes `while True:` costs 0.2 s to reject rather than a pod session.

WHAT THIS IS AND IS NOT
    It is not a security sandbox and the README says so plainly. The code came
    from our own prompt, is committed, and is reviewed by eye before it trains
    anything. Layer A exists to catch ACCIDENTS AND MIS-SPECIFICATION - a
    swapped argument order, a numpy import that silently breaks seed
    reproducibility, a `.item()` that costs a GPU sync every step of training -
    and to catch them somewhere cheaper than a rollout. Claiming containment
    would be false, and a false claim is worse than the honest limit.

FIVE CONSUMERS, ONE IMPLEMENTATION
    env_t3.py installs the generated code; probes.py, sampler_check.py and
    align.py each load it to measure it; test_loader.py checks that this file
    rejects every fixture in t3/fixtures/. If the checks lived in the env
    subclass, the tests could not reach them without a simulator.

Every rule is read from t3/spec.py, which is also what renders the prompt. The
model is told exactly what is checked, and the two cannot drift.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import (ALLOWED_ENV_ATTRS, ALLOWED_IMPORTS, ALLOWED_INFO_KEYS,  # noqa: E402
                  FORBIDDEN_ATTRS, FORBIDDEN_NAMES, MAX_AST_NODES,
                  MAX_SOURCE_LINES, REWARD_FILE, REWARD_MAX_NAME,
                  REWARD_MODULE_REQUIRES, SAMPLER_FILE, SAMPLER_MODULE_REQUIRES)

REQUIRES = {"reward": REWARD_MODULE_REQUIRES, "sampler": SAMPLER_MODULE_REQUIRES}
FILENAME = {"reward": REWARD_FILE, "sampler": SAMPLER_FILE}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """`import torch` has to work, and `import os` has to not.

    exec() with a stripped __builtins__ removes __import__ entirely, which makes
    even the allowed imports fail. Supplying a whitelisting one keeps the static
    rule and the runtime behaviour identical, so a module that passed
    check_static cannot then import something else at load time.
    """
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(
            f"'{name}' is not importable in T-III generated code "
            f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
    return __import__(name, globals, locals, fromlist, level)


# What the generated code is allowed to call. Small on purpose: a reward that
# needs `sorted` or `map` is doing something a batched tensor expression should
# be doing instead.
SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "abs": abs,
    "float": float, "int": int, "bool": bool, "enumerate": enumerate,
    "zip": zip, "sum": sum, "print": print, "round": round,
    "True": True, "False": False, "None": None,
    "__import__": _guarded_import,
    "__build_class__": None,   # `class` is rejected statically; belt and braces
}


def _dotted(node):
    """The dotted path of an Attribute chain, or None if it is not rooted at a
    bare name. `env.cubeA.pose.p` -> ('env', 'cubeA.pose.p')."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None, None
    return node.id, ".".join(reversed(parts))


def _env_attr_ok(path):
    """A maximal chain is fine if it is an allowed leaf, or a prefix of one.

    The prefix case admits an intermediate binding - `cubeA = env.cubeA` then
    `cubeA.pose.p` - which is idiomatic and which a leaves-only rule would
    reject. It does mean the alias itself is unchecked; that hole is closed at
    runtime by layer B's recording proxy, which sees the actual reads.
    """
    if path in ALLOWED_ENV_ATTRS:
        return True
    return any(a.startswith(path + ".") for a in ALLOWED_ENV_ATTRS)


def check_static(source, kind, name="<generated>"):
    """-> list of violation strings, empty when the source satisfies the contract.

    Returns rather than raises so one call reports EVERY problem: an LLM
    generation usually has two or three, and fixing them one exception at a time
    costs one API call each.
    """
    bad = []
    required = REQUIRES[kind]

    try:
        tree = ast.parse(source, filename=name)
    except SyntaxError as e:
        return [f"does not parse: line {e.lineno}: {e.msg}"]

    n_lines = len(source.splitlines())
    if n_lines > MAX_SOURCE_LINES:
        bad.append(f"too long: {n_lines} lines > {MAX_SOURCE_LINES}")
    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > MAX_AST_NODES:
        bad.append(f"too complex: {nodes} AST nodes > {MAX_AST_NODES}")

    # --- module-level shape ------------------------------------------------
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign,
                             ast.Expr, ast.Pass)):
            continue
        bad.append(f"line {getattr(node, 'lineno', '?')}: "
                   f"{type(node).__name__} is not allowed at module level")

    # --- the walk ----------------------------------------------------------
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")

        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    bad.append(f"line {ln}: forbidden import '{a.name}' "
                               f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                bad.append(f"line {ln}: forbidden import from '{node.module}' "
                           f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")

        elif isinstance(node, ast.While):
            # A batched reward has no use for one, and rejection sampling must
            # be a bounded `for` so a pathological region cannot hang training.
            bad.append(f"line {ln}: `while` is not allowed - use a bounded `for`")

        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global,
                               ast.Nonlocal, ast.Await, ast.Yield)):
            bad.append(f"line {ln}: {type(node).__name__} is not allowed")

        elif isinstance(node, ast.FunctionDef) and node.decorator_list:
            bad.append(f"line {ln}: decorators are not allowed on '{node.name}'")

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_NAMES:
                bad.append(f"line {ln}: forbidden name '{node.id}'")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                bad.append(f"line {ln}: dunder attribute '.{node.attr}' is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                bad.append(f"line {ln}: '.{node.attr}' is not allowed - it either "
                           f"mutates the simulator or forces a GPU sync every step")

    # --- env attribute chains ---------------------------------------------
    # Only MAXIMAL chains are checked: env.cubeA.pose.p is one read, not three.
    checked = set()
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Attribute):
                checked.add(id(child.value))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in checked:
            continue
        root, path = _dotted(node)
        if root == "env" and path and not _env_attr_ok(path):
            bad.append(f"line {getattr(node, 'lineno', '?')}: "
                       f"'env.{path}' is not on the allowed surface")

    # --- info[...] subscripts ---------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "info"):
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if key.value not in ALLOWED_INFO_KEYS:
                bad.append(f"line {node.lineno}: info['{key.value}'] is not a key "
                           f"evaluate() returns (allowed: "
                           f"{', '.join(sorted(ALLOWED_INFO_KEYS))})")
        else:
            bad.append(f"line {node.lineno}: info[...] must use a literal string key")

    # --- required functions, exact parameter names, in order ---------------
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fname, params in required.items():
        fn = defs.get(fname)
        if fn is None:
            bad.append(f"missing required function '{fname}({', '.join(params)})'")
            continue
        got = tuple(a.arg for a in fn.args.args)
        if got != tuple(params):
            # Names and ORDER, not arity. compute_reward(env, action, obs, info)
            # parses, imports, runs, and computes the reward from the wrong
            # tensor - which is a debugging session, not an error message.
            bad.append(f"'{fname}' has parameters {got}, expected {tuple(params)} "
                       f"(names and order both matter)")
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            bad.append(f"'{fname}' must not take *args/**kwargs/keyword-only args")

    # --- REWARD_MAX --------------------------------------------------------
    if kind == "reward":
        found = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == REWARD_MAX_NAME:
                        found = node.value
        if found is None:
            bad.append(f"missing module-level '{REWARD_MAX_NAME} = <positive number>'")
        elif not (isinstance(found, ast.Constant)
                  and isinstance(found.value, (int, float))
                  and not isinstance(found.value, bool)
                  and found.value > 0):
            bad.append(f"'{REWARD_MAX_NAME}' must be a positive numeric literal, "
                       f"so the gate can read it without importing the module")

    return bad


def load_source(source, kind, name="<generated>"):
    """check_static, then exec into a restricted namespace. -> namespace dict.

    Raises ValueError listing every violation. Static checks gate the import,
    never the other way round: importing first would run module-level code we
    have not looked at yet.
    """
    bad = check_static(source, kind, name)
    if bad:
        raise ValueError(f"{name} violates the T-III contract:\n" +
                         "\n".join(f"  - {b}" for b in bad))
    ns = {"__builtins__": dict(SAFE_BUILTINS), "__name__": "t3_generated"}
    exec(compile(source, name, "exec"), ns)          # noqa: S102 - see docstring
    for fname in REQUIRES[kind]:
        if not callable(ns.get(fname)):
            raise ValueError(f"{name}: '{fname}' is not callable after import")
    return ns


def load_file(path, kind):
    with open(path) as f:
        return load_source(f.read(), kind, os.path.basename(path))


def load_artifacts(run_dir):
    """Both modules from a generation directory. -> (reward_ns, sampler_ns)."""
    return (load_file(os.path.join(run_dir, REWARD_FILE), "reward"),
            load_file(os.path.join(run_dir, SAMPLER_FILE), "sampler"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__.strip().split("\n\n")[1])
    target = args[0]
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else None

    if os.path.isdir(target):
        jobs = [(os.path.join(target, REWARD_FILE), "reward"),
                (os.path.join(target, SAMPLER_FILE), "sampler")]
    else:
        if kind is None:
            kind = "sampler" if "sampler" in os.path.basename(target) else "reward"
        jobs = [(target, kind)]

    fails = 0
    for path, k in jobs:
        if not os.path.exists(path):
            print(f"  FAIL  {path} - missing")
            fails += 1
            continue
        with open(path) as f:
            bad = check_static(f.read(), k, os.path.basename(path))
        if bad:
            fails += 1
            print(f"  FAIL  {path}  ({k})")
            for b in bad:
                print(f"          {b}")
        else:
            print(f"  PASS  {path}  ({k})")

    if fails:
        print(f"\n{fails} artifact(s) violate the contract. The contract is "
              f"rendered into the prompt by t3/spec.py -\n"
              f"if a rule is wrong, fix it there and regenerate; do not edit the "
              f"generated file by hand\nunless the report says you did.")
        sys.exit(1)
    print("\nlayer A passed - the artifacts are code of the right shape.")


if __name__ == "__main__":
    main()
