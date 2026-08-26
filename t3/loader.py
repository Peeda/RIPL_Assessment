#!/usr/bin/env python
"""Check generated code against the contract, then install it. Stdlib.

    python3 t3/loader.py t3/artifacts/gap          # lint an artifact directory
    python3 t3/loader.py t3/fixtures/good_reward.py --kind reward

The cheapest check in T-III and the first one that runs. It walks the syntax
tree BEFORE the module is imported, so a generation that reaches for `os` or
writes `while True:` costs 0.2 s to reject rather than a pod session.

TWO CHANNELS, AND THE SPLIT IS THE POLICY
    check_static returns (errors, warnings). Errors refuse to load; warnings are
    printed, recorded, and proceed. Something is an ERROR only when it makes the
    artifact unusable or silently wrong - a wrong parameter ORDER computes the
    reward from the wrong tensor, a numpy import breaks reset(seed=s). Style,
    performance and reading an unexpected attribute are warnings, because a
    checker that refuses a working reward costs a regeneration cycle and blocks
    T-IV.

WHAT THIS IS AND IS NOT
    Not a security sandbox, and the README says so plainly. The code came from
    our own prompt, is committed, and is read by eye before it trains anything.
    This exists to catch ACCIDENTS AND MIS-SPECIFICATION somewhere cheaper than
    a rollout. Claiming containment would be false.

Every rule is read from t3/spec.py, which is also what renders the prompt, so
the model is told exactly what is checked.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import (ALLOWED_ENV_ROOTS, ALLOWED_IMPORTS, ALLOWED_INFO_KEYS,  # noqa: E402
                  DISCOURAGED_ATTRS, FORBIDDEN_ATTRS, FORBIDDEN_NAMES,
                  MAX_SOURCE_LINES, REWARD_FILE, REWARD_MAX_NAME,
                  REWARD_MODULE_REQUIRES, SAMPLER_FILE, SAMPLER_MODULE_REQUIRES)

REQUIRES = {"reward": REWARD_MODULE_REQUIRES, "sampler": SAMPLER_MODULE_REQUIRES}


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """`import torch` has to work, and `import os` has to not.

    exec() with a stripped __builtins__ removes __import__ entirely, which makes
    even the allowed imports fail. Supplying a whitelisting one keeps the static
    rule and the runtime behaviour identical.
    """
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"'{name}' is not importable in T-III generated code "
                          f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
    return __import__(name, globals, locals, fromlist, level)


# EVERY ORDINARY BUILTIN A NUMERICAL FUNCTION MIGHT REACCH FOR. The omission of
# `dict` cost a 46,463-token generation: the model ended its sampler with
# `return dict(cubeA_xyz=..., ...)` - which is idiomatic, and which the contract
# asks for in those words ("-> dict with exactly these four keys") - and it died
# with NameError at call time, after the static check had passed it. A `{...}`
# literal is a bytecode op and needs no builtin, so the sibling generation that
# happened to use one worked. That is an arbitrary distinction to fail a
# generation on.
#
# The rule for this list: include anything whose absence would surprise someone
# writing plain numerical Python. Exclude only what actually widens the reach -
# open/eval/exec/compile, the attribute and namespace reflection
# (getattr/setattr/globals/vars/dir), and object/type/super. Static checks
# already reject dunder attributes and `class`, so this is the second layer, not
# the only one.
SAFE_BUILTINS = {
    # constructors and containers
    "dict": dict, "list": list, "tuple": tuple, "set": set,
    "frozenset": frozenset, "slice": slice,
    # numbers and text
    "float": float, "int": int, "bool": bool, "complex": complex,
    "str": str, "repr": repr, "format": format, "hash": hash,
    "abs": abs, "round": round, "pow": pow, "divmod": divmod,
    # iteration and aggregation
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "all": all, "any": any,
    "iter": iter, "next": next,
    # the rest
    "isinstance": isinstance, "issubclass": issubclass, "print": print,
    "True": True, "False": False, "None": None,
    "__import__": _guarded_import,
    "__build_class__": None,   # `class` is rejected statically; belt and braces
}


def check_static(source, kind, name="<generated>"):
    """-> (errors, warnings), both lists of strings.

    Returns rather than raises so one call reports EVERY problem: a generation
    usually has two or three, and fixing them one exception at a time costs one
    API call each.
    """
    errors, warnings = [], []
    required = REQUIRES[kind]

    try:
        tree = ast.parse(source, filename=name)
    except SyntaxError as e:
        return [f"does not parse: line {e.lineno}: {e.msg}"], []

    n_lines = len(source.splitlines())
    if n_lines > MAX_SOURCE_LINES:
        errors.append(f"too long: {n_lines} lines > {MAX_SOURCE_LINES}")

    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass)):
            errors.append(f"line {getattr(node, 'lineno', '?')}: "
                          f"{type(node).__name__} is not allowed at module level")

    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for n in names:
                if n.split(".")[0] not in ALLOWED_IMPORTS:
                    # THE rule that earns its keep. sapien_env.py:951 seeds
                    # torch's global generator and nothing else before calling
                    # _initialize_episode, so a sampler drawing from numpy or
                    # random draws from a stream reset(seed=s) does not control.
                    errors.append(
                        f"line {ln}: forbidden import '{n}' "
                        f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")

        elif isinstance(node, ast.While):
            # Rejection sampling must be a bounded `for`, or a pathological
            # region hangs every environment reset in training.
            errors.append(f"line {ln}: `while` is not allowed - use a bounded `for`")

        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global,
                               ast.Nonlocal, ast.Await, ast.Yield)):
            errors.append(f"line {ln}: {type(node).__name__} is not allowed")

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_NAMES:
                errors.append(f"line {ln}: forbidden name '{node.id}'")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                errors.append(f"line {ln}: dunder attribute '.{node.attr}'")
            elif node.attr in FORBIDDEN_ATTRS:
                errors.append(f"line {ln}: '.{node.attr}' mutates the simulator "
                              f"from inside generated code")
            elif node.attr in DISCOURAGED_ATTRS:
                warnings.append(f"line {ln}: '.{node.attr}' forces a GPU->CPU "
                                f"sync every step")
            # Root-only surface check. Deliberately not a chain walk: the point
            # is to notice `env.scene` or `env._episode_seed`, not to police
            # every path through an alias.
            elif (isinstance(node.value, ast.Name) and node.value.id == "env"
                    and node.attr not in ALLOWED_ENV_ROOTS):
                warnings.append(f"line {ln}: 'env.{node.attr}' is outside the "
                                f"documented API surface")

        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "info":
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value not in ALLOWED_INFO_KEYS:
                        warnings.append(
                            f"line {ln}: info['{key.value}'] is not a key "
                            f"evaluate() returns - it will KeyError at run time")
                else:
                    warnings.append(f"line {ln}: info[...] with a non-literal key")

    # --- names that would NameError at call time ---------------------------
    # THE CHECK THAT WOULD HAVE SAVED A 46,463-TOKEN GENERATION. `dict` was
    # missing from SAFE_BUILTINS, so a sampler ending `return dict(...)` passed
    # every static rule and died on its own return statement, on the pod, after
    # nine minutes of generation. A free name is resolvable only from builtins,
    # so it is checkable here for nothing.
    #
    # Bound names are OVER-approximated - every Store anywhere in the module,
    # regardless of scope - because a false positive here rejects working code,
    # which is the failure this whole file was loosened to avoid.
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            fa = getattr(node, "args", None)
            if fa:
                for x in list(fa.args) + list(fa.posonlyargs) + list(fa.kwonlyargs):
                    bound.add(x.arg)
                for x in (fa.vararg, fa.kwarg):
                    if x:
                        bound.add(x.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id not in bound and node.id not in SAFE_BUILTINS):
            errors.append(f"line {getattr(node, 'lineno', '?')}: name "
                          f"'{node.id}' is neither defined nor available in the "
                          f"sandbox - it would NameError when called")

    # --- required functions, exact parameter names, in order ---------------
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fname, params in required.items():
        fn = defs.get(fname)
        if fn is None:
            errors.append(f"missing required function "
                          f"'{fname}({', '.join(params)})'")
            continue
        got = tuple(a.arg for a in fn.args.args)
        if got != tuple(params):
            # Names AND order, not arity. compute_reward(env, action, obs, info)
            # parses, imports, runs, and computes from the wrong tensor.
            errors.append(f"'{fname}' has parameters {got}, expected "
                          f"{tuple(params)} (names and order both matter)")
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            errors.append(f"'{fname}' must not take *args/**kwargs/keyword-only")

    # --- REWARD_MAX --------------------------------------------------------
    if kind == "reward":
        found = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == REWARD_MAX_NAME:
                        found = node.value
        if found is None:
            errors.append(f"missing module-level "
                          f"'{REWARD_MAX_NAME} = <positive number>'")
        elif not (isinstance(found, ast.Constant)
                  and isinstance(found.value, (int, float))
                  and not isinstance(found.value, bool)
                  and found.value > 0):
            errors.append(f"'{REWARD_MAX_NAME}' must be a positive numeric "
                          f"literal, so it can be read without importing")

    return errors, warnings


def load_source(source, kind, name="<generated>"):
    """check_static, then exec into a restricted namespace. -> namespace dict.

    Raises ValueError listing every error. Static checks gate the import, never
    the other way round: importing first would run module-level code nobody has
    looked at.
    """
    errors, _ = check_static(source, kind, name)
    if errors:
        raise ValueError(f"{name} violates the T-III contract:\n" +
                         "\n".join(f"  - {e}" for e in errors))
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
        sys.exit("usage: python3 t3/loader.py <dir|file> [--kind reward|sampler]")
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
            errors, warnings = check_static(f.read(), k, os.path.basename(path))
        print(f"  {'FAIL' if errors else 'PASS'}  {path}  ({k})")
        for e in errors:
            print(f"          ERROR    {e}")
        for w in warnings:
            print(f"          warning  {w}")
        fails += bool(errors)

    if fails:
        print(f"\n{fails} artifact(s) cannot be loaded. The contract is rendered "
              f"into the prompt by\nt3/spec.py - if a rule is wrong, fix it there "
              f"and regenerate; do not edit a\ngenerated file by hand unless the "
              f"report says you did.")
        sys.exit(1)
    print("\nloadable. Warnings above are for review, not a refusal.")


if __name__ == "__main__":
    main()
