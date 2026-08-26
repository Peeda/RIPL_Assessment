#!/usr/bin/env python
"""Flattening and unflattening a ManiSkill sim state, for the backend check.

Recovered from t2/t2_common.py, deleted in c1c010b when the discovery-era
harness was collapsed. Recovered rather than rewritten because the ordering
trap documented in state_dict_from_flat is the whole reason this pair exists,
and it is invisible until an episode quietly starts somewhere else.

Both scripts that need this - t4/capture_states.py writes, t4/backend_check.py
reads - share ONE definition here, because a flattener and an unflattener that
disagree fail silently by construction.
"""
import numpy as np
import torch


def flatten_state_dict(d, prefix=""):
    """(names, values) from a ManiSkill state dict, walked in sorted-key order.

    Sorted rather than insertion order so the layout is stable across ManiSkill
    versions and across actor build order. Names travel with the values so the
    saved array is self-describing - the point of logging this at all is that
    the analysis reproduces without a working ManiSkill install.
    """
    names, vals = [], []
    if isinstance(d, dict):
        for k in sorted(d):
            n, v = flatten_state_dict(d[k], f"{prefix}/{k}" if prefix else k)
            names += n
            vals += v
    else:
        flat = np.asarray(torch.as_tensor(d).reshape(-1).float().cpu())
        names += [f"{prefix}[{i}]" for i in range(flat.shape[0])]
        vals += list(flat)
    return names, vals


def state_dict_from_flat(names, values):
    """Invert flatten_state_dict: (names, (N, D) array) -> ManiSkill state dict.

    Needed because the flat arrays in *_states.npz CANNOT be fed to
    env.set_state(). flatten_state_dict above walks in SORTED key order on
    purpose (stability across ManiSkill versions), while BaseEnv.set_state
    reconstructs by iterating self._init_raw_state["actors"] in INSERTION order
    (sapien_env.py:1316-1328). The two orders differ, and the mismatch is
    silent: set_state would happily read cubeB's pose out of cubeA's slot and
    the episode would just start somewhere unexpected.

    So round-trip through the names, which travel with the values precisely so
    the layout is self-describing, and hand back a dict.
    env.reset(options={"reset_to_env_states": {"env_states": <dict>}}) accepts
    one and routes to set_state_dict, which keys by name and cannot be
    misordered.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float32))
    groups = {}                                  # (group, entity) -> [col idx]
    for col, nm in enumerate(np.asarray(names).reshape(-1)):
        head = str(nm).split("[")[0]             # "actors/cubeA[3]" -> "actors/cubeA"
        parts = head.split("/")
        if len(parts) != 2:
            raise ValueError(f"unexpected state name {nm!r}; expected 'group/entity[i]'")
        groups.setdefault((parts[0], parts[1]), []).append(col)

    out = {}
    for (group, entity), cols in groups.items():
        # Sorted-key order groups a given entity's columns contiguously and in
        # ascending index order, but do not rely on that - sort by the bracket.
        cols = sorted(cols, key=lambda c: int(str(np.asarray(names).reshape(-1)[c]).split("[")[1].rstrip("]")))
        # set_state_dict runs common.to_tensor over whatever it is handed.
        out.setdefault(group, {})[entity] = torch.as_tensor(values[:, cols])
    return out
