"""Copy a VAI_AIWORKER LeRobot v3.0 dataset keeping only some joints in action / observation.state.

Lift and head are held fixed during laundry folding, so their columns are constant and
only add normalization noise. This drops them (default) and keeps the other 16 joints in
JOINT_NAMES order -- the same subset VAI_AIWORKER's real_policy_inference.ipynb passes as
STATE_JOINT_NAMES (EXCLUDED_JOINTS = lift_joint, head_joint1, head_joint2).

    python tools/select_joints.py ~/aiworker_data/laundry_fold_tshirt_eef_final \
        ~/aiworker_data/laundry_fold_tshirt_eef_final_arm16

Videos, eef_pose and every other feature are copied unchanged. Per-joint statistics
(meta/stats.json and the per-episode stats in meta/episodes) are sliced, not recomputed:
each stat is computed per dimension, so slicing gives exactly the same values.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VECTOR_KEYS = ("action", "observation.state")
DEFAULT_EXCLUDE = ("lift_joint", "head_joint1", "head_joint2")


def slice_column(table, name, idx):
    col = table.column(name).combine_chunks()
    values = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1)[:, idx]
    child = col.type.value_type
    if pa.types.is_fixed_size_list(col.type):
        new = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=child), len(idx))
    else:
        new = pa.array(values.tolist(), type=pa.list_(child))
    return table.set_column(table.schema.get_field_index(name), name, new)


def rewrite_parquet(path, columns, idx, hf_lengths=None):
    table = pq.read_table(path)
    for name in columns:
        table = slice_column(table, name, idx)
    metadata = dict(table.schema.metadata or {})
    if hf_lengths and b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        for name in hf_lengths:
            hf["info"]["features"][name]["length"] = len(idx)
        metadata[b"huggingface"] = json.dumps(hf).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--exclude", nargs="+", default=list(DEFAULT_EXCLUDE), help="joint names to drop")
    args = parser.parse_args()
    src, dst = args.src.expanduser(), args.dst.expanduser()
    if dst.exists():
        raise SystemExit("%s 가 이미 있습니다." % dst)

    info = json.loads((src / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    if info["features"]["observation.state"]["names"] != names:
        raise SystemExit("action / observation.state joint order differs")
    unknown = set(args.exclude) - set(names)
    if unknown:
        raise SystemExit("unknown joints: %s" % sorted(unknown))
    idx = [i for i, n in enumerate(names) if n not in args.exclude]
    kept = [names[i] for i in idx]

    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store", "images", "rejected_episodes.json"))

    for key in VECTOR_KEYS:
        info["features"][key]["shape"] = [len(idx)]
        info["features"][key]["names"] = kept
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    stats = json.loads((dst / "meta/stats.json").read_text())
    for key in VECTOR_KEYS:
        for stat, value in stats[key].items():
            if stat != "count":
                stats[key][stat] = np.asarray(value)[idx].tolist()
    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    for path in sorted(glob.glob(str(dst / "data/*/*.parquet"))):
        rewrite_parquet(path, VECTOR_KEYS, idx, hf_lengths=VECTOR_KEYS)
    for path in sorted(glob.glob(str(dst / "meta/episodes/*/*.parquet"))):
        cols = [c for c in pq.read_schema(path).names
                if any(c.startswith("stats/%s/" % k) for k in VECTOR_KEYS) and not c.endswith("/count")]
        rewrite_parquet(path, cols, idx)

    print("%s -> %s" % (src, dst))
    print("kept %d joints: %s" % (len(kept), kept))


if __name__ == "__main__":
    main()
