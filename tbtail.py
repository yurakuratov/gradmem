#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.compat import tf


def list_event_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    files = []
    for name in os.listdir(path):
        if name.startswith("events.out.tfevents"):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                files.append(full)

    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {path}")

    # oldest -> newest, so "latest" is stable across multi-file runs
    files.sort(key=os.path.getmtime)
    return files


def _scalar_from_tensor_proto_fast(t) -> Optional[float]:
    """
    Fast-path decode for scalar TensorProto without tf.make_ndarray().
    Falls back to tf.make_ndarray only for uncommon cases.
    """
    # Accept scalar shape or 1-element shape.
    dims = [d.size for d in t.tensor_shape.dim]
    numel = 1
    for d in dims:
        numel *= d
    if dims not in ([], [1]) and numel != 1:
        return None

    dt = t.dtype

    # Common scalar encodings: float_val / double_val / int_val / int64_val / bool_val
    if dt == types_pb2.DT_FLOAT and t.float_val:
        v = float(t.float_val[0])
        return v if math.isfinite(v) else None

    if dt == types_pb2.DT_DOUBLE and t.double_val:
        v = float(t.double_val[0])
        return v if math.isfinite(v) else None

    if dt in (types_pb2.DT_INT8, types_pb2.DT_INT16, types_pb2.DT_INT32,
              types_pb2.DT_UINT8, types_pb2.DT_UINT16) and t.int_val:
        v = float(t.int_val[0])
        return v if math.isfinite(v) else None

    if dt in (types_pb2.DT_INT64, types_pb2.DT_UINT32, types_pb2.DT_UINT64) and t.int64_val:
        v = float(t.int64_val[0])
        return v if math.isfinite(v) else None

    if dt == types_pb2.DT_BOOL and t.bool_val:
        return 1.0 if t.bool_val[0] else 0.0

    # Sometimes scalar tensor data is packed into tensor_content.
    # Fall back for those rarer layouts.
    try:
        arr = tf.make_ndarray(t)
        if getattr(arr, "shape", ()) == ():
            v = arr.item()
        elif getattr(arr, "size", None) == 1:
            v = arr.reshape(()).item()
        else:
            return None
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def extract_scalar(summary_value) -> Optional[float]:
    kind = summary_value.WhichOneof("value")

    if kind == "simple_value":
        v = float(summary_value.simple_value)
        return v if math.isfinite(v) else None

    if kind == "tensor":
        return _scalar_from_tensor_proto_fast(summary_value.tensor)

    return None


def update_stats(stats: Dict[str, dict], tag: str, step: int, wall_time: float, value: float) -> None:
    cur = stats.get(tag)
    if cur is None:
        stats[tag] = {
            "tag": tag,
            "latest_value": value,
            "latest_step": step,
            "latest_wall_time": wall_time,
            "min_value": value,
            "min_step": step,
            "max_value": value,
            "max_step": step,
            "count": 1,
        }
        return

    cur["count"] += 1

    if step > cur["latest_step"] or (step == cur["latest_step"] and wall_time >= cur["latest_wall_time"]):
        cur["latest_value"] = value
        cur["latest_step"] = step
        cur["latest_wall_time"] = wall_time

    if value < cur["min_value"]:
        cur["min_value"] = value
        cur["min_step"] = step

    if value > cur["max_value"]:
        cur["max_value"] = value
        cur["max_step"] = step


def read_fast(path: str, wanted_tags: Optional[set] = None) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    files = list_event_files(path)

    for file_path in files:
        loader = RawEventFileLoader(file_path)
        for record in loader.Load():
            ev = event_pb2.Event.FromString(record)
            if not ev.HasField("summary"):
                continue

            step = int(ev.step)
            wall_time = float(ev.wall_time)

            for sv in ev.summary.value:
                tag = sv.tag
                if wanted_tags is not None and tag not in wanted_tags:
                    continue

                value = extract_scalar(sv)
                if value is None:
                    continue

                update_stats(stats, tag, step, wall_time, value)

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fast TensorBoard scalar reader (latest/min/max/count, O(tags) memory)"
    )
    ap.add_argument("path", help="Event file or directory")
    ap.add_argument("--tag", nargs="+", action="append", default=[], metavar="TAG",
                    help="Only include these tags. Example: --tag loss acc lr"
                    )
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    ap.add_argument("--sort", choices=["tag", "latest", "min", "max", "count"], default="tag", help="Sort output")
    args = ap.parse_args()

    wanted_tags = {tag for group in args.tag for tag in group} if args.tag else None

    print(f'path: {os.path.abspath(args.path)}')

    try:
        rows = list(read_fast(args.path, wanted_tags).values())
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.sort == "tag":
        rows.sort(key=lambda r: r["tag"])
    elif args.sort == "latest":
        rows.sort(key=lambda r: r["latest_value"])
    elif args.sort == "min":
        rows.sort(key=lambda r: r["min_value"])
    elif args.sort == "max":
        rows.sort(key=lambda r: r["max_value"])
    elif args.sort == "count":
        rows.sort(key=lambda r: r["count"], reverse=True)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No scalar summaries found.")
        return 0

    tag_w = max(len(r["tag"]) for r in rows)
    for r in rows:
        print(
            f'{r["tag"]:<{tag_w}}  '
            f'last={r["latest_value"]:<12g} step={r["latest_step"]:<8}  '
            f'min={r["min_value"]:<12g} step={r["min_step"]:<8}  '
            f'max={r["max_value"]:<12g} step={r["max_step"]:<8}  '
            f'n={r["count"]}'
        )
    return 0


"""
Extracts metrics from tensorboard event files.
USAGE example, from experiment folder:
python ~/data/test_time_gd/tbtail.py . --tag eval/exact_match eval/token_accuracy train/patience
eval/exact_match     last=0.4344       step=31000     min=0.3024       step=4000      max=0.478        step=0      n=63
eval/token_accuracy  last=0.696        step=31000     min=0.5166       step=4000      max=0.696        step=31000  n=63
train/patience       last=2            step=31000     min=0            step=0         max=12           step=700    n=125
"""
if __name__ == "__main__":
    raise SystemExit(main())
