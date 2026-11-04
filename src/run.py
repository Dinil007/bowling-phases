#!/usr/bin/env python3
"""
ScoutEdge Bowling Delivery Evidence Pipeline - single entry point.

    python -m src.run --manifest data/input_manifest.csv --out outputs/

Exit codes: 0 all clips processed; 1 manifest unusable / config / weights missing;
3 one or more clips failed (each failure is printed, logged in <out>/run_log.json and, for
no-person / ambiguous-bowler cases, written as a schema-valid all-not_observable prediction).
Per-clip failure codes: 10 missing file, 11 unreadable, 12 checksum mismatch, 13 no person,
14 ambiguous bowler, 20 unexpected error.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import yaml

from src.ingest.manifest import (PipelineError, ManifestError, NoPersonDetectedError, AmbiguousBowlerError,
                                 read_manifest, verify_entry)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pipeline_version(cfg) -> str:
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        if sha:
            dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no", "src", "configs"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    return cfg.get("pipeline_version_fallback", "nogit")


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_models(cfg):
    from src.processing.detector import YOLOXDetector
    from src.phase_detection.rtmpose import RTMPose
    for k in ("detector", "pose"):
        if not os.path.isfile(repo_path(cfg["models"][k])):
            raise FileNotFoundError(f"Model weights missing: {cfg['models'][k]}. Run: python scripts/fetch_weights.py")
    t = int(cfg["models"].get("threads", 2))
    return YOLOXDetector(repo_path(cfg["models"]["detector"]), threads=t), RTMPose(repo_path(cfg["models"]["pose"]), threads=t)


def process_clip(entry, cfg, out_dir, det, pose, version):
    from src.ingest.decoder import probe
    from src.processing.scan import scan_video
    from src.processing.isolate import find_anchor
    from src.phase_detection.pose import dense_refine
    from src.phase_detection.events import EventDetector
    from src.phase_detection.event_engine import build_records
    from src.evidence.schema import atomic_write_predictions

    t0 = time.time()
    sha = verify_entry(entry, cfg["ingest"]["strict_checksum"])
    meta = probe(entry.path)
    info = {"clip_id": entry.clip_id, "path": entry.path, "sha256": sha, "width": meta.width, "height": meta.height,
            "fps": meta.fps, "frame_count": meta.frame_count, "duration_s": meta.duration_s, "vfr": meta.is_vfr,
            "probe": meta.probe_source, "camera_view": entry.camera_view, "meta_notes": meta.notes}
    if entry.frame_rate and abs(entry.frame_rate - meta.fps) > 0.01:
        info["meta_notes"].append(f"Manifest frame_rate {entry.frame_rate} differs from container {meta.fps:.3f}; container used.")
    scan = scan_video(meta, det, pose, cfg["scan"])
    anchor = find_anchor(scan, meta.fps, cfg["isolation"])
    W = dense_refine(meta, scan, anchor, det, pose, cfg["dense"])
    ev_cfg = dict(cfg["events"])
    sfx = f" + yolox-s@megvii-0.1.1 + rtmpose-m-halpe26@mmpose-body7"
    estimates = EventDetector(ev_cfg, sfx).detect(W, scan, anchor, meta)
    extra = " ".join(meta.notes)
    recs = build_records(estimates, meta, entry, sha, version, extra)
    pred = os.path.join(out_dir, "predictions", f"{entry.clip_id}.json")
    atomic_write_predictions(recs, pred)
    info.update({"anchor_frame": anchor.frame, "anchor_method": anchor.method, "anchor_notes": anchor.notes,
                 "dense_window": [W.start, W.end], "prediction": pred})
    if cfg["visualizations"]["enabled"]:
        from src.evidence.visualizer import render_video
        from src.evidence.contact_sheet import make_sheets
        vdir = os.path.join(out_dir, "visualizations")
        vid, span = render_video(meta, recs, W, os.path.join(vdir, f"{entry.clip_id}_annotated.mp4"),
                                 cfg["visualizations"]["video_max_width"])
        make_sheets(meta, recs, W, vdir, cfg["visualizations"]["contact_sheet_radius"])
        info["annotated_video_span"] = list(span)
    info["wall_time_s"] = round(time.time() - t0, 2)
    info["s_per_video_s"] = round(info["wall_time_s"] / max(meta.duration_s, 1e-6), 3)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="ScoutEdge Bowling Delivery Evidence Pipeline")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default_pipeline.yaml"))
    ap.add_argument("--clips", default=None, help="Optional comma-separated subset of clip_ids (debugging)")
    args = ap.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        entries = read_manifest(args.manifest)
        if args.clips:
            keep = set(args.clips.split(","))
            entries = [e for e in entries if e.clip_id in keep]
        det, pose = load_models(cfg)
    except (ManifestError, FileNotFoundError, OSError, yaml.YAMLError) as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1
    for d in ("predictions", "visualizations", "evaluation"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    version = pipeline_version(cfg)
    log = {"pipeline_version": version, "manifest": os.path.abspath(args.manifest), "config": args.config,
           "hardware": {"machine": platform.machine(), "processor": platform.processor(), "cpus": os.cpu_count(),
                        "system": platform.platform(), "python": platform.python_version()},
           "clips": [], "failures": []}
    print(f"[*] ScoutEdge pipeline {version}: {len(entries)} clip(s) from {args.manifest}")
    for e in entries:
        print(f"[+] {e.clip_id} ...", flush=True)
        try:
            info = process_clip(e, cfg, args.out, det, pose, version)
            log["clips"].append(info)
            print(f"    OK  {info['wall_time_s']}s  anchor={info['anchor_frame']}  window={info['dense_window']}")
        except PipelineError as err:
            print(f"[!] {e.clip_id} FAILED ({err.tag}, code {err.code}): {err}", file=sys.stderr)
            log["failures"].append({"clip_id": e.clip_id, "code": err.code, "tag": err.tag, "message": str(err)})
            if isinstance(err, (NoPersonDetectedError, AmbiguousBowlerError)):
                from src.phase_detection.event_engine import failure_records
                from src.evidence.schema import atomic_write_predictions
                atomic_write_predictions(failure_records(e, e.sha256, version, err),
                                         os.path.join(args.out, "predictions", f"{e.clip_id}.json"))
        except Exception as err:  # unexpected: still explicit
            print(f"[!] {e.clip_id} FAILED (UNEXPECTED, code 20): {type(err).__name__}: {err}", file=sys.stderr)
            log["failures"].append({"clip_id": e.clip_id, "code": 20, "tag": "UNEXPECTED", "message": repr(err)})
    log_path = os.path.join(args.out, "run_log.json")
    if args.clips and os.path.isfile(log_path):  # subset run: merge into the existing log
        try:
            old = json.load(open(log_path, encoding="utf-8"))
            done = {c["clip_id"] for c in log["clips"]} | {c["clip_id"] for c in log["failures"]}
            log["clips"] = [c for c in old.get("clips", []) if c["clip_id"] not in done] + log["clips"]
            log["failures"] = [c for c in old.get("failures", []) if c["clip_id"] not in done] + log["failures"]
        except Exception:
            pass
    log["clips"].sort(key=lambda c: c["clip_id"])
    log["summary"] = {"total": len(log["clips"]) + len(log["failures"]), "ok": len(log["clips"]),
                      "failed": len(log["failures"])}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"[*] Done this run: {sum(1 for c in log['clips'] if c['clip_id'] in {e.clip_id for e in entries})}/{len(entries)} OK. Log: {log_path}")
    return 3 if log["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
