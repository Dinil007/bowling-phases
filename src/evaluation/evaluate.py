#!/usr/bin/env python3
"""
Evaluation: compare predictions with reference annotations (Brief section 6).

    python -m src.evaluation.evaluate --pred outputs/predictions --ref annotations/reference \
        --manifest data/input_manifest.csv --out outputs/evaluation

Safety: a prediction and a reference are only compared if both carry the SAME video_sha256
(and, with --manifest, that SHA matches the manifest). Mismatches are reported and excluded.

Metrics: per-event timing error (median / max, frames and ms, % within +-1/2/3 frames),
observability confusion with false-confident cases listed separately, frame-level phase agreement
with a confusion matrix over the 5 phases (+ 'none'), coverage, and robustness/processing time
from <out_root>/run_log.json if present. Every per-clip number is reported.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

EVENTS = ["RUS", "GTO", "BFC", "FFC", "BR", "FTE"]
PHASE_ORDER = ["Run-up", "Gather", "Delivery stride / release preparation", "Ball release", "Follow-through", "none"]
ALIAS = {"run-up": "Run-up", "run_up": "Run-up", "gather": "Gather", "delivery stride": PHASE_ORDER[2],
         "delivery_stride": PHASE_ORDER[2], "ball release": "Ball release", "ball_release": "Ball release",
         "follow-through": "Follow-through", "follow_through": "Follow-through"}


def canon(name):
    return name if name in PHASE_ORDER else ALIAS.get(name.strip().lower(), name)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def observable(rec):
    return rec.get("event_frame") is not None


def phase_labels(recs, n):
    lab = ["none"] * n
    ev = {r["name"]: r.get("event_frame") for r in recs if r["record_type"] == "event"}
    spans = [("Run-up", "RUS", "GTO"), ("Gather", "GTO", "BFC"), (PHASE_ORDER[2], "BFC", "BR"),
             ("Follow-through", "BR", "FTE")]
    for name, a, b in spans:
        if ev.get(a) is not None and ev.get(b) is not None:
            for f in range(max(0, ev[a]), min(n, ev[b])):
                lab[f] = name
    if ev.get("BR") is not None and 0 <= ev["BR"] < n:
        lab[ev["BR"]] = "Ball release"
    return lab


def evaluate(pred_dir, ref_dir, out_dir, manifest=None, run_log=None):
    os.makedirs(out_dir, exist_ok=True)
    man = {}
    if manifest:
        with open(manifest, encoding="utf-8-sig") as f:
            man = {r["clip_id"]: r for r in csv.DictReader(f)}
    preds = {f[:-5]: os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".json") and f.count(".") == 1}
    refs = {f[:-5]: os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".json") and f.count(".") == 1}
    log = load(run_log) if run_log and os.path.isfile(run_log) else {}
    info = {c["clip_id"]: c for c in log.get("clips", [])}

    excluded, clips = [], []
    for cid in sorted(set(preds) | set(refs)):
        if cid not in preds or cid not in refs:
            excluded.append({"clip_id": cid, "reason": "missing prediction" if cid not in preds else "missing reference"})
            continue
        P, R = load(preds[cid]), load(refs[cid])
        ps, rs = P[0]["video_sha256"], R[0]["video_sha256"]
        if ps != rs or (cid in man and man[cid]["sha256"].lower() != ps):
            excluded.append({"clip_id": cid, "reason": f"SHA mismatch pred={ps[:12]} ref={rs[:12]}"
                             + (f" manifest={man[cid]['sha256'][:12]}" if cid in man else "")})
            continue
        clips.append((cid, P, R))

    per_clip, errs = [], defaultdict(list)
    conf = {e: {"TP_obs": 0, "pred_obs_ref_not": 0, "pred_not_ref_obs": 0, "both_not": 0} for e in EVENTS}
    false_conf, phase_cm = [], defaultdict(lambda: defaultdict(int))
    agree_tot = [0, 0]
    for cid, P, R in clips:
        fps = info.get(cid, {}).get("fps") or (float(man[cid]["frame_rate"]) if cid in man and man[cid].get("frame_rate") else 30.0)
        n = info.get(cid, {}).get("frame_count") or 1 + max([r.get("event_frame") or 0 for r in P + R] +
                                                           [r.get("event_window_end") or 0 for r in P + R])
        pe = {r["name"]: r for r in P if r["record_type"] == "event"}
        re_ = {r["name"]: r for r in R if r["record_type"] == "event"}
        row = {"clip_id": cid, "fps": fps, "events": {}}
        for e in EVENTS:
            p, r = pe.get(e), re_.get(e)
            if p is None or r is None:
                row["events"][e] = {"status": "missing record"}
                continue
            po, ro = observable(p), observable(r)
            cell = {"pred": p.get("event_frame"), "ref": r.get("event_frame"),
                    "pred_vis": p["visibility_status"], "ref_vis": r["visibility_status"]}
            if po and ro:
                d = p["event_frame"] - r["event_frame"]
                cell.update({"err_frames": d, "err_ms": round(1000 * d / fps, 1)})
                errs[e].append((abs(d), abs(1000 * d / fps)))
                conf[e]["TP_obs"] += 1
            elif po and not ro:
                conf[e]["pred_obs_ref_not"] += 1
                false_conf.append({"clip_id": cid, "event": e, "pred_frame": p["event_frame"],
                                   "ref_vis": r["visibility_status"]})
            elif ro and not po:
                conf[e]["pred_not_ref_obs"] += 1
                w = (p.get("event_window_start"), p.get("event_window_end"))
                if w[0] is not None:
                    cell["ref_inside_pred_window"] = w[0] <= r["event_frame"] <= w[1]
            else:
                conf[e]["both_not"] += 1
            row["events"][e] = cell
        lp, lr = phase_labels(P, n), phase_labels(R, n)
        idx = [i for i in range(n) if lr[i] != "none"]
        for i in idx:
            phase_cm[lr[i]][lp[i]] += 1
        a = sum(lp[i] == lr[i] for i in idx)
        row["phase_agreement"] = round(a / len(idx), 3) if idx else None
        row["phase_frames_compared"] = len(idx)
        agree_tot[0] += a
        agree_tot[1] += len(idx)
        row["coverage"] = round(sum(observable(x) for x in pe.values()) / len(EVENTS), 3)
        per_clip.append(row)

    timing = {}
    for e in EVENTS:
        if errs[e]:
            f = np.array([x[0] for x in errs[e]])
            ms = np.array([x[1] for x in errs[e]])
            timing[e] = {"n": int(len(f)), "median_frames": float(np.median(f)), "max_frames": int(f.max()),
                         "median_ms": round(float(np.median(ms)), 1), "max_ms": round(float(ms.max()), 1),
                         "pct_within_1": round(100 * float(np.mean(f <= 1)), 1),
                         "pct_within_2": round(100 * float(np.mean(f <= 2)), 1),
                         "pct_within_3": round(100 * float(np.mean(f <= 3)), 1)}
        else:
            timing[e] = {"n": 0}
    all_p = [r for _, P, _ in clips for r in P if r["record_type"] == "event"]
    summary = {
        "label": "DEVELOPMENT RESULTS - the author saw these clips while building; not an independent test.",
        "clips_compared": len(clips), "excluded": excluded,
        "event_timing": timing, "observability_confusion": conf,
        "false_confident": false_conf, "false_confident_count": len(false_conf),
        "phase_agreement_overall": round(agree_tot[0] / agree_tot[1], 3) if agree_tot[1] else None,
        "phase_confusion": {r: dict(c) for r, c in phase_cm.items()},
        "coverage_overall": round(sum(observable(r) for r in all_p) / len(all_p), 3) if all_p else None,
        "robustness": log.get("summary"), "failures": log.get("failures"),
        "processing": [{"clip_id": c["clip_id"], "wall_time_s": c.get("wall_time_s"), "s_per_video_s": c.get("s_per_video_s")}
                       for c in log.get("clips", [])], "hardware": log.get("hardware"),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
    write_md(os.path.join(out_dir, "evaluation_report.md"), summary, per_clip)
    print(f"[OK] compared {len(clips)} clip(s); excluded {len(excluded)}; false-confident {len(false_conf)}")
    for x in excluded:
        print(f"  [excluded] {x['clip_id']}: {x['reason']}")
    return summary


def write_md(path, s, per_clip):
    L = ["# Evaluation report", "", f"**{s['label']}**", "",
         f"- Clips compared: {s['clips_compared']}  ", f"- Excluded: {len(s['excluded'])} "
         + ("; ".join(f"{x['clip_id']} ({x['reason']})" for x in s["excluded"]) if s["excluded"] else ""),
         f"- False-confident events (frame given, reference not observable): **{s['false_confident_count']}**",
         f"- Coverage (events output with a frame): {s['coverage_overall']}",
         f"- Frame-level phase agreement: {s['phase_agreement_overall']}", "",
         "## Event timing error (only where both prediction and reference give a frame)", "",
         "| Event | n | median fr | max fr | median ms | max ms | <=1 | <=2 | <=3 |", "|---|---|---|---|---|---|---|---|---|"]
    for e, m in s["event_timing"].items():
        L.append(f"| {e} | {m['n']} | " + (" | ".join(str(m[k]) for k in ("median_frames", "max_frames", "median_ms",
                 "max_ms", "pct_within_1", "pct_within_2", "pct_within_3")) if m["n"] else "- | - | - | - | - | - | -") + " |")
    L += ["", "## Observability confusion (per event)", "", "| Event | both observable | pred obs / ref not (false-confident) | pred not / ref obs | both not |", "|---|---|---|---|---|"]
    for e, c in s["observability_confusion"].items():
        L.append(f"| {e} | {c['TP_obs']} | {c['pred_obs_ref_not']} | {c['pred_not_ref_obs']} | {c['both_not']} |")
    if s["false_confident"]:
        L += ["", "False-confident cases: " + "; ".join(f"{x['clip_id']} {x['event']} @ {x['pred_frame']}" for x in s["false_confident"])]
    L += ["", "## Per-clip results (pred / ref, error in frames)", "", "| Clip | " + " | ".join(EVENTS) + " | phase agr. | coverage |",
          "|---" * (len(EVENTS) + 3) + "|"]
    for r in per_clip:
        cells = []
        for e in EVENTS:
            c = r["events"].get(e, {})
            if "err_frames" in c:
                cells.append(f"{c['pred']}/{c['ref']} ({c['err_frames']:+d})")
            elif c:
                cells.append(f"{c.get('pred')}/{c.get('ref')}" + (" [ref in win]" if c.get("ref_inside_pred_window") else ""))
            else:
                cells.append("-")
        L.append(f"| {r['clip_id']} | " + " | ".join(cells) + f" | {r['phase_agreement']} | {r['coverage']} |")
    L += ["", "## Phase confusion (rows = reference, cols = prediction, frames)", "",
          "| ref \\ pred | " + " | ".join(PHASE_ORDER) + " |", "|---" * (len(PHASE_ORDER) + 1) + "|"]
    for rname in PHASE_ORDER[:-1]:
        row = s["phase_confusion"].get(rname, {})
        L.append(f"| {rname} | " + " | ".join(str(row.get(c, 0)) for c in PHASE_ORDER) + " |")
    if s.get("robustness"):
        L += ["", "## Robustness and processing time", "", f"- {s['robustness']}", f"- Hardware: {s.get('hardware')}", ""]
        L += ["| Clip | wall s | s per video s |", "|---|---|---|"] + [f"| {p['clip_id']} | {p['wall_time_s']} | {p['s_per_video_s']} |" for p in s["processing"]]
        for x in s.get("failures") or []:
            L.append(f"- FAILURE {x['clip_id']}: {x['tag']} - {x['message']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate ScoutEdge predictions against references")
    ap.add_argument("--pred", default="outputs/predictions")
    ap.add_argument("--ref", default="annotations/reference")
    ap.add_argument("--out", default="outputs/evaluation")
    ap.add_argument("--manifest", default=None, help="Optional: also require SHA to match the manifest")
    ap.add_argument("--run-log", default=None, help="Default: <pred>/../run_log.json")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.pred) or not os.path.isdir(a.ref):
        print(f"[FATAL] missing directory: {a.pred if not os.path.isdir(a.pred) else a.ref}", file=sys.stderr)
        return 1
    run_log = a.run_log or os.path.join(os.path.dirname(os.path.abspath(a.pred)), "run_log.json")
    s = evaluate(a.pred, a.ref, a.out, a.manifest, run_log)
    return 0 if s["clips_compared"] else 2


if __name__ == "__main__":
    sys.exit(main())
