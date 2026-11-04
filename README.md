# ScoutEdge_CV — Bowling Delivery Evidence Pipeline

> **Notice:** This is a technical assessment forming part of our interview process for evaluating your suitability for a potential role at ScoutEdge. It is unpaid and does not constitute employment, a paid assignment, or a commitment to future engagement.

---

Turns a cricket bowling clip into timestamped, inspectable evidence of one delivery: 6 events, 5 phases, visibility and confidence for each. CPU only; Apache-2.0 models via onnxruntime; no PyTorch.

## 1. Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows   |   source venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt    # exact tested versions: requirements-lock.txt
python scripts/fetch_weights.py    # downloads YOLOX-s + RTMPose-m Halpe-26, verifies SHA-256
```
Optional: install FFmpeg so `ffprobe` is on PATH (used only to flag variable frame rate).

## 2. Run
```bash
python -m src.run --manifest data/input_manifest.csv --out outputs/
```
- Video paths in the manifest may be absolute, relative to the manifest file, or relative to the working directory.
- Writes `outputs/predictions/<clip_id>.json` (schema 1.0), `outputs/visualizations/<clip_id>_annotated.mp4`, `outputs/visualizations/<clip_id>_<EVENT>_contact_sheet.png` (frames −3…+3), and `outputs/run_log.json` (true metadata, VFR flags, anchors, timings, failures).
- Exit code: 0 = all clips OK; 3 = at least one clip failed (each failure is printed with its reason); 1 = manifest, config or weights unusable.
- Per-clip failure codes: 10 missing file, 11 unreadable, 12 checksum mismatch, 13 no person, 14 ambiguous bowler. Codes 13/14 still write a schema-valid all-`not_observable` prediction.

## 3. Evaluate
```bash
python -m src.evaluation.evaluate --pred outputs/predictions --ref annotations/reference --manifest data/input_manifest.csv --out outputs/evaluation
```
Only files whose `video_sha256` match each other and the manifest are compared. Writes `metrics_summary.json` and `evaluation_report.md`: timing error in frames and ms, observability confusion with false-confident cases, phase confusion, coverage, robustness and time.

## 4. Test
```bash
pytest -q tests
```

Design, licences and risks: `reports/technical_design.md`. Results: `reports/validation_report.md`. Limits: `reports/limitations.md`.
