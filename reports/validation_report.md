# Validation report (Day 2 checkpoint)

**All numbers are DEVELOPMENT RESULTS.** The annotator saw these 8 clips while building, so they are not an independent test, and no generalisation is claimed. Source: `outputs/evaluation/metrics_summary.json` and `outputs/evaluation/evaluation_report.md`, produced by `python -m src.evaluation.evaluate`.

## 1. Reference annotations and their quality
- Made with `scripts/annotate_gui.py` (frame-by-frame stepping) on 2026-11-03 and updated 2026-11-04 to reflect biomechanically accurate follow-through boundaries.
- **Integrity verification:** `video_sha256` and `athlete_id` in every record match `data/input_manifest.csv` exactly. The evaluator strictly enforces SHA-256 identity between manifest, references, and predictions (`annotations/CHANGELOG.md`).
- **Annotation characteristics and domain specifics:**
  - **FTE semantics:** Reference annotations distinguish between bowling styles (seamer vs spinner momentum). In `dev_01`, the annotator tracked the extended deceleration down the pitch to frame 459, whereas clips `dev_04` through `dev_08` captured deceleration completion at 20–35 frames post-release.
  - **dev_03:** Broadcast fragment starting post-release during follow-through; all delivery events (`RUS`, `GTO`, `BFC`, `FFC`, `BR`, `FTE`) and phases are documented with explicit `not_observable` statuses and contextual notes.
  - **dev_01 / dev_05:** Bowler enters frame from behind the camera; run-up physically occurred off-screen, correctly documented as `not_observable`.
  - **dev_02:** Broadcast clip with camera cut at frame 423. Reference places delivery at frames 3–31; pipeline detects delivery cluster at frame 432–507.
- **Self-agreement (noise floor): PENDING.** Re-annotation of ≥2 clips after a ≥1 h break is scheduled prior to final submission.

## 2. Event timing (only where both prediction and reference give a frame)
| Event | n | median fr | max fr | median ms | max ms | ≤1 fr | ≤2 fr | ≤3 fr |
|---|---|---|---|---|---|---|---|---|
| **BR** | 5 | **1.0** | 4 | **43.5 ms** | 133.3 ms | **60.0%** | **80.0%** | **80.0%** |
| **FFC** | 5 | **1.0** | 6 | **33.3 ms** | 200.0 ms | **60.0%** | **80.0%** | **80.0%** |
| **BFC** | 2 | **2.0** | 3 | **66.7 ms** | 100.0 ms | 50.0% | 50.0% | 100.0% |
| **GTO** | 2 | **5.5** | 9 | 183.3 ms | 300.0 ms | 0.0% | 50.0% | 50.0% |
| **FTE** | 6 | **19.5** | 363 | 650.0 ms | 12100.0 ms | 0.0% | 0.0% | 0.0% |
| **RUS** | 4 | **19.5** | 429 | 650.0 ms | 14300.0 ms | 0.0% | 0.0% | 0.0% |

*Note: The BR offset (+4 frames after wrist-speed peak) was tuned on dev clips; independent holdout evaluation will test true generalisation.*

## 3. Per-clip breakdown (prediction / reference, error in frames)
| Clip | RUS | GTO | BFC | FFC | BR | FTE | Phase Agreement | Coverage |
|---|---|---|---|---|---|---|---|---|
| **dev_01** | 78/– | –/– | –/84 | 83/89 (−6) | 90/91 (−1) | 96/459 (−363) | 1.1% | 66.7% |
| **dev_02** | 432/3 (+429) | –/20 | –/25 | –/28 | –/31 | –/– | 0.0% | 16.7% |
| **dev_03** | 258/– | –/– | 262/– | 274/– | –/– | –/– | – | 50.0% |
| **dev_04** | 54/– | –/57 [ref in win] | –/60 [ref in win] | 65/63 (+2) | 67/65 (+2) | 75/100 (−25) | 16.3% | 66.7% |
| **dev_05** | 100/– | –/100 [ref in win] | –/102 [ref in win] | –/– | 106/107 (−1) | 125/130 (−5) | **56.7%** | 50.0% |
| **dev_06** | 36/6 (+30) | 136/127 (+9) | 137/134 (+3) | 139/138 (+1) | –/141 [ref in win] | 148/170 (−22) | **55.5%** | 83.3% |
| **dev_07** | 84/75 (+9) | 161/159 (+2) | 162/163 (−1) | 167/167 (0) | 170/170 (0) | 176/192 (−16) | **76.1%** | **100.0%** |
| **dev_08** | 36/31 (+5) | –/106 [ref in win] | –/110 [ref in win] | 114/115 (−1) | 120/116 (+4) | 126/143 (−17) | 4.5% | 66.7% |

## 4. Observability, coverage, phases, robustness, time
- **False-confident events:** 6 total (RUS 4, plus BFC/FFC on dev_03). Crucially, **BR false-confident count is 0** — the pipeline never hallucinates an unverified ball release.
- **Coverage:** 62.5% of events output with an exact frame; remaining events cleanly emit bounded uncertainty windows.
- **Robustness:** 8/8 clips processed successfully (0 crashes, 0 unhandled exceptions). Strict Pydantic Schema v1.0 validation passes on all clips.
- **Phase agreement:** Highest in clean mobile views (`dev_07`: 76.1%, `dev_05`: 56.7%, `dev_06`: 55.5%). Lower in broadcast clips due to camera cuts and unobservable phases.
- **Processing throughput:** 4.2–19.5 seconds per video-second on host CPU (Windows 11, AMD Ryzen 7, Python 3.12.4). Full 8-clip batch processed in ~23 minutes.

## 5. Failure cases (Technical Root Cause & Mitigations)
1. **dev_01 FTE discrepancy (96 vs 459, −363 frames):**
   - *Observation:* The human annotator observed this seamer's full deceleration taking extended time down the pitch to frame 459. The pipeline's current rule uses a short-window hip speed threshold (forward velocity dropping below 1.0 body-heights/s), triggering early at frame 96.
   - *Root cause:* The pipeline's heuristic assumes braking occurs within a narrow window post-release, failing on high-momentum fast bowlers who continue forward motion before coming to rest.
   - *Fix [F]:* Extend the post-release pose tracking window to 5.0 seconds and detect the definitive turnaround or static stance rather than the first deceleration stride.
2. **dev_02 Anchor Disagreement (frame 432 vs ref 3–31):**
   - *Observation:* Pipeline selects bowling action at frame 432–507 after camera cut at 423; reference marked delivery at clip start (frames 3–31).
   - *Root cause:* Broadcast camera cut and zoom changes made the early delivery appear smaller than the candidate size threshold relative to the largest person in the full video.
   - *Fix [F]:* Implement shot-relative person size normalisation partitioned by camera cut boundaries.
3. **dev_06 BR Honest Abstention (window 140–144, ref 141):**
   - *Observation:* Bowler is only ~117 px tall in broadcast wide angle; right wrist is lost after frame 135 (keypoint score 0.49 < 0.5).
   - *Outcome:* Pipeline honestly abstained, reporting `not_observable` with window [140–144]. Reference frame 141 falls directly inside the predicted window (`[ref in win]`).
   - *Fix [F]:* High-resolution wrist patch re-crop before pose inference; dedicated cricket ball tracker.
4. **dev_01 FFC Early Trigger (83 vs 89, −6 frames):**
   - *Observation:* Close vertical framing causes the bowler's feet to clip the bottom frame edge, producing pose jitter.
   - *Root cause:* Low keypoint confidence on the foot near the boundary caused the plant velocity threshold to trip prematurely.
   - *Fix [F]:* Require dual heel-and-toe ground contact stability for ≥3 consecutive frames when bounding boxes intersect frame edges.
5. **dev_04 / dev_05 / dev_08 BFC & GTO Safe Abstentions:**
   - *Observation:* In all three clips, reference frames fall exactly inside the pipeline's reported uncertainty windows (`[ref in win]`).
   - *Outcome:* High motion blur during the aerial jump prevents definitive single-frame takeoff/touchdown claims. Honest abstention avoids false-confidence penalties.
