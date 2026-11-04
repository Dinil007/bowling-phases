# Technical design (Day 1 checkpoint)

Candidate: Dinil Raj · Pipeline v3 · Labels: **[D]** demonstrated by outputs in this repo, **[A]** assumption, **[F]** proposed future work.

## 1. Clip suitability (development set, 8 clips received)
| clip | view (manifest) | res | fps (container) | frames | notes |
|---|---|---|---|---|---|
| dev_01 | mobile_vertical | 720x1280 | 30 | 847 | close framing; feet partly at frame edge [D] |
| dev_02 | broadcast_wide | 1280x576 | 30 | 1397 | 46 s, camera cuts (423, 539); several arm-action clusters (delivery + replays/angles) [D] |
| dev_03 | broadcast_wide | 1280x576 | 30 | 1856 | 62 s; many people; reference marks almost everything not observable (to be re-checked) [D] |
| dev_04 | mobile_vertical | 478x850 | 30 | 163 | **VFR flagged**: 4 of 162 intervals >25% off nominal (up to 100 ms gaps) [D] |
| dev_05 | mobile_vertical | 720x1280 | **23** | 229 | sub-50 fps; 43.5 ms per frame [D] |
| dev_06 | broadcast_wide | 1280x592 | 30 | 360 | bowler ~117 px tall: small feet/hands [D] |
| dev_07 | mobile_vertical | 720x1296 | 30 | 502 | batter's stroke follows the release (handled, see §2.2) [D] |
| dev_08 | broadcast_wide | 1280x592 | 30 | 486 | VFR flagged: 1 interval of 45 ms [D] |

All clips are below 50 fps, so one frame = 33.3 ms (43.5 ms for dev_05). Timing errors are reported in frames and ms. The Brief says 6 development clips; 8 were received and all 8 are used. Manifest `camera_view` uses `mobile_vertical`/`broadcast_wide`, not the Brief's side-on/front-on/behind vocabulary; the pipeline does **not** depend on this column [D].

## 2. Approach (two-stage: cheap whole-clip scan, then native-frame refinement)
1. **Ingest** (`src/ingest`): SHA-256 per clip; true fps from the container (ffprobe `r_frame_rate`, OpenCV fallback); frame count by decoding every frame; VFR flagged from packet PTS. Timestamps = frame_index / true_fps (Brief). Frames are read strictly sequentially (no seeking), so the same input always yields the same frames. Originals are opened read-only [D].
2. **Scan** (`src/processing/scan.py`): every frame at 320 px → camera motion (LK features on background, RANSAC similarity) and shot cuts (HSV histogram). Every 6th frame (5 fps) → YOLOX-s persons → motion-compensated tracker (IoU + centre gate, ByteTrack-style two-stage). Up to 8 candidates, ranked by ground-relative motion, get RTMPose at 5 fps, then at 10 fps around raised-arm samples [D].
3. **Bowler + delivery anchor** (`src/processing/isolate.py`): a delivery is the moment a moving person swings a hand fast around the shoulder. Per pose sample: arm swing = change of (wrist − shoulder) in torso-lengths/s, plus wrist height above shoulder. Clusters are scored by swing + raise + prior run-up motion + size. The **earliest** cluster scoring ≥ 75% of the best is the first delivery (later replays are ignored and noted). `AMBIGUOUS_BOWLER` is raised only if two non-overlapping people score within 10% **and** peak within 0.2 s of each other. A batter's stroke follows the ball, so it is not simultaneous [D, dev_07].
4. **Dense refinement** (`src/phase_detection/pose.py`): every native frame from anchor −1.5 s to +2.0 s (clipped to the shot). Per frame: YOLOX persons and sports-ball detections, bowler tracked forward and backward with a bounded degraded state (≤10 frames, then "lost"), RTMPose Halpe-26 (heels and toes) and camera motion [D].
5. **Events** (`src/phase_detection/events.py`). All speeds are ground-relative (camera motion removed) and in body-heights/s, so they don't depend on resolution or zoom.
   - **BR:** the bowling arm is the wrist with the higher speed peak. Release = peak + 4 frames (offset calibrated on dev clips) with a ±2 frame window. It is `visible` only if a ball detection separates from the wrist in 2 consecutive frames. Otherwise it is `partially_visible` with the kinematic frame, and only if the arm keypoint score is ≥ 0.5, the track is not degraded and the peak is ≥ 2× the window median. Else it is `not_observable` with the window.
   - **FFC/BFC:** the front foot (opposite the bowling arm) or back foot changes from swing to planted (speed < 1.0 body-heights/s for ≥ 2 frames), searched in [BR−0.5 s, BR+1] and [FFC−0.5 s, FFC−1].
   - **GTO:** first frame with both feet off the ground before BFC.
   - **RUS:** onset of sustained bowler motion on the sparse track, with fragmented tracks linked; frame 0 / shot start if already moving.
   - **FTE:** hip speed below walking pace, or the last frame of the clip. `out_of_frame` if the bowler is lost or the camera cuts away first.

   Any missing evidence gives frame = null, a reason and the search window [D].
6. **Evidence:** one JSON per clip (6 events + 5 phases). Annotated MP4 covering all events and windows (frame, timestamp, phase, event markers, every event's visibility, bowler box and skeleton). One −3…+3 contact sheet per event: centred on the prediction, or on the window centre labelled NOT OBSERVABLE [D].

### Confidence (not a calibrated probability)
`confidence` = mean RTMPose SimCC peak score (0–1) of the keypoints the decision used, over the decision frame ±1:
- BR: bowling shoulder, elbow and wrist
- FFC/BFC: that foot's ankle, heel and toes
- GTO: both feet
- FTE: hips
- RUS: mean YOLOX box score at the onset

It is `null` whenever `event_frame` is null. Phase confidence = the minimum of its two boundary events. It is not calibrated, and is used only to rank evidence quality [D].

### Schema notes
Schema 1.0 unchanged. Phase `name`s are exactly the Brief's five. Optional `annotator` exists for references only. `partially_visible` + frame for BR means a "kinematic estimate, ball not resolved". The evaluation treats "observable" as "event_frame is not null" [D].

## 3. Built vs reused, licences
| component | source | licence | role |
|---|---|---|---|
| YOLOX-s ONNX weights + grid decode scheme | Megvii-BaseDetection/YOLOX release 0.1.1rc0 | Apache-2.0 | person + sports-ball detection |
| RTMPose-m Halpe-26 256x192 ONNX (body7) | OpenMMLab MMPose (mmdeploy export; fetched from the ossia/score-packages GitHub mirror, official zip as fallback, SHA-pinned) | Apache-2.0 | 26-keypoint pose incl. heels/toes |
| onnxruntime | Microsoft | MIT | inference |
| OpenCV (opencv-python) | OpenCV | Apache-2.0 | decode, LK flow, RANSAC, drawing |
| NumPy | NumPy | BSD-3 | maths |
| Pydantic | pydantic | MIT | schema validation |
| PyYAML / pytest | — | MIT | config / tests |
| ffprobe (optional, system) | FFmpeg | LGPL-2.1+ (GPL if built with GPL parts) | VFR detection only; called as a separate program, not linked |

**Built by me** (with AI-assistant help, disclosed): everything in `src/`, `scripts/`, `tests/` (pre-/post-processing, tracker, motion compensation, anchor, all event rules, schema, evidence renderers, evaluation). No third-party code was copied; ByteTrack/BoT-SORT/SimCC are re-implemented ideas. No public dataset was used for training or evaluation.

**Flagged risks** (licences):
- (a) The RTMPose body7 and YOLOX COCO weights are Apache-2.0, but their training datasets carry their own terms (e.g. COCO images, AI Challenger). Check them before any commercial use.
- (b) ffprobe can be a GPL build. It is optional: without it, VFR is not flagged.
- (c) The earlier draft used Ultralytics YOLOv8 (**AGPL-3.0**). It has been **removed**; no AGPL or GPL component remains in the runtime.

## 4. Risks and open questions
1. **Reference consistency:** several of my reference annotations break the fixed definitions. RUS is marked not_observable although the definition gives frame 0 if the bowler is already moving, and FTE is marked 1 frame after BR. dev_02/dev_03 need re-checking. These are fixed by re-annotation and reported as such; references are never tuned to predictions.
2. **BR offset** (+4 frames) is fitted on only 5 dev clips, so it may not hold for other bowlers or views [A].
3. **Small or distant bowlers** (broadcast) and feet at the frame edge (mobile) give noisy foot keypoints, so BFC/GTO often abstain [D].
4. **Long broadcast clips:** the "first delivery" choice depends on the anchor score [D].
5. **Runtime:** 2–10 s per video-second on a 2-core VM (AMD Ryzen 7 4800H host). That is acceptable but slow; detection every frame in the window is the main cost [D].

## 5. Commands
```
python scripts/fetch_weights.py
python -m src.run --manifest data/input_manifest.csv --out outputs/
python -m src.evaluation.evaluate --pred outputs/predictions --ref annotations/reference --manifest data/input_manifest.csv --out outputs/evaluation
pytest -q tests
```
