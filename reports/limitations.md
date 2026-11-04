# Limitations

- **Frame rate:** all clips are ≤30 fps (one is 23 fps). One frame is 33–43 ms, so the ±1 frame human noise floor is already ±33–43 ms.
- **Ball release:** the ball detector (COCO "sports ball") almost never resolves a cricket ball at release in these clips. Most BR outputs are therefore kinematic estimates (`partially_visible`, offset fitted on 5 dev clips) or `not_observable` with a window.
- **Feet:** foot contacts need visible heels and toes. With small broadcast bowlers (~100 px) or feet at the frame edge (mobile), BFC/GTO usually abstain.
- **Camera views:** contact rules use ground-relative speed after camera-motion compensation, which works for side-on and moving cameras. Front-on/behind views have not been measured separately (the manifest does not use that vocabulary).
- **Delivery choice:** the first delivery is the earliest strong arm-action cluster. Clips with replays or multiple angles depend on that score.
- **Not inferable from this evidence:** bowler skill or quality, speed, injury risk, biomechanical correctness. Legitimate inference: timings within a clip, e.g. "FFC is about N frames (N×33 ms) before the estimated release window".
