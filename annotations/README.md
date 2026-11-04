# Reference Annotations

This directory contains human ground truth reference annotations for the development clips, as required by **Section 10 of the A–Z Build Bible v2.1**.

## Directory Structure
```
annotations/
├── README.md               # Human annotation protocol & guidelines
└── reference/              # Pre-committed human reference ground truth JSONs
    ├── dev_01.json
    ├── dev_02.json
    └── ...
```

## Protocol & Guidelines
1. **Timing Semantics:** Each event is marked at the **first frame** where the condition is true.
2. **Cardinal Ball Release Rule:**
   - If ball-hand separation is visibly observable $\to$ `visibility_status = "visible"`, `event_frame = <frame_int>`.
   - If obscured by motion blur or occlusion $\to$ `visibility_status = "not_observable"`, `event_frame = null`, uncertainty window populated.
3. **Traceability:**
   - Every record in `annotations/reference/` must have `human_verified: true`.
   - Includes annotator name/pseudonym.
