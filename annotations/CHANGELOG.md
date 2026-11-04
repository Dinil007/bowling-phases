# Reference annotation change log

## 2026-11-04 - follow-through end (FTE) definition update and dev_03 clarification

1. **FTE alignment with Brief §4B:**
   - Previous references set FTE to BR + 1 frame (human annotator pressed 'e' immediately after release, logging the onset rather than the completion of follow-through).
   - Re-annotated FTE to the true cessation of bowling momentum / transition to walking pace across dev_04..dev_08 (~20-35 frames post-release).
   - In dev_01, the annotator recorded the complete physical follow-through deceleration down the pitch to frame 459.
2. **dev_03 broadcast clip:**
   - Visual inspection verified the clip starts post-release during follow-through; all delivery events (RUS, GTO, BFC, FFC, BR, FTE) and phases set to `not_observable` with explicit explanatory notes.

## 2026-11-04 - metadata-only correction (no frame changed)

The annotation tool copied `video_sha256` and `athlete_id` from the manifest that existed at
annotation time. That manifest's metadata columns were out of step with the video files (rows were
re-keyed later on 2026-11-03). The annotator (Dinil Raj) confirmed that each annotation file was made
by watching the video file of the same name (`data/raw/<clip_id>.mp4`).

Correction: `video_sha256` and `athlete_id` in every record were set to the current manifest values.
Event frames, phase frames, visibility statuses, notes and annotator fields are byte-for-byte unchanged
(checked by script).

| clip | before (sha / athlete) | after (sha / athlete) |
|---|---|---|
| dev_01 | 3c657c77be46... / ATH-PLACEHOLDER-01 | 3f193a5e4912... / ATH-PLACEHOLDER-08 |
| dev_02 | 25637da627cc... / ATH-PLACEHOLDER-02 | 3c657c77be46... / ATH-PLACEHOLDER-01 |
| dev_03 | 93e405bdaad9... / ATH-PLACEHOLDER-03 | 25637da627cc... / ATH-PLACEHOLDER-02 |
| dev_04 | d0822f378af6... / ATH-PLACEHOLDER-04 | 93e405bdaad9... / ATH-PLACEHOLDER-03 |
| dev_05 | 405fd446270a... / ATH-PLACEHOLDER-05 | d0822f378af6... / ATH-PLACEHOLDER-04 |
| dev_06 | ebce6ad56bae... / ATH-PLACEHOLDER-06 | 405fd446270a... / ATH-PLACEHOLDER-05 |
| dev_07 | ab42f1571356... / ATH-PLACEHOLDER-07 | ebce6ad56bae... / ATH-PLACEHOLDER-06 |
| dev_08 | 3f193a5e4912... / ATH-PLACEHOLDER-08 | ab42f1571356... / ATH-PLACEHOLDER-07 |
