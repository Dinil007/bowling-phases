"""
Stage 4: the six events (Brief 4B definitions), measured on native frames.

Each detector returns an EventEstimate. When the defining evidence is missing (feet out of frame,
low keypoint confidence, no flight phase, bowler lost) the estimate has frame=None and a
visibility status explaining why, plus the search window - it never forces a frame.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.phase_detection.kinematics import (ARM, FOOT, body_height, ground_speed, nanmedfilt,
                                            point_track, runs)
from src.phase_detection.rtmpose import KP
from src.processing.isolate import track_speed


@dataclass
class EventEstimate:
    name: str
    frame: Optional[int]
    window: Optional[tuple]
    visibility: str
    confidence: Optional[float]
    evidence: str
    method: str
    notes: str
    extra: dict = field(default_factory=dict)


def _conf(scores, i, r=1):
    seg = scores[max(0, i - r): i + r + 1]
    return round(float(np.clip(np.mean(seg), 0, 1)), 3) if len(seg) else None


class EventDetector:
    def __init__(self, cfg: dict, method_suffix: str):
        self.cfg = cfg
        self.sfx = method_suffix

    # ------------------------------------------------------------------ helpers
    def _foot_signals(self, W, H):
        out = {}
        for side, idxs in FOOT.items():
            pts, scs = point_track(W, idxs, self.cfg["kp_min_score"])
            spd = ground_speed(W, pts, H, centered=bool(self.cfg.get("centered_speed", False)))
            k = int(self.cfg.get("foot_speed_filter", 3))
            spd = nanmedfilt(spd, k) if k > 1 else spd
            out[side] = (pts, scs, spd)
        return out

    def _out_of_frame(self, W, pts, i):
        h, w = W.img_hw
        if np.isnan(pts[i]).any():
            return True
        return pts[i][1] > h - 3 or pts[i][0] < 2 or pts[i][0] > w - 3

    # ------------------------------------------------------------------ main
    def detect(self, W, scan, anchor, meta) -> List[EventEstimate]:
        c, fps, F = self.cfg, W.fps, W.frames
        n = len(F)
        H = body_height(W)
        idx = {f: i for i, f in enumerate(F)}
        a_i = idx.get(min(max(anchor.frame, F[0]), F[-1]), 0)
        ev = {}

        # ---- BR: bowling-arm wrist kinematics (+ optional ball separation check)
        arm = {}
        lo, hi = max(0, a_i - int(c["br_search_pre_s"] * fps)), min(n - 1, a_i + int(c["br_search_post_s"] * fps))
        for side, (s_i, e_i, w_i) in ARM.items():
            pts, scs = point_track(W, [w_i], c["kp_min_score"])
            spd = nanmedfilt(ground_speed(W, pts, H, centered=bool(c.get("centered_speed", False))), 3)
            seg = spd[lo:hi + 1]
            if np.all(np.isnan(seg)):
                continue
            pk = lo + int(np.nanargmax(seg))
            arm[side] = (np.nanmax(seg), pk, pts, scs, spd)
            # wrist apex relative to shoulder (top of the bowling arc) near the speed peak
            sh, _ = point_track(W, [s_i], c["kp_min_score"])
            rel = pts[:, 1] - sh[:, 1]
            a0, a1 = max(0, pk - 6), min(n - 1, pk + 6)
            if not np.all(np.isnan(rel[a0:a1 + 1])):
                arm[side] = arm[side] + (a0 + int(np.nanargmin(rel[a0:a1 + 1])),)
            else:
                arm[side] = arm[side] + (pk,)
        if not arm:
            br = EventEstimate("BR", None, (F[lo], F[hi]), "not_observable", None, "pose:wrists",
                               "release-wristpeak-v1" + self.sfx,
                               "Bowling wrist not tracked with sufficient keypoint confidence near the delivery.")
            bowl_side = None
        else:
            bowl_side = max(arm, key=lambda s: arm[s][0])
            vmax, pk, wpts, wsc, wspd, apex = arm[bowl_side]
            off = int(c["br_offset_frames"])
            cue_i = apex if c.get("br_cue", "speed_peak") == "apex" else pk
            est_i = int(np.clip(cue_i + off, 0, n - 1))
            r = c["br_window_frames"] + (1 if fps < 25 else 0)
            win = (F[max(0, est_i - r)], F[min(n - 1, est_i + r)])
            arm_ids = list(ARM[bowl_side])
            arm_conf = float(np.mean([W.sc[F[j]][arm_ids].mean() for j in range(max(0, pk - 2), min(n, pk + 3))
                                      if F[j] in W.sc])) if wsc[pk] > 0 else 0.0
            med = np.nanmedian(wspd[lo:hi + 1])
            prominence = vmax / med if med and med > 0 else 0.0
            tracked = all(W.status.get(F[j]) == "tracked" for j in range(max(0, pk - 2), min(n, pk + 3)))
            ball_f = self._ball_check(W, wpts, est_i, H) if c.get("ball_check", True) else None
            evidence = f"pose:{bowl_side}_wrist_speed_peak"
            notes = (f"Bowling arm = {bowl_side} (higher wrist-speed peak). Wrist speed peak at frame {F[pk]} "
                     f"({vmax:.1f} body-heights/s, {prominence:.1f}x window median); arm keypoint score {arm_conf:.2f}.")
            if ball_f is not None:
                br = EventEstimate("BR", ball_f, (ball_f - 1, ball_f + 1), "visible",
                                   round(min(1.0, arm_conf), 3), evidence + ",ball_det:yolox_sports_ball",
                                   "release-ballsep-v1" + self.sfx,
                                   notes + f" Ball detected separated from the hand first at frame {ball_f}.")
            elif (c["br_emit_estimate"] and arm_conf >= c["br_min_arm_conf"] and tracked
                  and prominence >= c["br_min_prominence"]):
                br = EventEstimate("BR", F[est_i], win, "partially_visible", round(arm_conf, 3), evidence,
                                   "release-wristpeak-v1" + self.sfx,
                                   notes + f" Ball not resolved by the detector; frame is a kinematic estimate "
                                           f"(peak {off:+d} frames, offset calibrated on development clips); "
                                           f"true release expected inside the window.")
            else:
                why = []
                if arm_conf < c["br_min_arm_conf"]:
                    why.append(f"arm keypoint score {arm_conf:.2f} < {c['br_min_arm_conf']}")
                if not tracked:
                    why.append("bowler track degraded around the peak")
                if prominence < c["br_min_prominence"]:
                    why.append(f"wrist-speed peak not distinct ({prominence:.1f}x)")
                if not c["br_emit_estimate"]:
                    why.append("ball not resolved and kinematic estimates disabled")
                br = EventEstimate("BR", None, win, "not_observable", None, evidence, "release-wristpeak-v1" + self.sfx,
                                   notes + " Not observable: " + "; ".join(why) + ". Release lies in the window.")
        ev["BR"] = br
        br_i = idx.get(br.frame) if br.frame is not None else (idx.get(int(round(np.mean(br.window)))) if br.window else a_i)
        if br_i is None:
            br_i = a_i

        # ---- foot contacts and gather take-off
        feet = self._foot_signals(W, H)
        plant = c["plant_speed"]
        front = {"right": "left", "left": "right"}.get(bowl_side, None)
        back = bowl_side
        planted = {s: (feet[s][2] < plant) for s in feet}

        def contact(side, lo_i, hi_i, name, ev_src):
            if side is None:
                return EventEstimate(name, None, (F[lo_i], F[hi_i]), "not_observable", None, ev_src,
                                     "contact-groundspeed-v1" + self.sfx, "Bowling arm unknown, so front/back foot unknown.")
            pts, scs, spd = feet[side]
            cand = [(s, e) for s, e in runs(planted[side]) if lo_i <= s <= hi_i and e - s + 1 >= c["plant_min_frames"]]
            # must follow a swing (foot moving before contact)
            cand = [(s, e) for s, e in cand if s > 0 and not np.isnan(spd[s - 1]) and spd[s - 1] >= plant]
            if not cand:
                oof = sum(self._out_of_frame(W, pts, j) for j in range(lo_i, hi_i + 1)) > 0.5 * (hi_i - lo_i + 1)
                lowc = np.mean(scs[lo_i:hi_i + 1]) < c["foot_min_score"]
                vis = "out_of_frame" if oof else ("occluded" if lowc else "not_observable")
                return EventEstimate(name, None, (F[lo_i], F[hi_i]), vis, None, f"pose:{side}_foot_ground_speed",
                                     "contact-groundspeed-v1" + self.sfx,
                                     f"No {side}-foot swing-to-stance transition found in frames {F[lo_i]}-{F[hi_i]} "
                                     f"({'foot out of frame' if oof else 'low foot keypoint score' if lowc else 'no clear contact'}).")
            s, e = cand[-1]
            conf = _conf(scs, s)
            if conf is not None and conf < c["foot_min_score"]:
                return EventEstimate(name, None, (F[max(0, s - 2)], F[min(n - 1, s + 2)]), "occluded", None,
                                     f"pose:{side}_foot_ground_speed", "contact-groundspeed-v1" + self.sfx,
                                     f"Candidate contact at frame {F[s]} rejected: foot keypoint score {conf:.2f} < {c['foot_min_score']}.")
            r = 1 + (1 if fps < 25 else 0)
            vis = "partially_visible" if self._out_of_frame(W, pts, s) else "visible"
            return EventEstimate(name, F[s], (F[max(0, s - r)], F[min(n - 1, s + r)]), vis, conf,
                                 f"pose:{side}_heel,{side}_toes,{side}_ankle;gmc",
                                 "contact-groundspeed-v1" + self.sfx,
                                 f"{side.capitalize()} foot ground-relative speed falls below {plant} body-heights/s "
                                 f"at frame {F[s]} after a swing; planted for {e - s + 1} frames.")

        ffc = contact(front, max(0, br_i - int(c["ffc_search_s"] * fps)), min(n - 1, br_i + 1), "FFC", "pose:front_foot")
        ev["FFC"] = ffc
        ffc_i = idx[ffc.frame] if ffc.frame is not None else max(0, br_i - int(0.12 * fps))
        bfc = contact(back, max(0, ffc_i - int(c["bfc_search_s"] * fps)), max(0, ffc_i - 1), "BFC", "pose:back_foot")
        ev["BFC"] = bfc

        # GTO: last frame any foot planted before BFC, +1 (first frame of the flight)
        if bfc.frame is not None:
            b_i = idx[bfc.frame]
            j = b_i - 1
            limit = max(0, b_i - int(c["gto_search_s"] * fps))
            anyp = planted["left"] | planted["right"]
            known = ~(np.isnan(feet["left"][2]) & np.isnan(feet["right"][2]))
            while j >= limit and not anyp[j] and known[j]:
                j -= 1
            flight = b_i - 1 - j
            if j >= limit and anyp[j] and flight >= 1:
                g = j + 1
                sc_both = (feet["left"][1] + feet["right"][1]) / 2
                ev["GTO"] = EventEstimate("GTO", F[g], (F[max(0, g - 1)], F[min(n - 1, g + 1)]), "visible",
                                          _conf(sc_both, g), "pose:both_feet_ground_speed;gmc",
                                          "takeoff-flight-v1" + self.sfx,
                                          f"Both feet off the ground from frame {F[g]} for {flight} frame(s) before back-foot contact.")
            else:
                ev["GTO"] = EventEstimate("GTO", None, (F[limit], bfc.frame), "not_observable", None,
                                          "pose:both_feet_ground_speed", "takeoff-flight-v1" + self.sfx,
                                          "No flight phase (both feet off ground) resolved before back-foot contact.")
        else:
            ev["GTO"] = EventEstimate("GTO", None, (F[max(0, ffc_i - int(0.8 * fps))], F[ffc_i]), bfc.visibility
                                      if bfc.visibility != "visible" else "not_observable", None,
                                      "pose:both_feet_ground_speed", "takeoff-flight-v1" + self.sfx,
                                      "Back-foot contact not observable, so the take-off before it cannot be bounded.")

        ev["RUS"] = self._rus(scan, anchor, meta, W)
        ev["FTE"] = self._fte(W, H, br_i, meta, scan, anchor)
        return [ev[k] for k in ("RUS", "GTO", "BFC", "FFC", "BR", "FTE")]

    # ------------------------------------------------------------------ BR ball check
    def _ball_check(self, W, wpts, est_i, H):
        c, F = self.cfg, W.frames
        hits = []
        for j in range(max(0, est_i - 3), min(len(F), est_i + 5)):
            f = F[j]
            b = W.balls.get(f)
            if b is None or not len(b) or np.isnan(wpts[j]).any():
                hits.append(None)
                continue
            ctr = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
            d = np.linalg.norm(ctr - wpts[j], axis=1) / H
            ok = (d > c["ball_sep_min"]) & (d < c["ball_sep_max"]) & (b[:, 4] >= c["ball_conf"])
            hits.append(f if ok.any() else None)
        for k in range(len(hits) - 1):
            if hits[k] is not None and hits[k + 1] is not None:
                return hits[k]
        return None

    # ------------------------------------------------------------------ RUS
    def _rus(self, scan, anchor, meta, W):
        c, fps = self.cfg, meta.fps
        tid = anchor.track_id
        chain = [tid]
        # link fragmented tracks backwards (same shot, end shortly before, spatially close)
        for _ in range(3):
            t0 = scan.tracks[chain[0]]
            best = None
            for oid, o in scan.tracks.items():
                if oid in chain or o["frames"][-1] >= t0["frames"][0]:
                    continue
                gap = t0["frames"][0] - o["frames"][-1]
                if gap > fps or any(scan.cuts[o["frames"][-1] + 1: t0["frames"][0] + 1]):
                    continue
                from src.processing.motion import compose, warp_box
                pb = warp_box(compose(scan.A, o["frames"][-1], t0["frames"][0]), o["boxes"][-1])
                nb = np.array(t0["boxes"][0])
                h = nb[3] - nb[1]
                d = np.hypot((pb[0] + pb[2] - nb[0] - nb[2]) / 2, (pb[1] + pb[3] - nb[1] - nb[3]) / 2)
                if d < c["link_dist"] * h * max(1, gap / (fps / c_detect(c))) and (best is None or d < best[1]):
                    best = (oid, d)
            if best is None:
                break
            chain.insert(0, best[0])
        sp = {}
        for t in chain:
            sp.update(track_speed(scan, t, fps))
        frames = sorted(f for f in sp if f <= anchor.frame)
        first_frame = scan.tracks[chain[0]]["frames"][0]
        step = max(1, int(round(fps / c_detect(c))))
        if not frames:
            return EventEstimate("RUS", first_frame, (max(0, first_frame - step), first_frame), "partially_visible", None,
                                 "track:bowler_box_ground_speed", "runup-onset-v1" + self.sfx,
                                 "Too few samples before the delivery; using first frame the bowler is detected.")
        thr = c["rus_move_speed"]
        k = len(frames) - 1
        dips = 0
        while k > 0:
            if sp[frames[k - 1]] >= thr:
                dips = 0
            else:
                dips += 1
                if dips > 1:
                    break
            k -= 1
        k = min(len(frames) - 1, k + dips) if dips > 1 else k
        onset = frames[k]
        shot_start = 0
        for i in range(anchor.frame, 0, -1):
            if scan.cuts[i]:
                shot_start = i
                break
        det_conf = round(float(np.mean(scan.tracks[chain[0]]["scores"][:3])), 3)
        if k == 0 and first_frame <= shot_start + step:
            f = shot_start
            note = ("Bowler already moving when first visible" + (" at the start of the clip; RUS = frame 0 by definition."
                    if shot_start == 0 else f"; delivery shot starts at cut frame {shot_start}."))
            return EventEstimate("RUS", f, (f, f + step), "visible", det_conf, "track:bowler_box_ground_speed;gmc",
                                 "runup-onset-v1" + self.sfx, note)
        if k == 0:
            return EventEstimate("RUS", first_frame, (max(0, first_frame - step), first_frame), "partially_visible",
                                 det_conf, "track:bowler_box_ground_speed;gmc", "runup-onset-v1" + self.sfx,
                                 f"Bowler first detected (already moving) at frame {first_frame}; earlier frames may "
                                 f"show the start of the run-up undetected.")
        return EventEstimate("RUS", onset, (max(0, onset - step), onset), "visible", det_conf,
                             "track:bowler_box_ground_speed;gmc", "runup-onset-v1" + self.sfx,
                             f"Ground-relative bowler speed rises above {thr} body-heights/s from frame {onset} and "
                             f"stays up until the delivery (sampled every {step} frames).")

    # ------------------------------------------------------------------ FTE
    def _fte(self, W, H, br_i, meta, scan, anchor):
        c, F, fps = self.cfg, W.frames, W.fps
        hip_pts, hip_sc = point_track(W, [KP["hip"], KP["left_hip"], KP["right_hip"]], c["kp_min_score"])
        spd = nanmedfilt(ground_speed(W, hip_pts, H, centered=bool(c.get("centered_speed", False))), 5)
        start = br_i + int(c["fte_min_after_br_s"] * fps)
        need = c["fte_slow_frames"]
        for i in range(start, len(F) - need + 1):
            seg = spd[i:i + need]
            if not np.isnan(seg).any() and (seg < c["fte_walk_speed"]).all():
                return EventEstimate("FTE", F[i], (F[max(0, i - 2)], F[min(len(F) - 1, i + 2)]), "visible",
                                     _conf(hip_sc, i), "pose:hip_ground_speed;gmc", "fte-walkpace-v1" + self.sfx,
                                     f"Hip ground speed below walking pace ({c['fte_walk_speed']} body-heights/s) from frame {F[i]}.")
        last_tracked = max([f for f in F if W.status.get(f) == "tracked"], default=F[-1])
        if W.end >= meta.frame_count - 1 and last_tracked >= meta.frame_count - 2:
            f = meta.frame_count - 1
            return EventEstimate("FTE", f, (f, f), "visible", None, "clip_end", "fte-walkpace-v1" + self.sfx,
                                 "Clip ends before walking pace is reached; FTE = last frame of the clip by definition.")
        lost = [f for f in F if f > F[br_i] and W.status.get(f) == "lost"]
        if lost or W.end < anchor.frame + int(self.cfg_post(fps)):
            f0 = lost[0] if lost else W.end
            return EventEstimate("FTE", None, (f0, min(meta.frame_count - 1, f0 + int(fps))), "out_of_frame", None,
                                 "pose:hip_ground_speed", "fte-walkpace-v1" + self.sfx,
                                 f"Bowler leaves the frame / camera cuts at frame {f0} before reaching walking pace.")
        return EventEstimate("FTE", None, (F[start] if start < len(F) else F[-1], F[-1]), "not_observable", None,
                             "pose:hip_ground_speed", "fte-walkpace-v1" + self.sfx,
                             "Walking pace not reached inside the analysis window.")

    def cfg_post(self, fps):
        return self.cfg.get("post_s_hint", 2.0) * fps - 2


def c_detect(c):
    return c.get("detect_fps", 5.0)
