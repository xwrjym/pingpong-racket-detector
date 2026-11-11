"""离线评估：检测指标 + 方案 §10/§12.2 要求的业务指标。

不做花哨的可视化，只输出一份能直接贴进报告的结构化结果：

* 检测：AP@0.5（按类）、mAP@0.5、指定置信度下的 precision / recall；
* 侧别：匹配上的框里 side 判对的比例，UNKNOWN 单独计数；
* 击球时刻：每个击球区间内是否检到对应侧的球拍（方案 §12.2 的核心检查点）；
* Proposal：TrackNet 提议区间内有有效检出的比例；
* 跟踪：断轨时长由 ``RacketTracker.stats()`` 给出，这里只做汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .dataset import FrameLabel, HitLabel
from .geometry import iou
from .schema import RacketObservation

BBox = tuple[float, float, float, float]


@dataclass
class PredBox:
    bbox_xyxy: BBox
    confidence: float
    side: str = "UNKNOWN"

    @property
    def class_name(self) -> str:
        return "racket_A" if self.side == "A" else "racket_B" if self.side == "B" else "racket_unknown"


@dataclass
class PredFrame:
    session_id: str
    frame_id: int
    timestamp_s: float
    boxes: list[PredBox] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return (self.session_id, self.frame_id)


def from_observations(observations: Iterable[RacketObservation]) -> list[PredFrame]:
    """把 racket.v1 观测日志转成评估用帧，visible=False 的帧保留（算漏检）。"""
    frames: dict[tuple[str, int], PredFrame] = {}
    for obs in observations:
        key = (obs.session_id, obs.frame_id)
        frame = frames.setdefault(
            key, PredFrame(obs.session_id, obs.frame_id, obs.timestamp_s, [])
        )
        if obs.visible and obs.bbox_xyxy and obs.confidence is not None:
            frame.boxes.append(PredBox(obs.bbox_xyxy, float(obs.confidence), obs.side))
    return sorted(frames.values(), key=lambda f: (f.session_id, f.frame_id))


# ------------------------------------------------------------------ 匹配
def match_boxes(
    gt: Sequence[tuple[BBox, str]],
    preds: Sequence[PredBox],
    iou_thr: float,
    *,
    require_same_side: bool = False,
) -> list[tuple[int, int, float]]:
    """贪心匹配：置信度高的预测先选 IoU 最大的未匹配 GT。"""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].confidence)
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pi in order:
        pred = preds[pi]
        best, best_iou = -1, 0.0
        for gi, (gt_bbox, gt_side) in enumerate(gt):
            if gi in used_gt:
                continue
            if require_same_side and gt_side != "UNKNOWN" and pred.side != "UNKNOWN" \
                    and gt_side != pred.side:
                continue
            value = iou(gt_bbox, pred.bbox_xyxy)
            if value > best_iou:
                best, best_iou = gi, value
        if best >= 0 and best_iou >= iou_thr:
            used_gt.add(best)
            matches.append((best, pi, best_iou))
    return matches


def _average_precision(records: Sequence[tuple[float, bool]], n_pos: int) -> float:
    """全点插值 AP。records 是 (confidence, is_tp)。"""
    if n_pos == 0 or not records:
        return 0.0
    ordered = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    points: list[tuple[float, float]] = []
    prev_recall = 0.0
    ap = 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for _conf, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # 精度包络 + 全点插值
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    for recall, precision in zip(recalls, precisions):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return max(0.0, min(1.0, ap))


def _pr_at(records: Sequence[tuple[float, bool]], n_pos: int, conf_thr: float) -> tuple[float, float, int]:
    tp = sum(1 for conf, is_tp in records if is_tp and conf >= conf_thr)
    fp = sum(1 for conf, is_tp in records if not is_tp and conf >= conf_thr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    return precision, recall, tp + fp


# ------------------------------------------------------------------ 检测
def evaluate_detection(
    gt_frames: Sequence[FrameLabel],
    pred_frames: Sequence[PredFrame],
    *,
    iou_thr: float = 0.5,
    conf_thresholds: Sequence[float] = (0.25, 0.5),
) -> dict[str, Any]:
    pred_index = {(f.session_id, f.frame_id): f for f in pred_frames}
    records: dict[str, list[tuple[float, bool]]] = {name: [] for name in ("racket_A", "racket_B", "racket_unknown")}
    n_pos: dict[str, int] = {name: 0 for name in records}
    side_total = side_correct = side_unknown = 0
    matched_ious: list[float] = []
    frames_with_gt = frames_with_hit = 0

    for gt_frame in gt_frames:
        gt_boxes = [(b.bbox_xyxy, b.side) for b in gt_frame.boxes if b.visible and b.bbox_xyxy]
        if gt_boxes:
            frames_with_gt += 1
        for _bbox, side in gt_boxes:
            n_pos["racket_A" if side == "A" else "racket_B" if side == "B" else "racket_unknown"] += 1

        pred_frame = pred_index.get((gt_frame.session_id, gt_frame.frame_id))
        preds = pred_frame.boxes if pred_frame else []
        matches = match_boxes(gt_boxes, preds, iou_thr)
        matched_gt = {gi for gi, _pi, _v in matches}
        if matches:
            frames_with_hit += 1
        for _gi, pi, value in matches:
            matched_ious.append(value)
            side_total += 1
            gt_side = gt_boxes[_gi][1]
            pred_side = preds[pi].side
            if pred_side == "UNKNOWN":
                side_unknown += 1
            elif gt_side == pred_side:
                side_correct += 1

        matched_pred = {pi for _gi, pi, _v in matches}
        for gi, (bbox, side) in enumerate(gt_boxes):
            name = "racket_A" if side == "A" else "racket_B" if side == "B" else "racket_unknown"
            if gi in matched_gt:
                # 找到与这个 GT 匹配的预测置信度
                conf = next(preds[pi].confidence for g, pi, _v in matches if g == gi)
                records[name].append((conf, True))
            else:
                records[name].append((0.0, False))
        for pi, pred in enumerate(preds):
            if pi in matched_pred:
                continue
            records[pred.class_name].append((pred.confidence, False))

    per_class_ap = {name: _average_precision(records[name], n_pos[name]) for name in records}
    present = [name for name in records if n_pos[name] > 0]
    map50 = sum(per_class_ap[name] for name in present) / len(present) if present else 0.0

    pr: dict[str, dict[str, float]] = {}
    for thr in conf_thresholds:
        all_records = [r for name in records for r in records[name]]
        total_pos = sum(n_pos.values())
        precision, recall, count = _pr_at(all_records, total_pos, thr)
        pr[f"{thr:g}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "detections": count,
        }

    return {
        "iou_thr": iou_thr,
        "frames_evaluated": len(gt_frames),
        "frames_with_gt": frames_with_gt,
        "frames_with_prediction": frames_with_hit,
        "frame_recall": round(frames_with_hit / frames_with_gt, 4) if frames_with_gt else 0.0,
        "gt_boxes": sum(n_pos.values()),
        "per_class_ap50": {k: round(v, 4) for k, v in per_class_ap.items()},
        "mAP50": round(map50, 4),
        "gt_per_class": n_pos,
        "precision_recall": pr,
        "side_accuracy": round(side_correct / side_total, 4) if side_total else 0.0,
        "side_unknown_rate": round(side_unknown / side_total, 4) if side_total else 0.0,
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
    }


# ------------------------------------------------------------------ 击球时刻
def evaluate_hit_windows(
    hits: Sequence[HitLabel],
    pred_frames: Sequence[PredFrame],
    *,
    window_s: float = 0.2,
    conf_thr: float = 0.25,
    side_required: bool = True,
) -> dict[str, Any]:
    """方案 §12.2 的「击球时刻附近的球拍召回率」。"""
    if not hits:
        return {"hits": 0, "note": "没有击球区间标注，跳过该项"}

    by_session: dict[str, list[PredFrame]] = {}
    for frame in pred_frames:
        by_session.setdefault(frame.session_id, []).append(frame)

    found = 0
    found_side = 0
    per_hit: list[dict[str, Any]] = []
    for hit in hits:
        candidates = [
            f for f in by_session.get(hit.session_id, [])
            if hit.t0_s - window_s <= f.timestamp_s <= hit.t1_s + window_s
        ]
        has_any = any(
            box.confidence >= conf_thr for f in candidates for box in f.boxes
        )
        has_side = any(
            box.confidence >= conf_thr and (not side_required or hit.side == "UNKNOWN" or box.side == hit.side)
            for f in candidates for box in f.boxes
        )
        found += int(has_any)
        found_side += int(has_side)
        per_hit.append({
            "hit_id": hit.hit_id,
            "side": hit.side,
            "detected": has_any,
            "detected_correct_side": has_side,
        })
    return {
        "hits": len(hits),
        "window_s": window_s,
        "conf_thr": conf_thr,
        "hit_recall": round(found / len(hits), 4),
        "hit_recall_correct_side": round(found_side / len(hits), 4),
        "missed_hit_ids": [h["hit_id"] for h in per_hit if not h["detected"]],
        "per_hit": per_hit,
    }


# ------------------------------------------------------------------ Proposal
def evaluate_proposals(
    proposals: Sequence[tuple[float, float]],
    pred_frames: Sequence[PredFrame],
    *,
    conf_thr: float = 0.25,
) -> dict[str, Any]:
    """Proposal 区间内的有效检出率：决定"只在区间内推理"是否够用。"""
    if not proposals:
        return {"proposals": 0, "note": "没有 Proposal 区间"}
    covered = 0
    lengths: list[float] = []
    for t0, t1 in proposals:
        lengths.append(t1 - t0)
        if any(
            t0 <= f.timestamp_s <= t1 and any(b.confidence >= conf_thr for b in f.boxes)
            for f in pred_frames
        ):
            covered += 1
    return {
        "proposals": len(proposals),
        "proposal_recall": round(covered / len(proposals), 4),
        "mean_proposal_s": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        "conf_thr": conf_thr,
    }


# ------------------------------------------------------------------ 报告
def format_report(
    detection: dict[str, Any],
    *,
    hits: dict[str, Any] | None = None,
    proposals: dict[str, Any] | None = None,
    tracking: dict[str, Any] | None = None,
    title: str = "球拍检测离线评估",
) -> str:
    section = 1
    lines = [f"# {title}", ""]
    lines += [
        f"## {section}. 检测指标",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 评估帧数 | {detection['frames_evaluated']} |",
        f"| GT 框数 | {detection['gt_boxes']} |",
        f"| mAP@0.5 | **{detection['mAP50']}** |",
        f"| AP@0.5 racket_A | {detection['per_class_ap50']['racket_A']} |",
        f"| AP@0.5 racket_B | {detection['per_class_ap50']['racket_B']} |",
        f"| 帧级召回 | {detection['frame_recall']} |",
        f"| 匹配平均 IoU | {detection['mean_matched_iou']} |",
        f"| 侧别准确率 | {detection['side_accuracy']} |",
        f"| 侧别 UNKNOWN 率 | {detection['side_unknown_rate']} |",
        "",
    ]
    lines += ["| 置信度阈值 | Precision | Recall |", "|---|---|---|"]
    for thr, values in detection["precision_recall"].items():
        lines.append(f"| {thr} | {values['precision']} | {values['recall']} |")
    lines.append("")

    if hits:
        section += 1
        lines += [f"## {section}. 击球时刻球拍召回", ""]
        if hits.get("hits"):
            lines += [
                "| 项目 | 值 |",
                "|---|---|",
                f"| 击球区间数 | {hits['hits']} |",
                f"| 时间窗 | ±{hits['window_s']} s |",
                f"| 击球时刻球拍召回 | **{hits['hit_recall']}** |",
                f"| 含正确侧别的召回 | **{hits['hit_recall_correct_side']}** |",
                "",
            ]
            if hits["missed_hit_ids"]:
                lines.append(f"漏检击球：{', '.join(hits['missed_hit_ids'])}")
                lines.append("")
        else:
            lines += [hits.get("note", "无数据"), ""]

    if proposals:
        section += 1
        lines += [f"## {section}. Proposal 区间覆盖", ""]
        if proposals.get("proposals"):
            lines += [
                "| 项目 | 值 |",
                "|---|---|",
                f"| 区间数 | {proposals['proposals']} |",
                f"| 区间内有检出比例 | **{proposals['proposal_recall']}** |",
                f"| 平均区间长度 | {proposals['mean_proposal_s']} s |",
                "",
            ]
        else:
            lines += [proposals.get("note", "无数据"), ""]

    if tracking:
        section += 1
        lines += [
            f"## {section}. 跟踪稳定性",
            "",
            "| 项目 | 值 |",
            "|---|---|",
            f"| 建立轨迹数 | {tracking.get('tracks_created')} |",
            f"| 断轨次数 | {tracking.get('gap_events')} |",
            f"| 最长断轨 | {tracking.get('max_gap_s')} s |",
            f"| 断轨帧占比 | {tracking.get('gap_ratio')} |",
            "",
        ]
    return "\n".join(lines)
