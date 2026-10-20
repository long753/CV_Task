"""Small shared conversions between detector-native boxes and COCO detections."""

from __future__ import annotations

from typing import Any, Iterable


def coco_detections_from_xyxy(
    boxes: Iterable[Iterable[float]],
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    image_id: int,
    label_to_category: dict[int, int],
    score_floor: float,
    max_detections: int,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Clip native ``xyxy`` boxes, retain valid detections, and emit COCO ``xywh``.

    ``label_to_category`` deliberately makes every adapter state its detector-internal
    class convention before writing the single public COCO category (``1``).
    """
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")

    candidates: list[tuple[float, int, dict[str, Any]]] = []
    discarded = 0
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
        label_id = int(label)
        if label_id not in label_to_category:
            raise RuntimeError(
                f"Unexpected predicted class {label_id} for image {image_id}; "
                f"expected one of {sorted(label_to_category)}"
            )
        score_value = float(score)
        if score_value < score_floor:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if image_size is not None:
            image_width, image_height = image_size
            x1 = min(max(x1, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            x2 = min(max(x2, 0.0), float(image_width))
            y2 = min(max(y2, 0.0), float(image_height))
        width, height = x2 - x1, y2 - y1
        if width <= 0.0 or height <= 0.0:
            discarded += 1
            continue
        candidates.append(
            (
                score_value,
                index,
                {
                    "image_id": image_id,
                    "category_id": label_to_category[label_id],
                    "bbox": [x1, y1, width, height],
                    "score": score_value,
                },
            )
        )

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return [candidate[2] for candidate in candidates[:max_detections]], discarded
