from pathlib import Path

import cv2
import math
import numpy as np
import torch

from vajra import Vajra


WEIGHTS = r"F:\material\baseline\VayuAI-main\zanbiya3\zanbiya_data\weights\best-vajra-v1-nano-seg.pt"
SOURCE = r"C:\Users\Administrator\Desktop\datasets\images\train"
OUTPUT_DIR = Path("merge-visual_zanbiya3_0.3") / "defect_only"
MERGE_GAP_PX = 10  # Merge masks with gaps up to this many pixels; 0 means touching/overlapping only.


def merge_adjacent_masks(result, merge_gap):
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or len(result) <= 1:
        return result

    mask_tensor = masks.data
    mask_array = (mask_tensor.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if mask_array.ndim != 3 or mask_array.shape[0] <= 1:
        return result

    mask_h, mask_w = mask_array.shape[1:]
    orig_h, orig_w = result.orig_shape
    scale = max(mask_w / max(orig_w, 1), mask_h / max(orig_h, 1))
    gap_in_mask = max(0, int(round(merge_gap * scale)))
    dilate_radius = int(math.ceil(gap_in_mask / 2))

    if dilate_radius > 0:
        kernel_size = dilate_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        connect_masks = np.stack([cv2.dilate(mask, kernel, iterations=1) for mask in mask_array])
    else:
        connect_masks = mask_array

    connect_map = connect_masks.any(axis=0).astype(np.uint8)
    component_count, component_labels = cv2.connectedComponents(connect_map, connectivity=8)
    if component_count <= 1:
        return result

    parent = list(range(mask_array.shape[0]))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    component_to_indices = {}
    for index, mask in enumerate(connect_masks):
        component_ids = np.unique(component_labels[mask.astype(bool)])
        for component_id in component_ids:
            if component_id:
                component_to_indices.setdefault(int(component_id), []).append(index)

    for indices in component_to_indices.values():
        first = indices[0]
        for index in indices[1:]:
            union(first, index)

    groups = {}
    for index in range(mask_array.shape[0]):
        groups.setdefault(find(index), []).append(index)

    if len(groups) == mask_array.shape[0]:
        return result

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls
    merged_boxes = []
    merged_masks = []

    for indices in groups.values():
        group_indices = torch.tensor(indices, device=xyxy.device)
        group_boxes = xyxy[group_indices]
        group_conf = conf[group_indices]
        best_index = group_indices[group_conf.argmax()]
        merged_xyxy = torch.stack(
            (
                group_boxes[:, 0].min(),
                group_boxes[:, 1].min(),
                group_boxes[:, 2].max(),
                group_boxes[:, 3].max(),
            )
        )
        merged_box = torch.cat((merged_xyxy, conf[best_index].view(1), cls[best_index].view(1)))
        merged_boxes.append(merged_box)
        merged_masks.append(mask_array[indices].any(axis=0))

    merged_boxes = torch.stack(merged_boxes).to(device=boxes.data.device, dtype=boxes.data.dtype)
    merged_masks = torch.as_tensor(np.stack(merged_masks), device=mask_tensor.device).to(dtype=mask_tensor.dtype)
    result.boxes = boxes.__class__(merged_boxes, result.orig_shape)
    result.masks = masks.__class__(merged_masks, result.orig_shape)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = Vajra(WEIGHTS)

    total = 0
    saved = 0
    merged = 0
    for result in model.predict(
        SOURCE,
        img_size=1600,
        device=0,
        conf=0.3,
        iou=0.7,
        batch=1,
        stream=True,
        retina_masks=True,
        save=False,
        verbose=True,
    ):
        total += 1
        if len(result) == 0:
            continue

        before_merge = len(result)
        result = merge_adjacent_masks(result, MERGE_GAP_PX)
        merged += before_merge - len(result)
        saved += 1
        save_path = OUTPUT_DIR / Path(result.path).name
        result.save(
            filename=str(save_path),
            labels=True,
            conf=False,
            boxes=True,
            masks=True,
        )

    print(f"processed: {total}")
    print(f"defect_images_saved: {saved}")
    print(f"merged_defect_instances: {merged}")
    print(f"merge_gap_px: {MERGE_GAP_PX}")
    print(f"save_dir: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
