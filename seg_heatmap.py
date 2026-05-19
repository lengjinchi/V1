from pathlib import Path

import cv2
import numpy as np
from vajra import Vajra


model_path = r"runs4/xiaorong/baseline e=3/weights/best.pt"
source = r"F:\path\to\your\image.jpg"  # 改成你的测试图片或图片文件夹

out_dir = Path("runs4/heatmaps")
out_dir.mkdir(parents=True, exist_ok=True)

model = Vajra(model_path)

results = model(
    source,
    conf=0.25,
    img_size=1600,
    retina_masks=True,  # 让 mask 尺寸尽量和原图一致
)

for i, result in enumerate(results):
    img = result.orig_img.copy()  # BGR

    if result.masks is None:
        print(f"{result.path}: no masks")
        continue

    masks = result.masks.data.float().cpu().numpy()  # [N, H, W]

    # 方式1：显示“哪里被分割到了”，多个目标重叠也不会更亮
    heat = masks.max(axis=0)

    # 方式2：显示“目标密集程度”，重叠/多个目标区域更亮
    # heat = masks.sum(axis=0)
    # heat = np.clip(heat, 0, 1)

    if heat.shape[:2] != img.shape[:2]:
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))

    heat = heat / (heat.max() + 1e-6)
    heat = (heat * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)

    save_path = out_dir / f"{Path(result.path).stem}_heatmap.jpg"
    cv2.imwrite(str(save_path), overlay)
    print(f"saved: {save_path}")
