import cv2
import numpy as np
import torch


class DetectInnerBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # 只接 MASK
            }
        }

    RETURN_TYPES = ("INT", "INT")  # 输出：宽、高（8 的倍数）
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = "DrawTools"

    def run(self, mask):
        # 1. unwrap 输入 Tensor → numpy 灰度图
        if isinstance(mask, list):
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]
        np_mask = mask.cpu().numpy()
        # 0~1 转 0~255
        if np_mask.max() <= 1.0:
            np_mask = (np_mask * 255).astype(np.uint8)
        else:
            np_mask = np_mask.astype(np.uint8)

        # 2. 高阈值二值化：只保留「几乎纯白」区域
        _, binary = cv2.threshold(np_mask, 250, 255, cv2.THRESH_BINARY)

        # 3. 连通组件
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return 0, 0

        # 4. 找最大连通块
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1

        # 5. 精确计算它的像素边界
        comp_mask = labels == largest_label
        ys, xs = np.where(comp_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # 6. 真实宽高
        w = int(x1 - x0 + 1)
        h = int(y1 - y0 + 1)

        # 7. 向上取整到最小的 8 的倍数
        w8 = ((w + 7) // 8) * 8
        h8 = ((h + 7) // 8) * 8

        return w8, h8
