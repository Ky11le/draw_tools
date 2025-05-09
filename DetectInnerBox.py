import cv2
import torch
import numpy as np


class DetectInnerBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # 只接 MASK
                "connectivity": ("INT", {"default": 4, "min": 4, "max": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT")  # width, height
    RETURN_NAMES = ("inner_w", "inner_h")
    FUNCTION = "run"
    CATEGORY = "DrawTools"

    # ---------------- 主逻辑 ---------------- #
    def run(self, mask, connectivity=4):
        if isinstance(mask, list):
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]

        # ① 二值化并反相
        inv = ((mask == 0).cpu().numpy().astype("uint8")) * 255  # 空白→255

        # ② 连通域分析
        conn = 4 if connectivity == 4 else 8
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, conn)
        # stats[i] = [x, y, w, h, area]

        if num <= 1:
            return 0, 0  # 只有背景

        # ③ 去掉 label=0（与外边框连通的背景）
        inner_stats = stats[1:]  # (num-1, 5)
        # ④ 取面积最大的那块
        idx = np.argmax(inner_stats[:, 4])
        w, h = int(inner_stats[idx, 2]), int(inner_stats[idx, 3])
        return w, h
