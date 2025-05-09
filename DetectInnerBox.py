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
        # ① tensor → numpy uint8 0/255
        if isinstance(mask, list):  # ComfyUI 有时包装成 list
            mask = mask[0]
        if mask.ndim == 3:  # [1,H,W] → [H,W]
            mask = mask[0]
        m = (mask.cpu().numpy() > 0).astype("uint8") * 255  # bool → 0/255

        # ② flood-fill 把边缘连通白色涂成黑
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flags = 4 if connectivity == 4 else 8
        cv2.floodFill(pad, None, (0, 0), 0, flags=flags | (255 << 8))
        inner = pad[1:-1, 1:-1]  # 去掉填充，得到中央白块

        # ③ 找中央白色像素坐标
        coords = cv2.findNonZero(inner)  # 返回 Nx1x2，或 None
        if coords is None:
            return 0, 0  # 没内容

        x, y, w, h = cv2.boundingRect(coords)
        return int(w), int(h)
