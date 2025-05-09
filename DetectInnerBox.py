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
        # ① tensor → numpy 单通道 0/255
        if isinstance(mask, list):  # ComfyUI 有时包在 list 里
            mask = mask[0]
        if mask.ndim == 3:  # [1,H,W] → [H,W]
            mask = mask[0]
        np_mask = mask.cpu().numpy().astype("uint8") * 255  # 0/255

        H, W = np_mask.shape[:2]

        # ② flood-fill 把连边的白色清零
        pad = cv2.copyMakeBorder(np_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flags = 4 if connectivity == 4 else 8
        cv2.floodFill(
            pad, None, (0, 0), 0, loDiff=0, upDiff=0, flags=flags | (255 << 8)
        )
        inner = pad[1:-1, 1:-1]

        # ③ 取剩余像素的 bbox
        ys, xs = np.where(inner == 255)
        if ys.size == 0:
            return 0, 0

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        return int(x1 - x0 + 1), int(y1 - y0 + 1)
