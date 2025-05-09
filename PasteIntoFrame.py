import cv2
import torch
import numpy as np


class PasteIntoFrame:
    """
    根据 mask 中央矩形尺寸缩放 content，再用 *原始 mask* 把内容贴进 frame。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_image": ("IMAGE",),
                "content_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "connectivity": ("INT", {"default": 4, "min": 4, "max": 8}),
                "keep_aspect": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "run"
    CATEGORY = "DrawTools"

    # ---------------- 主逻辑 ---------------- #
    def run(
        self,
        frame_image,
        content_image,
        mask_image,
        threshold=0.2,
        connectivity=4,
        keep_aspect=False,
    ):
        frame = self._tensor_to_np(frame_image)  # H,W,3 或 4
        content = self._tensor_to_np(content_image)  # Hc,Wc,3
        mask = self._tensor_to_np(
            mask_image,
            gray=True,  # 0/255
            thr=threshold,
        )

        H, W = mask.shape

        # ① 生成 inner mask 仅用于 bbox 计算
        pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flags = 4 if connectivity == 4 else 8
        cv2.floodFill(pad, None, (0, 0), 0, flags=flags | (255 << 8))
        inner = pad[1:-1, 1:-1]

        ys, xs = np.where(inner == 255)
        if ys.size == 0:
            raise ValueError("mask 中央区域为空！")

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        target_w, target_h = x1 - x0 + 1, y1 - y0 + 1

        # ② 把 content 缩放/填充到和 frame 同尺寸的 canvas
        canvas = frame[:, :, :3].copy()  # 复制框的 RGB (不含 alpha)
        if keep_aspect:
            h0, w0 = content.shape[:2]
            scale = min(target_w / w0, target_h / h0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            resized = cv2.resize(content, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_off = y0 + (target_h - new_h) // 2
            x_off = x0 + (target_w - new_w) // 2
            canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        else:
            resized = cv2.resize(
                content, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            canvas[y0 : y1 + 1, x0 : x1 + 1] = resized

        # ③ 用 **原始 mask** 贴图
        result = frame[:, :, :3].copy()
        cv2.copyTo(canvas, mask, result)  # mask=原始 => 保留全部异形区域

        # ④ 恢复透明度（如果 frame 有 α）
        if frame.shape[2] == 4:
            alpha = frame[:, :, 3:]
            result = np.concatenate([result, alpha], axis=2)

        return (self._np_to_tensor(result),)

    # --------- tensor ↔ numpy 工具 --------- #
    def _tensor_to_np(self, img, gray=False, thr=0.2):
        if isinstance(img, list):
            img = img[0]
        if img.ndim == 3 and img.shape[0] <= 4:  # [C,H,W] → [H,W,C]
            img = img.permute(1, 2, 0)
        arr = img.cpu().numpy()
        if arr.dtype != np.bool_ and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype("uint8")
        if gray:
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            arr = (arr > int(thr * 255)).astype("uint8") * 255
        return arr

    def _np_to_tensor(self, arr):
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.from_numpy(arr.astype("float32") / 255.0).permute(2, 0, 1)
        return tensor.unsqueeze(0)
