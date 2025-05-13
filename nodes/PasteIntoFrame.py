import cv2
import numpy as np
import torch
from PIL import Image
from ..utils.converters import novel_tensor2pil, novel_pil2tensor


class PasteIntoFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "patch": ("IMAGE",),  # 生成图
                "frame": ("IMAGE",),  # 原始卡片
                "mask": ("MASK",),  # 白=洞口，黑=框
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "DrawTools"

    def run(self, patch, frame, mask):
        results = []

        for p_t, f_t, m_t in zip(patch, frame, mask):
            # 1. Tensor → PIL RGBA / L
            pil_patch = novel_tensor2pil(p_t).convert("RGBA")
            pil_frame = novel_tensor2pil(f_t).convert("RGBA")
            pil_mask = novel_tensor2pil(m_t.squeeze(0)).convert("L")

            # 2. 连通域分析，找最大白色连通洞口
            np_mask = np.array(pil_mask)
            _, binary = cv2.threshold(np_mask, 127, 255, cv2.THRESH_BINARY)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            if num <= 1:
                results.append(novel_pil2tensor(pil_frame))
                continue

            areas = stats[1:, cv2.CC_STAT_AREA]
            lbl = int(np.argmax(areas)) + 1
            x0, y0 = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP]
            w, h = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
            x1, y1 = x0 + w, y0 + h

            # 3. 从 patch 居中裁切 (或替换成 resize 也行)
            pw, ph = pil_patch.size
            ox = max((pw - w) // 2, 0)
            oy = max((ph - h) // 2, 0)
            patch_crop = pil_patch.crop((ox, oy, ox + w, oy + h))

            # 4. 构造透明底图，把 patch 粘到洞口下层
            base = Image.new("RGBA", pil_frame.size, (0, 0, 0, 0))
            comp_mask = (labels == lbl).astype(np.uint8) * 255
            pil_comp = Image.fromarray(comp_mask[y0:y1, x0:x1]).convert("L")
            base.paste(patch_crop, (x0, y0), pil_comp)

            # 5. 用原始 pil_mask 制作 frame 的 alpha 通道
            #    洞口透明(0)，边框不透明(255)
            alpha_mask_full = pil_mask.point(lambda x: 255 - x)
            pil_frame.putalpha(alpha_mask_full)

            # 6. 最终合成：洞口保留 patch，其余地方由 frame 边框覆盖
            composite = Image.alpha_composite(base, pil_frame)

            # 7. 转回 Tensor
            results.append(novel_pil2tensor(composite))

        # batch 输出
        return (torch.stack(results, dim=0),)
