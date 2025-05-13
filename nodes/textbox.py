import math, re, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from ..utils.converters import novel_pil2tensor


class TextBoxAutoWrap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "font_path": ("STRING",),
                "box_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "box_height": ("INT", {"default": 256, "min": 64, "max": 4096}),
                "font_size": ("INT", {"default": 48, "min": 4, "max": 512}),
                "letter_spacing": ("INT", {"default": 0, "min": -20, "max": 100}),
                "base_chars_per_line": ("INT", {"default": 10, "min": 1, "max": 100}),
                "max_lines": ("INT", {"default": 3, "min": 1, "max": 10}),
                "min_hscale": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 1.0}),
                "font_hex": (
                    "STRING",
                    {"default": "#FFFFFFFF"},  # RRGGBBAA
                ),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20}),
                "stroke_hex": (
                    "STRING",
                    {"default": "#000000FF"},  # 描边颜色
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "run"
    CATEGORY = "DrawTools"

    # ---------- 工具 ----------
    @staticmethod
    def line_height(font):
        a, d = font.getmetrics()
        return a + d

    @staticmethod
    def wrap_by_pixel(text, font, max_w, max_lines):
        lines, buf = [], ""
        for ch in text.replace("\n", ""):
            test = buf + ch
            if font.getlength(test) <= max_w:
                buf = test
            else:
                lines.append(buf)
                buf = ch
                if len(lines) == max_lines:
                    return lines  # 超出行数上限，截掉
        if buf and len(lines) < max_lines:
            lines.append(buf)
        return lines

    @staticmethod
    def wrap_by_pixel_with_spacing(text, font, max_w, max_lines, letter_spacing):
        """包含字间距的换行函数"""
        lines, buf = [], ""
        for ch in text.replace("\n", ""):
            test = buf + ch
            # 计算包含字间距的文本宽度
            width = (
                sum(font.getlength(c) for c in test) + (len(test) - 1) * letter_spacing
            )
            if width <= max_w:
                buf = test
            else:
                lines.append(buf)
                buf = ch
                if len(lines) == max_lines:
                    return lines  # 超出行数上限，截掉
        if buf and len(lines) < max_lines:
            lines.append(buf)
        return lines

    @staticmethod
    def draw_text_with_spacing_and_stroke(
        draw, pos, text, font, fill, letter_spacing, stroke_width, stroke_fill
    ):
        """带字间距和描边的文本绘制"""
        x, y = pos

        # 没有描边时，直接绘制
        if stroke_width <= 0:
            for char in text:
                draw.text((x, y), char, fill=fill, font=font)
                x += font.getlength(char) + letter_spacing
            return

        # 有描边时，先绘制描边
        for char in text:
            # 绘制描边（通过在多个位置绘制字符实现）
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:  # 跳过中心点，中心点稍后用实际颜色绘制
                        draw.text((x + dx, y + dy), char, fill=stroke_fill, font=font)

            # 绘制中心文字
            draw.text((x, y), char, fill=fill, font=font)
            x += font.getlength(char) + letter_spacing

    # ---------- 颜色解析 ----------
    @staticmethod
    def hex_to_rgba(hex_str: str):
        """#RRGGBB or #RRGGBBAA → (R,G,B,A) 0‑255"""
        hs = hex_str.strip().lstrip("#")
        if len(hs) == 6:
            hs += "FF"
        if len(hs) != 8:
            return (255, 255, 255, 255)
        try:
            r, g, b, a = (
                int(hs[0:2], 16),
                int(hs[2:4], 16),
                int(hs[4:6], 16),
                int(hs[6:8], 16),
            )
            return (r, g, b, a)
        except ValueError:
            return (255, 255, 255, 255)

    def run(
        self,
        text,
        font_path,
        box_width,
        box_height,
        font_size,
        letter_spacing,
        base_chars_per_line,
        max_lines,
        min_hscale,
        font_hex,
        stroke_width,
        stroke_hex,
    ):
        font_rgba = self.hex_to_rgba(font_hex)
        stroke_rgba = self.hex_to_rgba(stroke_hex)
        # 1) 计算行字符数、缩放比
        char_cnt = len(re.sub(r"\s+", "", text))
        chars_per = max(base_chars_per_line, math.ceil(char_cnt / max_lines))
        hscale = max(base_chars_per_line / chars_per, min_hscale)
        virt_w = int(round(box_width / hscale))  # 扩大的虚拟宽度

        # 2) 准备字体
        font = ImageFont.truetype(font_path, font_size)
        lh = self.line_height(font)

        # 3) 像素级换行 (在虚拟宽度内)
        lines = self.wrap_by_pixel_with_spacing(
            text, font, virt_w, max_lines, letter_spacing
        )

        # 4) 先画到虚拟画布
        img_v = Image.new("RGBA", (virt_w, box_height), (0, 0, 0, 0))
        msk_v = Image.new("L", (virt_w, box_height), 0)
        d_v = ImageDraw.Draw(img_v)
        m_v = ImageDraw.Draw(msk_v)

        y = 0
        for ln in lines:
            # 使用带描边的绘制方法
            self.draw_text_with_spacing_and_stroke(
                d_v,
                (0, y),
                ln,
                font,
                font_rgba,
                letter_spacing,
                stroke_width,
                stroke_rgba,
            )
            # 对于蒙版，我们可能不需要描边，但为了保持一致的外观，也应用描边
            self.draw_text_with_spacing_and_stroke(
                m_v, (0, y), ln, font, 255, letter_spacing, stroke_width, 255
            )
            y += lh

        # 5) 整幅图缩回目标宽度
        img = img_v.resize((box_width, box_height), resample=Image.LANCZOS)
        # 应用锐化滤镜提高清晰度
        img = img.filter(ImageFilter.SHARPEN)

        mask = msk_v.resize((box_width, box_height), resample=Image.NEAREST)
        # 蒙版也可以考虑锐化以保持边缘清晰
        mask = mask.filter(ImageFilter.SHARPEN)

        # 6) 转 tensor
        img_tensor = novel_pil2tensor(img)
        mask_rgb = torch.stack(
            [torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)], dim=0
        )

        return (img_tensor.unsqueeze(0), mask_rgb.unsqueeze(0))
