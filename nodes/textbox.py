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
                "font_hex": ("STRING", {"default": "#FFFFFFFF"}),  # RRGGBBAA
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20}),
                "stroke_hex": ("STRING", {"default": "#000000FF"}),  # 描边颜色
                "enable_small_caps": ("BOOLEAN", {"default": False}),
                "small_caps_scale": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 1.0}),
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
    def wrap_by_pixel_with_spacing(
        text,
        font,
        max_w,
        max_lines,
        letter_spacing,
        enable_small_caps=False,
        small_font=None,
    ):
        """逐像素宽度换行：数字→小字号，其余→大字号"""
        lines, buf = [], ""
        for ch in text.replace("\n", ""):
            test = buf + ch

            width = 0
            for c in test:
                is_digit = c.isdigit()
                use_font = (
                    small_font
                    if (enable_small_caps and is_digit and small_font)
                    else font
                )
                width += use_font.getlength(c)
            width += max(0, len(test) - 1) * letter_spacing

            if width <= max_w:
                buf = test
            else:
                lines.append(buf)
                buf = ch
                if len(lines) == max_lines:
                    return lines
        if buf and len(lines) < max_lines:
            lines.append(buf)
        return lines

    # ---------- 绘制 ----------
    @staticmethod
    def draw_text_with_spacing_and_stroke(
        draw,
        pos,
        text,
        font,
        fill,
        letter_spacing,
        stroke_width,
        stroke_fill,
        enable_small_caps=False,
        small_font=None,
    ):
        """按字符绘制：数字小字号，其余大字号；保持 baseline 对齐"""
        x, y0 = pos
        big_a, _ = font.getmetrics()
        sml_a, _ = small_font.getmetrics() if small_font else font.getmetrics()
        y_offset = max(0, big_a - sml_a)

        for char in text:
            is_digit = char.isdigit()
            use_small = enable_small_caps and is_digit and small_font is not None
            cur_font = small_font if use_small else font
            y = y0 + (y_offset if use_small else 0)

            # ---- 描边 ----
            if stroke_width > 0:
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx or dy:
                            draw.text(
                                (x + dx, y + dy), char, fill=stroke_fill, font=cur_font
                            )
            # ---- 主文字 ----
            draw.text((x, y), char, fill=fill, font=cur_font)
            x += cur_font.getlength(char) + letter_spacing

    # ---------- 颜色解析 ----------
    @staticmethod
    def hex_to_rgba(hex_str: str):
        hs = hex_str.strip().lstrip("#")
        if len(hs) == 6:
            hs += "FF"
        if len(hs) != 8:
            return (255, 255, 255, 255)
        try:
            return tuple(int(hs[i : i + 2], 16) for i in (0, 2, 4, 6))
        except ValueError:
            return (255, 255, 255, 255)

    # ---------- 运行 ----------
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
        enable_small_caps,
        small_caps_scale,
    ):
        font_rgba = self.hex_to_rgba(font_hex)
        stroke_rgba = self.hex_to_rgba(stroke_hex)

        font = ImageFont.truetype(font_path, font_size)
        small_font = (
            ImageFont.truetype(font_path, max(1, int(font_size * small_caps_scale)))
            if enable_small_caps
            else None
        )

        # 缩放比
        char_cnt = len(re.sub(r"\s+", "", text))
        chars_per = max(base_chars_per_line, math.ceil(char_cnt / max_lines))
        hscale = max(base_chars_per_line / chars_per, min_hscale)
        virt_w = int(round(box_width / hscale))

        # 换行
        lines = self.wrap_by_pixel_with_spacing(
            text,
            font,
            virt_w,
            max_lines,
            letter_spacing,
            enable_small_caps,
            small_font,
        )

        lh = max(self.line_height(font), self.line_height(small_font or font))

        # 虚拟画布
        img_v = Image.new("RGBA", (virt_w, box_height), (0, 0, 0, 0))
        msk_v = Image.new("L", (virt_w, box_height), 0)
        d_v = ImageDraw.Draw(img_v)
        m_v = ImageDraw.Draw(msk_v)

        y = 0
        for ln in lines:
            # 正片
            self.draw_text_with_spacing_and_stroke(
                d_v,
                (0, y),
                ln,
                font,
                font_rgba,
                letter_spacing,
                stroke_width,
                stroke_rgba,
                enable_small_caps,
                small_font,
            )
            # 蒙版
            self.draw_text_with_spacing_and_stroke(
                m_v,
                (0, y),
                ln,
                font,
                255,
                letter_spacing,
                stroke_width,
                255,
                enable_small_caps,
                small_font,
            )
            y += lh

        # 缩放 & 锐化
        img = img_v.resize((box_width, box_height), resample=Image.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
        mask = msk_v.resize((box_width, box_height), resample=Image.NEAREST)
        mask = mask.filter(ImageFilter.SHARPEN)

        img_tensor = novel_pil2tensor(img)
        mask_rgb = torch.stack(
            [torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)], dim=0
        )

        return (img_tensor.unsqueeze(0), mask_rgb.unsqueeze(0))
