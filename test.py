import cv2
import numpy as np

# 1. 读取并二值化
img = cv2.imread("ComfyUI_temp_padxu_00001_.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

# 2. 连通域标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

# 3. 遍历每个连通域（从 1 开始跳过背景 0）
for label in range(1, num_labels):
    # 提取这个连通域的掩码
    comp_mask = (labels == label).astype(np.uint8) * 255

    # 3a. 如果你想把它单独保存或显示
    cv2.imwrite(f"component_{label}.png", comp_mask)

    # 3b. 如果你想用轮廓来可视化真实的边界
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)  # 绿色描边
    cv2.imwrite(f"contour_{label}.png", out)

    print(f"Label={label}, 面积={stats[label, cv2.CC_STAT_AREA]}")
