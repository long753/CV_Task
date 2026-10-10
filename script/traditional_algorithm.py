from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_dir = next((p for p in (Path('images'), Path('../images')) if (p / 'coins_original.png').is_file()), None)
if image_dir is None:
    raise FileNotFoundError('找不到 images/coins_original.png；请从项目根目录或 documents/ 启动 JupyterLab')

gray = cv2.imread(str(image_dir / 'coins_original.png'), cv2.IMREAD_GRAYSCALE)
coffee_bgr = cv2.imread(str(image_dir / 'coffee_original.png'), cv2.IMREAD_COLOR)

if gray is None or coffee_bgr is None:
    raise FileNotFoundError('示例图片无法读取')
coffee_rgb = cv2.cvtColor(coffee_bgr, cv2.COLOR_BGR2RGB)

def show(images, titles, *, figsize=None):
    if len(images) != len(titles):
        raise ValueError('图片与标题数量不同')
    fig, axes = plt.subplots(1, len(images), figsize=figsize or (4 * len(images), 4), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(image)  # 彩色图像须先转换为 RGB
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

# print(f'OpenCV {cv2.__version__}; coins {gray.shape}; coffee {coffee_rgb.shape}')
# show([gray, coffee_rgb], ['Coins (grayscale)', 'Coffee (RGB)'])

coffee_gray = cv2.cvtColor(coffee_bgr, cv2.COLOR_BGR2GRAY)
brighter = cv2.convertScaleAbs(gray, alpha=1.20, beta=25)
darker = cv2.convertScaleAbs(gray, alpha=0.80, beta=0)
show([gray, brighter, darker, coffee_gray], ['Original', 'Brighter', 'Darker', 'Coffee grayscale'])