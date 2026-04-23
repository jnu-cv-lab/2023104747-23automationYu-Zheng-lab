import cv2
import numpy as np
import matplotlib.pyplot as plt



img_size = 500
test_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # 白色背景

# 画矩形
cv2.rectangle(test_img, (100,100), (400,400), (0,0,0), 2)
# 画圆
cv2.circle(test_img, (250,250), 100, (255,0,0), 2)
# 画平行线（水平2条）
cv2.line(test_img, (120,150), (380,150), (0,255,0), 2)
cv2.line(test_img, (120,200), (380,200), (0,255,0), 2)
# 画垂直线（垂直2条，和上面水平线垂直）
cv2.line(test_img, (150,120), (150,380), (0,0,255), 2)
cv2.line(test_img, (350,120), (350,380), (0,0,255), 2)

print("✅ 自制测试图完成：包含矩形、圆、平行线、垂直线")


h, w = test_img.shape[:2]
center = (w//2, h//2)


angle = 30
scale = 0.8
rad = np.radians(angle)
# 相似变换：旋转+缩放+平移
similar_mat = np.array([
    [scale*np.cos(rad), -scale*np.sin(rad), 50],  # x平移50
    [scale*np.sin(rad), scale*np.cos(rad), 30]   # y平移30
], dtype=np.float32)

# 仿射变换：剪切+缩放+平移
affine_mat = np.array([
    [0.8, 0.2, 40],   # 水平剪切
    [0.1, 0.7, 20]
], dtype=np.float32)

# 透视变换：更复杂的变换，保持直线但不保持平行和垂直
perspective_mat = np.array([
    [1.0, 0.2, 0],
    [0.1, 1.1, 0],
    [0.0012, 0.0008, 1.0]
], dtype=np.float32)


img_similar = cv2.warpAffine(test_img, similar_mat, (w,h))
img_affine  = cv2.warpAffine(test_img, affine_mat, (w,h))
img_perspective = cv2.warpPerspective(test_img, perspective_mat, (w,h))




import os

# 自动获取当前代码所在的文件夹（解决VSCode路径错误）
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "paper.jpg")

print("正在读取图片：", img_path)  # 你会看到正确路径

try:
    # 用正确路径读取图片，Linux/Windows/Mac 都通用
    src_img = cv2.imread(img_path)

    if src_img is None:
        print("\n❌ 读取失败！请确认：")
        print("1. 图片名字必须是：paper.jpg")
        print("2. 图片必须和这个 .py 文件在同一个文件夹")
        print("3. 后缀不是 .jpeg / .png，必须是 .jpg")
        exit()

    # 坐标修复（必须 4个点，每个点两个数字）
    src_points = np.float32([
        [423,598],
        [2423,651],
        [2934,3772],
        [110, 3887]
    ])
    dst_w, dst_h = 800, 1136  # B5标准比例 176:250
    dst_points = np.float32([[0,0], [dst_w,0], [dst_w,dst_h], [0,dst_h]])

    warp_mat = cv2.getPerspectiveTransform(src_points, dst_points)
    correct_img = cv2.warpPerspective(src_img, warp_mat, (dst_w, dst_h))

    save_path = os.path.join(current_dir, "paper_correct.jpg")
    cv2.imwrite(save_path, correct_img)

    print("\n✅ 成功！校正后的图片已保存！")

except Exception as e:
    print("\n⚠️ 错误信息：", e)



plt.figure(figsize=(16,9)) # 和你现在窗口大小一致

plt.subplot(2,3,1)
plt.imshow(test_img)
plt.title("原始测试图") 

plt.subplot(2,3,2)
plt.imshow(img_similar)
plt.title("相似变换") 

plt.subplot(2,3,3)
plt.imshow(img_affine)
plt.title("仿射变换") 

plt.subplot(2,3,4)
plt.imshow(img_perspective)
plt.title("透视变换") 

plt.tight_layout() 
plt.show()