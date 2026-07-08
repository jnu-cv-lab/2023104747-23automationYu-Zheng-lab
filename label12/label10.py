import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ====================== 1. 标定参数配置（根据你的棋盘格修改） ======================
chessboard_size = (9, 6)
square_size = 25
# 关键修改：添加通配符匹配图片文件
img_path = "/home/favian/work14/image-zy/*.jpg"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ====================== 2. 生成棋盘格三维世界坐标 ======================
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

obj_points = []
img_points = []
img_list = glob.glob(img_path)
print(f"一共读取 {len(img_list)} 张标定图片")

# ====================== 3. 遍历图片检测角点+亚像素优化 ======================
for idx, img_file in enumerate(img_list):
    img = cv2.imread(img_file)
    # 新增空值校验
    if img is None:
        print(f"警告：无法读取图片 {img_file}，跳过该文件")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_sub)
        cv2.drawChessboardCorners(img, chessboard_size, corners_sub, ret)
        cv2.imwrite(f"./corner_detect_{idx}.jpg", img)
        print(f"图片{idx+1} 角点检测成功")
    else:
        print(f"图片{idx+1} 角点检测失败，丢弃该图")

# ====================== 4. 相机标定，求解内参、畸变、外参、重投影误差 ======================
ret_err, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# ====================== 5. 输出标定结果 ======================
print("=" * 50)
print(f"平均重投影误差: {ret_err:.4f} 像素")
print("=" * 50)
print("相机内参矩阵 K：")
print(K)
print("=" * 50)
print("畸变系数 D = [k1, k2, p1, p2, k3]：")
print(D.ravel())
print("=" * 50)

# 提取内参分量用于报告分析
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
img_w, img_h = gray.shape[::-1]
img_center_x = img_w / 2
img_center_y = img_h / 2
print(f"fx={fx:.2f}, fy={fy:.2f}")
print(f"cx={cx:.2f}, cy={cy:.2f}, 图像中心({img_center_x}, {img_center_y})")

# ====================== 6. 图像去畸变处理 ======================
# 取第一张有效图片做去畸变演示
demo_img = cv2.imread(img_list[0])
h, w = demo_img.shape[:2]
# 计算优化内参，去除黑边/保留原图两种模式可选
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
# 去畸变
undist_img = cv2.undistort(demo_img, K, D, None, new_K)
# 裁剪黑边（可选）
x, y, w_roi, h_roi = roi
undist_crop = undist_img[y:y + h_roi, x:x + w_roi]

# 保存原图、去畸变图
cv2.imwrite("original_demo.jpg", demo_img)
cv2.imwrite("undistort_demo.jpg", undist_img)
cv2.imwrite("undistort_crop.jpg", undist_crop)

# ====================== 7. 绘图对比原图&去畸变图 ======================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Undistorted Image")
plt.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
#plt.show()