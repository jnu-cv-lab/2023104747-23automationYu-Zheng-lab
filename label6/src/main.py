import cv2
import numpy as np

# ------------------------------
# 任务1：ORB关键点检测
# ------------------------------
def detect_orb_features(image_path, output_path, nfeatures=1000):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    img_draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 小圆圈画关键点
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_draw, (x, y), 2, (0, 255, 0), 1)

    cv2.imwrite(output_path, img_draw)

    kp_count = len(keypoints)
    print(f"[任务1] {image_path} 关键点数量: {kp_count}")
    if descriptors is not None:
        desc_dim = descriptors.shape[1]
        print(f"[任务1] {image_path} 描述子维度: {desc_dim}")
    else:
        desc_dim = None
    return img, keypoints, descriptors, kp_count, desc_dim

# ------------------------------
# 任务2：ORB特征匹配
# ------------------------------
def orb_feature_matching(img1, kp1, des1, img2, kp2, des2, top_n=30):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:top_n]

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print(f"[任务2] 总匹配数量: {len(matches)}")
    print(f"[任务2] 可视化前 {top_n} 个最佳匹配")

    # 这里返回所有匹配，而不只是前50个
    return match_img, matches

# ------------------------------
# 任务3：RANSAC剔除错误匹配
# ------------------------------
def ransac_filter_matches(img1, kp1, img2, kp2, matches, reproj_thresh=5.0):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    mask = mask.ravel().tolist()
    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

    ransac_img = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    total_matches = len(matches)
    inlier_count = len(inlier_matches)
    inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0.0

    print(f"[任务3] Homography矩阵:\n{H}")
    print(f"[任务3] 总匹配数量: {total_matches}")
    print(f"[任务3] RANSAC内点数量: {inlier_count}")
    print(f"[任务3] 内点比例: {inlier_ratio:.4f}")

    return ransac_img, H, total_matches, inlier_count, inlier_ratio

# ------------------------------
# 任务4：目标定位
# ------------------------------
def locate_object(img1, img2, H):
    # 1. 获取box.png的四个角点
    h1, w1 = img1.shape[:2]
    corners = np.float32([
        [0, 0],         # 左上角
        [w1, 0],        # 右上角
        [w1, h1],       # 右下角
        [0, h1]         # 左下角
    ]).reshape(-1, 1, 2)  # 格式: (4, 1, 2)

    # 2. 使用cv2.perspectiveTransform()进行角点投影
    projected_corners = cv2.perspectiveTransform(corners, H)
    # 转为int32，方便画框
    projected_corners = np.int32(projected_corners)

    # 3. 使用cv2.polylines()在场景图中画出四边形边框
    img2_with_box = img2.copy()
    cv2.polylines(
        img2_with_box, 
        [projected_corners], 
        isClosed=True, 
        color=(0, 0, 255),  # 红色边框
        thickness=1
    )

    return img2_with_box, projected_corners

# ==============================================================================
# ====================== SIFT ==============
# ==============================================================================

def detect_sift_features(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    img_draw = img.copy()
    for point in kp:
        x, y = int(point.pt[0]), int(point.pt[1])
        cv2.circle(img_draw, (x, y), 2, (255, 0, 0), 1)
    cv2.imwrite(output_path, img_draw)

    print(f"[SIFT] {image_path} 关键点数量: {len(kp)}")
    print(f"[SIFT] {image_path} 描述子维度: {des.shape[1]}")
    return img, kp, des

def sift_knn_matching_with_ratio(img1, kp1, des1, img2, kp2, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)  # 题目要求：NORM_L2
    knn_matches = bf.knnMatch(des1, des2, k=2)  # KNN 匹配
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:  # Lowe ratio test
            good.append(m)
    print(f"[SIFT] KNN+Ratio 筛选后匹配数: {len(good)}")
    return good

# ==============================================================================
# ============================ 主程序 ==============================
# ==============================================================================
if __name__ == "__main__":
    img1_path = "/home/lenovo/cv-course/label6/src/box.png"
    img2_path = "/home/lenovo/cv-course/label6/src/box_in_scene.png"

    # 任务1
    img1, kp1, des1, kp_count1, dim1 = detect_orb_features(img1_path, "box_orb_small.png", nfeatures=1000)
    img2, kp2, des2, kp_count2, dim2 = detect_orb_features(img2_path, "box_in_scene_orb_small.png", nfeatures=1000)

    # 任务2
    match_img, all_matches = orb_feature_matching(img1, kp1, des1, img2, kp2, des2, top_n=50)
    cv2.imwrite("orb_matches.png", match_img)

    # 任务3
    ransac_img, H, total_matches, inlier_count, inlier_ratio = ransac_filter_matches(
        img1, kp1, img2, kp2, all_matches, reproj_thresh=5.0
    )
    cv2.imwrite("orb_ransac_matches.png", ransac_img)
    
    # 任务4调用
    box_in_scene_with_box, proj_corners = locate_object(img1, img2, H)
    cv2.imwrite("box_in_scene_with_box.png", box_in_scene_with_box)

    # 输出
    print("\n===== 最终结果 =====")
    print(f"box.png 关键点数量: {kp_count1}, 描述子维度: {dim1}")
    print(f"box_in_scene.png 关键点数量: {kp_count2}, 描述子维度: {dim2}")
    print(f"ORB总匹配数量: {total_matches}")
    print(f"RANSAC内点数量: {inlier_count}")
    print(f"内点比例: {inlier_ratio:.4f}")
    print("Homography矩阵:\n", H)
    print("\n[任务4] 目标定位完成，已保存结果图: box_in_scene_with_box.png")
    print(f"[任务4] 投影后的四个角点坐标:\n{proj_corners}")

    # ==========================================================================
    #                                    SIFT 
    # ==========================================================================
    print("\n" + "="*50)
    print("                    新增 SIFT 特征匹配任务                     ")
    print("="*50)

    # SIFT 检测
    img1_s, kp1_s, des1_s = detect_sift_features(img1_path, "box_sift.png")
    img2_s, kp2_s, des2_s = detect_sift_features(img2_path, "box_in_scene_sift.png")

    # SIFT KNN + Ratio Test 匹配
    sift_good_matches = sift_knn_matching_with_ratio(img1_s, kp1_s, des1_s, img2_s, kp2_s, des2_s)

    # SIFT 使用 RANSAC 过滤
    sift_ransac_img, H_sift, sift_total, sift_inlier, sift_ratio = ransac_filter_matches(
        img1_s, kp1_s, img2_s, kp2_s, sift_good_matches
    )
    cv2.imwrite("sift_ransac_result.png", sift_ransac_img)

    # SIFT 目标定位
    sift_result_img, _ = locate_object(img1_s, img2_s, H_sift)
    cv2.imwrite("sift_box_in_scene.png", sift_result_img)

    # 最终对比输出
    print("\n==================== ORB vs SIFT 对比 ====================")
    print(f"ORB 总匹配数: {total_matches}   内点: {inlier_count}   内点率: {inlier_ratio:.4f}")
    print(f"SIFT总匹配数: {sift_total}   内点: {sift_inlier}   内点率: {sift_ratio:.4f}")

    # 显示两张定位图
    cv2.imshow("ORB 目标定位", box_in_scene_with_box)
    cv2.imshow("SIFT 目标定位", sift_result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()