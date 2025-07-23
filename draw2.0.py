import math
import os
import cv2
import dlib
import numpy as np

# ---------------------------- 配置 -----------------------------
PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
IMAGE_PATH     = "22.jpg"
MAX_SIDE       = 10000          # 缩放阈值
# ---------------------------------------------------------------

# 1. 检验文件
for p in (PREDICTOR_PATH, IMAGE_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"❌ 找不到 {p}")

# 2. 读图 & 缩放
img = cv2.imread(IMAGE_PATH)
h0, w0 = img.shape[:2]
scale  = min(1.0, MAX_SIDE / max(h0, w0))
img_small = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

# 3. 人脸关键点检测
gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
faces = detector(gray_small.copy())
if not faces: raise RuntimeError("未检测到人脸")
face = faces[0]
shape = predictor(gray_small, face)
pts_small = np.array([[p.x, p.y] for p in shape.parts()])
pts = (pts_small / scale).astype(int)

# 4. 提取关键点 (这里会提取更多用于中线计算的点)
# 左右眉毛点
left_eyebrow_pts = pts[17:22]
right_eyebrow_pts = pts[22:27]

# 左右眼睛中点 (内眼角和外眼角连线的中点)
left_eye_center = (pts[36] + pts[39]) // 2
right_eye_center = (pts[42] + pts[45]) // 2

# 鼻梁点 (dlib 的鼻梁点 27-30)
nose_bridge_pts = pts[27:31]

# 嘴唇中点 (上唇珠和下唇中点)
pt_lip_top_center = pts[51] # 上唇珠
pt_lip_bottom_center = pts[57] # 下唇中点

left_corner = tuple(pts[48]) # 左嘴角 (保持不变，用于平行线)
right_corner = tuple(pts[54]) # 右嘴角 (保持不变，用于平行线)


# --- 新增：计算人脸中线 ---
# 收集所有用于计算中线的X坐标
midline_x_coords = []

# 眉毛中点X坐标
if len(left_eyebrow_pts) > 0 and len(right_eyebrow_pts) > 0:
    midline_x_coords.append(np.mean([p[0] for p in left_eyebrow_pts])) # 左眉毛的平均X
    midline_x_coords.append(np.mean([p[0] for p in right_eyebrow_pts])) # 右眉毛的平均X
    # 也可以只取眉毛整体的中心X：
    # midline_x_coords.append(np.mean([p[0] for p in left_eyebrow_pts + right_eyebrow_pts]))

# 眼睛中点X坐标
midline_x_coords.append((left_eye_center[0] + right_eye_center[0]) / 2)

# 鼻梁点X坐标的平均
if len(nose_bridge_pts) > 0:
    midline_x_coords.append(np.mean([p[0] for p in nose_bridge_pts]))

# 嘴唇中点X坐标
midline_x_coords.append((pt_lip_top_center[0] + pt_lip_bottom_center[0]) / 2)


if not midline_x_coords:
    raise RuntimeError("无法计算人脸中线，请检查关键点提取。")

# 计算最终的人脸中线X坐标平均值
avg_midline_x = int(np.mean(midline_x_coords))

# 确定人脸中线的 Y 范围
# 我们可以从眉毛上方延伸到下巴下方，以便画出足够长的中线
# 这里微调一下，确保中线包含所有相关点
midline_y_top = min(p[1] for p in (left_eyebrow_pts[0], right_eyebrow_pts[4])) - 50 # 眉毛最上方的点再往上一些
midline_y_bottom = pts[8][1] + 50 # 下巴最底部的点再往下一些 (pts[8]是下巴最低点)

# 定义人脸中线的两个点
pt_midline_top = (avg_midline_x, int(midline_y_top))
pt_midline_bottom = (avg_midline_x, int(midline_y_bottom))
# --- 人脸中线计算结束 ---


# 5. 在原图上画主线 (现在是人脸中线) & 平行线
out = img.copy()

# 主线延长并加粗 (现在使用新计算的 pt_midline_top 和 pt_midline_bottom)
def draw_extended(pt1, pt2, color, thickness):
    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
    norm = np.hypot(dx, dy)+1e-6
    ux, uy = dx/norm, dy/norm
    L = max(img.shape[:2])*2
    x1, y1 = int(pt1[0]-ux*L), int(pt1[1]-uy*L)
    x2, y2 = int(pt2[0]+ux*L), int(pt2[1]+uy*L)
    cv2.line(out, (x1,y1), (x2,y2), color, thickness)

# 绘制新的人脸中线，颜色保持洋红色 (255,0,255)
draw_extended(pt_midline_top, pt_midline_bottom, (255,0,255), 3)

# --- 新增：可视化用于计算中线的关键点 ---
# 绘制左右眉毛点 (蓝色)
for p in left_eyebrow_pts:
    cv2.circle(out, tuple(p), 4, (255,0,0), -1)
for p in right_eyebrow_pts:
    cv2.circle(out, tuple(p), 4, (255,0,0), -1)

# 绘制眼睛中点 (绿色)
cv2.circle(out, tuple(left_eye_center), 4, (0,255,0), -1)
cv2.circle(out, tuple(right_eye_center), 4, (0,255,0), -1)

# 绘制鼻梁点 (黄色)
for p in nose_bridge_pts:
    cv2.circle(out, tuple(p), 4, (0,255,255), -1)

# 绘制嘴唇中点 (红色)
cv2.circle(out, tuple(pt_lip_top_center), 4, (0,0,255), -1)
cv2.circle(out, tuple(pt_lip_bottom_center), 4, (0,0,255), -1)

# 绘制计算出的中线两个端点 (白色)
cv2.circle(out, pt_midline_top, 6, (255,255,255), -1)
cv2.circle(out, pt_midline_bottom, 6, (255,255,255), -1)

# --- 可视化点：新中线和相关关键点 ---
cv2.imshow("Midline and Keypoints", out.copy()) # 使用 out.copy() 以免影响后续绘制
cv2.waitKey(0)
# --- 可视化结束 ---


# 平行线函数需要返回方向向量 (保持不变)
def draw_parallel(pt1, pt2, through, color, thickness):
    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
    norm = np.hypot(dx, dy)+1e-6
    ux, uy = dx/norm, dy/norm
    L = max(img.shape[:2])*2
    x1, y1 = int(through[0]-ux*L), int(through[1]-uy*L)
    x2, y2 = int(through[0]+ux*L), int(through[1]+uy*L)
    cv2.line(out, (x1,y1), (x2,y2), color, thickness)
    return (ux, uy)

ux, uy = None, None
# 重新计算 ux, uy 基于新中线 (保持不变)
midline_vec_y = (pt_midline_bottom[1] - pt_midline_top[1])
midline_vec_x = (pt_midline_bottom[0] - pt_midline_top[0])

norm_midline = np.hypot(midline_vec_x, midline_vec_y) + 1e-6
ux, uy = midline_vec_x / norm_midline, midline_vec_y / norm_midline

# 左嘴角平行线
ux, uy = draw_parallel(pt_midline_top, pt_midline_bottom, left_corner,  (0,255,255), 2)
# 右嘴角平行线（ux,uy 相同） (保持不变)
draw_parallel(pt_midline_top, pt_midline_bottom, right_corner, (0,255,255), 2)

# 6. 检测托盘：简单 BGR 白色阈值
#    假定托盘主要是纯白，BGR 三通道都 >200
mask_white = cv2.inRange(img, np.array([200,200,200]), np.array([255,255,255]))
# 形态学去噪
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)
# 找最大连通域，认为是托盘
contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours: raise RuntimeError("未检测到白色托盘")
tray = max(contours, key=cv2.contourArea)
# 托盘的上边缘 y_tray
y_tray = min(pt[1] for pt in tray.reshape(-1,2))

# 7. 画第4条水平线
cv2.line(out, (0,y_tray), (img.shape[1], y_tray), (0,255,0), 2)

# 8. 计算交点：solving (x,y_tray) lies on parallel lines through left/right corners
ints = []
for through in (left_corner, right_corner):
    # 求 t 使 y = through.y + uy*t = y_tray
    t = (y_tray - through[1]) / (uy + 1e-6)
    x_int = int(through[0] + ux * t)
    ints.append((x_int, y_tray))
    # 标出交点
    cv2.circle(out, (x_int, y_tray), 6, (0,0,255), -1)

# 假设你已有：
#   img        – 原图
#   out        – 绘好了前四条线和两个交点的图像
#   pts        – 68 个 landmark (原图坐标) ndarray
#   ints       – 两个交点列表 [(xL,y_tray),(xR,y_tray)]
#   pt_lip     – 唇珠中点

import numpy as np
import cv2

# --- 开始：脸颊轮廓提取新方法 ---

# 假设原图 img 未缩放，我们直接对原图进行处理，因为 dlib 的关键点是原图坐标
img_orig_copy = img.copy()  # 用于轮廓检测的图像副本

# 1. 肤色检测 (HSV 空间)
# 根据经验值，以下是常见肤色在 HSV 空间的大致范围
# 您可能需要根据实际图片效果进行微调
hsv = cv2.cvtColor(img_orig_copy, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0, 52, 70], dtype="uint8")  # 调低V值以适应较暗肤色
upper_skin = np.array([20, 255, 225], dtype="uint8")
skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

# --- 添加可视化点 2：HSV 肤色掩膜 ---
cv2.imshow("Skin Mask (HSV)", skin_mask)
cv2.waitKey(0)
# --- 可视化点 2 结束 ---

# 2. 传统方法辅助检测 (YCbCr 空间)
# Cb 和 Cr 分量在人脸区域通常有稳定的范围
ycbcr = cv2.cvtColor(img_orig_copy, cv2.COLOR_BGR2YCrCb)
lower_skin_ycbcr = np.array([0, 140, 77], dtype="uint8")
upper_skin_ycbcr = np.array([230, 173, 127], dtype="uint8")
skin_mask_ycbcr = cv2.inRange(ycbcr, lower_skin_ycbcr, upper_skin_ycbcr)

# --- 添加可视化点 3：YCbCr 肤色掩膜 ---
cv2.imshow("Skin Mask (YCbCr)", skin_mask_ycbcr)
cv2.waitKey(0)
# --- 可视化点 3 结束 ---

# 3. 结合两种掩膜（可以根据实际效果选择逻辑或或逻辑与）
# 这里我们选择逻辑与，确保检测结果更严格
final_skin_mask = cv2.bitwise_and(skin_mask, skin_mask_ycbcr)

# --- 添加可视化点 4：结合后的肤色掩膜 ---
cv2.imshow("Combined Skin Mask", final_skin_mask)
cv2.waitKey(0)
# --- 可视化点 4 结束 ---

# 4. 形态学操作：去除噪声，填充小孔，连接断开区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 椭圆形内核
final_skin_mask = cv2.erode(final_skin_mask, kernel, iterations=2)  # 腐蚀
final_skin_mask = cv2.dilate(final_skin_mask, kernel, iterations=3)  # 膨胀

# --- 添加可视化点 5：形态学操作后的掩膜 ---
cv2.imshow("Mask After Morphology", final_skin_mask)
cv2.waitKey(0)
# --- 可视化点 5 结束 ---


# 5. 寻找最大连通域（人脸）
contours, _ = cv2.findContours(final_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

face_contour_pts = []
if contours:
    # 找到最大轮廓，假设它是人脸
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似多边形，减少点数并平滑轮廓，epsilon 值需要根据图片调整
    # 这里的 0.001 * arcLength 是一个经验值，您可以调整它以获得更平滑或更细节的轮廓
    epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 提取近似轮廓的顶点
    face_contour_pts = [tuple(p[0]) for p in approx_contour]

if not face_contour_pts:
    raise RuntimeError("未通过传统方法检测到人脸轮廓，请检查肤色范围或图片。")

# 为了与旧代码兼容，我们将整个近似轮廓作为 'jaw_pts'
# 并根据 x 坐标划分左右脸颊（这里仅作概念上的划分，实际使用时需要调整索引）
# 这里我们假设 approx_contour 包含了从下巴一侧到另一侧的完整轮廓。
# 对于切线计算，我们将使用 face_contour_pts 作为整个脸颊的轮廓点。
jaw_pts = face_contour_pts

# 可视化新的轮廓线（绿色）
cv2.polylines(out, [np.array(jaw_pts, dtype=np.int32)], True, (0, 255, 0), 3)

# --- 添加可视化点 6：绘制在图片上的最终脸颊轮廓 ---
cv2.imshow("Face Contour on Image", out) # out 是绘制了轮廓的图像
cv2.waitKey(0)
# --- 可视化点 6 结束 ---

# 由于我们现在有了完整的脸颊轮廓，dlib 的 landmark 划分不再适用。
# 我们需要为切线检测定义新的左右索引，或者直接遍历整个轮廓点。
# 考虑到您原始代码的逻辑，我们仍然需要区分左右。
# 简单的处理方式是找到唇珠的X坐标，然后将 jaw_pts 分为左右两部分。
# 更好的方法可能是直接在 `is_supporting_line` 中使用 `face_contour_pts`。
# 但为了保持原有的 `left_idx` 和 `right_idx` 结构，我们在这里进行一个近似划分。

# 找到最左和最右的 landmark，用于大致界定左右脸颊轮廓的范围
leftmost_dlib_x = min(pts[i][0] for i in range(0, 8))  # 下巴左侧的dlib点
rightmost_dlib_x = max(pts[i][0] for i in range(8, 17))  # 下巴右侧的dlib点

# 根据唇珠的x坐标或整个脸部中心来划分左右脸颊点
# 假设唇珠是脸部中心的良好指示器
center_x = avg_midline_x

# 将新的 `jaw_pts` （完整脸颊轮廓点）划分为左右两部分
left_face_pts = [p for p in jaw_pts if p[0] < center_x]
right_face_pts = [p for p in jaw_pts if p[0] >= center_x]

# 注意：`left_idx` 和 `right_idx` 在这种新方法下不再是 dlib 的特定索引，
# 它们现在代表的是 `jaw_pts` 列表中左右脸颊部分的索引。
# 简单起见，我们直接将左右脸颊的完整点列表作为新的 `jaw_pts`，
# 然后在切线计算时，根据交点 P 的位置选择对应的点集。
# 这意味着 `left_idx` 和 `right_idx` 不再是整数索引，而是直接指向了点集。
# 为了兼容 `is_supporting_line` 和后续的 `tangents` 逻辑，
# 我们可以创建一个新的 `all_jaw_pts` 列表，并根据交点 P 的 X 坐标来判断使用哪个子集。

# 更新 jaw_pts 为我们通过传统方法获得的脸颊轮廓点
jaw_pts = face_contour_pts  # 完整的脸颊轮廓点列表

# 可视化新的轮廓线（绿色）
# 由于现在 jaw_pts 是一个近似多边形，我们直接画出它
cv2.polylines(out, [np.array(jaw_pts, dtype=np.int32)], True, (0, 255, 0), 3)  # True 表示闭合轮廓

# --- 结束：脸颊轮廓提取新方法 ---
THICK = 2
LARGE = max(img.shape[:2]) * 2
TAN_COLOR = (0,0,255)

def is_supporting_line(P, Q, contour):
    """
    判断直线 PQ 是否是对 contour 的支持线：
    对 contour 上的所有点 R，(R-P)x(Q-P) 要么全 >=0，要么全 <=0
    """
    v = np.array(Q) - np.array(P)
    signs = []
    for R in contour:
        w = np.array(R) - np.array(P)
        cross = v[0]*w[1] - v[1]*w[0]
        signs.append(cross)
    signs = np.array(signs)
    return np.all(signs >= -1e-6) or np.all(signs <= 1e-6)


# --- 在 for P in ints: 循环外部，添加脸颊垂直范围的定义 ---
# 获取下巴最低点的Y坐标，或其附近点
# pts[8] 是下巴最底部的点，我们希望切点在它上方或附近，避免切到脖子
# 我们可以考虑 dlib 脸颊上的点 (例如 pts[4] 和 pts[12]) 的 Y 坐标作为上限
# 这里设定一个相对灵活的范围，可以根据实际情况调整
jawline_bottom_y = pts[8][1]- 50 # 下巴最低点的 Y 坐标
cheek_upper_y_left = pts[4][1] # 左脸颊Dlib点
cheek_upper_y_right = pts[12][1] # 右脸颊Dlib点

# 为了统一，我们可以取左右脸颊Dlib点中较高的那个作为脸颊的有效上边界
# 或者设定一个基于唇珠Y坐标的相对值，以适应不同姿态
cheek_upper_bound_y = min(pts[3][1], pts[4][1], pts[5][1], pts[11][1], pts[12][1], pts[13][1])-50
pt_nose = tuple(pts[27]) # 鼻峰
# 确保这个上限不会高于鼻尖
cheek_upper_bound_y = max(cheek_upper_bound_y, pt_nose[1]) # 避免过高

# 临时可视化代码，放在绘制绿色轮廓之后
cv2.line(out, (0, jawline_bottom_y), (img.shape[1], jawline_bottom_y), (0, 0, 0), 2) # 黑色线表示下边界
cv2.line(out, (0, cheek_upper_bound_y), (img.shape[1], cheek_upper_bound_y), (0, 0, 0), 2) # 黑色线表示上边界
cv2.imshow("Face Contour with Bounds", out)
cv2.waitKey(0)

# --- 以下代码在 for P in ints: 循环内部 ---
for side, P in enumerate(ints):
    tangents = []

    # 定义当前交点 P 对应的脸颊上边界
    current_cheek_upper_y = cheek_upper_bound_y # 统一使用一个上边界

    is_left_side_P = (P[0] < avg_midline_x)

    # 遍历完整的脸颊轮廓点 jaw_pts 来寻找支持线
    for Q_candidate in jaw_pts:
        # 增加对候选点 Y 坐标的更严格限制
        # Q_candidate 必须在交点 P 的下方，但也要在合理脸颊区域内 (高于下巴最低点，低于脸颊上边界)
        if Q_candidate[1] > P[1] and Q_candidate[1] < jawline_bottom_y and Q_candidate[1] > current_cheek_upper_y:
            if is_left_side_P:
                # 对于左侧交点 P，我们期望切点 Q_candidate 在 P 的左下方，且在脸颊区域内
                if Q_candidate[0] < P[0]: # 确保在左侧
                    if is_supporting_line(P, Q_candidate, jaw_pts):
                        tangents.append((Q_candidate, Q_candidate))
            else:
                # 对于右侧交点 P，我们期望切点 Q_candidate 在 P 的右下方，且在脸颊区域内
                if Q_candidate[0] > P[0]: # 确保在右侧
                    if is_supporting_line(P, Q_candidate, jaw_pts):
                        tangents.append((Q_candidate, Q_candidate))

    if not tangents:
        print(f"⚠️ 警告：对于交点 {P} 未能找到合适的脸颊支持线。请检查轮廓点筛选逻辑或图片。")
        # 再次尝试放宽条件，但仍然限制在下巴最低点上方
        for Q_full_contour in jaw_pts:
            if Q_full_contour[1] < jawline_bottom_y and Q_full_contour[1] > current_cheek_upper_y: # 仍限制垂直范围
                 if is_left_side_P:
                    if Q_full_contour[0] < P[0]:
                        if is_supporting_line(P, Q_full_contour, jaw_pts):
                            tangents.append((Q_full_contour, Q_full_contour))
                 else:
                    if Q_full_contour[0] > P[0]:
                        if is_supporting_line(P, Q_full_contour, jaw_pts):
                            tangents.append((Q_full_contour, Q_full_contour))

        if not tangents:
            raise RuntimeError(f"仍然找不到支持线切点 for P={P}，即使放宽筛选。")


    # 若有多个，挑离 P 最近的那个
    # 这里存储的是 (点, 点) 的元组，所以 key 是 x[0]
    j_sel_tuple, Q_sel = min(tangents, key=lambda x: np.hypot(x[1][0] - P[0], x[1][1] - P[1]))

    # 画从交点 P 出发的切线（双向延长）
    v = np.array(Q_sel) - np.array(P)
    v = v / (np.linalg.norm(v) + 1e-6)
    pt1 = (int(P[0] - v[0] * LARGE), int(P[1] - v[1] * LARGE))
    pt2 = (int(P[0] + v[0] * LARGE), int(P[1] + v[1] * LARGE))
    cv2.line(out, pt1, pt2, TAN_COLOR, THICK)
    # 标记切点
    cv2.circle(out, Q_sel, 12, TAN_COLOR, -1)


# ---------------------------------------------------------

# 11. 计算平行方向向量 (现在基于新的人脸中线)
dx, dy = pt_midline_bottom[0]-pt_midline_top[0], pt_midline_bottom[1]-pt_midline_top[1]
para_vec = np.array([dx, dy]) / (np.hypot(dx, dy)+1e-6)

# 参数区
ARC_COLOR    = (0, 0, 255)      # 红色
ARC_THICK    = 4
FONT         = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE   = 1.2
TEXT_THICK   = 4
RADIUS       = 80
TEXT_DIST    = 120
BG_COLOR     = (255, 255, 255)  # 文字背景白色
BG_ALPHA     = 0.6              # 半透明度

def draw_angle(img, center, vec1, vec2, color):
    """在 img 上以 center 为圆心，画出 vec1->vec2 间的锐角弧和标注角度。"""
    # 1. 计算锐角
    cosang = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle  = math.degrees(math.acos(cosang))  # 0-180
    if angle > 90:                           # 强制锐角
        angle = 180 - angle

    # 2. 起止角度（以 vec1 为 0 起点，atan2 返回 [-180,180]）
    a1 = math.degrees(math.atan2(vec1[1], vec1[0])) % 360
    a2 = math.degrees(math.atan2(vec2[1], vec2[0])) % 360
    diff = (a2 - a1) % 360
    if diff > 180:
        diff -= 360
    # 保证走最短方向
    if abs(diff) > 180 - abs(diff):
        diff = -np.sign(diff)*(360-abs(diff))
    # 再次保证锐角
    if abs(diff) > 90:
        diff = -np.sign(diff)*(180-abs(diff))
    start = a1
    end   = (a1 + diff) % 360

    # 3. 画弧
    cv2.ellipse(img, tuple(center), (RADIUS, RADIUS),
                0, float(start), float(end), color, ARC_THICK)

    # 4. 文字
    txt = f"{angle:.3f}°"
    # 文本尺寸
    (w, h), _ = cv2.getTextSize(txt, FONT, TEXT_SCALE, TEXT_THICK)
    # 文本背景位置
    mid_ang = math.radians(start + diff/2)
    bx = int(center[0] + math.cos(mid_ang)*(RADIUS + TEXT_DIST)) - w//2
    by = int(center[1] + math.sin(mid_ang)*(RADIUS + TEXT_DIST)) + h//2

    # 绘制半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (bx-5, by-h-5), (bx+w+5, by+5), BG_COLOR, -1)
    cv2.addWeighted(overlay, BG_ALPHA, img, 1-BG_ALPHA, 0, img)

    # 绘制文字（前景）
    cv2.putText(img, txt, (bx, by), FONT, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)

    return angle


# 13. 对左右交点分别计算
for side, P in enumerate(ints):
    tangents = []

    # 定义当前交点 P 对应的脸颊上边界 (这部分是之前调整的，保持不变)
    current_cheek_upper_y = cheek_upper_bound_y # 统一使用一个上边界

    is_left_side_P = (P[0] < avg_midline_x)

    # 遍历完整的脸颊轮廓点 jaw_pts 来寻找支持线 (这部分是之前调整的，保持不变)
    for Q_candidate in jaw_pts:
        if Q_candidate[1] > P[1] and Q_candidate[1] < jawline_bottom_y and Q_candidate[1] > current_cheek_upper_y:
            if is_left_side_P:
                if Q_candidate[0] < P[0]:
                    if is_supporting_line(P, Q_candidate, jaw_pts):
                        tangents.append((Q_candidate, Q_candidate))
            else:
                if Q_candidate[0] > P[0]:
                    if is_supporting_line(P, Q_candidate, jaw_pts):
                        tangents.append((Q_candidate, Q_candidate))

    if not tangents:
        print(f"⚠️ 警告：对于交点 {P} 未能找到合适的脸颊支持线。请检查轮廓点筛选逻辑或图片。")
        # 再次尝试放宽条件 (保持不变)
        for Q_full_contour in jaw_pts:
            if Q_full_contour[1] < jawline_bottom_y and Q_full_contour[1] > current_cheek_upper_y:
                 if is_left_side_P:
                    if Q_full_contour[0] < P[0]:
                        if is_supporting_line(P, Q_full_contour, jaw_pts):
                            tangents.append((Q_full_contour, Q_full_contour))
                 else:
                    if Q_full_contour[0] > P[0]:
                        if is_supporting_line(P, Q_full_contour, jaw_pts):
                            tangents.append((Q_full_contour, Q_full_contour))

        if not tangents:
            raise RuntimeError(f"仍然找不到支持线切点 for P={P}，即使放宽筛选。")

    # 若有多个支持线，挑离 P 最近的那个作为切点 (保持不变)
    _, Q = min(tangents, key=lambda x: np.hypot(x[1][0] - P[0], x[1][1] - P[1]))

    # 计算切线方向向量 (保持不变)
    tan_vec = (np.array(Q) - np.array(P)) / (np.linalg.norm(np.array(Q) - np.array(P)) + 1e-6)

    # 绘制切线 (保持不变)
    v = np.array(Q) - np.array(P)
    v = v / (np.linalg.norm(v) + 1e-6)
    pt1 = (int(P[0] - v[0] * LARGE), int(P[1] - v[1] * LARGE))
    pt2 = (int(P[0] + v[0] * LARGE), int(P[1] + v[1] * LARGE))
    cv2.line(out, pt1, pt2, TAN_COLOR, THICK)
    cv2.circle(out, Q, 12, TAN_COLOR, -1)

    # --- 添加或修改这一行以绘制夹角 ---
    # draw_angle 函数需要：图像，中心点（交点P），向量1（平行线方向），向量2（切线方向），颜色
    draw_angle(out, P, para_vec, tan_vec, ARC_COLOR)
    # --- 角度可视化结束 ---
# 14. 显示结果
cv2.imshow("Sharp Angles", out)
cv2.waitKey(0)
cv2.destroyAllWindows()