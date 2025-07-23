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

# 4. 提取关键点
pt_nose = tuple(pts[27])                           # 鼻峰
pt_lip = tuple(((pts[51])))        # 唇珠
left_corner = tuple(pts[48])                           # 左嘴角
right_corner = tuple(pts[54])                           # 右嘴角

# 5. 在原图上画主线 & 平行线
out = img.copy()

# 主线延长并加粗
def draw_extended(pt1, pt2, color, thickness):
    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
    norm = np.hypot(dx, dy)+1e-6
    ux, uy = dx/norm, dy/norm
    L = max(img.shape[:2])*2
    x1, y1 = int(pt1[0]-ux*L), int(pt1[1]-uy*L)
    x2, y2 = int(pt2[0]+ux*L), int(pt2[1]+uy*L)
    cv2.line(out, (x1,y1), (x2,y2), color, thickness)
draw_extended(pt_nose, pt_lip, (255,0,255), 3)

# 平行线函数需要返回方向向量
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
# 左嘴角平行线
ux, uy = draw_parallel(pt_nose, pt_lip, left_corner,  (0,255,255), 2)
# 右嘴角平行线（ux,uy 相同）
draw_parallel(pt_nose, pt_lip, right_corner, (0,255,255), 2)


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

# 9. 可视化：画关键点
for pt,color in [(pt_nose,(255,0,0)), (pt_lip,(0,255,0)), (left_corner,(255,255,0)), (right_corner,(255,255,0))]:
    cv2.circle(out, pt, 4, color, -1)

# 假设你已有：
#   img        – 原图
#   out        – 绘好了前四条线和两个交点的图像
#   pts        – 68 个 landmark (原图坐标) ndarray
#   ints       – 两个交点列表 [(xL,y_tray),(xR,y_tray)]
#   pt_lip     – 唇珠中点

import numpy as np
import cv2

# 只保留左/右脸颊的 3-7 和 9-13
jaw_pts = [tuple(pts[i]) for i in [3, 4, 5, 6, 7, 9, 10, 11, 12, 13]]
left_idx  = list(range(0, 5))   # 0-4
right_idx = list(range(5, 10))  # 5-9

# 轮廓线
cv2.polylines(out, [np.array([jaw_pts[:5]], dtype=np.int32)], False, (0,255,0), 3)   # 左绿
cv2.polylines(out, [np.array([jaw_pts[5:]], dtype=np.int32)], False, (255,0,0), 3)   # 右蓝

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

for P in ints:
    if P[0] < pt_lip[0]:
        idxs = left_idx
    else:
        idxs = right_idx

    tangents = []
    # 遍历候选点
    for j in idxs:
        Q = jaw_pts[j]
        if is_supporting_line(P, Q, jaw_pts):
            tangents.append((j, Q))
    if not tangents:
        raise RuntimeError(f"找不到支持线切点 for P={P}")

    # 若有多个，挑离 P 最近的那个
    j_sel, Q_sel = min(tangents, key=lambda x: np.hypot(x[1][0]-P[0], x[1][1]-P[1]))

    # 画从交点 P 出发的切线（双向延长）
    v = np.array(Q_sel) - np.array(P)
    v = v / (np.linalg.norm(v) + 1e-6)
    pt1 = (int(P[0] - v[0]*LARGE), int(P[1] - v[1]*LARGE))
    pt2 = (int(P[0] + v[0]*LARGE), int(P[1] + v[1]*LARGE))
    cv2.line(out, pt1, pt2, TAN_COLOR, THICK)
    # 标记切点
    cv2.circle(out, Q_sel, 12, TAN_COLOR, -1)


# ---------------------------------------------------------
# 11. 计算平行方向向量
dx, dy = pt_lip[0]-pt_nose[0], pt_lip[1]-pt_nose[1]
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
    idxs = left_idx if P[0] < pt_lip[0] else right_idx
    tangents = [(j, jaw_pts[j]) for j in idxs
                if is_supporting_line(P, jaw_pts[j], jaw_pts)]
    if not tangents:
        print(f"⚠️  未找到 {['左','右'][side]}侧支持线")
        continue
    _, Q = min(tangents, key=lambda x: np.hypot(x[1][0]-P[0], x[1][1]-P[1]))
    tan_vec = (np.array(Q) - np.array(P)) / (np.linalg.norm(np.array(Q) - np.array(P)) + 1e-6)
    draw_angle(out, P, para_vec, tan_vec, ARC_COLOR)

# 14. 显示结果
cv2.imshow("Sharp Angles", out)
cv2.waitKey(0)
cv2.destroyAllWindows()