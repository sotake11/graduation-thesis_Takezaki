# 可変型16ピース

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import glob
import time
import datetime
from collections import deque, Counter
from threading import Thread  # ★追加: 別スレッド用ライブラリ

# ==========================================
# ★設定・定数
# ==========================================
SHAPE_FOLDER = "shapes"

CAM_ID_MAIN = 1
CAM_ID_SUB  = 0

FLIP_MODE_MAIN = -1
FLIP_MODE_SUB  = -1

INVERT_MASK = False

MIN_AREA = 2500
MAX_AREA = 50000
MATCH_THRESHOLD = 0.5

# 画面の端対策
BORDER_MARGIN = 20

# カメラの入力解像度設定
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# 手がピースにこれくらい近づいたらターゲットとみなす距離（ピクセル）
PROXIMITY_THRESHOLD = 150 

# ★設定: 切り替えにかかる時間（秒）
# 0.5秒待たないと画面が切り替わりません（誤作動防止）
DWELL_THRESHOLD = 0.5

# ==========================================
# ★画面レイアウト設定
# ==========================================
FRAME_WIDTH = 850       
FRAME_HEIGHT = 600      
SIDE_PANEL_WIDTH = 650  

TOTAL_WIDTH = FRAME_WIDTH + SIDE_PANEL_WIDTH
TOTAL_HEIGHT = 720

# 追跡パラメータ
MEMORY_LIMIT = 15
MAX_DIST = 100
ID_HISTORY_LEN = 15

# ==========================================
# ★パズルの配置定義 (ID 0～11)
# ==========================================
PUZZLE_SHAPES = {
    0: np.array([[0.01, 0.00], [0.00, 0.02], [0.00, 0.17], [0.08, 0.18], [0.12, 0.20], [0.14, 0.22], [0.15, 0.24], [0.16, 0.27], [0.17, 0.30], [0.20, 0.31], [0.24, 0.30], [0.27, 0.26], [0.29, 0.22], [0.30, 0.14], [0.27, 0.13], [0.25, 0.10], [0.23, 0.06], [0.23, 0.01], [0.02, 0.00]]),
    1: np.array([[0.23, 0.00], [0.23, 0.06], [0.26, 0.12], [0.29, 0.14], [0.34, 0.13], [0.37, 0.12], [0.39, 0.14], [0.40, 0.18], [0.40, 0.21], [0.41, 0.24], [0.43, 0.26], [0.47, 0.25], [0.51, 0.23], [0.51, 0.19], [0.51, 0.14], [0.53, 0.10], [0.55, 0.08], [0.59, 0.05], [0.60, 0.00], [0.23, 0.00]]),
    2: np.array([[0.60, 0.00], [0.59, 0.05], [0.56, 0.08], [0.54, 0.09], [0.52, 0.13], [0.51, 0.18], [0.52, 0.23], [0.54, 0.27], [0.58, 0.26], [0.61, 0.24], [0.64, 0.23], [0.66, 0.24], [0.67, 0.28], [0.68, 0.33], [0.71, 0.35], [0.73, 0.36], [0.75, 0.36], [0.75, 0.33], [0.75, 0.29], [0.77, 0.26], [0.78, 0.23], [0.77, 0.20], [0.76, 0.16], [0.77, 0.14], [0.79, 0.10], [0.81, 0.09], [0.83, 0.07], [0.84, 0.04], [0.84, 0.00], [0.61, 0.00]]),
    3: np.array([[0.84, 0.00], [0.83, 0.07], [0.79, 0.11], [0.76, 0.14], [0.76, 0.18], [0.78, 0.21], [0.79, 0.24], [0.82, 0.27], [0.85, 0.27], [0.89, 0.24], [0.89, 0.21], [0.91, 0.19], [0.92, 0.19], [0.94, 0.21], [0.95, 0.25], [0.96, 0.27], [0.98, 0.29], [1.00, 0.29], [1.00, 0.03], [0.99, 0.01], [0.98, 0.00], [0.84, 0.00]]),
    4: np.array([[0.01, 0.17], [0.07, 0.18], [0.13, 0.20], [0.15, 0.25], [0.17, 0.30], [0.20, 0.32], [0.23, 0.30], [0.26, 0.28], [0.28, 0.29], [0.29, 0.36], [0.31, 0.43], [0.30, 0.49], [0.29, 0.55], [0.28, 0.58], [0.24, 0.59], [0.23, 0.52], [0.21, 0.48], [0.18, 0.47], [0.17, 0.47], [0.14, 0.52], [0.11, 0.52], [0.09, 0.50], [0.08, 0.45], [0.05, 0.42], [0.01, 0.41], [0.00, 0.41], [0.00, 0.18]]),
    5: np.array([[0.27, 0.27], [0.29, 0.33], [0.30, 0.40], [0.30, 0.44], [0.36, 0.45], [0.42, 0.47], [0.45, 0.51], [0.47, 0.56], [0.50, 0.57], [0.53, 0.56], [0.55, 0.52], [0.56, 0.48], [0.56, 0.43], [0.55, 0.37], [0.55, 0.27], [0.53, 0.24], [0.52, 0.22], [0.47, 0.24], [0.43, 0.25], [0.41, 0.23], [0.40, 0.17], [0.39, 0.14], [0.36, 0.12], [0.34, 0.13], [0.30, 0.14], [0.29, 0.18], [0.29, 0.23], [0.27, 0.27]]),
    6: np.array([[0.55, 0.27], [0.55, 0.37], [0.56, 0.45], [0.55, 0.49], [0.59, 0.52], [0.62, 0.56], [0.64, 0.61], [0.66, 0.63], [0.68, 0.63], [0.71, 0.61], [0.74, 0.59], [0.77, 0.58], [0.81, 0.57], [0.80, 0.51], [0.77, 0.45], [0.76, 0.41], [0.75, 0.36], [0.72, 0.35], [0.69, 0.34], [0.67, 0.29], [0.66, 0.24], [0.64, 0.23], [0.60, 0.25], [0.58, 0.26], [0.55, 0.26]]),
    7: np.array([[0.78, 0.22], [0.75, 0.28], [0.75, 0.35], [0.76, 0.39], [0.78, 0.46], [0.80, 0.51], [0.83, 0.49], [0.86, 0.47], [0.89, 0.49], [0.92, 0.50], [0.94, 0.49], [0.96, 0.46], [0.97, 0.46], [0.99, 0.45], [1.00, 0.30], [0.97, 0.28], [0.94, 0.24], [0.93, 0.20], [0.91, 0.19], [0.89, 0.22], [0.87, 0.25], [0.83, 0.27], [0.80, 0.25], [0.78, 0.23]]),
    8: np.array([[0.01, 0.41], [0.04, 0.41], [0.06, 0.43], [0.09, 0.48], [0.10, 0.51], [0.11, 0.53], [0.12, 0.52], [0.14, 0.51], [0.15, 0.48], [0.17, 0.47], [0.19, 0.47], [0.21, 0.49], [0.23, 0.53], [0.24, 0.58], [0.23, 0.61], [0.23, 0.68], [0.21, 0.72], [0.18, 0.76], [0.16, 0.71], [0.13, 0.69], [0.11, 0.69], [0.09, 0.72], [0.06, 0.76], [0.04, 0.77], [0.00, 0.78], [0.00, 0.41]]),
    9: np.array([[0.24, 0.59], [0.24, 0.63], [0.23, 0.67], [0.22, 0.70], [0.25, 0.74], [0.28, 0.76], [0.31, 0.78], [0.35, 0.76], [0.39, 0.74], [0.43, 0.74], [0.46, 0.75], [0.48, 0.76], [0.50, 0.79], [0.53, 0.75], [0.52, 0.71], [0.50, 0.68], [0.47, 0.65], [0.45, 0.62], [0.44, 0.58], [0.45, 0.55], [0.46, 0.54], [0.43, 0.48], [0.38, 0.46], [0.35, 0.45], [0.30, 0.44], [0.30, 0.49], [0.30, 0.53], [0.28, 0.57], [0.27, 0.58], [0.24, 0.59]]),
    10: np.array([[0.46, 0.55], [0.49, 0.56], [0.51, 0.57], [0.53, 0.56], [0.55, 0.53], [0.55, 0.50], [0.58, 0.51], [0.60, 0.52], [0.61, 0.55], [0.65, 0.61], [0.66, 0.63], [0.68, 0.63], [0.70, 0.62], [0.73, 0.60], [0.77, 0.58], [0.81, 0.57], [0.81, 0.62], [0.80, 0.67], [0.78, 0.72], [0.74, 0.75], [0.72, 0.77], [0.71, 0.78], [0.68, 0.77], [0.65, 0.78], [0.63, 0.80], [0.61, 0.85], [0.56, 0.87], [0.51, 0.86], [0.52, 0.82], [0.50, 0.79], [0.53, 0.73], [0.51, 0.69], [0.47, 0.65], [0.45, 0.62], [0.44, 0.58], [0.46, 0.55]]),
    11: np.array([[0.80, 0.50], [0.81, 0.55], [0.81, 0.58], [0.81, 0.64], [0.84, 0.66], [0.85, 0.69], [0.86, 0.73], [0.88, 0.76], [0.90, 0.78], [0.92, 0.77], [0.94, 0.73], [0.96, 0.71], [1.00, 0.70], [1.00, 0.46], [0.97, 0.45], [0.96, 0.46], [0.93, 0.50], [0.91, 0.50], [0.88, 0.49], [0.86, 0.48], [0.84, 0.48], [0.82, 0.49], [0.81, 0.50]]),
    12: np.array([[0.00, 0.79], [0.04, 0.78], [0.06, 0.76], [0.09, 0.72], [0.11, 0.69], [0.13, 0.69], [0.15, 0.70], [0.18, 0.74], [0.18, 0.75], [0.17, 0.77], [0.15, 0.80], [0.15, 0.85], [0.15, 0.88], [0.18, 0.89], [0.23, 0.90], [0.26, 0.92], [0.28, 0.95], [0.28, 1.00], [0.02, 1.00], [0.00, 0.98], [0.00, 0.79]]),
    13: np.array([[0.22, 0.70], [0.25, 0.74], [0.29, 0.78], [0.33, 0.77], [0.37, 0.75], [0.43, 0.73], [0.48, 0.76], [0.51, 0.80], [0.52, 0.85], [0.50, 0.89], [0.47, 0.93], [0.46, 0.95], [0.46, 1.00], [0.28, 1.00], [0.27, 0.95], [0.25, 0.91], [0.21, 0.89], [0.16, 0.88], [0.15, 0.84], [0.15, 0.81], [0.17, 0.77], [0.18, 0.75], [0.20, 0.73], [0.22, 0.71]]),
    14: np.array([[0.51, 0.86], [0.55, 0.87], [0.60, 0.85], [0.63, 0.81], [0.65, 0.78], [0.69, 0.77], [0.72, 0.78], [0.71, 0.82], [0.71, 0.87], [0.75, 0.90], [0.78, 0.91], [0.80, 0.92], [0.83, 0.95], [0.83, 1.00], [0.46, 1.00], [0.45, 0.97], [0.47, 0.94], [0.49, 0.91], [0.51, 0.87]]),
    15: np.array([[0.81, 0.64], [0.84, 0.66], [0.86, 0.72], [0.87, 0.76], [0.90, 0.78], [0.92, 0.76], [0.95, 0.73], [0.96, 0.71], [0.99, 0.69], [1.00, 0.99], [0.98, 1.00], [0.84, 1.00], [0.81, 0.93], [0.79, 0.92], [0.75, 0.90], [0.72, 0.88], [0.71, 0.84], [0.71, 0.80], [0.72, 0.77], [0.75, 0.74], [0.79, 0.70], [0.81, 0.65]]),
}

# ==========================================
# ★位置合わせ・座標計算用の関数
# ==========================================
def select_puzzle_corners(cap):
    win_name = "Calibration: Click 4 corners (Sub Camera)"
    cv2.namedWindow(win_name)
    points = []

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(win_name, mouse_event)
    print("\n【位置合わせ開始】")
    print("サブカメラのウィンドウで、パズルの角を [左上→右上→右下→左下] の順にクリックしてください。")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP_MODE_SUB is not None:
            frame = cv2.flip(frame, FLIP_MODE_SUB)
            
        display = frame.copy()
        
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 8, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        if len(points) == 4:
            cv2.polylines(display, [np.array(points)], True, (0, 255, 0), 2)
            cv2.putText(display, "Press [SPACE] to Start", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Click corners: {len(points)}/4", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(1)
        
        if len(points) == 4 and key == 32: break
        if key == ord('r'): points = [] 
        if key == 27: 
            cv2.destroyAllWindows()
            return None

    cv2.destroyWindow(win_name)
    return points

def get_target_poly(pid, corners):
    if pid not in PUZZLE_SHAPES: return None
    shape_ratios = PUZZLE_SHAPES[pid]
    
    def interp(rx, ry):
        p0, p1, p2, p3 = corners 
        top_x = p0[0] + (p1[0] - p0[0]) * rx
        top_y = p0[1] + (p1[1] - p0[1]) * rx
        btm_x = p3[0] + (p2[0] - p3[0]) * rx
        btm_y = p3[1] + (p2[1] - p3[1]) * rx
        x = top_x + (btm_x - top_x) * ry
        y = top_y + (btm_y - top_y) * ry
        return int(x), int(y)

    poly_points = []
    for rx, ry in shape_ratios:
        poly_points.append(interp(rx, ry))
        
    return np.array(poly_points, dtype=np.int32)

def draw_fit_image(canvas, img, x, y, w, h, bg_color=(30, 30, 30)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    if img is None: return None
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    top = y + dy
    bottom = top + nh
    left = x + dx
    right = left + nw
    canvas[top:bottom, left:right] = resized
    return (left, top, nw, nh)

def is_hand_over_polygon(hand_pos, polygon):
    if hand_pos is None: return False
    return cv2.pointPolygonTest(polygon, (float(hand_pos[0]), float(hand_pos[1])), False) >= 0

def get_polygon_center(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0: return (0,0)
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def calculate_match_score(contour1, contour2):
    try:
        epsilon1 = 0.01 * cv2.arcLength(contour1, True)
        approx1 = cv2.approxPolyDP(contour1, epsilon1, True)
        return cv2.matchShapes(approx1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    except:
        return 999

class SimpleTracker:
    def __init__(self):
        self.tracks = [] 

    def update(self, detections, hand_pos):
        for t in self.tracks:
            t['life'] -= 1
            t['updated'] = False
            t['locked'] = False

        unmatched_detections = []
        
        for det in detections:
            best_idx = -1
            best_dist = MAX_DIST
            for i, t in enumerate(self.tracks):
                if t['updated']: continue
                dist = math.dist(t['center'], det['center'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                track = self.tracks[best_idx]
                track['center'] = det['center']
                track['poly'] = det['poly']
                track['life'] = MEMORY_LIMIT
                track['updated'] = True
                if det['id'] != -1:
                    track['id_history'].append(det['id'])
                if track['id_history']:
                    most_common = Counter(track['id_history']).most_common(1)
                    track['id'] = most_common[0][0]
                    track['score'] = det['score'] 
            else:
                unmatched_detections.append(det)

        for det in unmatched_detections:
            initial_history = deque(maxlen=ID_HISTORY_LEN)
            if det['id'] != -1:
                initial_history.append(det['id'])
            self.tracks.append({
                'id': det['id'],
                'id_history': initial_history,
                'center': det['center'],
                'poly': det['poly'],
                'score': det['score'],
                'life': MEMORY_LIMIT,
                'updated': True,
                'locked': False
            })

        if hand_pos:
            for t in self.tracks:
                if is_hand_over_polygon(hand_pos, t['poly']):
                    t['life'] = MEMORY_LIMIT
                    t['locked'] = True

        self.tracks = [t for t in self.tracks if t['life'] > 0]

        id_map = {}
        for t in self.tracks:
            if t['id'] == -1: continue 
            if t['id'] not in id_map: id_map[t['id']] = []
            id_map[t['id']].append(t)
        
        for pid, obj_list in id_map.items():
            if len(obj_list) > 1:
                obj_list.sort(key=lambda x: x.get('score', 999))
                for i in range(1, len(obj_list)):
                    obj_list[i]['id'] = -1
                    obj_list[i]['id_history'].clear() 

        return self.tracks

# ----------------------------------------------------
# ★追加: カメラ読み込みを高速化するクラス
# ----------------------------------------------------
class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        # カメラを準備
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # 最初の1フレームを読んでおく
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        # 読み込み担当スレッド（裏方）を開始
        # daemon=True にすると、プログラム終了時に勝手に死んでくれる
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        # 裏でひたすら最新フレームを取り続けるループ
        while not self.stopped:
            ret, frame = self.capture.read()
            if ret:
                self.ret, self.frame = ret, frame
            else:
                # カメラが抜けたりしたら停止
                self.stopped = True

    def read(self):
        # メイン処理が「今の画像」を欲しがったときに渡す
        return self.ret, self.frame

    def isOpened(self):
        # カメラが開いているか確認用（cv2.VideoCapture互換）
        return self.capture.isOpened()

    def release(self):
        # 終了処理
        self.stopped = True
        self.capture.release()

# ==========================================
# ★メイン関数
# ==========================================
def main():
    # ------------------------------------------
    # 初期化処理
    # ------------------------------------------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.3, # 感度調整
        min_tracking_confidence=0.3,  
        model_complexity=1            
    )

    tracker = SimpleTracker()

    shapes_db = []
    if os.path.exists(SHAPE_FOLDER):
        print(f"📂 '{SHAPE_FOLDER}' 読み込み中...")
        files = glob.glob(os.path.join(SHAPE_FOLDER, "*.jpg")) + glob.glob(os.path.join(SHAPE_FOLDER, "*.png"))
        for f in files:
            try:
                sid = int(os.path.splitext(os.path.basename(f))[0])
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(max_cnt)
                    ratio = min(w, h) / max(w, h)
                    epsilon = 0.01 * cv2.arcLength(max_cnt, True)
                    approx_cnt = cv2.approxPolyDP(max_cnt, epsilon, True)
                    shapes_db.append({"id": sid, "contour": approx_cnt, "ratio": ratio})
                    print(f"   ✅ ID: {sid} 登録完了 (比率: {ratio:.2f})")
            except: pass

    # カメラ接続
    print("🎥 カメラ接続中...")
    
    # ★変更: 高速化クラスを使用
    cap = ThreadedCamera(CAM_ID_MAIN, width=CAP_WIDTH, height=CAP_HEIGHT).start()
    
    # ※サブカメラは使用頻度が低いため通常のVideoCaptureのままとします
    cap_sub = cv2.VideoCapture(CAM_ID_SUB)
    cap_sub.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap_sub.set(cv2.CAP_PROP_FPS, 30)

    # ThreadedCameraにisOpenedメソッドを追加したため、そのままチェック可能
    if not cap.isOpened():
        print(f"❌ メインカメラ(ID:{CAM_ID_MAIN}) が開けません。")
        return

    use_sub_camera = cap_sub.isOpened()
    if use_sub_camera:
        print(f"✅ サブカメラ(ID:{CAM_ID_SUB}) 接続成功")
    else:
        print(f"⚠️ サブカメラ(ID:{CAM_ID_SUB}) が見つかりません。")

    # 起動時位置合わせ
    puzzle_corners = []
    if use_sub_camera:
        puzzle_corners = select_puzzle_corners(cap_sub)
        if puzzle_corners is None:
            print("位置合わせがキャンセルされました。")
            return
        print("✅ 位置合わせ完了！メイン処理を開始します。")

    WINDOW_NAME = "Step Guide"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("="*40)
    print("🟢 組立ガイドシステム")
    print("   - [ENTER]キー: 録画開始/停止")
    print("   - [ESC]キー:   終了")
    print("="*40)

    # ------------------------------------------
    # 色設定
    # ------------------------------------------
    # 肌色検出用の設定
    SKIN_LOWER = np.array([0, 60, 80], dtype=np.uint8)
    SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)

    # 変数の初期化
    is_tracking_locked = False
    locked_pieces_data = []    
    latest_wipe_id = -1
    is_recording = False
    out_main = None
    out_sub = None
    start_time = 0.0
    elapsed_str = "00:00.00"

    # ★追加: 滞留時間判定用の変数
    pending_id = -1          # 今、手の下にある候補ID
    pending_start_time = 0   # その候補の上に手が乗った時刻
    
    # ★削除済み: 手の退出検知用の変数はもう使わないので削除しました
    
    # ------------------------------------------
    # メインループ
    # ------------------------------------------
    was_in_left_area = False  # 手が左エリアに入ったことを覚えるフラグ
    RESET_THRESHOLD = 150     # 左画面の境界線（X座標など）

    while True:
        ret, frame = cap.read()
        if not ret: break

        if FLIP_MODE_MAIN is not None:
            frame = cv2.flip(frame, FLIP_MODE_MAIN)
        
        frame_sub = None
        if use_sub_camera:
            ret_sub, temp_sub = cap_sub.read()
            if ret_sub:
                if FLIP_MODE_SUB is not None:
                    temp_sub = cv2.flip(temp_sub, FLIP_MODE_SUB)
                frame_sub = temp_sub

        frame_display = frame.copy() 
        h_frame, w_frame = frame_display.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == 13:
            if not is_recording:
                now = datetime.datetime.now()
                ts = now.strftime("%Y%m%d_%H%M%S")
                fname_main = f"main_{ts}.avi"
                fname_sub  = f"sub_{ts}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_main = cv2.VideoWriter(fname_main, fourcc, 21.0, (TOTAL_WIDTH, TOTAL_HEIGHT))
                if use_sub_camera and frame_sub is not None:
                    sh, sw = frame_sub.shape[:2]
                    out_sub = cv2.VideoWriter(fname_sub, fourcc, 21.0, (sw, sh))
                start_time = time.time()
                is_recording = True
                print(f"\n🔴 録画開始")
            else:
                is_recording = False
                if out_main: out_main.release(); out_main = None
                if out_sub: out_sub.release(); out_sub = None
                print(f"\n⏹️ 録画終了 (最終時間: {elapsed_str})")

        # ----------------------------------------------------
        # 手の検出（ハイブリッド版：AI + 色認識）
        # ----------------------------------------------------
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        # --- 1. AI (MediaPipe) ---
        res_hand = hands.process(frame_rgb)
        
        hand_pos = None
        hand_landmarks_list = []
        
        if res_hand.multi_hand_landmarks:
            for landmarks in res_hand.multi_hand_landmarks:
                lm = landmarks.landmark[8] # 人差し指
                cx = int(lm.x * w_frame)
                cy = int(lm.y * h_frame)
                hand_pos = (cx, cy)
                
                cv2.circle(frame_display, hand_pos, 10, (0, 255, 255), -1) 
                
                h_list = []
                for p in landmarks.landmark:
                    px = int(p.x * w_frame)
                    py = int(p.y * h_frame)
                    h_list.append((px, py))
                hand_landmarks_list.append(h_list)

        # --- 2. 色認識で救済 ---
        if hand_pos is None:
            roi_h = int(h_frame * 0.3) 
            roi_y_start = h_frame - roi_h
            roi = frame_display[roi_y_start:h_frame, 0:w_frame]
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_skin = cv2.inRange(hsv_roi, SKIN_LOWER, SKIN_UPPER)
            
            kernel = np.ones((5,5), np.uint8)
            mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
            
            contours_skin, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_skin) > 0:
                c = max(contours_skin, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    topmost = tuple(c[c[:,:,1].argmin()][0])
                    fx = topmost[0]
                    fy = topmost[1] + roi_y_start
                    hand_pos = (fx, fy)
                    cv2.circle(frame_display, hand_pos, 10, (0, 255, 255), -1) 
        
        # ----------------------------------------------------
        # ★修正版：左に入って、右に出たときだけリセット
        # ----------------------------------------------------
        if hand_pos is not None:
            # hand_pos は (x, y)
            current_hand_x = hand_pos[0]
            
            # ▼ 条件：手が左の境界線より左にいるか？
            if current_hand_x < RESET_THRESHOLD:
                # 今、左にいるなら「左に入ったこと」を覚えるだけ（まだ消さない！）
                was_in_left_area = True

            else:
                # ▼ 今、境界線の右側（作業エリア）にいる
                if was_in_left_area:
                    # 「さっきまで左にいた」のに「今右にいる」＝「出てきた瞬間」
                    latest_wipe_id = -1
                    pending_id = -1
                    
                    # フラグを戻す（これで連続リセットを防ぐ）
                    was_in_left_area = False
        
        # ----------------------------------------------------
        # ロック状態の更新（検出前に行う）
        # ----------------------------------------------------
        if hand_pos is not None and not is_tracking_locked:
            if tracker.tracks:
                new_locked = []
                for t in tracker.tracks:
                    if t['id'] != -1:
                        new_locked.append({'id': t['id'], 'poly': t['poly'].copy(), 'center': t['center']})
                if new_locked:
                    is_tracking_locked = True
                    locked_pieces_data = new_locked
        elif hand_pos is None and is_tracking_locked:
            is_tracking_locked = False
            locked_pieces_data = []

        # ----------------------------------------------------
        # 判定ロジック
        # ----------------------------------------------------
        
        # 判定用の候補IDをリセット（これを後段のタイマーに渡します）
        target_candidate_id = -1

        # =========================================================
        # A. 手がフリーな場合（通常トラッキング）
        # =========================================================
        if not is_tracking_locked:
            detections = []
            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if INVERT_MASK:
                mask = cv2.bitwise_not(mask)

            if hand_landmarks_list:
                for h_points in hand_landmarks_list:
                    hull = cv2.convexHull(np.array(h_points))
                    cv2.drawContours(mask, [hull], -1, 0, thickness=cv2.FILLED)
                    cv2.drawContours(mask, [hull], -1, 0, thickness=40)

            kernel_erode = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel_erode, iterations=2)
            kernel_open = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.dilate(mask, kernel_erode, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA or area > MAX_AREA: continue
                
                x, y, w, h = cv2.boundingRect(cnt)
                if (x < BORDER_MARGIN) or (y < BORDER_MARGIN) or \
                   (x + w > w_frame - BORDER_MARGIN) or (y + h > h_frame - BORDER_MARGIN):
                    continue

                aspect_ratio = float(w) / h
                if aspect_ratio < 0.4 or aspect_ratio > 3.0: continue

                poly = cnt
                center = get_polygon_center(poly)
                target_ratio = min(w, h) / max(w, h)

                best_id, best_score = -1, 999
                for item in shapes_db:
                    if abs(target_ratio - item["ratio"]) > 0.25:
                        continue
                    score = calculate_match_score(poly, item["contour"])
                    if score < best_score:
                        best_score = score
                        best_id = item["id"]
                
                if best_score > MATCH_THRESHOLD: 
                    best_id = -1
                
                detections.append({'poly': poly, 'center': center, 'id': best_id, 'score': best_score})

            tracker.tracks = tracker.update(detections, hand_pos)

            # --- 距離判定の準備 ---
            closest_dist_val = 99999
            closest_dist_id = -1
            if hand_pos:
                for t in tracker.tracks:
                    if t['id'] != -1:
                        dist = math.dist(hand_pos, t['center'])
                        if dist < closest_dist_val:
                            closest_dist_val = dist
                            closest_dist_id = t['id']

            # --- 候補の算出 ---
            for t in tracker.tracks:
                color = (0, 0, 255)
                text = "Unk"
                if t['id'] != -1:
                    if t['locked']: color = (255, 0, 255); text = f"ID:{t['id']}"
                    elif t['updated']: color = (0, 255, 0); text = f"ID:{t['id']}"
                    else: color = (0, 165, 255); text = f"ID:{t['id']}"
                    
                    if hand_pos:
                        # 1. 距離が近い
                        if closest_dist_id == t['id'] and closest_dist_val < PROXIMITY_THRESHOLD:
                            target_candidate_id = t['id']
                        # 2. ポリゴン内
                        elif is_hand_over_polygon(hand_pos, t['poly']):
                             if target_candidate_id == -1: 
                                target_candidate_id = t['id']

                cv2.polylines(frame_display, [t['poly']], True, color, 3)
                x, y, w, h = cv2.boundingRect(t['poly'])
                cv2.putText(frame_display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # =========================================================
        # B. ロックモード（ピースを持って移動中）
        # =========================================================
        else:
            closest_dist_val = 99999
            closest_dist_id = -1
            if hand_pos:
                for p in locked_pieces_data:
                    dist = math.dist(hand_pos, p['center'])
                    if dist < closest_dist_val:
                        closest_dist_val = dist
                        closest_dist_id = p['id']

            for p in locked_pieces_data:
                color = (255, 255, 0) # ロック中の基本色
                
                is_hovered = False
                if hand_pos:
                    if (closest_dist_id == p['id'] and closest_dist_val < PROXIMITY_THRESHOLD) or \
                       is_hand_over_polygon(hand_pos, p['poly']):
                           is_hovered = True

                if is_hovered:
                    target_candidate_id = p['id']  # 候補に入れるだけ（即更新しない）
                    color = (0, 255, 255)          # 選ばれている色

                cv2.polylines(frame_display, [p['poly']], True, color, 3)
                x, y, w, h = cv2.boundingRect(p['poly'])
                cv2.putText(frame_display, f"ID:{p['id']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # =========================================================
        # ★共通: 滞留時間（Dwell Time）判定ロジック（修正済）
        # =========================================================
        
        # 1. 候補が変わった場合（移動した、または手が離れた）
        if target_candidate_id != pending_id:
            pending_id = target_candidate_id
            pending_start_time = time.time() # タイマーリセット
        
        # 2. 同じ候補の上に一定時間留まり続けたか？
        if pending_id != -1:
            elapsed = time.time() - pending_start_time
            
            # 閾値を超えたら画面を更新（確定）
            if elapsed > DWELL_THRESHOLD:
                # ★修正: 候補がある時だけ更新（＝手が何もない所に移動しても前のガイドを消さない）
                latest_wipe_id = pending_id
                
                # 確定サイン（緑の二重丸）
                if hand_pos:
                       cv2.circle(frame_display, hand_pos, 15, (0, 255, 0), 2)
                       cv2.circle(frame_display, hand_pos, 20, (0, 255, 0), 2)
            
            # 待ち時間の可視化（まだ確定していない移動中）
            elif pending_id != latest_wipe_id:
                progress = min(1.0, elapsed / DWELL_THRESHOLD)
                if hand_pos:
                    # グレーの背景円
                    cv2.ellipse(frame_display, hand_pos, (25, 25), 0, 0, 360, (100, 100, 100), 4)
                    # 白いゲージ進行
                    cv2.ellipse(frame_display, hand_pos, (25, 25), 0, 0, 360 * progress, (255, 255, 255), 4)
        
        else:
            # pending_id が -1 の場合（手がピースから離れた場合）
            # ここでは何もしない → latest_wipe_id は維持される
            pass 

        # ----------------------------------------
        # 4. 画面合成
        # ----------------------------------------
        canvas = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

        # 左側パネルの背景を描画
        cv2.rectangle(canvas, (0, 0), (SIDE_PANEL_WIDTH, TOTAL_HEIGHT), (30, 30, 30), -1)

        # メインカメラ
        if frame_display is not None:
            draw_fit_image(
                canvas, 
                frame_display, 
                SIDE_PANEL_WIDTH, 0,      # 開始位置 X, Y
                FRAME_WIDTH, TOTAL_HEIGHT # エリアの幅, 高さ
            )

        panel_x_start = 0

        # --- 録画情報の表示 ---
        if is_recording:
            current_time = time.time()
            diff_time = current_time - start_time
            m = int(diff_time // 60)
            s = int(diff_time % 60)
            ms = int((diff_time - int(diff_time)) * 100)
            elapsed_str = f"{m:02d}:{s:02d}.{ms:02d}"
            print(f"\r⏱️ 経過時間: {elapsed_str}", end="")
            cv2.circle(canvas, (panel_x_start + 30, 40), 10, (0, 0, 255), -1)
            cv2.putText(canvas, f"REC  {elapsed_str}", (panel_x_start + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(canvas, "[ENTER] to REC", (panel_x_start + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # サブカメラ
        if use_sub_camera and frame_sub is not None:
            wipe_view = frame_sub.copy()

            if len(puzzle_corners) == 4 and latest_wipe_id != -1:
                target_poly = get_target_poly(latest_wipe_id, puzzle_corners)
                if target_poly is not None:
                    overlay = wipe_view.copy()
                    cv2.fillPoly(overlay, [target_poly], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.4, wipe_view, 0.6, 0, wipe_view)
                    cv2.polylines(wipe_view, [target_poly], True, (0, 0, 255), 5)

            area_x = panel_x_start + 10
            area_y = 100 
            area_w = SIDE_PANEL_WIDTH - 20
            area_h = TOTAL_HEIGHT - 120 

            rect = draw_fit_image(canvas, wipe_view, area_x, area_y, area_w, area_h, bg_color=(50, 50, 50))
            
            if rect is not None:
                rx, ry, rw, rh = rect
                
                border_color = (100, 100, 100)
                if latest_wipe_id != -1:
                    if is_tracking_locked: border_color = (255, 255, 0)
                    else: border_color = (0, 255, 255)
                
                cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), border_color, 2)

                if latest_wipe_id != -1:
                    label = f"Place ID: {latest_wipe_id}"
                    tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    tx = rx + (rw - tsize[0]) // 2 
                    text_y = ry + rh + 35
                    if text_y > TOTAL_HEIGHT - 10:
                        text_y = TOTAL_HEIGHT - 10
                    cv2.putText(canvas, label, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if is_recording:
            if out_main: out_main.write(canvas)
            if out_sub and frame_sub is not None: out_sub.write(frame_sub)
        
        cv2.imshow(WINDOW_NAME, canvas)

        # ★削除: ここにあったエラーの原因となる行は消しました

    if is_recording:
        if out_main: out_main.release()
        if out_sub: out_sub.release()

    cap.release()
    if use_sub_camera: cap_sub.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()