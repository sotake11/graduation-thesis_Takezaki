# 改善前の可変型

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import glob
import time
import datetime
from collections import deque, Counter

# ==========================================
# ★設定・定数
# ==========================================
SHAPE_FOLDER = "shapes2"

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

# ★変更点: 手がピースにこれくらい近づいたらターゲットとみなす距離（ピクセル）
PROXIMITY_THRESHOLD = 150 

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
    0: np.array([[0.09, 0.07], [0.07, 0.08], [0.07, 0.34], [0.09, 0.34], [0.11, 0.36], [0.13, 0.37], [0.15, 0.36], [0.17, 0.34], [0.18, 0.28], [0.20, 0.26], [0.24, 0.23], [0.29, 0.20], [0.30, 0.17], [0.30, 0.14], [0.29, 0.11], [0.29, 0.07], [0.29, 0.06], [0.09, 0.06]]),
    1: np.array([[0.30, 0.06], [0.30, 0.10], [0.31, 0.15], [0.32, 0.19], [0.30, 0.21], [0.27, 0.22], [0.28, 0.26], [0.29, 0.30], [0.36, 0.32], [0.38, 0.26], [0.39, 0.22], [0.41, 0.22], [0.43, 0.23], [0.45, 0.25], [0.47, 0.27], [0.50, 0.26], [0.51, 0.24], [0.51, 0.20], [0.52, 0.17], [0.52, 0.14], [0.54, 0.11], [0.54, 0.07], [0.54, 0.06], [0.32, 0.06]]),
    2: np.array([[0.55, 0.06], [0.54, 0.09], [0.55, 0.13], [0.58, 0.13], [0.60, 0.14], [0.61, 0.16], [0.61, 0.18], [0.60, 0.22], [0.60, 0.25], [0.62, 0.27], [0.66, 0.29], [0.70, 0.30], [0.73, 0.28], [0.78, 0.26], [0.77, 0.22], [0.78, 0.18], [0.80, 0.15], [0.81, 0.13], [0.82, 0.11], [0.82, 0.06], [0.55, 0.06]]),
    3: np.array([[0.82, 0.06], [0.82, 0.12], [0.80, 0.14], [0.79, 0.16], [0.78, 0.20], [0.78, 0.25], [0.80, 0.29], [0.83, 0.32], [0.85, 0.33], [0.86, 0.36], [0.85, 0.39], [0.82, 0.42], [0.80, 0.47], [0.80, 0.52], [0.81, 0.55], [0.83, 0.54], [0.85, 0.54], [0.88, 0.57], [0.93, 0.58], [0.93, 0.12], [0.93, 0.08], [0.91, 0.07], [0.83, 0.06]]),
    4: np.array([[0.08, 0.35], [0.08, 0.71], [0.10, 0.71], [0.12, 0.72], [0.13, 0.74], [0.14, 0.77], [0.15, 0.80], [0.17, 0.80], [0.18, 0.80], [0.20, 0.78], [0.20, 0.75], [0.19, 0.73], [0.18, 0.69], [0.18, 0.66], [0.19, 0.64], [0.22, 0.62], [0.24, 0.62], [0.27, 0.63], [0.28, 0.55], [0.27, 0.49], [0.26, 0.45], [0.23, 0.44], [0.21, 0.43], [0.18, 0.43], [0.17, 0.41], [0.15, 0.37], [0.14, 0.38], [0.12, 0.36], [0.10, 0.35], [0.08, 0.34]]),
    5: np.array([[0.28, 0.23], [0.21, 0.26], [0.19, 0.29], [0.19, 0.32], [0.17, 0.35], [0.15, 0.37], [0.16, 0.40], [0.18, 0.43], [0.20, 0.44], [0.24, 0.44], [0.26, 0.45], [0.27, 0.48], [0.28, 0.52], [0.28, 0.56], [0.27, 0.62], [0.27, 0.63], [0.29, 0.65], [0.30, 0.67], [0.32, 0.70], [0.33, 0.74], [0.34, 0.77], [0.37, 0.75], [0.40, 0.73], [0.41, 0.70], [0.43, 0.67], [0.44, 0.67], [0.45, 0.61], [0.46, 0.56], [0.46, 0.51], [0.46, 0.46], [0.43, 0.44], [0.40, 0.44], [0.37, 0.44], [0.36, 0.42], [0.36, 0.38], [0.36, 0.33], [0.32, 0.31], [0.29, 0.30], [0.28, 0.23]]),
    6: np.array([[0.39, 0.23], [0.38, 0.28], [0.37, 0.32], [0.36, 0.36], [0.36, 0.41], [0.38, 0.44], [0.41, 0.45], [0.44, 0.45], [0.46, 0.47], [0.47, 0.51], [0.50, 0.51], [0.52, 0.47], [0.54, 0.43], [0.58, 0.40], [0.61, 0.40], [0.63, 0.40], [0.63, 0.35], [0.63, 0.31], [0.63, 0.28], [0.60, 0.25], [0.60, 0.22], [0.61, 0.18], [0.61, 0.15], [0.59, 0.13], [0.56, 0.13], [0.55, 0.12], [0.54, 0.10], [0.53, 0.12], [0.51, 0.16], [0.51, 0.19], [0.51, 0.24], [0.49, 0.26], [0.47, 0.27], [0.45, 0.24], [0.42, 0.22], [0.40, 0.22], [0.39, 0.23]]),
    7: np.array([[0.63, 0.28], [0.63, 0.30], [0.63, 0.34], [0.62, 0.40], [0.64, 0.42], [0.66, 0.45], [0.66, 0.49], [0.66, 0.54], [0.65, 0.57], [0.66, 0.61], [0.69, 0.61], [0.73, 0.60], [0.77, 0.61], [0.78, 0.57], [0.81, 0.55], [0.80, 0.50], [0.81, 0.46], [0.83, 0.41], [0.85, 0.39], [0.86, 0.35], [0.85, 0.32], [0.81, 0.31], [0.79, 0.28], [0.78, 0.26], [0.73, 0.28], [0.69, 0.29], [0.63, 0.28]]),
    8: np.array([[0.46, 0.52], [0.45, 0.58], [0.44, 0.67], [0.48, 0.67], [0.51, 0.69], [0.51, 0.73], [0.52, 0.77], [0.54, 0.78], [0.56, 0.77], [0.59, 0.76], [0.63, 0.77], [0.69, 0.86], [0.71, 0.84], [0.73, 0.82], [0.77, 0.81], [0.79, 0.78], [0.79, 0.74], [0.77, 0.70], [0.77, 0.67], [0.77, 0.62], [0.74, 0.60], [0.70, 0.61], [0.67, 0.61], [0.65, 0.59], [0.65, 0.55], [0.65, 0.50], [0.65, 0.43], [0.63, 0.41], [0.60, 0.40], [0.56, 0.41], [0.53, 0.44], [0.52, 0.48], [0.50, 0.50], [0.46, 0.52]]),
    9: np.array([[0.08, 0.72], [0.08, 0.92], [0.09, 0.95], [0.35, 0.95], [0.35, 0.82], [0.34, 0.77], [0.31, 0.68], [0.28, 0.64], [0.24, 0.62], [0.20, 0.62], [0.18, 0.64], [0.18, 0.67], [0.19, 0.71], [0.20, 0.73], [0.20, 0.77], [0.19, 0.79], [0.18, 0.80], [0.17, 0.81], [0.15, 0.79], [0.14, 0.76], [0.12, 0.73], [0.10, 0.71], [0.07, 0.71]]),
    10: np.array([[0.34, 0.77], [0.35, 0.85], [0.35, 0.94], [0.68, 0.95], [0.68, 0.91], [0.68, 0.88], [0.69, 0.86], [0.65, 0.79], [0.60, 0.76], [0.57, 0.76], [0.55, 0.77], [0.53, 0.78], [0.51, 0.76], [0.51, 0.73], [0.51, 0.69], [0.48, 0.67], [0.45, 0.66], [0.43, 0.67], [0.41, 0.70], [0.39, 0.73], [0.34, 0.76]]),
    11: np.array([[0.68, 0.89], [0.68, 0.94], [0.90, 0.94], [0.92, 0.92], [0.93, 0.58], [0.89, 0.57], [0.86, 0.55], [0.82, 0.54], [0.79, 0.56], [0.77, 0.61], [0.77, 0.66], [0.78, 0.71], [0.79, 0.75], [0.79, 0.80], [0.76, 0.81], [0.72, 0.81], [0.69, 0.88], [0.68, 0.89]]),
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
        min_detection_confidence=0.3, # ★ 感度調整
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
    cap = cv2.VideoCapture(CAM_ID_MAIN)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌ メインカメラ(ID:{CAM_ID_MAIN}) が開けません。")
        return

    cap_sub = cv2.VideoCapture(CAM_ID_SUB)
    cap_sub.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap_sub.set(cv2.CAP_PROP_FPS, 30)

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
    # ★ ここで色設定を行います (ループの外)
    # ------------------------------------------
    # 肌色検出用の設定（MediaPipeがダメなときの予備）
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

    # 変数の初期化
    is_tracking_locked = False
    locked_pieces_data = []    
    latest_wipe_id = -1
    
    # ★追加: 滞留時間判定用の変数
    pending_id = -1          # 今、手の下にある候補ID
    pending_start_time = 0   # その候補の上に手が乗った時刻
    
    # ★設定: 切り替えにかかる時間（秒）
    # 0.4〜0.6秒くらいが「誤反応せず、かつ待たされすぎない」丁度いい値です
    DWELL_THRESHOLD = 0.5

    # ------------------------------------------
    # メインループ
    # ------------------------------------------
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
                out_main = cv2.VideoWriter(fname_main, fourcc, 30.0, (TOTAL_WIDTH, TOTAL_HEIGHT))
                if use_sub_camera and frame_sub is not None:
                    sh, sw = frame_sub.shape[:2]
                    out_sub = cv2.VideoWriter(fname_sub, fourcc, 30.0, (sw, sh))
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
        
        # --- 1. まずはAI (MediaPipe) で挑戦 ---
        res_hand = hands.process(frame_rgb)
        
        hand_pos = None
        hand_landmarks_list = []
        
        # AIが見つけた場合
        if res_hand.multi_hand_landmarks:
            for landmarks in res_hand.multi_hand_landmarks:
                lm = landmarks.landmark[8] # 人差し指
                cx = int(lm.x * w_frame)
                cy = int(lm.y * h_frame)
                hand_pos = (cx, cy)
                
                # 描画
                cv2.circle(frame_display, hand_pos, 10, (0, 255, 255), -1) # 黄色
                
                # ロック判定用のリスト作成
                h_list = []
                for p in landmarks.landmark:
                    px = int(p.x * w_frame)
                    py = int(p.y * h_frame)
                    h_list.append((px, py))
                hand_landmarks_list.append(h_list)

        # --- 2. AIがダメで、かつ画面の下の方なら「色認識」で救済 ---
        if hand_pos is None:
            # 画面の下30%だけを調査エリアにする
            roi_h = int(h_frame * 0.3) 
            roi_y_start = h_frame - roi_h
            roi = frame_display[roi_y_start:h_frame, 0:w_frame]
            
            # HSVに変換して肌色を抽出
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_skin = cv2.inRange(hsv_roi, SKIN_LOWER, SKIN_UPPER)
            
            # ノイズ除去
            kernel = np.ones((5,5), np.uint8)
            mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
            
            # 輪郭を見つける
            contours_skin, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_skin) > 0:
                # 一番大きい塊（手）を見つける
                c = max(contours_skin, key=cv2.contourArea)
                
                # ある程度の大きさがある場合のみ採用（ゴミ除去）
                if cv2.contourArea(c) > 500:
                    # その塊の一番「上」の点を指先とする
                    topmost = tuple(c[c[:,:,1].argmin()][0])
                    
                    # 座標を全体座標に戻す
                    fx = topmost[0]
                    fy = topmost[1] + roi_y_start
                    
                    hand_pos = (fx, fy)
                    
                    # ★変更点: 色認識モードである印（黄色丸に変更・文字削除）
                    cv2.circle(frame_display, hand_pos, 10, (0, 255, 255), -1) 
                    # cv2.putText(frame_display, "Color Mode", ...) ← 削除

        # ----------------------------------------------------
        # 以下、ロック判定などの続き...
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
        # 画像処理パイプライン
        # ----------------------------------------------------
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

            # ★変更点: 距離ベースで最も近いピースIDを算出しておく
            closest_dist_val = 99999
            closest_dist_id = -1
            if hand_pos:
                for t in tracker.tracks:
                    if t['id'] != -1:
                        dist = math.dist(hand_pos, t['center'])
                        if dist < closest_dist_val:
                            closest_dist_val = dist
                            closest_dist_id = t['id']

            current_hover_id = -1
            for t in tracker.tracks:
                color = (0, 0, 255)
                text = "Unk"
                if t['id'] != -1:
                    if t['locked']: color = (255, 0, 255); text = f"ID:{t['id']}"
                    elif t['updated']: color = (0, 255, 0); text = f"ID:{t['id']}"
                    else: color = (0, 165, 255); text = f"ID:{t['id']}"
                    
                    # 判定ロジック:
                    # 1. 距離が閾値以下なら、そのIDを強制的にターゲットとする
                    # 2. それ以外なら、従来のポリゴン内判定を使う
                    if hand_pos:
                        if closest_dist_id == t['id'] and closest_dist_val < PROXIMITY_THRESHOLD:
                            current_hover_id = t['id']
                        elif is_hand_over_polygon(hand_pos, t['poly']):
                             # 距離判定が発動していない場合に限り、従来の重なり判定も考慮
                             if current_hover_id == -1: 
                                current_hover_id = t['id']

                cv2.polylines(frame_display, [t['poly']], True, color, 3)
                x, y, w, h = cv2.boundingRect(t['poly'])
                cv2.putText(frame_display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ---------------------------------------------------------
            # ★変更: 滞留時間による確定ロジック
            # ---------------------------------------------------------
            
            # 1. 候補が変わった瞬間（別のピースに移動した、または手が離れた）
            if current_hover_id != pending_id:
                pending_id = current_hover_id
                pending_start_time = time.time() # タイマーリセット
            
            # 2. 同じ候補の上に一定時間留まり続けたか？
            if pending_id != -1:
                elapsed = time.time() - pending_start_time
                
                # 閾値を超えたら「確定」として画面を更新
                if elapsed > DWELL_THRESHOLD:
                    latest_wipe_id = pending_id
                    
                    # 視覚効果：確定したら枠を太くしたり色を変えて知らせる（任意）
                    if hand_pos:
                         cv2.circle(frame_display, hand_pos, 15, (0, 255, 0), 2)
            
            # 3. 待ち時間の可視化（任意ですが、あると分かりやすいです）
            elif pending_id != -1 and pending_id != latest_wipe_id:
                # まだ確定していない（移動中かもしれない）場合
                elapsed = time.time() - pending_start_time
                progress = min(1.0, elapsed / DWELL_THRESHOLD)
                
                # 手の近くに「ゲージ」を表示（円が描かれていくアニメーション）
                if hand_pos:
                    # ゲージの背景（グレー）
                    cv2.ellipse(frame_display, hand_pos, (20, 20), 0, 0, 360, (100, 100, 100), 3)
                    # ゲージの進行（オレンジ）
                    cv2.ellipse(frame_display, hand_pos, (20, 20), 0, 0, 360 * progress, (0, 165, 255), 3)

            # ---------------------------------------------------------

        else:
            # 固定（ロック）モード
            hovered_piece_id_in_freeze = -1
            
            # ロックモードでも距離判定用の準備
            closest_dist_val = 99999
            closest_dist_id = -1
            if hand_pos:
                for p in locked_pieces_data:
                    dist = math.dist(hand_pos, p['center'])
                    if dist < closest_dist_val:
                        closest_dist_val = dist
                        closest_dist_id = p['id']

            for p in locked_pieces_data:
                color = (255, 255, 0)
                
                # ロックモードの判定ロジック
                is_hovered = False
                if hand_pos:
                    # 距離が近い or ポリゴン内
                    if (closest_dist_id == p['id'] and closest_dist_val < PROXIMITY_THRESHOLD) or \
                       is_hand_over_polygon(hand_pos, p['poly']):
                        is_hovered = True

                if is_hovered:
                    hovered_piece_id_in_freeze = p['id']
                    color = (0, 255, 255)

                cv2.polylines(frame_display, [p['poly']], True, color, 3)
                x, y, w, h = cv2.boundingRect(p['poly'])
                cv2.putText(frame_display, f"ID:{p['id']}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if hovered_piece_id_in_freeze != -1:
                latest_wipe_id = hovered_piece_id_in_freeze

        # ----------------------------------------
        # 4. 画面合成
        # ----------------------------------------
        canvas = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

        # 左側パネルの背景を描画
        cv2.rectangle(canvas, (0, 0), (SIDE_PANEL_WIDTH, TOTAL_HEIGHT), (30, 30, 30), -1)

        # ==========================================
        # ▼ 右側：メインカメラ（比率維持・中央配置）
        # ==========================================
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

        # ==========================================
        # ▼ 左側：サブカメラ（比率維持・配置）
        # ==========================================
        if use_sub_camera and frame_sub is not None:
            
            # 表示用のコピーを作成
            wipe_view = frame_sub.copy()

            # 赤枠ハイライト処理
            if len(puzzle_corners) == 4 and latest_wipe_id != -1:
                target_poly = get_target_poly(latest_wipe_id, puzzle_corners)
                if target_poly is not None:
                    overlay = wipe_view.copy()
                    cv2.fillPoly(overlay, [target_poly], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.4, wipe_view, 0.6, 0, wipe_view)
                    cv2.polylines(wipe_view, [target_poly], True, (0, 0, 255), 5)

            # 描画領域の設定（左側のパネル全体を使う）
            area_x = panel_x_start + 10
            area_y = 100 
            area_w = SIDE_PANEL_WIDTH - 20
            area_h = TOTAL_HEIGHT - 120 

            # 描画実行（比率維持で最大化）
            rect = draw_fit_image(canvas, wipe_view, area_x, area_y, area_w, area_h, bg_color=(50, 50, 50))
            
            # 枠線とテキストの描画
            if rect is not None:
                rx, ry, rw, rh = rect
                
                # 枠線
                border_color = (100, 100, 100)
                if latest_wipe_id != -1:
                    if is_tracking_locked: border_color = (255, 255, 0)
                    else: border_color = (0, 255, 255)
                
                cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), border_color, 2)

                # テキスト表示（画像の下に配置）
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

    if is_recording:
        if out_main: out_main.release()
        if out_sub: out_sub.release()

    cap.release()
    if use_sub_camera: cap_sub.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()