固定型練習用12ピース_ランダムに埋める

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import glob
import time
import datetime
import threading
from collections import deque, Counter

# ==========================================
# ★設定・定数
# ==========================================
SHAPE_FOLDER = "shapes2"

# ★カメラ設定
MAIN_CAM_ID = 1   # メイン（手元・右画面）
SUB_CAM_ID = 0    # サブ（全体・左画面）

# ★映像の反転 (1:左右, 0:上下, -1:両方)
FLIP_MODE_MAIN = -1
FLIP_MODE_SUB  = -1

# ★レイアウト
FRAME_WIDTH = 850       
FRAME_HEIGHT = 600      
SIDE_PANEL_WIDTH = 650  
TOTAL_WIDTH = FRAME_WIDTH + SIDE_PANEL_WIDTH
TOTAL_HEIGHT = 720

# ★画像処理設定
MIN_AREA = 2500        
MAX_AREA = 50000
BORDER_MARGIN = 20
INVERT_MASK = False     
MATCH_THRESHOLD = 0.5   
EPSILON_FACTOR = 0.005  

# ★肌色検出パラメータ
SKIN_LOWER = np.array([0, 60, 80], dtype=np.uint8)
SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# ★ロジック用しきい値
PROXIMITY_THRESHOLD = 999  
MAX_DIST = 100             
MEMORY_LIMIT = 15
ID_HISTORY_LEN = 10

# ★手が画面から消えたとみなすまでのフレーム数（誤検知防止用 0.5秒程度）
HAND_MISSING_WAIT = 15

NEXT_STEP_THRESHOLD_RATIO = 0.2

# ==========================================
# ★パズル定義
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
# ★ヘルパー関数
# ==========================================
def crop_center_to_aspect(img, target_w, target_h):
    h_orig, w_orig = img.shape[:2]
    target_aspect = target_w / target_h
    orig_aspect = w_orig / h_orig

    if orig_aspect > target_aspect:
        new_w = int(h_orig * target_aspect)
        start_x = (w_orig - new_w) // 2
        return img[:, start_x:start_x+new_w]
    else:
        new_h = int(w_orig / target_aspect)
        start_y = (h_orig - new_h) // 2
        return img[start_y:start_y+new_h, :]

def draw_fit_image(canvas, img, x, y, w, h, bg_color=(30, 30, 30)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    if img is None: return None
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    dx, dy = (w - nw) // 2, (h - nh) // 2
    top, left = y + dy, x + dx
    canvas[top:top+nh, left:left+nw] = resized
    return (left, top, nw, nh)

def get_polygon_center(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0: return (0,0)
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def is_hand_over_polygon(hand_pos, polygon):
    if hand_pos is None: return False
    return cv2.pointPolygonTest(polygon, (float(hand_pos[0]), float(hand_pos[1])), False) >= 0

def calculate_match_score(contour1, contour2):
    try:
        epsilon1 = EPSILON_FACTOR * cv2.arcLength(contour1, True)
        approx1 = cv2.approxPolyDP(contour1, epsilon1, True)
        return cv2.matchShapes(approx1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    except: return 999

def select_puzzle_corners(cap):
    win_name = "Calibration"
    cv2.namedWindow(win_name)
    points = []
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: points.append((x, y))
    cv2.setMouseCallback(win_name, mouse_event)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP_MODE_SUB is not None: frame = cv2.flip(frame, FLIP_MODE_SUB)
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
        if key == 27: cv2.destroyAllWindows(); return None
    cv2.destroyWindow(win_name)
    return points

def get_target_poly(pid, corners):
    if pid not in PUZZLE_SHAPES: return None
    shape_ratios = PUZZLE_SHAPES[pid]
    poly_points = []
    p0, p1, p2, p3 = corners 
    for rx, ry in shape_ratios:
        top_x = p0[0] + (p1[0] - p0[0]) * rx
        top_y = p0[1] + (p1[1] - p0[1]) * rx
        btm_x = p3[0] + (p2[0] - p3[0]) * rx
        btm_y = p3[1] + (p2[1] - p3[1]) * rx
        poly_points.append((int(top_x + (btm_x - top_x) * ry), int(top_y + (btm_y - top_y) * ry)))
    return np.array(poly_points, dtype=np.int32)

class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.status = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            if self.status:
                self.thread = threading.Thread(target=self.update, args=(), daemon=True)
                self.thread.start()
    def update(self):
        while self.running:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
            time.sleep(0.005)
    def read(self):
        with self.lock:
            return self.status, self.frame.copy() if self.frame is not None else None
    def release(self):
        self.running = False
        if hasattr(self, 'thread'): self.thread.join()
        self.capture.release()

class SimpleTracker:
    def __init__(self):
        self.tracks = []
    def update(self, detections, hand_pos):
        for t in self.tracks: t['life'] -= 1; t['updated'] = False; t['locked'] = False
        unmatched = []
        for det in detections:
            best_idx, best_dist = -1, MAX_DIST
            for i, t in enumerate(self.tracks):
                if t['updated']: continue
                dist = math.dist(t['center'], det['center'])
                if dist < best_dist: best_dist = dist; best_idx = i
            if best_idx != -1:
                t = self.tracks[best_idx]
                t['center'], t['poly'], t['life'], t['updated'] = det['center'], det['poly'], MEMORY_LIMIT, True
                if det['id'] != -1: t['id_history'].append(det['id'])
                if t['id_history']: t['id'] = Counter(t['id_history']).most_common(1)[0][0]
                t['score'] = det['score']
            else: unmatched.append(det)
        for det in unmatched:
            hist = deque(maxlen=ID_HISTORY_LEN)
            if det['id'] != -1: hist.append(det['id'])
            self.tracks.append({'id': det['id'], 'id_history': hist, 'center': det['center'], 'poly': det['poly'], 'score': det['score'], 'life': MEMORY_LIMIT, 'updated': True, 'locked': False})
        
        if hand_pos:
            for t in self.tracks:
                if is_hand_over_polygon(hand_pos, t['poly']):
                    t['life'] = MEMORY_LIMIT
        self.tracks = [t for t in self.tracks if t['life'] > 0]
        return self.tracks

def run_recognition_test(cap, shapes_db):
    """
    テストモード：認識成功したものを表示
    - Green: 認識成功（Unique）
    - Red: ID重複（Duplicate）
    - SPACE または R キーで終了
    """
    win_name = "TEST MODE (Main Camera)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    print("=== TEST MODE START ===")
    print("Green: OK, Red: DUPLICATE ID detected")
    print("Press [SPACE] or [R] to confirm and start/restart.")
    
    # 連打防止用ウェイト
    time.sleep(0.5)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if FLIP_MODE_MAIN is not None:
            frame = cv2.flip(frame, FLIP_MODE_MAIN)

        frame_cropped = crop_center_to_aspect(frame, FRAME_WIDTH, FRAME_HEIGHT)
        frame_display = cv2.resize(frame_cropped, (FRAME_WIDTH, FRAME_HEIGHT))

        # --- 画像処理 ---
        gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if INVERT_MASK: mask = cv2.bitwise_not(mask)
        
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display_img = frame_display.copy()
        
        # 候補抽出
        candidates = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # フィルタ（メインと同じ条件）
            if area < MIN_AREA or area > MAX_AREA:
                continue

            poly = cnt
            best_score = 999
            best_id = -1
            
            for item in shapes_db:
                score = calculate_match_score(poly, item["contour"])
                if score < best_score:
                    best_score = score
                    best_id = item["id"]

            if best_score < MATCH_THRESHOLD:
                candidates.append({
                    'poly': poly,
                    'id': best_id,
                    'score': best_score
                })

        # 重複チェックと描画
        id_counts = Counter([c['id'] for c in candidates])

        for c in candidates:
            cid = c['id']
            poly = c['poly']
            score = c['score']
            x, y, w, h = cv2.boundingRect(poly)

            # 重複判定
            if id_counts[cid] > 1:
                color = (0, 0, 255) # 赤
                text = f"ID:{cid} DUP!"
                thickness = 4
            else:
                color = (0, 255, 0) # 緑
                # ★修正：スコアの表示を消して「ID:番号」だけにしました
                text = f"ID:{cid}" 
                thickness = 2

            cv2.drawContours(display_img, [poly], -1, color, thickness)
            cv2.putText(display_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 案内テキスト
        # ★修正：余計なテキストを消して、Restartの案内だけを一番上(高さ40)に表示します
        cv2.putText(display_img, "Press [SPACE] to Restart", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, display_img)

        key = cv2.waitKey(1)
        # SPACEキー で終了
        if key == 32:
            break
        if key == 27: # ESC
            cv2.destroyAllWindows()
            return False

    cv2.destroyWindow(win_name)
    return True

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    shapes_db = []
    available_ids = []
    if os.path.exists(SHAPE_FOLDER):
        print(f"📂 Reading {SHAPE_FOLDER}...")
        files = glob.glob(os.path.join(SHAPE_FOLDER, "*.jpg")) + glob.glob(os.path.join(SHAPE_FOLDER, "*.png"))
        for f in files:
            try:
                sid = int(os.path.splitext(os.path.basename(f))[0])
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    rect = cv2.boundingRect(cnt)
                    ratio = min(rect[2], rect[3]) / max(rect[2], rect[3])
                    approx = cv2.approxPolyDP(cnt, EPSILON_FACTOR * cv2.arcLength(cnt, True), True)
                    shapes_db.append({"id": sid, "contour": approx, "ratio": ratio})
                    available_ids.append(sid)
            except: pass
    
    # ★修正：指定順序への並べ替え
    # ID0→ID1→ID2→ID3→ID11→ID10→ID9→ID4→ID5→ID6→ID7→ID8
    target_order = [0, 1, 2, 3, 11, 10, 9, 4, 5, 6, 7, 8]
    # available_idsにあるものを、target_orderの順番に従ってリスト化する
    available_ids = [pid for pid in target_order if pid in available_ids]

    # カメラ起動
    cap = ThreadedCamera(MAIN_CAM_ID, 1280, 720)
    cap_sub = cv2.VideoCapture(SUB_CAM_ID)
    cap_sub.set(3, 1280); cap_sub.set(4, 720)
    use_sub = cap_sub.isOpened()
    
    # ★起動時テストモード
    if not run_recognition_test(cap, shapes_db):
        return
    
    puzzle_corners = select_puzzle_corners(cap_sub) if use_sub else []
    if use_sub and not puzzle_corners: return

    tracker = SimpleTracker()
    current_step_index = 0
    target_id = available_ids[current_step_index] if available_ids else -1
    
    latest_wipe_id = -1
    is_tracking_locked = False
    locked_pieces_data = [] 
    
    is_recording = False
    out_main, out_sub = None, None
    start_time = 0

    # ★進行管理用フラグ
    hand_missing_timer = 0
    ready_for_next_on_return = False
    
    # ★【追加】次のステップへ進むための滞留時間管理用
    next_step_dwell_timer = 0       # 計測開始時間
    is_waiting_for_next_step = False # 計測中フラグ
    DWELL_THRESHOLD = 0.4         # 0.4秒待機         
    
    # ★【追加】「左から来たか」を判定するフラグ
    hand_entered_from_left = False

    # ★【追加】状態管理用フラグ
    #hand_entered_from_left = False 
    #step_completed = False  # ★滞留完了したかを記録するフラグ
    
    cv2.namedWindow("System", cv2.WINDOW_NORMAL)

    while True:
        ret, frame_full = cap.read()
        if not ret: time.sleep(0.01); continue
        
        ret_sub, frame_sub = cap_sub.read() if use_sub else (False, None)

        if FLIP_MODE_MAIN is not None: frame_full = cv2.flip(frame_full, FLIP_MODE_MAIN)
        if use_sub and ret_sub and FLIP_MODE_SUB is not None: frame_sub = cv2.flip(frame_sub, FLIP_MODE_SUB)
        
        frame_cropped = crop_center_to_aspect(frame_full, FRAME_WIDTH, FRAME_HEIGHT)
        frame_display = cv2.resize(frame_cropped, (FRAME_WIDTH, FRAME_HEIGHT))
        
        h_frame, w_frame = frame_display.shape[:2]

        next_step_threshold_x = int(w_frame * NEXT_STEP_THRESHOLD_RATIO)

        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        res_hand = hands.process(frame_rgb)
        hand_pos = None
        hand_landmarks_list = []
        
        if res_hand.multi_hand_landmarks:
            for lms in res_hand.multi_hand_landmarks:
                lm = lms.landmark[8] 
                hand_pos = (int(lm.x * w_frame), int(lm.y * h_frame))
                hand_landmarks_list.append([(int(p.x * w_frame), int(p.y * h_frame)) for p in lms.landmark])

        if hand_pos is None:
            roi_h = int(h_frame * 0.3) 
            roi_y_start = h_frame - roi_h
            roi = frame_display[roi_y_start:h_frame, 0:w_frame]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_skin = cv2.inRange(hsv_roi, SKIN_LOWER, SKIN_UPPER)
            kernel = np.ones((5,5), np.uint8)
            mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
            cnts_skin, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_skin:
                c = max(cnts_skin, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    topmost = tuple(c[c[:,:,1].argmin()][0])
                    hand_pos = (topmost[0], topmost[1] + roi_y_start)

        # -----------------------------------------------------------
        # ★ロジック：ロック中、手が「左から右へ」移動して0.3秒留まったら次へ
        # -----------------------------------------------------------
        if is_tracking_locked:
            # 座標計算
            next_step_threshold_x = int(w_frame * NEXT_STEP_THRESHOLD_RATIO)
            
            # エリア判定
            is_in_right_zone = (hand_pos is not None and hand_pos[0] > next_step_threshold_x)
            is_in_left_zone = (hand_pos is not None and hand_pos[0] <= next_step_threshold_x)

            # 1. 手が左エリアにいるなら「フラグを立てる」（右に行く準備完了）
            if is_in_left_zone:
                hand_entered_from_left = True
                is_waiting_for_next_step = False 

            # 2. 手が右エリアにあり、かつ「左から来ている」場合のみ計測
            if is_in_right_zone and hand_entered_from_left:
                # エリアに入った瞬間（計測開始）
                if not is_waiting_for_next_step:
                    is_waiting_for_next_step = True
                    next_step_dwell_timer = time.time()
                
                # 経過時間を計算
                elapsed = time.time() - next_step_dwell_timer
                
                # ★UI描画処理（ご提示のコードデザインを適用）
                if hand_pos:
                    # まだ時間が経過していない（進行中）
                    if elapsed < DWELL_THRESHOLD:
                        progress = min(1.0, elapsed / DWELL_THRESHOLD)
                        
                        # グレーの背景円（リング）
                        cv2.ellipse(frame_display, hand_pos, (25, 25), 0, 0, 360, (100, 100, 100), 4)
                        
                        # 白いゲージ進行
                        # 0度(3時方向)から時計回りに進行
                        cv2.ellipse(frame_display, hand_pos, (25, 25), 0, 0, 360 * progress, (255, 255, 255), 4)
                    
                    # 時間経過済み（確定状態）
                    else:
                        # 確定サイン（緑の二重丸）
                        cv2.circle(frame_display, hand_pos, 15, (0, 255, 0), 2)
                        cv2.circle(frame_display, hand_pos, 20, (0, 255, 0), 2)

                # 設定時間経過したら次へ処理実行
                if elapsed > DWELL_THRESHOLD:
                    if current_step_index < len(available_ids) - 1:
                        # 少しだけ待ってから切り替える（描画を一瞬見せるため）
                        # ※即座に切り替えてよければここは不要ですが、緑丸を見せるために記述
                        
                        current_step_index += 1
                        target_id = available_ids[current_step_index]
                        latest_wipe_id = -1
                        is_tracking_locked = False
                        locked_pieces_data = []
                        
                        # 完了フラグのリセット
                        hand_entered_from_left = False
                        is_waiting_for_next_step = False
                        
                        print(f"✅ NEXT STEP -> {target_id}")
                    else:
                        cv2.putText(frame_display, "FINISHED!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            
            # 手が消えた場合などはタイマーリセット
            elif not is_in_right_zone:
                is_waiting_for_next_step = False

        # ----------------------------------------------------
        # ★認識ロジック
        # ----------------------------------------------------
        if current_step_index < len(available_ids):
            target_id = available_ids[current_step_index]

        if not is_tracking_locked:
            detections = []
            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if INVERT_MASK: mask = cv2.bitwise_not(mask)

            if hand_landmarks_list:
                for h_points in hand_landmarks_list:
                    hull = cv2.convexHull(np.array(h_points))
                    cv2.drawContours(mask, [hull], -1, 0, thickness=cv2.FILLED)
                    cv2.drawContours(mask, [hull], -1, 0, thickness=40)

            mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA or area > MAX_AREA: continue
                
                # ★メインループのフィルタ処理もコメントアウトしてテストモードと条件を一致させました
                # x,y,w,h = cv2.boundingRect(cnt)
                # if x<BORDER_MARGIN or y<BORDER_MARGIN or x+w>w_frame-BORDER_MARGIN or y+h>h_frame-BORDER_MARGIN: continue
                
                # aspect = float(w)/h
                # if aspect < 0.4 or aspect > 3.0: continue
                
                poly = cnt
                center = get_polygon_center(poly)
                
                final_id = -1
                best_score = 999
                best_id_candidate = -1

                for item in shapes_db:
                    score = calculate_match_score(poly, item["contour"])
                    if score < best_score:
                        best_score = score
                        best_id_candidate = item["id"]

                if best_score < MATCH_THRESHOLD:
                    final_id = best_id_candidate
                
                detections.append({'poly': poly, 'center': center, 'id': final_id, 'score': best_score})
            
            tracker.tracks = tracker.update(detections, hand_pos)

            if hand_pos:
                for t in tracker.tracks:
                    if t['id'] == target_id:
                        is_tracking_locked = True
                        latest_wipe_id = t['id']
                        ready_for_next_on_return = False
                        locked_pieces_data = [{'id': t['id'], 'poly': t['poly'].copy(), 'center': t['center']}]
                        break 

            for t in tracker.tracks:
                color = (255, 255, 0)
                text = f"ID:{t['id']}" 
                if t['id'] == target_id:
                    color = (0, 255, 255)
                    text = f"TARGET:{t['id']}"

                cv2.polylines(frame_display, [t['poly']], True, color, 3)
                if t['id'] != -1:
                    cv2.putText(frame_display, text, (t['center'][0], t['center'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        else:
            for p in locked_pieces_data:
                cv2.polylines(frame_display, [p['poly']], True, (0, 255, 255), 3)

        canvas = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (SIDE_PANEL_WIDTH, TOTAL_HEIGHT), (30, 30, 30), -1)

        if frame_display is not None:
            draw_fit_image(canvas, frame_display, SIDE_PANEL_WIDTH, 0, FRAME_WIDTH, TOTAL_HEIGHT)

        panel_x_start = 0
        if is_recording:
            current_time = time.time()
            diff_time = current_time - start_time
            m = int(diff_time // 60)
            s = int(diff_time % 60)
            ms = int((diff_time - int(diff_time)) * 100)
            elapsed_str = f"{m:02d}:{s:02d}.{ms:02d}"
            cv2.circle(canvas, (panel_x_start + 30, 40), 10, (0, 0, 255), -1)
            cv2.putText(canvas, f"REC  {elapsed_str}", (panel_x_start + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(canvas, "[ENTER] to REC", (panel_x_start + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if use_sub and frame_sub is not None:
            wipe_view = frame_sub.copy()
            if len(puzzle_corners) == 4 and target_id != -1:
                target_poly = get_target_poly(target_id, puzzle_corners)
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
                if is_tracking_locked: border_color = (0, 255, 255)
                
                cv2.rectangle(canvas, (rx, ry), (rx+rw, ry+rh), border_color, 2)

                if target_id != -1:
                    label = f"Pick ID: {target_id}"
                    tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    tx = rx + (rw - tsize[0]) // 2 
                    text_y = ry + rh + 35
                    if text_y > TOTAL_HEIGHT - 10: text_y = TOTAL_HEIGHT - 10
                    cv2.putText(canvas, label, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if is_recording:
            if out_main: out_main.write(canvas)
            if out_sub and frame_sub is not None: out_sub.write(frame_sub)
        
        cv2.imshow("System", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if k == 13: 
            if not is_recording:
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_main = cv2.VideoWriter(f"main_{now}.avi", cv2.VideoWriter_fourcc(*'XVID'), 21, (TOTAL_WIDTH, TOTAL_HEIGHT))
                if use_sub and frame_sub is not None:
                    h, w = frame_sub.shape[:2]
                    out_sub = cv2.VideoWriter(f"sub_{now}.avi", cv2.VideoWriter_fourcc(*'XVID'), 21, (w, h))
                start_time = time.time(); is_recording = True
            else:
                elapsed = time.time() - start_time
                print(f"⏹️ Recording Stopped. Time: {elapsed:.2f} seconds")

                is_recording = False; out_main.release(); 
                if out_sub: out_sub.release()
        
        # ... (ループの最後の方) ...
        if k == ord('r') or k == 32:
            print("Entering Test Mode...")
            
            run_recognition_test(cap, shapes_db)
            
            current_step_index = 0
            if available_ids: target_id = available_ids[0]
            latest_wipe_id = -1
            is_tracking_locked = False
            locked_pieces_data = []
            ready_for_next_on_return = False
            
            # ★追加: 滞留時間用変数のリセット
            is_waiting_for_next_step = False
            next_step_dwell_timer = 0
            hand_entered_from_left = False  # ★ここを追加
            #step_completed = False
            
            print("Reset and Restarted from ID 0")

    cap.release()
    if use_sub: cap_sub.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()