# 改善前の固定型

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
WIPE_FOLDER = "wipes2"  # ワイプ画像のフォルダ

# ★映像の反転設定
FLIP_MODE = -1

# ★カメラID設定
MAIN_CAM_ID = 0  # メインカメラ（ガイド表示用）
SUB_CAM_ID = 1   # サブカメラ（別アングル録画用）

# カメラ撮影サイズ
CAP_WIDTH = 1280
CAP_HEIGHT = 720

# サブカメラ録画サイズ
SUB_WIDTH = 640
SUB_HEIGHT = 480

# 画面表示サイズ（メイン）
FRAME_WIDTH = 960       
FRAME_HEIGHT = 540      

SIDE_PANEL_WIDTH = 320
WIPE_BASE_WIDTH = 280

TOTAL_WIDTH = FRAME_WIDTH + SIDE_PANEL_WIDTH
TOTAL_HEIGHT = FRAME_HEIGHT

# ★画面端の誤検出防止（ピクセル数）
BORDER_MARGIN = 20

# 画像処理パラメータ
MIN_AREA = 1000        
MAX_AREA = 40000
MATCH_THRESHOLD = 0.4
LOWER_WHITE = np.array([0, 0, 140])
UPPER_WHITE = np.array([180, 80, 255])

# 手の誤検出防止
HAND_MASK_DILATE_SIZE = 40 

# 追跡パラメータ
MEMORY_LIMIT = 15
MAX_DIST = 100
ID_HISTORY_LEN = 15
HAND_PRESENCE_THRESH = 10
HAND_ABSENCE_THRESH = 10

# ==========================================
# 初期化
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- データベース読み込み ---
shapes_db = []
available_ids = []

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
                epsilon = 0.005 * cv2.arcLength(max_cnt, True)
                approx_cnt = cv2.approxPolyDP(max_cnt, epsilon, True)
                shapes_db.append({"id": sid, "contour": approx_cnt})
                available_ids.append(sid)
                print(f"   ✅ ID: {sid} 登録完了")
        except: pass
else:
    print(f"⚠️ '{SHAPE_FOLDER}' が見つかりません")

available_ids.sort()

# ワイプ画像読み込み
wipe_db = {}
if os.path.exists(WIPE_FOLDER):
    files = glob.glob(os.path.join(WIPE_FOLDER, "*.jpg")) + glob.glob(os.path.join(WIPE_FOLDER, "*.png"))
    for f in files:
        try:
            sid = int(os.path.splitext(os.path.basename(f))[0])
            img = cv2.imread(f)
            if img is not None:
                h, w = img.shape[:2]
                target_w = WIPE_BASE_WIDTH
                img = cv2.resize(img, (target_w, int(target_w*h/w)))
                wipe_db[sid] = img
        except: pass

# ==========================================
# ヘルパー関数 & クラス
# ==========================================
def get_polygon_center(polygon):
    M = cv2.moments(polygon)
    if M["m00"] == 0: return (0,0)
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def calculate_match_score(contour1, contour2):
    try:
        epsilon1 = 0.005 * cv2.arcLength(contour1, True)
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
            if det['id'] != -1: initial_history.append(det['id'])
            self.tracks.append({
                'id': det['id'], 'id_history': initial_history,
                'center': det['center'], 'poly': det['poly'],
                'score': det['score'], 'life': MEMORY_LIMIT, 'updated': True
            })

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

tracker = SimpleTracker()
current_step_index = 0
hand_visible_counter = 0
hand_lost_counter = 0
is_hand_in_session = False

# ==========================================
# 録画・タイマー関連変数
# ==========================================
is_recording = False
out_main = None  # メイン画面用
out_sub = None   # サブカメラ用
start_time = 0.0
elapsed_str = "00:00.00"

# ==========================================
# カメラ初期化
# ==========================================
cap = cv2.VideoCapture(MAIN_CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

cap_sub = cv2.VideoCapture(SUB_CAM_ID)
if cap_sub.isOpened():
    print(f"📸 サブカメラ(ID:{SUB_CAM_ID}) 準備OK")
    cap_sub.set(cv2.CAP_PROP_FRAME_WIDTH, SUB_WIDTH)
    cap_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, SUB_HEIGHT)
else:
    print(f"⚠️ サブカメラ(ID:{SUB_CAM_ID}) が見つかりません。録画はメインのみ行われます。")

WINDOW_NAME = "Step Guide System"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("="*40)
print("🟢 組立手順ガイドシステム")
print("   - [ENTER]: 録画開始/停止")
print("   - 録画モード: メイン画面とサブカメラを個別のファイルに保存")
print("="*40)

# ==========================================
# メインループ
# ==========================================
while True:
    # 1. 画像読み込み
    ret, frame = cap.read()
    if not ret: break

    # サブカメラ読み込み
    ret_sub, frame_sub = cap_sub.read() if cap_sub.isOpened() else (False, None)
    
    # 反転処理
    if FLIP_MODE is not None:
        frame = cv2.flip(frame, FLIP_MODE)
        # サブカメラも必要なら反転（お好みでコメントアウトしてください）
        # if ret_sub and frame_sub is not None:
        #     frame_sub = cv2.flip(frame_sub, FLIP_MODE)

    # リサイズ（メイン）
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    h_frame, w_frame = frame.shape[:2]

    # キー入力
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == 13: # ENTER
        if not is_recording:
            # --- 録画開始処理 ---
            now = datetime.datetime.now()
            time_str = now.strftime("%Y%m%d_%H%M%S")
            
            # ファイル名決定
            file_main = f"main_rec_{time_str}.avi"
            file_sub = f"sub_rec_{time_str}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            # メイン画面用のライター作成
            out_main = cv2.VideoWriter(file_main, fourcc, 30.0, (TOTAL_WIDTH, TOTAL_HEIGHT))
            
            # サブカメラ用のライター作成（カメラが有効な場合のみ）
            if ret_sub and frame_sub is not None:
                # サブカメラの実際のサイズを取得
                sh, sw = frame_sub.shape[:2]
                out_sub = cv2.VideoWriter(file_sub, fourcc, 30.0, (sw, sh))
            
            start_time = time.time()
            is_recording = True
            print(f"\n🔴 録画開始")
            print(f"   Main: {file_main}")
            if out_sub: print(f"   Sub : {file_sub}")

        else:
            # --- 録画停止処理 ---
            end_time = time.time()
            duration = end_time - start_time
            m = int(duration // 60)
            s = int(duration % 60)
            ms = int((duration - int(duration)) * 100)
            formatted_time = f"{m:02d}:{s:02d}.{ms:02d}"
            
            is_recording = False
            
            # ライターの開放
            if out_main is not None:
                out_main.release()
                out_main = None
            if out_sub is not None:
                out_sub.release()
                out_sub = None
                
            print(f"\n⏹️ 録画終了")
            print(f"⏱️ 計測時間: {formatted_time}")

    # ==========================
    # メインカメラの解析処理
    # ==========================
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_hand = hands.process(frame_rgb)
    hand_pos = None
    hand_landmarks_list = []
    
    if res_hand.multi_hand_landmarks:
        for landmarks in res_hand.multi_hand_landmarks:
            lm = landmarks.landmark[8]
            hand_pos = (int(lm.x * w_frame), int(lm.y * h_frame))
            cv2.circle(frame, hand_pos, 10, (0, 255, 255), -1)
            h_list = []
            for p in landmarks.landmark:
                h_list.append((int(p.x * w_frame), int(p.y * h_frame)))
            hand_landmarks_list.append(h_list)

    if current_step_index < len(available_ids):
        if hand_pos is not None:
            hand_visible_counter += 1
            hand_lost_counter = 0
            if hand_visible_counter > HAND_PRESENCE_THRESH:
                is_hand_in_session = True
        else:
            hand_lost_counter += 1
            hand_visible_counter = 0
            if is_hand_in_session and hand_lost_counter > HAND_ABSENCE_THRESH:
                is_hand_in_session = False
                current_step_index += 1
                if current_step_index < len(available_ids):
                    print(f"✅ 次へ: ID {available_ids[current_step_index]}")
                else:
                    print("🎉 全工程完了！")

    target_id = -1
    if current_step_index < len(available_ids):
        target_id = available_ids[current_step_index]

    if hand_pos is None:
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

        if hand_landmarks_list:
            hand_mask_layer = np.zeros_like(mask)
            for h_points in hand_landmarks_list:
                hull = cv2.convexHull(np.array(h_points))
                cv2.fillPoly(hand_mask_layer, [hull], 255)
            kernel_hand = np.ones((HAND_MASK_DILATE_SIZE, HAND_MASK_DILATE_SIZE), np.uint8)
            hand_mask_layer = cv2.dilate(hand_mask_layer, kernel_hand, iterations=1)
            mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(hand_mask_layer))

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if (x < BORDER_MARGIN) or (y < BORDER_MARGIN) or \
               (x + w > w_frame - BORDER_MARGIN) or (y + h > h_frame - BORDER_MARGIN):
                continue
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.4 or aspect_ratio > 2.5: continue
            hull = cv2.convexHull(cnt)
            if cv2.contourArea(hull) == 0: continue
            solidity = float(area) / cv2.contourArea(hull)
            if solidity < 0.7: continue
            poly = cnt
            center = get_polygon_center(poly)
            best_id, best_score = -1, 999
            for item in shapes_db:
                score = calculate_match_score(poly, item["contour"])
                if score < best_score:
                    best_score = score
                    best_id = item["id"]
            if best_score > MATCH_THRESHOLD: best_id = -1
            detections.append({'poly': poly, 'center': center, 'id': best_id, 'score': best_score})
        tracker.tracks = tracker.update(detections, hand_pos)

    found_target = False
    for t in tracker.tracks:
        poly = t['poly']
        x, y, w, h = cv2.boundingRect(poly)
        if t['id'] == target_id:
            # ★変更箇所: ターゲットIDの色を黄色(0, 255, 255)から水色(255, 255, 0)に変更
            color = (255, 255, 0)
            text = f"TARGET: {t['id']}"
            thickness = 3
            found_target = True
        else:
            if t['id'] != -1 and t['score'] < MATCH_THRESHOLD + 0.1:
                color = (0, 165, 255); text = f"ID:{t['id']}"; thickness = 1
            else:
                color = (0, 0, 255); text = "."; thickness = 1
        cv2.polylines(frame, [poly], True, color, thickness)
        if t['id'] != -1: cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ==========================
    # 画面構築
    # ==========================
    canvas_base = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
    canvas_base[0:FRAME_HEIGHT, 0:FRAME_WIDTH] = frame
    cv2.rectangle(canvas_base, (FRAME_WIDTH, 0), (TOTAL_WIDTH, TOTAL_HEIGHT), (40, 40, 40), -1)

    panel_x_start = FRAME_WIDTH
    info_y_start = 100
    wipe_y_pos = info_y_start + 80
    
    if is_recording:
        current_time = time.time()
        diff_time = current_time - start_time
        m = int(diff_time // 60)
        s = int(diff_time % 60)
        ms = int((diff_time - int(diff_time)) * 100)
        elapsed_str = f"{m:02d}:{s:02d}.{ms:02d}"
        print(f"\r⏱️ 経過時間: {elapsed_str}", end="")
        cv2.circle(canvas_base, (panel_x_start + 30, 40), 10, (0, 0, 255), -1)
        cv2.putText(canvas_base, f"REC  {elapsed_str}", (panel_x_start + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(canvas_base, "[ENTER] to REC", (panel_x_start + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if current_step_index < len(available_ids):
        step_text = f"STEP: {current_step_index + 1}/{len(available_ids)}"
        cv2.putText(canvas_base, step_text, (panel_x_start + 15, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        id_text = f"Target: {target_id}"
        # ★ターゲットIDのテキストも水色に変更
        cv2.putText(canvas_base, id_text, (panel_x_start + 15, info_y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    else:
        cv2.putText(canvas_base, "FINISH!", (panel_x_start + 15, info_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    canvas_display = canvas_base.copy()
    if target_id != -1:
        wipe_img = wipe_db.get(target_id)
        if wipe_img is not None:
            wh, ww = wipe_img.shape[:2]
            x_pos = panel_x_start + (SIDE_PANEL_WIDTH - ww) // 2
            y_pos = wipe_y_pos
            if y_pos + wh < TOTAL_HEIGHT:
                canvas_display[y_pos:y_pos+wh, x_pos:x_pos+ww] = wipe_img
                # ★ワイプ枠も水色に変更
                border_color = (255, 255, 0) if found_target else (0, 255, 255)
                cv2.rectangle(canvas_display, (x_pos, y_pos), (x_pos+ww, y_pos+wh), border_color, 3)
                status_text = "FOUND!" if found_target else "SEARCH..."
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = panel_x_start + (SIDE_PANEL_WIDTH - text_size[0]) // 2
                cv2.putText(canvas_display, status_text, (text_x, y_pos + wh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)

    # ==========================
    # 録画書き込み処理
    # ==========================
    if is_recording:
        # 1. メイン画面の保存（ワイプ含むそのままの画面）
        if out_main is not None:
            out_main.write(canvas_display)
        
        # 2. サブカメラの保存（生映像）
        if out_sub is not None and frame_sub is not None:
            out_sub.write(frame_sub)

    cv2.imshow(WINDOW_NAME, canvas_display)

# 終了処理
if is_recording:
    if out_main is not None: out_main.release()
    if out_sub is not None: out_sub.release()

cap.release()
if cap_sub.isOpened():
    cap_sub.release()

cv2.destroyAllWindows()