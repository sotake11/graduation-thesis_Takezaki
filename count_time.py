# 支援なし時に時間を計測するためのコード

import cv2
import numpy as np
import time

# ==========================================
# 設定
# ==========================================
WINDOW_NAME = "Black Screen Timer"
WIDTH = 640
HEIGHT = 480
FPS = 20.0  # 録画のフレームレート

# 色設定 (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# ==========================================
# カメラ初期化
# ==========================================
cap_main = cv2.VideoCapture(0) # メインカメラ
cap_sub  = cv2.VideoCapture(1) # サブカメラ

# 録画用のオブジェクト
writer_main = None
writer_sub = None

# ==========================================
# 変数初期化
# ==========================================
is_running = False
start_time = 0.0
elapsed_str = "00:00.00"

cv2.namedWindow(WINDOW_NAME)

print("="*40)
print("⏱️ 2カメラ録画タイマー (180度反転モード)")
print("   - [ENTER]: 計測＆録画開始 / 停止")
print("   - [ESC]:   終了")
print("="*40)

while True:
    # カメラからフレームを取得
    ret_m, frame_m = cap_main.read()
    ret_s, frame_s = cap_sub.read()

    # ★ 取得したフレームを180度反転させる (-1 は上下左右反転)
    if ret_m:
        frame_m = cv2.flip(frame_m, -1)
    if ret_s:
        frame_s = cv2.flip(frame_s, -1)

    # 1. 真っ暗な表示用画面を作成 (黒背景)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # 現在の経過時間を計算
    if is_running:
        current_time = time.time()
        diff = current_time - start_time
        
        m = int(diff // 60)
        s = int(diff % 60)
        ms = int((diff - int(diff)) * 100)
        
        elapsed_str = f"{m:02d}:{s:02d}.{ms:02d}"
        print(f"\r⏱️ {elapsed_str}", end="")
        
        # --- 録画処理 (反転済みのフレームを書き込み) ---
        if ret_m and writer_main is not None:
            writer_main.write(frame_m)
        if ret_s and writer_sub is not None:
            writer_sub.write(frame_s)
        
        # 画面中央に表示
        text = elapsed_str
        font_scale = 3
        thickness = 5
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (WIDTH - text_w) // 2
        y = (HEIGHT + text_h) // 2
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_GREEN, thickness)
        
        cv2.circle(img, (30, 30), 10, COLOR_RED, -1)
        cv2.putText(img, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        
    else:
        msg = "Press [ENTER] to Start"
        (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (WIDTH - text_w) // 2
        y = (HEIGHT + text_h) // 2
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)

    cv2.imshow(WINDOW_NAME, img)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break
    
    elif key == 13:  # Enter
        if not is_running:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            now = time.strftime("%Y%m%d_%H%M%S")
            
            if ret_m:
                h, w = frame_m.shape[:2]
                writer_main = cv2.VideoWriter(f'main_{now}.avi', fourcc, FPS, (w, h))
            if ret_s:
                h, w = frame_s.shape[:2]
                writer_sub = cv2.VideoWriter(f'sub_{now}.avi', fourcc, FPS, (w, h))

            start_time = time.time()
            is_running = True
            print("\n▶️ 計測・録画スタート")
        else:
            duration = time.time() - start_time
            if writer_main: writer_main.release(); writer_main = None
            if writer_sub: writer_sub.release(); writer_sub = None

            is_running = False
            print(f"\n⏹️ 計測終了")
            print(f"📁 保存完了 (180度反転済み)")
            print("-" * 20)

cap_main.release()
cap_sub.release()
cv2.destroyAllWindows()