# 改善前の支援なし時時間計測

import cv2
import numpy as np
import time

# ==========================================
# 設定
# ==========================================
WINDOW_NAME = "Black Screen Timer"
WIDTH = 640
HEIGHT = 480

# 色設定 (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# ==========================================
# 変数初期化
# ==========================================
is_running = False
start_time = 0.0
elapsed_str = "00:00.00" # 初期表示も桁合わせ

cv2.namedWindow(WINDOW_NAME)

print("="*40)
print("⏱️ ブラック画面タイマー (高精度版)")
print("   - [ENTER]: 計測開始 / 停止 (結果をここに表示)")
print("   - [ESC]:   終了")
print("="*40)

while True:
    # 1. 真っ暗な画面を作成 (黒背景)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # 現在の経過時間を計算
    if is_running:
        current_time = time.time()
        diff = current_time - start_time
        
        # 分・秒・ミリ秒の計算
        m = int(diff // 60)
        s = int(diff % 60)
        ms = int((diff - int(diff)) * 100) # 小数点以下2桁
        
        elapsed_str = f"{m:02d}:{s:02d}.{ms:02d}"
        
        # ★コマンドプロンプトにリアルタイム表示（同じ行を上書き）
        print(f"\r⏱️ {elapsed_str}", end="")
        
        # 画面中央に大きく時間を表示
        text = elapsed_str
        font_scale = 3
        thickness = 5
        color = COLOR_GREEN
        
        # テキストサイズを取得して中央揃え
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = (WIDTH - text_w) // 2
        y = (HEIGHT + text_h) // 2
        
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # 録画中(計測中)マーク
        cv2.circle(img, (30, 30), 10, COLOR_RED, -1)
        cv2.putText(img, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        
    else:
        # 待機中の表示
        msg = "Press [ENTER] to Start"
        (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (WIDTH - text_w) // 2
        y = (HEIGHT + text_h) // 2
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)

    cv2.imshow(WINDOW_NAME, img)

    # ==========================================
    # キー入力処理
    # ==========================================
    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESCキーで終了
        break
    
    elif key == 13:  # Enterキー
        if not is_running:
            # --- 計測開始 ---
            start_time = time.time()
            is_running = True
            print("\n▶️ 計測スタート")
        else:
            # --- 計測停止 ---
            end_time = time.time()
            duration = end_time - start_time
            
            # 時間計算（高精度）
            m = int(duration // 60)
            s = int(duration % 60)
            ms = int((duration - int(duration)) * 100)
            result_time = f"{m:02d}:{s:02d}.{ms:02d}"
            
            is_running = False
            
            # コマンドプロンプトに表示
            print(f"\n⏹️ 計測終了")
            print(f"⏱️ 計測時間: {result_time}")
            print("-" * 20)

cv2.destroyAllWindows()