# ピースの輪郭抽出

import cv2
import numpy as np
import os

# ==========================================
# ★設定・定数
# ==========================================
SHAPE_FOLDER = "shapes3" 
OUTPUT_SIZE = 200      # 保存する輪郭画像のサイズ
LOWER_WHITE = np.array([0, 0, 140])  # 白抽出のHSV下限
UPPER_WHITE = np.array([180, 80, 255]) # 白抽出のHSV上限
MIN_AREA = 2000        # ピースとみなす最小面積
MAX_ID = 20            # 登録可能なIDの最大値 (0から20まで、合計21種類)

# ==========================================
# フォルダ準備
# ==========================================
if not os.path.exists(SHAPE_FOLDER):
    os.makedirs(SHAPE_FOLDER)
    print(f"📂 フォルダ '{SHAPE_FOLDER}' を作成しました。")

# ==========================================
# ヘルパー関数
# ==========================================
def preprocess_frame(frame):
    """フレームをHSV変換し、白を抽出してマスクを返す"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    
    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_largest_piece_contour(mask):
    """マスクから最大の輪郭（ピース）を見つける"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    # 面積でフィルタリング
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    
    if not valid_contours:
        return None, None
        
    # 最大面積の輪郭を取得
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # 輪郭の近似（スムージング）処理を施す
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return approx_contour, cv2.contourArea(approx_contour)

def normalize_contour_image(contour, frame_shape):
    """輪郭を切り取り、正方形に正規化された画像として返す"""
    x, y, w, h = cv2.boundingRect(contour)
    
    # 輪郭を囲む長方形の中心を求める
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 正方形のサイズを決定 (周囲に少し余裕を持たせる)
    size = max(w, h) + 20
    
    # 黒い背景の画像を作成
    normalized_image = np.zeros((size, size), dtype=np.uint8)
    
    # 輪郭座標を新しい画像内に移動
    offset_x = size // 2 - center_x
    offset_y = size // 2 - center_y
    
    # 輪郭を描画するために、座標をオフセット
    shifted_contour = contour.copy()
    shifted_contour[:, 0, 0] += offset_x
    shifted_contour[:, 0, 1] += offset_y
    
    # 輪郭の内部を白く塗る
    cv2.drawContours(normalized_image, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    
    # 指定サイズにリサイズして返す
    final_image = cv2.resize(normalized_image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return final_image

# ==========================================
# メイン処理
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けません。")
    exit()

print("=" * 40)
print("🟢 パズルピース輪郭登録ツール (make_shapes)")
print(f"   - 輪郭画像は '{SHAPE_FOLDER}' フォルダに保存されます。")
print("   - カメラに白いピースを一つ置いてください。")
print(f"   - IDは 0 から {MAX_ID} まで設定可能です。")
print("   - [S]キー: ピースの輪郭を現在のIDで保存")
print("   - [ ] ]キー (または ^ キー): IDを増やす")
print("   - [ [ ]キー (または @ キー): IDを減らす")
print("   - [ESC]キー: 終了")
print("=" * 40)

current_id = 0
status_message = "ピースを置いて、IDを決めてSキーを押してください。"

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    
    # 1. 前処理と輪郭検出
    mask = preprocess_frame(frame)
    contour, area = find_largest_piece_contour(mask)
    
    if contour is not None:
        # 輪郭の描画と情報表示
        cv2.drawContours(display_frame, [contour], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(display_frame, f"Area: {area:.0f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 2. 輪郭の正規化画像を生成してプレビュー
        preview_img = normalize_contour_image(contour, frame.shape)
        
        # 3. プレビューを画面右上に表示
        h_prev, w_prev = preview_img.shape
        display_frame[10:10+h_prev, frame.shape[1]-w_prev-10:frame.shape[1]-10] = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(display_frame, f"Preview (ID: {current_id})", (frame.shape[1]-w_prev-10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 4. ステータス表示
    cv2.putText(display_frame, f"ID: {current_id} | Status: {status_message}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Shape Registration Tool", display_frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # ESCキーで終了
        break
    
    elif key == ord(']') or key == ord('^'): # ]または^キーでIDを増やす
        if current_id < MAX_ID:
            current_id += 1
            status_message = f"IDを {current_id} に設定しました。Sキーで保存。"
        else:
            status_message = f"IDが上限 ({MAX_ID}) に達しました。"

    elif key == ord('[') or key == ord('@'): # [または@キーでIDを減らす
        if current_id > 0:
            current_id -= 1
            status_message = f"IDを {current_id} に設定しました。Sキーで保存。"
        else:
            status_message = "IDは 0 が最小値です。"

    elif key == ord('s') or key == ord('S'): # Sキーで保存
        if contour is not None and area > MIN_AREA:
            filename = os.path.join(SHAPE_FOLDER, f"{current_id}.png")
            cv2.imwrite(filename, preview_img)
            status_message = f"輪郭を ID={current_id} として保存しました: {filename}"
            print(status_message)
            # 保存後、次のIDに進める (ただし上限を超えないように)
            if current_id < MAX_ID:
                 current_id += 1
        else:
            status_message = "ピースが認識できていません。画面を確認してください。"

cap.release()
cv2.destroyAllWindows()