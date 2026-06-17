# ピースの座標決定サポート

import cv2
import numpy as np

# ==========================================
# 設定
# ==========================================
CAM_ID = 0           # サブカメラ（ガイド用）のID
FLIP_MODE = -1       # 上下左右反転が必要な場合（なければ None）

# 作業用の解像度（正方形に変換するサイズ）
WORK_SIZE = 800  

# ==========================================
# グローバル変数
# ==========================================
points_calibration = []
pieces_db = {}
current_piece_points = []
current_piece_id = 0
warp_matrix = None
warped_img_static = None

def mouse_event_calib(event, x, y, flags, param):
    global points_calibration
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_calibration) < 4:
            points_calibration.append([x, y])

def mouse_event_piece(event, x, y, flags, param):
    global current_piece_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_piece_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右クリックでも戻れるように
        if current_piece_points:
            current_piece_points.pop()

def main():
    # ★修正箇所: ここに points_calibration を追加しました
    global warp_matrix, warped_img_static, current_piece_points, current_piece_id, points_calibration

    # 1. カメラ起動
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"カメラ(ID:{CAM_ID})が開けません。設定を確認してください。")
        return

    win_calib = "Step 1: Calibration"
    cv2.namedWindow(win_calib)
    cv2.setMouseCallback(win_calib, mouse_event_calib)

    print("【Step 1】パズル全体の四隅をクリックしてください")
    print("順序: [左上] -> [右上] -> [右下] -> [左下]")
    print("※ [R]キーでリセット、4点決まったら [SPACE] で次へ")

    raw_frame = None

    # --- Step 1: 基準枠（四隅）の決定 ---
    while True:
        ret, frame = cap.read()
        if not ret: break
        if FLIP_MODE is not None:
            frame = cv2.flip(frame, FLIP_MODE)
        
        raw_frame = frame.copy()
        display = frame.copy()

        # クリックした点の描画
        for i, pt in enumerate(points_calibration):
            cv2.circle(display, tuple(pt), 10, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if i > 0:
                cv2.line(display, tuple(points_calibration[i-1]), tuple(pt), (0, 255, 0), 2)
        
        if len(points_calibration) == 4:
            cv2.line(display, tuple(points_calibration[3]), tuple(points_calibration[0]), (0, 255, 0), 2)
            cv2.putText(display, "Press [SPACE] to Next", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(win_calib, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            points_calibration = []
            print("リセットしました")
        elif key == 32 and len(points_calibration) == 4: # SPACE
            break
        elif key == 27: # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow(win_calib)

    # --- 射影変換行列の計算 ---
    # クリックした4点を、WORK_SIZE x WORK_SIZE の正方形に引き伸ばす
    pts1 = np.float32(points_calibration)
    pts2 = np.float32([[0, 0], [WORK_SIZE, 0], [WORK_SIZE, WORK_SIZE], [0, WORK_SIZE]])
    warp_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 画像を変換（これで真上から見たような画像になる）
    warped_img_static = cv2.warpPerspective(raw_frame, warp_matrix, (WORK_SIZE, WORK_SIZE))

    # --- Step 2: 各ピースの座標定義 ---
    win_piece = "Step 2: Define Pieces"
    cv2.namedWindow(win_piece)
    cv2.setMouseCallback(win_piece, mouse_event_piece)

    print("\n【Step 2】各ピースの形状をクリックしてください")
    print("マウス左クリック: 点を追加")
    print("キー操作:")
    print("  [n] : 次のピースへ進む（現在の形状を確定）")
    print("  [z] : 直前の点を取り消し")
    print("  [q] : 終了してコードを出力")

    while True:
        display = warped_img_static.copy()

        # 1. 登録済みピースの描画（薄く表示）
        for pid, pts in pieces_db.items():
            pts_np = np.array(pts, dtype=np.int32)
            # WORK_SIZE座標系に戻す
            pts_px = (pts_np * WORK_SIZE).astype(np.int32)
            cv2.polylines(display, [pts_px], True, (200, 200, 200), 1)
            # 重心に番号
            M = cv2.moments(pts_px)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display, str(pid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 2. 作成中のピースの描画
        if len(current_piece_points) > 0:
            pts_np = np.array(current_piece_points, dtype=np.int32)
            cv2.polylines(display, [pts_np], False, (0, 0, 255), 2)
            for pt in current_piece_points:
                cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)

        # ガイド表示
        info_text = f"ID: {current_piece_id} | Points: {len(current_piece_points)}"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "[n]:Next [z]:Undo [q]:Finish", (10, WORK_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win_piece, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'): # Next Piece
            if len(current_piece_points) >= 3:
                # 0.0〜1.0 の正規化座標に変換して保存
                norm_points = []
                for pt in current_piece_points:
                    nx = round(pt[0] / WORK_SIZE, 4) # 小数点4桁
                    ny = round(pt[1] / WORK_SIZE, 4)
                    norm_points.append([nx, ny])
                
                pieces_db[current_piece_id] = norm_points
                print(f"  -> ID {current_piece_id} 登録完了 ({len(norm_points)}点)")
                
                current_piece_id += 1
                current_piece_points = []
            else:
                print("⚠ 点が足りません（最低3点必要です）")

        elif key == ord('z'): # Undo
            if current_piece_points:
                current_piece_points.pop()
                print("  点を取り消しました")

        elif key == ord('q'): # Quit
            break
        
        elif key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- コード出力 ---
    print("\n" + "="*40)
    print("✅ 以下のコードをメインプログラムの PUZZLE_SHAPES 部分に貼り付けてください")
    print("="*40)
    print("PUZZLE_SHAPES = {")
    for pid, pts in pieces_db.items():
        pts_str = ", ".join([f"[{p[0]:.2f}, {p[1]:.2f}]" for p in pts])
        print(f"    {pid}: np.array([{pts_str}]),")
    print("}")
    print("="*40)

if __name__ == "__main__":
    main()