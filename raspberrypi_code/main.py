from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# ==========================================
# CONFIGURACIÓN (RANGOS DE COLOR)
# ==========================================
RED_L1 = np.array([0,   70, 55])
RED_U1 = np.array([12, 255, 255])
RED_L2 = np.array([163, 70, 55])
RED_U2 = np.array([180, 255, 255])
BLUE_L = np.array([85,  35, 75])
BLUE_U = np.array([118, 255, 255])
WHITE_L = np.array([0,   0,  165])
WHITE_U = np.array([180, 55, 255])
COLOR_THRESH = 0.055
FOCUS_MIN_VAR = 85.0
TARGET_WARP_SIDE = 520

T_SOFTMAX = 0.5
SEARCH_DEPTH = 2

def order_points(pts):
    pts = np.array(pts, dtype="float32").reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def focus_variance(gray_img):
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

def unsharp_mask(img, sigma=1.1, amount=1.35):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def red_score(hsv_patch, bgr_patch):
    m1 = cv2.inRange(hsv_patch, RED_L1, RED_U1)
    m2 = cv2.inRange(hsv_patch, RED_L2, RED_U2)
    strong_mask = cv2.bitwise_or(m1, m2)

    red_relaxed_l1 = np.array([0, 45, 40], dtype=np.uint8)
    red_relaxed_u1 = np.array([15, 255, 255], dtype=np.uint8)
    red_relaxed_l2 = np.array([160, 45, 40], dtype=np.uint8)
    red_relaxed_u2 = np.array([180, 255, 255], dtype=np.uint8)
    mr1 = cv2.inRange(hsv_patch, red_relaxed_l1, red_relaxed_u1)
    mr2 = cv2.inRange(hsv_patch, red_relaxed_l2, red_relaxed_u2)
    relaxed_mask = cv2.bitwise_or(mr1, mr2)

    b, g, r = cv2.split(bgr_patch)
    red_dom = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16) > 22) & (r > 55)

    area = max(hsv_patch.shape[0] * hsv_patch.shape[1], 1)
    strong_ratio = np.sum(strong_mask > 0) / area
    relaxed_ratio = np.sum(relaxed_mask > 0) / area
    red_dom_ratio = np.sum(red_dom) / area

    return max(strong_ratio, 0.85 * relaxed_ratio, 0.9 * red_dom_ratio)

def blue_score(hsv_patch, bgr_patch):
    strong_mask = cv2.inRange(hsv_patch, BLUE_L, BLUE_U)

    blue_relaxed_l = np.array([80, 25, 55], dtype=np.uint8)
    blue_relaxed_u = np.array([125, 255, 255], dtype=np.uint8)
    relaxed_mask = cv2.inRange(hsv_patch, blue_relaxed_l, blue_relaxed_u)

    b, g, r = cv2.split(bgr_patch)
    blue_dom = (b.astype(np.int16) - np.maximum(g, r).astype(np.int16) > 18) & (b > 50)

    area = max(hsv_patch.shape[0] * hsv_patch.shape[1], 1)
    strong_ratio = np.sum(strong_mask > 0) / area
    relaxed_ratio = np.sum(relaxed_mask > 0) / area
    blue_dom_ratio = np.sum(blue_dom) / area

    return max(strong_ratio, 0.85 * relaxed_ratio, 0.9 * blue_dom_ratio)

def white_fraction(bgr_patch):
    return np.sum(cv2.inRange(cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV), WHITE_L, WHITE_U) > 0) / max(bgr_patch.shape[0] * bgr_patch.shape[1], 1)

def analyze_precapture(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    brightness = float(np.mean(v))
    contrast = float(np.std(v))
    focus_var = focus_variance(gray)
    highlights = float(np.mean((v > 245) & (s < 70)))
    shadows = float(np.mean(v < 40))

    issues = []
    if brightness < 60:
        issues.append('oscura')
    elif brightness > 210:
        issues.append('sobreexpuesta')
    if contrast < 30:
        issues.append('poco_contraste')
    if highlights > 0.08:
        issues.append('reflejos')
    if shadows > 0.22:
        issues.append('sombras_fuertes')
    if focus_var < FOCUS_MIN_VAR:
        issues.append('desenfoque')

    quality_ok = len(issues) == 0
    return {
        'brightness': brightness,
        'contrast': contrast,
        'focus_var': focus_var,
        'highlights_ratio': highlights,
        'shadows_ratio': shadows,
        'issues': issues,
        'ok': quality_ok,
    }

def apply_adaptive_enhancement(img, capture_info):
    enhanced = img.copy()
    brightness = capture_info['brightness']
    contrast = capture_info['contrast']
    focus_var = capture_info['focus_var']

    if brightness < 80:
        gamma = 0.78
    elif brightness > 200:
        gamma = 1.25
    else:
        gamma = 1.0

    if abs(gamma - 1.0) > 1e-3:
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)

    if contrast < 38:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if focus_var < FOCUS_MIN_VAR:
        enhanced = unsharp_mask(enhanced)

    return enhanced

def warp_panel(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_L, WHITE_U)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    white_closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_close)
    white_clean  = cv2.morphologyEx(white_closed, cv2.MORPH_OPEN, k_open)
    cnts, _ = cv2.findContours(white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panel_cnt = None
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < (h * w) * 0.04: continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if min(bw, bh) / max(bw, bh) > 0.5:
            panel_cnt = cnt
            break
            
    if panel_cnt is None: return None, None
    
    rect = cv2.minAreaRect(panel_cnt)
    box = cv2.boxPoints(rect)
    
    # --- CORRECCIÓN AQUÍ ---
    box = np.int32(box) 
    # -----------------------
    
    src = order_points(box)
    side = TARGET_WARP_SIDE
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    M   = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (side, side), flags=cv2.INTER_CUBIC)
    return warped, M

def find_grid_bbox(warped):
    side = warped.shape[0]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(4, int(side * 0.016))
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    black_d = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k_dil)
    cnts, _ = cv2.findContours(black_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    best_box = None
    for cnt in cnts[:8]:
        area = cv2.contourArea(cnt)
        if area < (side ** 2) * 0.05: continue
        gx, gy, gw, gh = cv2.boundingRect(cnt)
        if min(gw, gh) / max(gw, gh) < 0.7: continue
        interior = warped[gy:gy + gh, gx:gx + gw]
        if white_fraction(interior) > 0.4:
            best_box = (gx, gy, gw, gh)
            break
            
    if best_box is None: best_box = (side // 10, side // 10, int(side * 0.8), int(side * 0.8))
    return best_box

def classify_cells(warped, gx, gy, gw, gh):
    side = warped.shape[0]
    whsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    wbgr = warped
    board = [[None] * 3 for _ in range(3)]
    cell_h, cell_w = gh // 3, gw // 3
    MARGIN_H, MARGIN_W = max(int(cell_h * 0.15), 5), max(int(cell_w * 0.15), 5)
    grid_gray = cv2.cvtColor(warped[gy:gy + gh, gx:gx + gw], cv2.COLOR_BGR2GRAY)
    focus_on_grid = focus_variance(grid_gray)
    red_thresh = max(0.032, COLOR_THRESH * 0.72) if focus_on_grid < FOCUS_MIN_VAR else COLOR_THRESH
    blue_thresh = max(0.030, COLOR_THRESH * 0.70) if focus_on_grid < FOCUS_MIN_VAR else COLOR_THRESH
    dominance_ratio = 1.18 if focus_on_grid < FOCUS_MIN_VAR else 1.3

    for row in range(3):
        for col in range(3):
            y1 = max(0, gy + row * cell_h + MARGIN_H)
            y2 = min(side, gy + (row + 1) * cell_h - MARGIN_H)
            x1 = max(0, gx + col * cell_w + MARGIN_W)
            x2 = min(side, gx + (col + 1) * cell_w - MARGIN_W)
            
            patch_hsv = whsv[y1:y2, x1:x2]
            patch_bgr = wbgr[y1:y2, x1:x2]
            if patch_hsv.size == 0: continue
            
            rs = red_score(patch_hsv, patch_bgr)
            bs = blue_score(patch_hsv, patch_bgr)
            
            if rs > red_thresh and rs > bs * dominance_ratio: board[row][col] = 'X'
            elif bs > blue_thresh and bs > rs * dominance_ratio: board[row][col] = 'O'
            
    return board

def board_to_string(board):
    mapping = {None: '0', 'X': '1', 'O': '2'}
    rows = []
    for row in board:
        rows.append(','.join(mapping[cell] for cell in row))
    return 'tablero={' + ';'.join(rows) + '}'

def possible_moves_numeric(board):
    aux = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == 0:
                aux.append((i, j))
    return aux

def evaluate_numeric(board):
    lines = []
    lines.extend(board)
    lines.extend(board.T)
    lines.append(np.diag(board))
    lines.append(np.diag(np.fliplr(board)))

    score = 0
    for line in lines:
        if np.all(line == 2):
            return 100
        if np.all(line == 1):
            return -100

        if np.count_nonzero(line == 2) == 2 and np.count_nonzero(line == 0) == 1:
            score += 10
        if np.count_nonzero(line == 1) == 2 and np.count_nonzero(line == 0) == 1:
            score -= 8

    return score

def minimax_numeric(board, depth, maximizing):
    score = evaluate_numeric(board)

    if abs(score) == 100 or depth == 0:
        return score

    moves = possible_moves_numeric(board)
    if not moves:
        return score

    if maximizing:
        best = -np.inf
        for (i, j) in moves:
            b = board.copy()
            b[i, j] = 2
            best = max(best, minimax_numeric(b, depth - 1, False))
        return best

    best = np.inf
    for (i, j) in moves:
        b = board.copy()
        b[i, j] = 1
        best = min(best, minimax_numeric(b, depth - 1, True))
    return best

def evaluate_moves_numeric(board, depth):
    moves = possible_moves_numeric(board)
    scores = []

    for (i, j) in moves:
        new_board = board.copy()
        new_board[i, j] = 2
        score = minimax_numeric(new_board, depth, maximizing=False)
        scores.append(score)

    return moves, np.array(scores, dtype=float)

def softmax(scores, temperature=T_SOFTMAX):
    if len(scores) == 0:
        return np.array([])
    safe_t = max(float(temperature), 1e-6)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted / safe_t)
    return exp_scores / np.sum(exp_scores)

def choose_move_softmax(board, temperature=T_SOFTMAX, base_depth=SEARCH_DEPTH):
    empty = int(np.count_nonzero(board == 0))

    if empty >= 7:
        depth = max(1, base_depth - 1)
        temp = temperature + 0.3
    elif empty >= 4:
        depth = base_depth
        temp = temperature
    else:
        depth = base_depth + 2
        temp = max(0.05, temperature - 0.3)

    moves, scores = evaluate_moves_numeric(board, depth)
    if len(moves) == 0:
        return None, None, depth, temp, []

    noisy_scores = scores + np.random.normal(0, 0.3, size=len(scores))
    probs = softmax(noisy_scores, temp)
    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx], float(noisy_scores[idx]), depth, temp, probs.tolist()

# ==========================================
# SERVIDOR WEB (FLASK)
# ==========================================
@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        data = request.data
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None: return "Error imagen", 400

        print("Recibida imagen. Analizando precaptura...")
        capture_info = analyze_precapture(img)
        preprocessed = apply_adaptive_enhancement(img, capture_info)

        warped, _ = warp_panel(preprocessed)
        if warped is None: 
            print("No se encontró tablero")
            return "tablero={Error: No Panel}", 200
        
        gx, gy, gw, gh = find_grid_bbox(warped)
        board = classify_cells(warped, gx, gy, gw, gh)
        resultado = board_to_string(board)
        
        print(f"Precaptura ok={capture_info['ok']} issues={capture_info['issues']}")
        print(f"RESULTADO: {resultado}")
        return resultado

    except Exception as e:
        print(f"Error: {e}")
        return "Error Server", 500

@app.route('/precaptura', methods=['POST'])
def precaptura():
    try:
        data = request.data
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Error imagen'}), 400

        capture_info = analyze_precapture(img)
        return jsonify(capture_info), 200

    except Exception as e:
        print(f"Error en /precaptura: {e}")
        return jsonify({'error': 'Error Server'}), 500

@app.route('/movimiento', methods=['POST'])
def movimiento():
    try:
        payload = request.get_json(silent=True)
        if not payload or 'matriz' not in payload:
            return jsonify({'error': 'Debes enviar JSON con la clave "matriz"'}), 400

        matrix = np.array(payload['matriz'])
        if matrix.shape != (3, 3):
            return jsonify({'error': 'La matriz debe ser de tamaño 3x3'}), 400

        if not np.isin(matrix, [0, 1, 2]).all():
            return jsonify({'error': 'La matriz solo puede contener valores 0, 1 y 2'}), 400

        matrix = matrix.astype(int)
        move, score, depth, temperature, probabilities = choose_move_softmax(matrix)

        if move is None:
            return jsonify({
                'movimiento': None,
                'score': None,
                'mensaje': 'No hay movimientos posibles'
            }), 200

        return jsonify({
            'movimiento': {'fila': int(move[0]), 'columna': int(move[1])},
            'score': score,
            'depth': depth,
            'temperatura': temperature,
            'probabilidades': probabilities
        }), 200

    except Exception as e:
        print(f"Error en /movimiento: {e}")
        return jsonify({'error': 'Error Server'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)