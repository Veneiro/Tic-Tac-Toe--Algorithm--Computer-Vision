from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# CONFIGURACIÓN DE IA
# ==========================================
# Cargar el modelo entrenado (deberás poner tu archivo .pt aquí cuando lo entrenes)
model = YOLO('best.pt') 

# T_SOFTMAX Y SEARCH_DEPTH (Mantenemos tu configuración)
T_SOFTMAX = 0.5
SEARCH_DEPTH = 2

# ==========================================
# LÓGICA DEL JUEGO (Mantenida intacta)
# ==========================================
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
        if np.all(line == 2): return 100
        if np.all(line == 1): return -100
        if np.count_nonzero(line == 2) == 2 and np.count_nonzero(line == 0) == 1: score += 10
        if np.count_nonzero(line == 1) == 2 and np.count_nonzero(line == 0) == 1: score -= 8
    return score

def minimax_numeric(board, depth, maximizing):
    score = evaluate_numeric(board)
    if abs(score) == 100 or depth == 0: return score

    moves = possible_moves_numeric(board)
    if not moves: return score

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
    if len(scores) == 0: return np.array([])
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
# NUEVA VISIÓN POR COMPUTADOR (YOLO)
# ==========================================
def analizar_tablero_yolo(img):
    """
    Pasa la imagen por YOLO, detecta el tablero y las fichas, 
    y devuelve la matriz 3x3 formateada como string.
    """
    # 1. Hacer la inferencia con YOLO
    resultados = model(img, conf=0.5)[0] # conf=0.5 filtra detecciones dudosas
    
    tablero_box = None
    fichas = []

    # 2. Extraer las cajas delimitadoras (Bounding Boxes)
    for box in resultados.boxes:
        clase = int(box.cls[0]) # 0: tablero, 1: roja, 2: azul (según entrenes)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        if clase == 0:
            tablero_box = (x1, y1, x2, y2)
        else:
            # Guardamos la clase (1 o 2) y el centroide de la ficha
            centro_x = (x1 + x2) / 2
            centro_y = (y1 + y2) / 2
            fichas.append((clase, centro_x, centro_y))

    if tablero_box is None:
        return "tablero={Error: No Panel}"

    # 3. Crear matriz 3x3 vacía
    matriz = np.zeros((3, 3), dtype=int)
    
    # 4. Calcular el tamaño de las celdas basado en la caja del tablero
    tx1, ty1, tx2, ty2 = tablero_box
    ancho_celda = (tx2 - tx1) / 3
    alto_celda = (ty2 - ty1) / 3

    # 5. Colocar cada ficha en su celda correspondiente
    for clase, cx, cy in fichas:
        # Verificar que el centro de la ficha esté dentro del tablero
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            columna = int((cx - tx1) // ancho_celda)
            fila = int((cy - ty1) // alto_celda)
            
            # Asegurar que los índices no se salgan de [0, 1, 2] por márgenes
            fila = max(0, min(2, fila))
            columna = max(0, min(2, columna))
            
            matriz[fila][columna] = clase

    # 6. Formatear la salida al string que esperas
    rows_str = []
    for row in matriz:
        rows_str.append(','.join(str(cell) for cell in row))
    
    return 'tablero={' + ';'.join(rows_str) + '}'


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

        print("Recibida imagen. Analizando con YOLO...")
        
        # Nueva llamada directa a YOLO
        resultado = analizar_tablero_yolo(img)
        
        print(f"RESULTADO: {resultado}")
        return resultado, 200

    except Exception as e:
        print(f"Error: {e}")
        return "Error Server", 500

# Mantenemos las rutas de movimiento intactas porque ya procesan JSON correctamente
@app.route('/movimiento', methods=['POST'])
def movimiento():
    try:
        payload = request.get_json(silent=True)
        if not payload or 'matriz' not in payload:
            return jsonify({'error': 'Debes enviar JSON con la clave "matriz"'}), 400

        matrix = np.array(payload['matriz'])
        if matrix.shape != (3, 3):
            return jsonify({'error': 'La matriz debe ser de tamaño 3x3'}), 400

        matrix = matrix.astype(int)
        move, score, depth, temperature, probabilities = choose_move_softmax(matrix)

        if move is None:
            return jsonify({'movimiento': None, 'mensaje': 'Fin'}), 200

        return jsonify({
            'movimiento': {'fila': int(move[0]), 'columna': int(move[1])},
            'score': score
        }), 200

    except Exception as e:
        print(f"Error en /movimiento: {e}")
        return jsonify({'error': 'Error Server'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)