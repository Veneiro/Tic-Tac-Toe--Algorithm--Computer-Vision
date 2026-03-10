import numpy as np

'''
TABLEROS DE EJEMPLO
'''
x1 = np.array([[1,1,0], 
               [1,2,0], 
               [2,0,2]])

x2 = np.array([[1,1,0], 
               [1,2,0],
               [2,0,0]])

x3 = np.array([[0,0,0],
               [0,1,0],
               [0,0,0]])

x4 = np.array([[0,0,0],
               [0,0,0],
               [0,0,0]])

'''
PARÁMETROS DE CONFIGURACIÓN
'''
T = 0.5 #Temperatura de softmax, valor más pequeño = más determinista, valor más grande = más aleatorio
sel_board = x4 # Tablero seleccionado
s_depth = 2 # Profundidad de búsqueda para minimax, valor más grande = mejor jugada pero más lento, valor más pequeño = peor jugada pero más rápido

'''
CÁLCULO DE MOVIMIENTOS POSIBLES PARA O
'''
def possible_moves(board):
    aux = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == 0:
                aux += [(i,j)]  
    return aux

'''
EVALUACIÓN DEL TABLERO Y POSIBLES MOVIMIENTOS
'''
def evaluate(board):
    lines = []

    lines.extend(board)                      # filas
    lines.extend(board.T)                    # columnas
    lines.append(np.diag(board))             # diagonal
    lines.append(np.diag(np.fliplr(board)))  # diagonal inversa

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

def minimax(board, depth, maximizing):
    score = evaluate(board)

    # estados terminales o corte
    if abs(score) == 100 or depth == 0:
        return score

    if maximizing:  # turno de O
        best = -np.inf
        for (i, j) in possible_moves(board):
            b = board.copy()
            b[i, j] = 2
            best = max(best, minimax(b, depth-1, False))
        return best

    else:  # turno de X
        best = np.inf
        for (i, j) in possible_moves(board):
            b = board.copy()
            b[i, j] = 1
            best = min(best, minimax(b, depth-1, True))
        return best

def evaluate_moves(moves, depth=s_depth):
    scores = []

    for (i, j) in moves:
        new_board = sel_board.copy()
        new_board[i, j] = 2  # juega O
        score = minimax(new_board, depth, maximizing=False)
        scores.append(score)

    return moves, np.array(scores)

p_moves = possible_moves(sel_board)

print("Possible moves: \n" + str(p_moves))

e_moves, e_scores = evaluate_moves(p_moves)

print("Evaluated scores, without noise: \n" + str(e_scores))

def softmax(scores, T=T):
    scores = scores - np.max(scores)  # estabilidad numérica
    exp_scores = np.exp(scores / T)
    return exp_scores / np.sum(exp_scores)

print("Softmax probabilities: \n" + str(softmax(e_scores)))

def choose_move_softmax(board, T=T):
    empty = np.count_nonzero(board == 0)

    # Apertura de la partida
    if empty >= 7:
        depth = s_depth - 1
        T = T + 0.3

    # Medio juego
    elif empty >= 4:
        depth = s_depth
        T = T

    # Final
    else:
        depth = s_depth + 2
        T = T - 0.3

    moves, scores = evaluate_moves(board, depth)

    scores = scores.astype(float)
    scores += np.random.normal(0, 0.3, size=len(scores))

    probs = softmax(scores, T)
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx], scores[idx]


c_move, c_score = choose_move_softmax(e_moves)

print("Chosen move: \n" + str(c_move) + " with score: " + str(c_score))