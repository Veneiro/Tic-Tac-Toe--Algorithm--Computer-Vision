"""
Tres en Raya — Detección con Cámara en Vivo
============================================
Captura fotos desde la cámara del ordenador y detecta el estado del tablero.

Uso:
    python tictactoe_camera.py
    python tictactoe_camera.py --camera 0      # seleccionar cámara específica
    python tictactoe_camera.py --debug         # modo debug
    python tictactoe_camera.py --visualize     # guardar imágenes anotadas

Controles:
    ESPACIO  - Capturar foto y analizar tablero
    Q        - Salir
    S        - Guardar última captura
    C        - Limpiar resultado mostrado

Dependencias:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Importar la función de análisis del módulo principal
from tictactoe_vision_v3 import analyze, board_to_string


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def draw_result_overlay(frame, result):
    """
    Dibuja el resultado del análisis sobre el frame de la cámara.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    if not result['board_found']:
        # Mensaje de "tablero no encontrado"
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 200), 2)
        cv2.putText(overlay, "TABLERO NO ENCONTRADO", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Dibujar el tablero detectado
    board = result['board']
    conf = result['confidence']
    
    # Panel de resultado en la esquina superior izquierda
    panel_w = 280
    panel_h = 320
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 255, 0), 2)
    
    # Título
    cv2.putText(overlay, "TABLERO DETECTADO", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Dibujar cuadrícula 3x3
    cell_size = 70
    grid_x = 30
    grid_y = 60
    
    for row in range(3):
        for col in range(3):
            x = grid_x + col * cell_size
            y = grid_y + row * cell_size
            
            piece = board[row][col]
            confidence = conf[row][col]
            
            # Color según la pieza
            if piece == 'X':
                color = (0, 0, 255)  # Rojo
                text = 'X'
            elif piece == 'O':
                color = (255, 150, 0)  # Azul claro
                text = 'O'
            else:
                color = (100, 100, 100)  # Gris
                text = ''
            
            # Dibujar celda
            cv2.rectangle(overlay, (x, y), (x + cell_size, y + cell_size), color, 2)
            
            # Dibujar pieza
            if text:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x = x + (cell_size - text_size[0]) // 2
                text_y = y + (cell_size + text_size[1]) // 2
                cv2.putText(overlay, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                
                # Mostrar confianza debajo
                conf_text = f"{confidence:.2f}"
                cv2.putText(overlay, conf_text, (x + 5, y + cell_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # String del tablero
    board_str = board_to_string(board)
    cv2.putText(overlay, board_str, (20, grid_y + 3 * cell_size + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Mezclar overlay con el frame original
    return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)


def draw_instructions(frame):
    """
    Dibuja las instrucciones de uso en la parte inferior del frame.
    """
    h, w = frame.shape[:2]
    instructions = [
        "ESPACIO: Capturar y analizar",
        "S: Guardar captura",
        "C: Limpiar resultado",
        "Q: Salir"
    ]
    
    y_start = h - 120
    cv2.rectangle(frame, (10, y_start - 10), (w - 10, h - 10), (0, 0, 0), -1)
    
    for i, text in enumerate(instructions):
        y = y_start + i * 25
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_camera_detection(camera_id=0, debug=False, visualize=False):
    """
    Ejecuta la aplicación de detección con cámara en vivo.
    """
    # Abrir cámara
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la cámara {camera_id}", file=sys.stderr)
        sys.exit(1)
    
    # Configurar resolución (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"[INFO] Cámara {camera_id} abierta correctamente")
    print("[INFO] Presiona ESPACIO para capturar y analizar")
    print("[INFO] Presiona Q para salir")
    
    window_name = "Tic Tac Toe - Detección con Cámara"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    last_result = None
    last_capture = None
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] No se puede leer el frame de la cámara")
            break
        
        # Crear copia para mostrar
        display_frame = frame.copy()
        
        # Si hay un resultado previo, dibujarlo
        if last_result is not None:
            display_frame = draw_result_overlay(display_frame, last_result)
        
        # Dibujar instrucciones
        display_frame = draw_instructions(display_frame)
        
        # Mostrar frame
        cv2.imshow(window_name, display_frame)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            # Salir
            print("[INFO] Saliendo...")
            break
        
        elif key == ord(' '):
            # Capturar y analizar
            print("\n" + "=" * 60)
            print("[INFO] Capturando foto...")
            
            # Guardar captura temporal
            temp_path = f"_temp_capture_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, frame)
            last_capture = temp_path
            capture_count += 1
            
            print(f"[INFO] Analizando captura #{capture_count}...")
            
            try:
                # Analizar la imagen
                result = analyze(temp_path, debug=debug, visualize=visualize)
                
                # Mostrar resultado en consola
                print("\n📸 RESULTADO DEL ANÁLISIS:")
                print(f"   Tablero encontrado: {result['board_found']}")
                
                if result['board_found']:
                    # Solo guardar el resultado si se detectó un tablero
                    last_result = result
                    
                    print("\n   Tablero detectado:")
                    syms = {None: '.', 'X': 'X', 'O': 'O'}
                    for row in result['board']:
                        print("   " + " | ".join(syms[c] for c in row))
                    
                    print(f"\n   String: {board_to_string(result['board'])}")
                else:
                    # No guardar resultado si no se detectó tablero
                    print("   ⚠️  No se pudo detectar el tablero en la imagen")
                    print("   💡 Asegúrate de que el tablero esté completamente visible")
                
            except Exception as e:
                print(f"[ERROR] Error al analizar: {e}")
                last_result = None
            
            print("=" * 60 + "\n")
        
        elif key == ord('s') or key == ord('S'):
            # Guardar última captura
            if last_capture:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"capture_{timestamp}.jpg"
                
                # Leer la captura temporal y guardarla con nombre permanente
                img = cv2.imread(last_capture)
                cv2.imwrite(save_path, img)
                
                print(f"[INFO] 💾 Captura guardada: {save_path}")
            else:
                print("[WARN] No hay ninguna captura para guardar")
        
        elif key == ord('c') or key == ord('C'):
            # Limpiar resultado
            last_result = None
            print("[INFO] Resultado limpiado")
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    
    # Eliminar archivos temporales
    for temp_file in Path(".").glob("_temp_capture_*.jpg"):
        try:
            temp_file.unlink()
        except:
            pass
    
    print("[INFO] Aplicación cerrada")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detección de tablero de tres en raya con cámara en vivo")
    parser.add_argument("--camera", type=int, default=0,
                        help="ID de la cámara a usar (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="Activar modo debug (guardar imágenes intermedias)")
    parser.add_argument("--visualize", action="store_true",
                        help="Guardar imágenes anotadas con resultados")
    args = parser.parse_args()
    
    run_camera_detection(args.camera, args.debug, args.visualize)


if __name__ == "__main__":
    main()
