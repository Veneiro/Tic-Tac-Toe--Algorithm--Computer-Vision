import cv2
import os
import random

def crear_dataset_aumentado(directorio_raiz, directorio_destino, 
                            variaciones_por_imagen=1):
    """
    Genera variaciones realistas para entrenamiento de modelos de IA.
    Aplanando la estructura y evitando distorsiones extremas.
    """
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.webp')
    
    # RANGOS PARA DATASET (Sutiles y realistas)
    # Alpha (Contraste): 0.8 es ligeramente lavado, 1.2 es un poco más de fuerza.
    # Beta (Brillo): -20 es sombra leve, 20 es luz de día clara.
    RANGO_ALPHA = (0.85, 1.15) 
    RANGO_BETA = (-15, 15)

    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    print(f"Generando dataset en: {directorio_destino}")
    contador = 0

    for root, _, files in os.walk(directorio_raiz):
        for nombre_archivo in files:
            if nombre_archivo.lower().endswith(extensiones_validas):
                ruta_entrada = os.path.join(root, nombre_archivo)
                img = cv2.imread(ruta_entrada)

                if img is not None:
                    # Hacemos N variaciones por cada imagen original
                    for i in range(variaciones_por_imagen):
                        alpha = random.uniform(*RANGO_ALPHA)
                        beta = random.randint(*RANGO_BETA)

                        # Aplicación de la transformación
                        resultado = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                        
                        # Nombre único para el dataset: ID_Alpha_Beta_NombreOriginal
                        nombre_final = f"{contador}_a{alpha:.2f}_b{beta}_{nombre_archivo}"
                        cv2.imwrite(os.path.join(directorio_destino, nombre_final), resultado)
                        contador += 1
                
                if contador % 100 == 0:
                    print(f"Procesadas {contador} imágenes...")

    print(f"\n✔ Dataset finalizado con {contador} imágenes.")

# --- PARÁMETROS ---
crear_dataset_aumentado(
    directorio_raiz='capturas', 
    directorio_destino='dataset_preparado',
    variaciones_por_imagen=2  # Si quieres duplicar el tamaño de tu dataset
)