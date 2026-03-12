# Modelo ligero usando tus runs de debug

Este pipeline reutiliza las imágenes que ya generas en `debug_steps/run_*` para entrenar un clasificador de celdas (`vacía`, `X`, `O`) sin red compleja.

## 1) Instalar dependencias

```bash
pip install -r requirements.txt
```

## Flujo recomendado (etiquetas manuales desde imagen original)

1) Exportar imágenes originales + CSV template:

```bash
python ml_prepare_manual_labels.py --debug-dir debug_steps --out-dir ml_manual
```

2) Editar `ml_manual/labels_manual.csv` y rellenar `c00..c22` con:

- `0` = vacía
- `1` = X
- `2` = O

3) Construir dataset usando ese CSV (sin depender de `classification_board` de runs):

```bash
python ml_build_dataset.py --debug-dir debug_steps --labels-csv ml_manual/labels_manual.csv --out ml_data/cell_dataset_manual.npz --min-tokens 1
```

4) Entrenar con dataset manual:

```bash
python ml_train_cell_classifier.py --dataset ml_data/cell_dataset_manual.npz --model-out ml_models/cell_classifier_manual.joblib
```

## 2) Construir dataset desde runs

```bash
python ml_build_dataset.py --debug-dir debug_steps --out ml_data/cell_dataset.npz --min-tokens 1
```

- Usa `*classification_board.txt` como etiqueta del tablero.
- Usa `*cell_<fila>_<col>_patch.png` como muestra de celda.

## 3) Entrenar modelo

```bash
python ml_train_cell_classifier.py --dataset ml_data/cell_dataset.npz --model-out ml_models/cell_classifier.joblib
```

Esto genera:

- `ml_models/cell_classifier.joblib`
- `ml_models/cell_classifier_metrics.json`

## 4) Evaluar el modelo sobre todas las runs

```bash
python ml_evaluate_runs.py --model ml_models/cell_classifier.joblib --debug-dir debug_steps --out ml_models/eval_runs_report.json
```

Métricas que verás:

- `board_exact_accuracy`: porcentaje de runs donde el tablero completo coincide.
- `cell_accuracy`: accuracy por celda sobre todas las runs.
- `confusion_matrix` y `classification_report` por clase.

## Notas importantes

- Las etiquetas salen de tu pipeline actual (pseudo-etiquetas). Si hay runs con mala detección, pueden meter ruido.
- Para mejorar calidad, borra runs claramente malos o sube `--min-tokens`.
- Cuando quieras, siguiente paso: integrar el modelo en `main.py` para que la decisión final de cada celda sea por modelo (o híbrida: heurística + modelo).
