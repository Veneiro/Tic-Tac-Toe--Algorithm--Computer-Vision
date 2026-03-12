import argparse
import glob
import json
import os
import re

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ml_build_dataset import (
    extract_features,
    resolve_board_labels,
    extract_patches_from_original,
)


INT_TO_LABEL = {0: None, 1: 'X', 2: 'O'}


def evaluate_runs(model_path, debug_dir, out_json=None):
    model = joblib.load(model_path)
    run_dirs = sorted([path for path in glob.glob(os.path.join(debug_dir, 'run_*')) if os.path.isdir(path)])
    pattern = re.compile(r'cell_(\d)_(\d)_patch\.png$')

    y_true = []
    y_pred = []
    run_results = []
    skipped = {
        'missing_or_invalid_board': 0,
        'no_patches_or_extract_fail': 0,
        'incomplete_cells': 0,
    }

    for run_dir in run_dirs:
        try:
            board_true = resolve_board_labels(run_dir)
        except Exception:
            board_true = None

        if board_true is None:
            skipped['missing_or_invalid_board'] += 1
            continue

        patch_files = sorted(glob.glob(os.path.join(run_dir, '*cell_*_patch.png')))
        cells = []

        if patch_files:
            for patch_path in patch_files:
                match = pattern.search(patch_path)
                if not match:
                    continue

                row = int(match.group(1))
                col = int(match.group(2))
                if row > 2 or col > 2:
                    continue

                image = cv2.imread(patch_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                cells.append((row, col, image, patch_path))
        else:
            cells = extract_patches_from_original(run_dir)

        if not cells:
            skipped['no_patches_or_extract_fail'] += 1
            continue

        board_pred = [[0] * 3 for _ in range(3)]
        seen = set()

        for row, col, image, _source in cells:
            if row > 2 or col > 2:
                continue

            features = extract_features(image).astype(np.float32).reshape(1, -1)
            pred_int = int(model.predict(features)[0])
            true_int = int(board_true[row][col])

            board_pred[row][col] = pred_int
            y_true.append(true_int)
            y_pred.append(pred_int)
            seen.add((row, col))

        if len(seen) < 9:
            skipped['incomplete_cells'] += 1
            continue

        diff = sum(1 for r in range(3) for c in range(3) if int(board_true[r][c]) != int(board_pred[r][c]))
        run_results.append({
            'run': os.path.basename(run_dir),
            'board_true': board_true,
            'board_pred': board_pred,
            'board_true_labels': [[INT_TO_LABEL.get(int(v), None) for v in row] for row in board_true],
            'board_pred_labels': [[INT_TO_LABEL.get(int(v), None) for v in row] for row in board_pred],
            'diff_cells': int(diff),
            'board_exact': bool(diff == 0),
        })

    if not y_true:
        raise RuntimeError('No se pudieron evaluar runs válidas')

    cell_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=['empty', 'X', 'O'], digits=4)

    exact_boards = sum(1 for item in run_results if item['board_exact'])
    total_boards = len(run_results)
    board_acc = exact_boards / max(total_boards, 1)

    metrics = {
        'model': model_path,
        'debug_dir': debug_dir,
        'runs_total': len(run_dirs),
        'runs_evaluated': total_boards,
        'board_exact_matches': int(exact_boards),
        'board_exact_accuracy': float(board_acc),
        'cell_samples': int(len(y_true)),
        'cell_accuracy': float(cell_acc),
        'confusion_matrix_0empty_1X_2O': cm.tolist(),
        'skipped': skipped,
        'runs': run_results,
    }

    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f'Runs total: {len(run_dirs)}')
    print(f'Runs evaluadas: {total_boards}')
    print(f"Runs saltadas: {len(run_dirs) - total_boards} -> {skipped}")
    print(f'Board exact accuracy: {board_acc:.4f} ({exact_boards}/{total_boards})')
    print(f'Cell accuracy: {cell_acc:.4f} ({len(y_true)} celdas)')
    print('Confusion matrix (0=empty,1=X,2=O):')
    print(cm)
    print('\nClassification report:')
    print(report)

    worst = sorted(run_results, key=lambda item: item['diff_cells'], reverse=True)[:5]
    if worst:
        print('\nTop runs con más diferencias:')
        for item in worst:
            print(f"- {item['run']}: diff_cells={item['diff_cells']}")

    if out_json:
        print(f'\nReporte guardado en: {out_json}')


def main():
    parser = argparse.ArgumentParser(description='Evalúa el modelo de celdas sobre run_* (capturas o debug_steps)')
    parser.add_argument('--model', default='ml_models/cell_classifier.joblib', help='Ruta del modelo entrenado')
    parser.add_argument('--debug-dir', default='capturas', help='Carpeta con run_* a evaluar')
    parser.add_argument('--out', default='ml_models/eval_runs_report.json', help='JSON de salida con métricas y detalle por run')
    args = parser.parse_args()

    evaluate_runs(args.model, args.debug_dir, out_json=args.out)


if __name__ == '__main__':
    main()
