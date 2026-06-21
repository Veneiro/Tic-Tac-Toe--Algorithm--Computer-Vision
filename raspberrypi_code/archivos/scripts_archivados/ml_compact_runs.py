import argparse
import ast
import glob
import os
import re

import cv2
import joblib

from ml_build_dataset import extract_features


INT_TO_LABEL = {0: None, 1: 'X', 2: 'O'}
LABEL_TO_INT = {None: 0, 'X': 1, 'O': 2}


def find_original_image(run_dir):
    patterns = [
        '*_01_original_image.png',
        '*original_image.png',
    ]
    for pattern in patterns:
        candidates = sorted(glob.glob(os.path.join(run_dir, pattern)))
        if candidates:
            return candidates[0]
    pngs = sorted(glob.glob(os.path.join(run_dir, '*.png')))
    return pngs[0] if pngs else None


def parse_board_file(run_dir):
    board_files = glob.glob(os.path.join(run_dir, '*classification_board.txt'))
    if not board_files:
        return None
    try:
        board = ast.literal_eval(open(board_files[0], 'r', encoding='utf-8').read().strip())
    except Exception:
        return None
    if not (isinstance(board, list) and len(board) == 3 and all(isinstance(row, list) and len(row) == 3 for row in board)):
        return None
    return board


def predict_board_from_patches(run_dir, model):
    if model is None:
        return None

    patch_paths = sorted(glob.glob(os.path.join(run_dir, '*cell_*_patch.png')))
    if len(patch_paths) < 9:
        return None

    board = [[None] * 3 for _ in range(3)]
    pattern = re.compile(r'cell_(\d)_(\d)_patch\.png$')
    filled = 0

    for path in patch_paths:
        match = pattern.search(path)
        if not match:
            continue
        row = int(match.group(1))
        col = int(match.group(2))
        if row > 2 or col > 2:
            continue

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        features = extract_features(image).reshape(1, -1)
        pred = int(model.predict(features)[0])
        board[row][col] = INT_TO_LABEL.get(pred, None)
        filled += 1

    if filled < 9:
        return None
    return board


def board_to_text(board):
    rows = ['[' + ', '.join('None' if value is None else repr(value) for value in row) + ']' for row in board]
    pretty = '[\n  ' + ',\n  '.join(rows) + '\n]'
    numeric_rows = []
    for row in board:
        numeric_rows.append(','.join(str(LABEL_TO_INT.get(value, 0)) for value in row))
    numeric = 'tablero={' + ';'.join(numeric_rows) + '}'
    return pretty + '\n' + numeric + '\n'


def compact_runs(debug_dir, model_path=None, dry_run=False):
    model = None
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)

    runs = sorted([path for path in glob.glob(os.path.join(debug_dir, 'run_*')) if os.path.isdir(path)])
    kept = 0
    skipped = 0

    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        original_image = find_original_image(run_dir)
        if original_image is None:
            skipped += 1
            print(f'[SKIP] {run_name}: no se encontró imagen original')
            continue

        board = predict_board_from_patches(run_dir, model)
        source = 'model_patches'

        if board is None:
            board = parse_board_file(run_dir)
            source = 'classification_board'

        if board is None:
            skipped += 1
            print(f'[SKIP] {run_name}: no se pudo inferir tablero')
            continue

        state_path = os.path.join(run_dir, 'estado_real_tablero.txt')

        if not dry_run:
            with open(state_path, 'w', encoding='utf-8') as file:
                file.write(board_to_text(board))

            keep_set = {os.path.abspath(original_image), os.path.abspath(state_path)}
            for file_name in os.listdir(run_dir):
                file_path = os.path.join(run_dir, file_name)
                if os.path.isdir(file_path):
                    continue
                if os.path.abspath(file_path) in keep_set:
                    continue
                os.remove(file_path)

        kept += 1
        print(f'[OK] {run_name}: {source}')

    print(f'Runs procesadas: {len(runs)}')
    print(f'Runs compactadas: {kept}')
    print(f'Runs omitidas: {skipped}')


def main():
    parser = argparse.ArgumentParser(description='Deja cada run con imagen original + estado_real_tablero.txt')
    parser.add_argument('--debug-dir', default='debug_steps', help='Directorio con run_*')
    parser.add_argument('--model', default='ml_models/cell_classifier.joblib', help='Modelo para inferir tablero desde patches')
    parser.add_argument('--dry-run', action='store_true', help='Solo simula, no borra archivos')
    args = parser.parse_args()

    compact_runs(args.debug_dir, model_path=args.model, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
