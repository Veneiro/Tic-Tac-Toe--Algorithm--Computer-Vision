import argparse
import csv
import os


def build_board_from_row(row):
    board = [[None] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            key = f'c{r}{c}'
            value = (row.get(key) or '').strip()
            if value == '1':
                board[r][c] = 'X'
            elif value == '2':
                board[r][c] = 'O'
            elif value == '0':
                board[r][c] = None
            else:
                return None
    return board


def board_text(board):
    lines = []
    lines.append(str(board))
    mapping = {None: '0', 'X': '1', 'O': '2'}
    rows = []
    for row in board:
        rows.append(','.join(mapping[cell] for cell in row))
    lines.append('tablero={' + ';'.join(rows) + '}')
    return '\n'.join(lines) + '\n'


def apply_sheet(debug_dir, csv_path, require_full=True):
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    updated = 0
    skipped = 0

    for row in rows:
        run_name = (row.get('run') or '').strip()
        if not run_name:
            skipped += 1
            continue

        board = build_board_from_row(row)
        if board is None:
            if require_full:
                skipped += 1
                continue
            else:
                continue

        run_dir = os.path.join(debug_dir, run_name)
        if not os.path.isdir(run_dir):
            skipped += 1
            continue

        out_path = os.path.join(run_dir, 'estado_real_tablero.txt')
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write(board_text(board))
        updated += 1

    print(f'Runs actualizadas: {updated}')
    print(f'Runs omitidas: {skipped}')


def main():
    parser = argparse.ArgumentParser(description='Aplica etiquetas revisadas CSV a estado_real_tablero.txt de cada run')
    parser.add_argument('--debug-dir', default='debug_steps')
    parser.add_argument('--csv', default='ml_manual/labels_review.csv')
    parser.add_argument('--allow-partial', action='store_true', help='No requiere todas las celdas completas')
    args = parser.parse_args()

    apply_sheet(args.debug_dir, args.csv, require_full=not args.allow_partial)


if __name__ == '__main__':
    main()
