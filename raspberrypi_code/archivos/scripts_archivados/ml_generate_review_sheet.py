import argparse
import ast
import csv
import glob
import os


def parse_estado(path):
    try:
        content = open(path, 'r', encoding='utf-8').read().strip().splitlines()
    except Exception:
        return None

    if not content:
        return None

    # Busca primero bloque tipo lista Python
    try:
        for i in range(len(content)):
            chunk = '\n'.join(content[i:]).strip()
            if chunk.startswith('['):
                board = ast.literal_eval(chunk.split('\n')[0] if chunk.count('[') == 1 else chunk)
                if isinstance(board, list) and len(board) == 3:
                    return board
    except Exception:
        pass

    # Fallback: formato tablero={...}
    for line in content:
        if line.startswith('tablero={') and line.endswith('}'):
            inner = line[len('tablero={'):-1]
            rows = inner.split(';')
            if len(rows) != 3:
                return None
            board = []
            for row in rows:
                vals = row.split(',')
                if len(vals) != 3:
                    return None
                mapped = []
                for v in vals:
                    v = v.strip()
                    if v == '1':
                        mapped.append('X')
                    elif v == '2':
                        mapped.append('O')
                    else:
                        mapped.append(None)
                board.append(mapped)
            return board

    return None


def board_to_ints(board):
    if not board or len(board) != 3:
        return [None] * 9
    out = []
    for r in range(3):
        for c in range(3):
            v = board[r][c]
            if v == 'X':
                out.append('1')
            elif v == 'O':
                out.append('2')
            else:
                out.append('0')
    return out


def generate_sheet(debug_dir, out_csv, with_suggestions=False):
    runs = sorted(glob.glob(os.path.join(debug_dir, 'run_*')))
    rows = []

    for run in runs:
        run_name = os.path.basename(run)
        originals = sorted(glob.glob(os.path.join(run, '*original_image.png')))
        if not originals:
            originals = sorted(glob.glob(os.path.join(run, '*_01_original_image.png')))
        if not originals:
            continue

        if with_suggestions:
            estado_path = os.path.join(run, 'estado_real_tablero.txt')
            board = parse_estado(estado_path) if os.path.exists(estado_path) else None
            sugg = board_to_ints(board)
        else:
            sugg = [''] * 9

        row = {
            'run': run_name,
            'image': os.path.relpath(originals[0], os.path.dirname(out_csv) or '.'),
        }
        for i, key in enumerate([f'c{r}{c}' for r in range(3) for c in range(3)]):
            row[key] = ''
            row[f's_{key}'] = sugg[i] if sugg[i] is not None else ''
        rows.append(row)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    fields = ['run', 'image']
    fields += [f'c{r}{c}' for r in range(3) for c in range(3)]
    fields += [f's_c{r}{c}' for r in range(3) for c in range(3)]

    with open(out_csv, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Runs incluidas: {len(rows)}')
    print(f'CSV generado: {out_csv}')
    if with_suggestions:
        print('Rellena c00..c22 con 0/1/2. Puedes usar s_c00..s_c22 como sugerencia.')
    else:
        print('Rellena c00..c22 con 0/1/2 usando solo la imagen original (sin sugerencias).')


def main():
    parser = argparse.ArgumentParser(description='Genera hoja CSV para corrección masiva de etiquetas')
    parser.add_argument('--debug-dir', default='debug_steps')
    parser.add_argument('--out', default='ml_manual/labels_review.csv')
    parser.add_argument('--with-suggestions', action='store_true', help='Rellena columnas s_c** usando estado_real_tablero.txt')
    args = parser.parse_args()
    generate_sheet(args.debug_dir, args.out, with_suggestions=args.with_suggestions)


if __name__ == '__main__':
    main()
