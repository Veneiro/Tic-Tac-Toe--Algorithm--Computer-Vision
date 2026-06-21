"""
ml_remove_blurry_runs.py
------------------------
Mide el desenfoque de la imagen original de cada run (varianza del Laplaciano)
y elimina las que estén por debajo del umbral.

Por defecto hace un DRY-RUN (solo muestra qué se borraría).
Usa --delete para borrar de verdad.

Uso:
  python ml_remove_blurry_runs.py
  python ml_remove_blurry_runs.py --threshold 80
  python ml_remove_blurry_runs.py --threshold 80 --delete
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def blur_score(image_path: Path) -> float:
    """Varianza del Laplaciano: cuanto mayor, más nítida la imagen."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", default="debug_steps")
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Umbral de varianza del Laplaciano (default: 60). "
             "Runs con score < umbral se consideran desenfocadas.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Borrar de verdad las runs desenfocadas (sin este flag es dry-run).",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    runs = sorted(debug_dir.glob("run_*"))

    if not runs:
        print(f"No se encontraron runs en '{debug_dir}'.")
        return

    scores = []
    for run in runs:
        img_path = run / "001_01_original_image.png"
        if not img_path.exists():
            # Buscar cualquier imagen PNG en el run
            pngs = list(run.glob("*.png"))
            img_path = pngs[0] if pngs else None

        score = blur_score(img_path) if img_path else 0.0
        scores.append((run, score))

    # Ordenar de más desenfocada a más nítida
    scores.sort(key=lambda x: x[1])

    print(f"\n{'Run':<45} {'Score':>8}  {'Estado'}")
    print("-" * 70)
    to_delete = []
    for run, score in scores:
        estado = "BORRAR" if score < args.threshold else "ok"
        marker = " <--" if score < args.threshold else ""
        print(f"{run.name:<45} {score:>8.1f}  {estado}{marker}")
        if score < args.threshold:
            to_delete.append(run)

    print("-" * 70)
    print(f"\nTotal runs: {len(scores)}")
    print(f"Umbral:     {args.threshold}")
    print(f"A borrar:   {len(to_delete)}")

    if not to_delete:
        print("\nNinguna run supera el umbral de desenfoque.")
        return

    if not args.delete:
        print(
            "\n[DRY-RUN] No se ha borrado nada. "
            "Añade --delete para borrar de verdad."
        )
        return

    print("\nBorrando runs desenfocadas...")
    for run in to_delete:
        shutil.rmtree(run)
        print(f"  Borrada: {run.name}")
    print(f"\nEliminadas {len(to_delete)} runs.")


if __name__ == "__main__":
    main()
