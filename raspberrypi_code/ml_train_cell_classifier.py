import argparse
import json
import os

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Índices del vector de features (debe coincidir con extract_features en ml_build_dataset.py)
# gray(576) + H-hist(24) + S-hist(8) + V-hist(8) + color_explicito(9) = 625
_GRAY_END  = 24 * 24        # 576
_H_END     = _GRAY_END + 24  # 600
_S_END     = _H_END    + 8   # 608
_V_END     = _S_END    + 8   # 616
_COL_END   = _V_END    + 9   # 625

# Índices dentro del bloque de color explícito (relativo a _V_END)
# [0]=red_ratio [1]=blue_ratio [2]=green_ratio [3]=white_ratio [4]=mean_sat
# [5]=c_red_ratio [6]=c_blue_ratio [7]=c_mean_s [8]=c_mean_v
_COL_OFFSET = _V_END


def augment_dataset(x, y, copies_per_sample=2, noise_std=0.015, random_state=42):
    """Augmenta features. Aplica hue-shift en histograma H para simular variación de
    temperatura de color de la ESP-CAM. Los tokens X/O reciben copias extra."""
    if copies_per_sample <= 0:
        return x, y

    rng = np.random.default_rng(random_state)
    x_aug = [x]
    y_aug = [y]

    # Oversampling: X y O reciben el doble de copias que empty
    mask_x  = (y == 1)
    mask_o  = (y == 2)
    mask_em = (y == 0)

    for copy_i in range(int(copies_per_sample)):
        xn = x.copy().astype(np.float32)

        # ── Gris: brillo + ruido ────────────────────────────────────────────────
        gray = xn[:, :_GRAY_END]
        brightness = rng.uniform(0.85, 1.15, size=(gray.shape[0], 1)).astype(np.float32)
        gray_noise = rng.normal(0.0, noise_std, size=gray.shape).astype(np.float32)
        xn[:, :_GRAY_END] = np.clip((gray * brightness) + gray_noise, 0.0, 1.0)

        # ── H histogram: shift circular ± 2 bins para simular hue drift ────────
        h_hist = xn[:, _GRAY_END:_H_END].copy()
        shift  = rng.integers(-2, 3, size=h_hist.shape[0])  # -2..+2 bins
        for idx, sh in enumerate(shift):
            if sh != 0:
                h_hist[idx] = np.roll(h_hist[idx], sh)
        h_sum = np.sum(h_hist, axis=1, keepdims=True)
        h_sum = np.maximum(h_sum, 1e-6)
        xn[:, _GRAY_END:_H_END] = h_hist / h_sum

        # ── S, V histograms: ruido suave ────────────────────────────────────────
        for start, end in [(_H_END, _S_END), (_S_END, _V_END)]:
            hist = xn[:, start:end]
            hist += rng.normal(0.0, noise_std * 0.3, size=hist.shape).astype(np.float32)
            hist  = np.clip(hist, 0.0, None)
            hs    = np.maximum(np.sum(hist, axis=1, keepdims=True), 1e-6)
            xn[:, start:end] = hist / hs

        # ── Features de color explícitas: ruido muy pequeño (son ratios 0-1) ───
        col = xn[:, _COL_OFFSET:_COL_END]
        col += rng.normal(0.0, noise_std * 0.15, size=col.shape).astype(np.float32)
        xn[:, _COL_OFFSET:_COL_END] = np.clip(col, 0.0, 1.0)

        x_aug.append(xn)
        y_aug.append(y)

        # Copia extra para X y O (clases más difíciles)
        if copy_i == 0:
            for mask_hard in [mask_x, mask_o]:
                if np.sum(mask_hard) > 0:
                    xh = x[mask_hard].copy().astype(np.float32)
                    # Variación de color más agresiva para que el modelo sea robusto
                    colored_brightness = rng.uniform(0.80, 1.20, size=(xh.shape[0], 1)).astype(np.float32)
                    xh[:, :_GRAY_END] = np.clip(
                        xh[:, :_GRAY_END] * colored_brightness
                        + rng.normal(0.0, noise_std * 1.5, size=xh[:, :_GRAY_END].shape).astype(np.float32),
                        0.0, 1.0,
                    )
                    # Shift hue más fuerte para tokens
                    h2 = xh[:, _GRAY_END:_H_END].copy()
                    sh2 = rng.integers(-3, 4, size=h2.shape[0])
                    for idx, sh in enumerate(sh2):
                        if sh != 0:
                            h2[idx] = np.roll(h2[idx], sh)
                    h2s = np.maximum(np.sum(h2, axis=1, keepdims=True), 1e-6)
                    xh[:, _GRAY_END:_H_END] = h2 / h2s
                    col2 = xh[:, _COL_OFFSET:_COL_END]
                    col2 += rng.normal(0.0, noise_std * 0.25, size=col2.shape).astype(np.float32)
                    xh[:, _COL_OFFSET:_COL_END] = np.clip(col2, 0.0, 1.0)
                    x_aug.append(xh)
                    y_aug.append(y[mask_hard])

    return np.vstack(x_aug), np.concatenate(y_aug)


class ColorFeatureSelector(BaseEstimator, TransformerMixin):
    """Selecciona sólo los features de color (H/S/V hist + ratios explícitos).
    Clase sklearn (no lambda) para que joblib pueda serializarla correctamente.
    """
    def fit(self, x, y=None):  # noqa: ARG002
        return self

    def transform(self, x):
        return x[:, _GRAY_END:]  # 49 features: H(24)+S(8)+V(8)+color(9)


def build_model(few_data_mode=False, random_state=42):
    if not few_data_mode:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=5000,
                class_weight='balanced',
                C=1.5,
            )),
        ])

    # LR: bueno con features lineales (las ratios de color son muy lineales)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', C=1.5)),
    ])
    # SVM global (gris + color)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            C=4.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=random_state,
        )),
    ])
    # RF: resistente a ruido, aprovecha las features explícitas de color
    rf = RandomForestClassifier(
        n_estimators=600,
        class_weight='balanced_subsample',
        max_depth=22,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
    )
    # SVM solo-color: ve únicamente los 49 features de H/S/V hist + ratios.
    # Sin el ruido de 576 pixels de gris, discrimina X(rojo) vs O(azul) directamente.
    color_svm = Pipeline([
        ('select', ColorFeatureSelector()),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            C=6.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=random_state,
        )),
    ])

    # color_svm tiene el peso más alto: es el más discriminativo para X vs O
    return VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf), ('color_svm', color_svm)],
        voting='soft',
        weights=[1, 2, 2, 4],
        n_jobs=-1,
    )


def train_model(
    dataset_path,
    model_out,
    test_size=0.2,
    random_state=42,
    few_data_mode=False,
    augment_copies=2,
    augment_noise=0.02,
    kfold=0,
):
    data = np.load(dataset_path, allow_pickle=True)
    x = data['x']
    y = data['y']

    if x.shape[0] < 30:
        raise RuntimeError('Muy pocas muestras para entrenar. Genera más runs primero.')

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    cv_metrics = None
    if int(kfold) and int(kfold) > 1:
        min_class_count = int(np.min(counts))
        effective_folds = min(int(kfold), min_class_count)
        if effective_folds < 2:
            print('K-fold omitido: no hay suficientes muestras por clase.')
        else:
            if effective_folds != int(kfold):
                print(f'K-fold ajustado de {kfold} a {effective_folds} por tamaño de clase mínima ({min_class_count}).')

            skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
            fold_items = []

            for fold_id, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]

                x_train_fit = x_train
                y_train_fit = y_train
                if few_data_mode:
                    x_train_fit, y_train_fit = augment_dataset(
                        x_train,
                        y_train,
                        copies_per_sample=augment_copies,
                        noise_std=augment_noise,
                        random_state=random_state + fold_id,
                    )

                model_fold = build_model(few_data_mode=few_data_mode, random_state=random_state + fold_id)
                model_fold.fit(x_train_fit, y_train_fit)
                pred_val = model_fold.predict(x_val)

                fold_acc = float(accuracy_score(y_val, pred_val))
                fold_cm = confusion_matrix(y_val, pred_val, labels=[0, 1, 2]).tolist()
                fold_items.append({
                    'fold': fold_id,
                    'samples_train': int(x_train.shape[0]),
                    'samples_val': int(x_val.shape[0]),
                    'samples_train_fit': int(x_train_fit.shape[0]),
                    'accuracy': fold_acc,
                    'confusion_matrix_0empty_1X_2O': fold_cm,
                })
                print(f'Fold {fold_id}/{effective_folds} acc: {fold_acc:.4f}')

            cv_accs = [item['accuracy'] for item in fold_items]
            cv_metrics = {
                'enabled': True,
                'requested_folds': int(kfold),
                'effective_folds': int(effective_folds),
                'mean_accuracy': float(np.mean(cv_accs)),
                'std_accuracy': float(np.std(cv_accs)),
                'folds': fold_items,
            }
            print(f"CV mean acc: {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")

    holdout_metrics = None
    if test_size and float(test_size) > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        x_train_fit = x_train
        y_train_fit = y_train
        if few_data_mode:
            x_train_fit, y_train_fit = augment_dataset(
                x_train,
                y_train,
                copies_per_sample=augment_copies,
                noise_std=augment_noise,
                random_state=random_state,
            )

        model_holdout = build_model(few_data_mode=few_data_mode, random_state=random_state)
        model_holdout.fit(x_train_fit, y_train_fit)
        pred = model_holdout.predict(x_test)

        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        report = classification_report(y_test, pred, labels=[0, 1, 2], target_names=['empty', 'X', 'O'], digits=4)
        holdout_metrics = {
            'enabled': True,
            'test_size': float(test_size),
            'samples_train': int(x_train.shape[0]),
            'samples_test': int(x_test.shape[0]),
            'samples_train_fit': int(x_train_fit.shape[0]),
            'accuracy': float(acc),
            'confusion_matrix_0empty_1X_2O': cm.tolist(),
        }

        print(f'Accuracy test: {acc:.4f}')
        print('Confusion matrix (0=empty,1=X,2=O):')
        print(cm)
        print('\nClassification report:')
        print(report)

    x_fit = x
    y_fit = y
    if few_data_mode:
        x_fit, y_fit = augment_dataset(
            x,
            y,
            copies_per_sample=augment_copies,
            noise_std=augment_noise,
            random_state=random_state + 999,
        )

    model = build_model(few_data_mode=few_data_mode, random_state=random_state + 999)
    model.fit(x_fit, y_fit)

    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(model, model_out)

    metrics = {
        'dataset': dataset_path,
        'samples_total': int(x.shape[0]),
        'samples_fit_final': int(x_fit.shape[0]),
        'class_counts_0empty_1X_2O': class_counts,
        'few_data_mode': bool(few_data_mode),
        'augment_copies': int(augment_copies if few_data_mode else 0),
        'augment_noise': float(augment_noise if few_data_mode else 0.0),
        'cv': cv_metrics,
        'holdout': holdout_metrics,
    }
    metrics_path = os.path.splitext(model_out)[0] + '_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f'\nModelo guardado en: {model_out}')
    print(f'Métricas guardadas en: {metrics_path}')


def main():
    parser = argparse.ArgumentParser(description='Entrena clasificador ligero de celdas (empty/X/O)')
    parser.add_argument('--dataset', default='ml_data/cell_dataset_capturas.npz', help='Dataset .npz generado por ml_build_dataset.py')
    parser.add_argument('--model-out', default='ml_models/cell_classifier.joblib', help='Ruta del modelo de salida')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción test')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--few-data', action='store_true', help='Activa modo pocos datos (ensemble + augmentación)')
    parser.add_argument('--augment-copies', type=int, default=2, help='Copias sintéticas por muestra para modo pocos datos')
    parser.add_argument('--augment-noise', type=float, default=0.015, help='Ruido gaussiano de augmentación (modo pocos datos)')
    parser.add_argument('--kfold', type=int, default=5, help='Nº de folds para validación estratificada (0/1 desactiva)')
    args = parser.parse_args()

    train_model(
        args.dataset,
        args.model_out,
        test_size=args.test_size,
        random_state=args.seed,
        few_data_mode=args.few_data,
        augment_copies=args.augment_copies,
        augment_noise=args.augment_noise,
        kfold=args.kfold,
    )


if __name__ == '__main__':
    main()
