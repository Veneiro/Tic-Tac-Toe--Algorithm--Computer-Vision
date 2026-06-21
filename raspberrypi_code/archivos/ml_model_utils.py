"""
Utilidades compartidas para el modelo ML: constantes de features y transformadores
sklearn serializables con joblib desde cualquier script.

Importar SIEMPRE este módulo antes de hacer joblib.load() con modelos que usen
ColorFeatureSelector.
"""
from sklearn.base import BaseEstimator, TransformerMixin

# ─── Índices del vector de features ──────────────────────────────────────────
# gray(576) + H-hist(24) + S-hist(8) + V-hist(8) + color_explicito(9) = 625
GRAY_END  = 24 * 24        # 576
H_END     = GRAY_END + 24  # 600
S_END     = H_END    + 8   # 608
V_END     = S_END    + 8   # 616
COL_END   = V_END    + 9   # 625
COL_OFFSET = V_END         # inicio del bloque de color explícito

FEATURE_DIM = COL_END      # 625


class ColorFeatureSelector(BaseEstimator, TransformerMixin):
    """Selecciona sólo los features de color (H/S/V hist + ratios explícitos).

    Clase sklearn (no lambda/función local) para que joblib pueda serializarla
    y deserializarla correctamente desde cualquier script que importe este módulo.
    """

    def fit(self, x, y=None):  # noqa: ARG002
        return self

    def transform(self, x):
        # Devuelve los 49 features de color: H(24)+S(8)+V(8)+color_explicito(9)
        return x[:, GRAY_END:]
