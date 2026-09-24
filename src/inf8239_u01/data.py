from pathlib import Path
import re
import pandas as pd

# Raíz del proyecto (src/inf8239_u01/data.py -> dos niveles arriba)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Datasets aprobados por el docente (formato owner/slug en Kaggle)
APPROVED = {
    "mujtabamatin/air-quality-and-pollution-assessment",
    "zeyadmohamed26/netflix-customer-churn-and-engagement-analytics",
}


def download_csv(url: str, destination="data/raw/dataset.csv") -> Path:
    """Descarga reproducible del dataset aprobado y lo guarda en `destination`.

    - Si `url` es una página de dataset de Kaggle (requiere autenticación), se
      extrae el slug owner/dataset y se descarga con kagglehub (API).
    - Si `url` es una URL directa a un .csv, se lee con pandas.
    Solo se permiten los datasets aprobados por el docente.
    """
    if not url.startswith(("https://", "http://")):
        raise ValueError("La fuente debe ser una URL HTTP(S)")

    path = _resolve(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "kaggle.com/datasets/" in url:
        frame = _read_from_kaggle(url)
    else:
        frame = pd.read_csv(url)

    if frame.empty:
        raise ValueError("El dataset descargado está vacío")
    frame.to_csv(path, index=False)
    return path


def _resolve(destination) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    p = Path(destination)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read_from_kaggle(url: str) -> pd.DataFrame:
    """Extrae el slug de una URL de Kaggle y descarga el CSV vía API (kagglehub)."""
    match = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?#]+)", url)
    if not match:
        raise ValueError("No se pudo extraer el slug del dataset de Kaggle")
    slug = f"{match.group(1)}/{match.group(2)}"
    if slug not in APPROVED:
        raise ValueError(f"Dataset no aprobado: {slug}")

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Falta kagglehub. Instala con: pip install kagglehub") from exc

    dataset_dir = Path(kagglehub.dataset_download(slug))
    csvs = sorted(dataset_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontró ningún CSV en {dataset_dir}")
    return pd.read_csv(csvs[0])