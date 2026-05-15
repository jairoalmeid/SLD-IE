from pathlib import Path

import pandas as pd
import yaml

COLLECTIONS_DIR = Path("data/collections")


def list_collections() -> list[str]:
    if not COLLECTIONS_DIR.exists():
        return []
    return sorted(p.name for p in COLLECTIONS_DIR.iterdir() if p.is_dir())


def _parse_frontmatter(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
    if end is None:
        return {}
    try:
        return yaml.safe_load("\n".join(lines[1:end])) or {}
    except Exception:
        return {}


def load_collection(name: str) -> pd.DataFrame:
    path = COLLECTIONS_DIR / name
    records = []
    for md_file in sorted(path.glob("*.md")):
        data = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
        if data:
            data["_arquivo"] = md_file.stem
            records.append(data)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    return df
