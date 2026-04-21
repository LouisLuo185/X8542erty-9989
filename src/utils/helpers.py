from pathlib import Path


def list_csv_files(directory: str | Path) -> list[Path]:
    path = Path(directory)
    return sorted(path.glob("*.csv"))
