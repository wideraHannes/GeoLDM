from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()

ANALYSIS_DIR = project_root / "analysis"
EGNN_DIR = project_root / "egnn"
egnn_model = EGNN_DIR / "egnn"
qm9 = project_root / "qm9"
