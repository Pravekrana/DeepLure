"""Initialize an empty project-owned private workspace without claiming existing data."""
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'private';marker=p/'.project-created'
if p.is_symlink():raise SystemExit('Refusing a symlinked private workspace')
if p.exists() and any(p.iterdir()) and not marker.is_file():
    raise SystemExit('Existing unmarked private files found; preserve them and choose a clean checkout. Ownership is not assumed.')
p.mkdir(exist_ok=True)
if not marker.exists():marker.write_text('Project-created working copies only; never place original user files here.\n')
print('Project-owned private workspace initialized')
