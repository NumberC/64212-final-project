"""Final configurations: the structures to build, as which brick goes where.

One CSV line per brick: W, L, x, y, rotation[, level[, color]]
  W, L      size in studs, W >= L (a 2x4 is a 4x2 rotated)
  x, y      stud cell of the brick's corner on the board
  rotation  0: covers [x, x+W) x [y, y+L);  1 (90 degrees): [x, x+L) x [y, y+W)
  level     stacking layer, 0 = on the baseplate (default 0)
  color     one of specifications.COLORS (default: cycles through them)
Blank lines and lines starting with '#' are ignored.
"""

import csv
from pathlib import Path

from physical.lego.specifications import COLORS

DIR = Path(__file__).parent / "final_configurations"


def names():
    return sorted(path.stem for path in DIR.glob("*.csv"))


def load(name):
    """Bricks (W, L, x, y, rotation, level, color) from a name in DIR or a CSV path."""
    path = Path(name) if Path(name).exists() else DIR / f"{name}.csv"
    final_configuration = []
    with open(path) as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row]
            if not row or not row[0] or row[0].startswith("#"):
                continue
            W, L, x, y, rot = map(int, row[:5])
            if W < L:
                raise ValueError(f"{path}: W must be >= L, got {W}x{L}")
            if rot not in (0, 1):
                raise ValueError(f"{path}: rotation must be 0 or 1, got {rot}")
            level = int(row[5]) if len(row) > 5 and row[5] else 0
            cycled = list(COLORS)[len(final_configuration) % len(COLORS)]
            color = row[6] if len(row) > 6 and row[6] else cycled
            final_configuration.append((W, L, x, y, rot, level, color))
    return final_configuration
