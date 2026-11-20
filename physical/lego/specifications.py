"""Every LEGO number: real brick dimensions, how bricks are simulated, and
where the board and the supply plate sit."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Real LEGO dimensions (mm), multiplied by SimParams.scale in the simulation.
PITCH_MM = 8.0            # stud-to-stud spacing
SIDE_GAP_MM = 0.1         # each outer face is inset 0.1 mm from the grid
BRICK_H_MM = 9.6          # brick height (without stud)
PLATE_H_MM = 3.2          # plate height (1/3 brick)
STUD_D_MM = 4.8           # stud diameter
STUD_H_MM = 1.7           # stud height
TOP_THICK_MM = 1.0        # thickness of the top deck
# Outer wall thickness is derived so that a stud just touches the inner wall
# (half-width 3.9 - stud reach 2.4 = 1.5 mm); clearance is then removed.
ABS_DENSITY = 1050.0      # kg/m^3
# Real 2x4 brick weighs ~2.3 g, i.e. ~46% of its solid envelope.
FILL_FRACTION = 0.46

# Colors (RGBA, 0..1). Distinct hues to make the classifier's job tractable.
COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (0.80, 0.07, 0.07, 1.0),
    "blue":   (0.05, 0.30, 0.75, 1.0),
    "yellow": (0.98, 0.80, 0.05, 1.0),
    "green":  (0.10, 0.55, 0.20, 1.0),
    "white":  (0.95, 0.95, 0.95, 1.0),
    "black":  (0.08, 0.08, 0.08, 1.0),
    "orange": (0.95, 0.45, 0.05, 1.0),
    "purple": (0.45, 0.20, 0.60, 1.0),
}


@dataclass(frozen=True)
class SimParams:
    """How the brick is simulated (shared by every brick in a scene)."""
    scale: float = 2.0
    collision: str = "interlock"        # "interlock" | "box"
    clearance_mm: float = 0.15          # real-scale mm, removed from walls/tubes
    mu_static: float = 0.6              # ABS-on-ABS is ~0.3-0.5; a bit higher helps grasping
    mu_dynamic: float = 0.5
    contact_model: str = "point"        # "point" | "hydroelastic"
    hydroelastic_modulus: float = 5.0e7 # Pa, only used for hydroelastic
    hunt_crossley_dissipation: float = 1.0
    mass_override: Optional[float] = None  # kg; None = derived from size


PARAMS = SimParams()
PITCH = PITCH_MM * PARAMS.scale * 1e-3  # stud spacing in the simulation (m)

# Baseplate size in studs, and where its (0, 0) corner sits in the world; its
# top surface is at z = 0, centered about 0.6 m in front of the iiwa base.
BOARD_STUDS = (16, 16)
BOARD_ORIGIN = [0.6 - BOARD_STUDS[0] * PITCH / 2, -BOARD_STUDS[1] * PITCH / 2, 0.0]

# Supply plate: where bricks start, at rotation 0, to the robot's right of the
# board. SUPPLY_GAP free studs between neighbors let the open fingers fit.
SUPPLY_STUDS = (16, 12)
SUPPLY_ORIGIN = [0.5 - SUPPLY_STUDS[0] * PITCH / 2, -0.35 - SUPPLY_STUDS[1] * PITCH / 2, 0.0]
SUPPLY_GAP = 2

SPAWN_CLEARANCE = 0.001  # bricks spawn this far above their seat and settle (m)
GENERATED = Path(__file__).parent / "generated_models"  # SDF cache
