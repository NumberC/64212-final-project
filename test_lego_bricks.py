"""Sanity tests for lego_bricks.py.

1. Every preset parses in Drake.
2. Stacking: bricks dropped onto a baseplate settle at their grid poses.
3. Lateral push: a sideways force on a seated top brick. 'interlock' should
   resist (studs hit walls/tubes), 'box' should slide.

Run: python test_lego_bricks.py [--meshcat]
"""
import sys
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, Simulator,
    ExternallyAppliedSpatialForce, SpatialForce, AbstractValue, LeafSystem,
    RigidTransform, StartMeshcat, AddDefaultVisualization,
)
from lego_bricks import (BrickSpec, SimParams, brick_info, add_brick, grid_pose,
                         generate_baseplate_sdf, all_preset_specs, generate_brick_sdf)


class Pusher(LeafSystem):
    """Applies a constant horizontal force to one body after t_on."""
    def __init__(self, body_index, force, t_on):
        super().__init__()
        self.b, self.f, self.t_on = body_index, np.array(force), t_on
        self.DeclareAbstractOutputPort(
            "forces", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.calc)

    def calc(self, context, out):
        F = ExternallyAppliedSpatialForce()
        F.body_index = self.b
        F.p_BoBq_B = np.array([0, 0, 0.0])
        f = self.f if context.get_time() >= self.t_on else np.zeros(3)
        F.F_Bq_W = SpatialForce(np.zeros(3), f)
        out.set_value([F])


def test_parse_all():
    for coll in ["interlock", "box"]:
        for cm in ["point", "hydroelastic"]:
            p = SimParams(collision=coll, contact_model=cm)
            builder = DiagramBuilder()
            plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
            parser = Parser(plant)
            for k, spec in enumerate(all_preset_specs()):
                add_brick(plant, spec, name=f"b{k}", params=p, parser=parser)
            plant.Finalize()
    print("[ok] all presets parse (interlock/box x point/hydroelastic)")


def run_stack(params, push_N=0.0, meshcat=None, T=2.0):
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=5e-4)
    plant.set_discrete_contact_approximation(
        __import__("pydrake.multibody.plant", fromlist=["x"]).DiscreteContactApproximation.kLagged)
    parser = Parser(plant)
    (bp,) = parser.AddModelsFromString(generate_baseplate_sdf(8, 8, params), "sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("plate", bp))

    s24, s22 = BrickSpec(4, 2, "red"), BrickSpec(2, 2, "blue")
    i24, i22 = brick_info(s24, params), brick_info(s22, params)
    bricks = [  # (name, spec, info, gx, gy, level, yaw)
        ("A", s24, i24, 1, 1, 0, 0),
        ("B", s24, i24, 2, 1, 1, 0),    # offset by one stud
        ("C", s22, i22, 3, 1, 2, 90),   # rotated 2x2 on top
    ]
    for name, spec, *_ in bricks:
        add_brick(plant, spec, name=name, params=params, parser=parser)
    plant.Finalize()

    top_body = plant.GetBodyByName("brick", plant.GetModelInstanceByName("C"))
    pusher = builder.AddSystem(Pusher(top_body.index(), [push_N, 0, 0], t_on=T / 2))
    builder.Connect(pusher.get_output_port(), plant.get_applied_spatial_force_input_port())
    if meshcat:
        AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    sim = Simulator(diagram)
    ctx = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())

    targets = {}
    for name, spec, info, gx, gy, lvl, yaw in bricks:
        X = grid_pose(info, gx, gy, lvl, yaw)
        targets[name] = X
        drop = RigidTransform(X.rotation(), X.translation() + [0, 0, 0.001 + 0.0005 * lvl])
        plant.SetFreeBodyPose(ctx, plant.GetBodyByName("brick", plant.GetModelInstanceByName(name)), drop)

    if meshcat:
        meshcat.StartRecording()
    sim.AdvanceTo(T / 2)
    settle = {}
    for name in targets:
        X = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName("brick", plant.GetModelInstanceByName(name)))
        settle[name] = np.linalg.norm(X.translation() - targets[name].translation())
    sim.AdvanceTo(T)
    Xc = plant.EvalBodyPoseInWorld(ctx, top_body)
    push_disp = Xc.translation()[0] - targets["C"].translation()[0]
    if meshcat:
        meshcat.StopRecording(); meshcat.PublishRecording()
    return settle, push_disp


if __name__ == "__main__":
    test_parse_all()
    mc = StartMeshcat() if "--meshcat" in sys.argv else None
    for coll in ["interlock", "box"]:
        p = SimParams(collision=coll)
        settle, _ = run_stack(p)
        ok = all(v < 2e-3 for v in settle.values())
        print(f"[{'ok' if ok else 'FAIL'}] {coll:9s} settle error (mm): "
              + ", ".join(f"{k}={v*1e3:.2f}" for k, v in settle.items()))
        # Push with ~1.5x the top brick's friction-limited force.
        m = brick_info(BrickSpec(2, 2), p).mass
        F = 1.5 * p.mu_static * m * 9.81
        _, d = run_stack(p, push_N=F)
        print(f"     {coll:9s} lateral push {F*1e3:.0f} mN -> top brick moved {d*1e3:.2f} mm")
    if mc:
        # Replay the interlock stacking + push test in the browser.
        from importlib.metadata import version
        try:
            print(f"\nDrake version: {version('drake')}")
        except Exception:
            print("\nDrake version: unknown (not installed via pip)")
        run_stack(SimParams(), push_N=0.08, meshcat=mc)
        try:
            mc.SetCameraPose([0.20, -0.16, 0.16], [0.06, 0.04, 0.02])
        except Exception as e:
            print(f"(couldn't set camera: {e})")
        # Also save a standalone copy you can double-click to open.
        with open("lego_viewer.html", "w") as f:
            f.write(mc.StaticHtml())
        print(f"Live viewer: {mc.web_url()}  (Open Controls -> Animations -> play)")
        print("Offline copy: lego_viewer.html (double-click to open)")
        input("Keep this running while you look. Press Enter to exit...")
