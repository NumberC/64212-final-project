"""Tests of the simulated brick models (physical/lego/brick_models.py).

Run: uv run pytest tests/test_brick_models.py -s

  test_parse_all  every preset brick loads in Drake, in each collision and contact mode
  test_stack      three stacked bricks settle on their grid poses; studs resist
                  a sideways push that slides a plain box
"""
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, Simulator,
    ExternallyAppliedSpatialForce, SpatialForce, AbstractValue, LeafSystem,
    RigidTransform, AddDefaultVisualization,
)
from physical.lego.brick_models import (BrickSpec, brick_info, add_brick, grid_pose,
                                 generate_baseplate_sdf, all_preset_specs)
from physical.lego.specifications import SimParams


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


def test_stack():
    moved = {}
    for coll in ["interlock", "box"]:
        p = SimParams(collision=coll)
        settle, _ = run_stack(p)
        # Push with ~1.5x the top brick's friction-limited force.
        F = 1.5 * p.mu_static * brick_info(BrickSpec(2, 2), p).mass * 9.81
        moved[coll] = abs(run_stack(p, push_N=F)[1])
        errors = ", ".join(f"{k}={v*1e3:.2f}" for k, v in settle.items())
        print(f"{coll:9s} settle error (mm): {errors}"
              f"; a {F*1e3:.0f} mN push moves the top brick {moved[coll]*1e3:.2f} mm")
        assert coll == "box" or all(v < 2e-3 for v in settle.values())
    assert moved["interlock"] < 1e-3 < moved["box"]  # studs resist, a plain box slides
