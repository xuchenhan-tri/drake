import numpy as np

from pydrake.all import *

# Desired duration of the simulation [s].
simulation_time = 15.0
# Desired real time rate.
realtime_rate = 1.0
# Discrete time step for the system [s]. Must be positive.
time_step = 1e-2
# Young's modulus of the deformable body [Pa].
E = 5e3
# Poisson's ratio of the deformable body, unitless.
nu = 0.4
# Mass density of the deformable body [kg/m³].
density = 8e2
# Stiffness damping coefficient for the deformable body [1/s].
beta = 0.001

def run_demo():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1.0e-3)

    model = DeformableModel(plant)
    # Add a deformable body to the model.
    deformable_body_config = DeformableBodyConfig()
    geometry = GeometryInstance(X_PG=RigidTransform(),
                                shape=Sphere(1.), name="sphere")
    props = ProximityProperties()
    props.AddProperty("material", "coulomb_friction",
                      CoulombFriction_[float](1.0, 1.0))
    geometry.set_proximity_properties(props)
    model.RegisterDeformableBody(
        geometry_instance=geometry,
        config=deformable_body_config,
        resolution_hint=1.0)

    # Add the model to the plant.
    plant.AddPhysicalModel(model)
    # Turn on SAP and finalize.
    plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)
    plant.Finalize()

    builder.Connect(model.vertex_positions_port(),
                    scene_graph.get_source_configuration_port(
                        plant.get_source_id()))
    DrakeVisualizer().AddToBuilder(builder, scene_graph)

    diagram = builder.Build()
    # Ensure we can simulate this system.
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.01)

run_demo()