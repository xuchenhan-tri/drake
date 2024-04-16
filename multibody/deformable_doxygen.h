/** @file
 Doxygen-only documentation for @ref deformable.  */

namespace drake {
namespace multibody {

/**
  Simulation of deformable bodies in Drake has been in development for a number
  of years. It has been developed as an experimental feature to allow developers
  to quickly iterate designs, free from the constraint of our stable API
  promise. As the feature matures, we are now closer to promoting to a full
  member of the Drake feature set. There are a number of key missing items that
  we want to achieve before we pull the trigger. These items have been in
  discussions within the development team but is no where documented.
  In this document, we aim to collect
  1. all the existing features of deformable body simulation in Drake as of
  today (April 2024);
  2. the features that are not immediately planned as pre-requisites for
  promoting deformable body simulation to a full Drake feature, and
  3. the features/fixes that are considered as a pre-requisite.

  This document places of all of that in a single place so they can be easily
  referred to, tracked, and checked off when they are resolved. I'm open to
  suggestions on what the format of this document should be (e.g. should this be
  modified into a Drake doxygen module and/or epic issue for external users to
  discover easily). I'm also open to suggestions to modification of the content
  listed below.

  Supported features:
  * Volumetric mesh-based deformable bodies modeled using Finite Element Method
  (FEM). Two constitutive models are supported: LinearCorotated (preferred) and
  Corotated (nonlinear). See `multibody::fem::MaterialModel` and
  `multibody::fem::DeformableBodyConfig` in general.
  * Zero displacement and zero velocity Dirichlet boundary condition. See
    multibody::DeformableModel::SetWallBoundaryCondition().
  * Fixed constraint between a deformable body and a rigid body. See
    multibody::DeformableModel::AddFixedConstraint() and the bubble gripper
    example in `//examples/multibody/deformable:bubble_gripper`.
  * Contact between deformable bodies and rigid bodies in MultibodyPlant when
    contact approximation is set to anything other than kTamsi.
  * Contact between deformable bodies and deformable bodies in MultibodyPlant
    when contact approximation is set to anything other than kTamsi.
  * Filtering contact that involves deformable bodies via
    `geometry::CollisionFilterDeclaration`.
  * Rendering deformable geometries via RenderEngineGl. In particular, users can
    choose between rendering the untextured surface of the simulated volume mesh
    or rendering a separate (potentially textured) surface mesh that is
    completely enclosed by, and moves along with, the simulated volumetric mesh.
    See the bubble gripper example in
    `//examples/multibody/deformable:bubble_gripper`.
  * Visualization of the surface of the simulated volume mesh via Meldis. See
    examples in the folder `drake/examples/multibody/deformable`.
  * Applying external forces to deformable bodies. See
    `multibody::DeformableModel::AddExternalForce()` and the suction cup example
    in `//examples/multibody/deformable:deformable_torus`.
  * Reporting contact results for deformable vs. rigid and deformable vs.
    deformable contact using the MultibodyPlant's contact_results output port.

  Unsupported features: These are the set of features that are _not_ considered
  as pre-requisites for promoting deformable body simulation to a full Drake
  feature. However, we might consider supporting them in the future if they turn
  out to be important.
  * Simulation of codimentional deformable objects (e.g. cloth, shell, rods).
  * Self-collision for deformable bodies.
  * Actuating deformable bodies.
  * Plastic deformation. Currently, the deformable bodies are modeled as
  * hyperelastic materials. We don't plan to support plastic deformation for
  * FEM based deformable bodies yet.
  * A deformable link in a robot description connected to a rigid link by a
    joint. Note that using deformable bodies as an end-effector (e.g. bubble
    gripper) is still possible with fixed constraints between deformable and
    rigid bodies.

  We should note that some of these features (e.g. cloth and plastic
  deformation) are planned to be supported by Material Point Method (MPM) that
  we are currently developing because modelling them in MPM is more natural.
  Nevertheless, enriching the FEM-based deformable body simulation with these
  features is still considered a valuable extension.

  Asset creation and specification.
  - [] Support for specifying deformable bodies through format files.
  - [] Creating deformable geometries from primitives.
       Currently, we only support parsing deformable geometries from a `.vtk`
       mesh file. Sometimes it may be convenient for the user to be able to
       specify a deformable geometry using a set of primitives (e.g., spheres,
       boxes, cylinders, etc.). This would allow the user to specify a
  deformable geometry in a more compact and human-readable way.
  - [] User guide for generating custom deformable geometries. The process is
       similar to creating compliant hydroelastic geometries. From my
       experience, our solver is robust to handle a wide range of mesh quality
       and mesh created by TetWild can be used directly after applying the
       subdivision fix. However, my sample size is limited and the guide should
       adapt as we have more user experience in the wild.

  Physics
  - [] Large number of contact constraints may result in instabilities when
       deformable bodies are extremely deformed.

  Rendering and visualization
  - [] Support rendering deformable geometries in all render engines.
  - [] Support visualizing deformable geometries in MeshcatVisualizer.
  - [] Move RenderMesh out of internal namespace.

  Miscellaneous user experience improvements
  - [] Add a tutorial on how to use deformable body simulation in Drake.
  - [] Automatically allow for deformable body simulation with
       AddMultibodyPlant()/AddMultibodyPlantSceneGraph(). Right now, users need
       to manually add a DeformableModel and connect its output port to the
       corresponding input port in the SceneGraph associated with the
       MultibodyPlant owning the DeformableModel.
  - [] Complete python bindings for all public deformable body APIs.
  - [] Allow specifying initial pose of deformable bodies with respect to frames
       other than the world frame.
  - [] Allow initializing deformable bodies with non-rest configurations. This
       would allow, for example, initializing a deformable body that is
       stretched or compressed at the beginning of the simulation.
  - [] Fail gracefully when the user tries run deformable body simulation with a
       MultibodyPlant using Tamsi. Right now, we allow registration of
       deformable bodies pre-Finalize() and the user gets a segfault after
       running the Simulator.
  - [] Move the entire `multibody::fem` namespace into the internal namespace.
       When we first designed the FEM APIs, we envisioned them to be general for
       solving a generic FEM problem. That turned out to be cumbersome and not
       very useful. Since then, we have pivoted to a more specialized design,
       specifically targeting elasticity problems. With that, a set of public
       FEM APIs are less useful and users should also interact with deformable
       bodies through the `multibody::DeformableModel` APIs.
*/

}  // namespace multibody
}  // namespace drake
