Fixed size Finite Element Method (FEM)
================================================================================

The second iteration of the FEM solver. The contents are meant to replace the
solver in the [first iteration](../../fem/dev/README.md) per the
plan of actions in
[issue #14330](https://github.com/RobotLocomotion/drake/issues/14330).

To run the "noodle" demo. First ensure that you have the visualizer and the
demo itself built:

```
bazel build //tools:drake_visualizer
bazel build //multibody/fixed_fem/dev:run_kinematic_pick_up
```

Then, in one terminal, launch the visualizer
```
bazel-bin/tools/drake_visualizer
```

In another terminal, launch the demo
```
bazel-bin/multibody/fixed_fem/dev/run_kinematic_pick_up

