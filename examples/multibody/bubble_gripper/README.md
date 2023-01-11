# Deformable bubble gripper

This is an example of simulation of deformable bodies in Drake.
The example drops a rigid cube and picks it up with a bubble grippers on schunk.

## Run visualizer

```
bazel run //tools:meldis -- --open-window &
```

## Run the example

```
bazel run --config=omp --copt=-march=native //examples/multibody/bubble_gripper:run_bubble_gripper
```

## Options

There are a few command-line options that you can use to adjust the physical
properties of the deformable body. Use `--help` to see the list.