Running FEM demo
================================================================================
The implementation of the FEM solver follows [Sifakis, 2012]. The corotated
linear constitutive model can be found in [Müller, 2004].

All instructions assume that you are launching from the `drake` workspace
directory.
```
cd drake
```

Prerequisites
-------------

Ensure that you have the visualizer and the FEM library built:

```
bazel build //tools:drake_visualize //fem
```


Pancake demo
---------------------
In one terminal, launch the visualizer
```
bazel-bin/tools/drake_visualizer
```

In another terminal, launch the mass spring cloth simulation
```
bazel-bin/fem/run_pancake
```

You can toggle between two demos (1) the pancake falling onto the ground and (2)
A rectangular block with fixed center. The first demo showcases the integrattion with 
the contact solver. The second demo shows the use of user prescribed boundary conditions.

References
-------------------------------------------------
 - [Sifakis, 2012] Sifakis, Eftychios, and Jernej Barbic. "FEM simulation of 3D 
 deformable solids: a practitioner's guide to theory, discretization and model
 reduction." Acm siggraph 2012 courses. 2012. 1-50.
 - [Müller, 2004] Müller, Matthias, and Markus H. Gross. "Interactive Virtual
  Materials." Graphics interface. Vol. 2004. 2004.