# Drake. Beta testing branch

This branch contains a prototype of the  SAP solver described in [Castro et al.
2021](https://arxiv.org/abs/2110.10107).

## Contact Model Parameters

For point contact the normal force can be simply described as a spring-damper
where the normal force fₙ is modeled as
```
fₙ = (−k⋅ϕ−c⋅vₙ)₊
```
with `k` the spring stiffness (in N/m) and `c` the dissipation (in Ns/m).
Operator `(⋅)₊` takes the positive part to ensure normal forces are repulsive.
We write the dissipation as `c = τ⋅k`, where `τ` is the *dissipation time
scale*, in seconds. You can think of `τ` as the time the material takes to
retract back to its resting position after it has been pressed. 

For modeling inelastic constant, the default value of `τ` is set to the
`MultibodyPlant` time step. This is a really good start for most applications.
While any non-negative values are allowed, values of `τ` larger than say, five
time the time step, usually lead to strange physical behavior. Therefore we
recommend values between zero and a small multiple of the time step.

### Setting Contact Parameters

Stiffness `k` and friction coefficient `μ` can be specified in SDF within a
`<collision>` element with:
```
<drake:proximity_properties>
    <drake:point_contact_stiffness>1.0e5</drake:point_contact_stiffness>
    <drake:mu_dynamic>0.5</drake:mu_dynamic>
</drake:proximity_properties>
```

Dissipation time scale can only be specified using
`geometry::ProximityProperties`, see the [clutter
demo](https://github.com/amcastro-tri/drake/tree/sap_v0.35.0/examples/multibody/mp_convex_solver/README.md)
for an example. However, the default value of `τ` (equal to the `MultibodyPlant`
time step) should work for most applications.

For Drake's Hydroelastic contact model, we use the same parameterization of
dissipation in terms of `τ`, and the same guidelines apply. Refer to Section
`Hydroelastic contact` in [MultibodyPlant's
documentation](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html) for guidelines on how to set the `hydroelastic_modulus` parameter.

## How to use the new solver in your application.

Please refer to section `Experimental contact solver` of the `Clutter` demo in
[drake/examples/multibody/mp_convex_solver/README.md](https://github.com/amcastro-tri/drake/tree/sap_v0.35.0/examples/multibody/mp_convex_solver/README.md).
You will find that the set of changes you need to run with the new solver is
very minimal. 

The clutter demo also shows how to set contact parameters and even custom
contact solver parameters (though default contact solver parameters should work
for most situations).

## Drake version

This entire experimental branch consists on a single commit on top of Drake
release `v0.35.0` from Oct 21 2021. If you have a newer version of Drake that
conflicts with this branch, please contact Alejandro Castro to update the
experimental branch to the latest Drake.
