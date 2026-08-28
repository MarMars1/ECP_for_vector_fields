# Supported dynamical systems
The framework currently supports four types of dynamical systems.

# Linear Autonomous Systems
## Mathematical model
The linear autonomous system is defined by:
$$
\frac{dx}{dt}=ax+by
$$
$$
\frac{dy}{dt}=cx+dy.
$$


## Vector field
For every point $(x,y)$, the vector field is $F(x,y)=(u,v)$ with:
$$
u=ax+by
$$
and:
$$
v=cx+dy.
$$

## Predefined examples
The module contains canonical examples representing:
- stable nodes,
- unstable nodes,
- saddles,
- stable focuses,
- unstable focuses,
- centers.

These examples are used as reference systems for evaluating the classification capabilities of ECP-based distances.

---
# Hopf Bifurcation
## Mathematical model
The project uses the Hopf normal form:
$$
\frac{dx}{dt}
=
\beta x-y-x(x^2+y^2)
$$
$$
\frac{dy}{dt}
=
x+\beta y-y(x^2+y^2).
$$
The bifurcation parameter is $\beta$. The default parameter range is $ \beta\in[-1,1]$, with step $\Delta\beta=0.1$.

---

# FitzHugh–Nagumo System
## Mathematical model
The project uses the following FitzHugh–Nagumo formulation:
$$
\frac{dx}{dt}
=
x-\frac{x^3}{3}-y+I
$$
$$
\frac{dy}{dt}
=
\frac{1}{\tau}(x+B-Ry)
$$

The model describes an excitable dynamical system inspired by the behavior of neuronal membrane dynamics. The model contains four parameters:
- $B$,
- $I$,
- $R$,
- $\tau$.

The default values are $R = 0.1$ and $\tau = 12.5$.


# Lorenz System
## Mathematical model
The Lorenz system is:
$$
\frac{dx}{dt}
=
\sigma(y-x)
$$
$$
\frac{dy}{dt}
=
x(\rho-z)-y
$$
$$
\frac{dz}{dt}
=
xy-\beta z.
$$
It is a three-dimensional nonlinear dynamical system. The default parameters are $\sigma = 10$ and $\beta = 8/3$.
