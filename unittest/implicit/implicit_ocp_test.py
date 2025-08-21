from casadi import *

K = 100
m = 1.0
dt = 0.05
MAKE_EXPLICIT = False

opti = Opti()

xx = []
uu = []
for k in range(K):
    xx.append(opti.variable(4, 1))  # state at time k
    uu.append(opti.variable(2, 1))  # control at time k
xx.append(opti.variable(4, 1))  # state at time K

x = hcat(xx)
u = hcat(uu)

# initial constraints
opti.subject_to(x[0,0] == 0)
opti.subject_to(x[1,0] == 0)
opti.subject_to(x[2,0] == 0)
opti.subject_to(x[3,0] == 0)

# dynamics
for k in range(K):
    if MAKE_EXPLICIT:
        opti.subject_to(x[0,k+1] == x[0,k] + dt * x[2,k])
        opti.subject_to(x[1,k+1] == x[1,k] + dt * x[3,k])
        opti.subject_to(x[2,k+1] == x[2,k] + dt * (u[0,k] / m))
        opti.subject_to(x[3,k+1] == x[3,k] + dt * (u[1,k] / m))
    else:
        opti.subject_to(x[0,k+1] == x[0,k] + dt * x[2,k+1])
        opti.subject_to(x[1,k+1] == x[1,k] + dt * x[3,k+1])
        opti.subject_to(x[2,k+1] == x[2,k] + dt * (u[0,k] / m))
        opti.subject_to(x[3,k+1] == x[3,k] + dt * (u[1,k] / m))

    opti.subject_to(-2 <= (u[0,k] <= 2))
    opti.subject_to(-2 <= (u[1,k] <= 2))

# final constraints
opti.subject_to(x[0,K] == 1)
opti.subject_to(x[1,K] == 1)
opti.subject_to(x[2,K] == 0)
opti.subject_to(x[3,K] == 0)

# objective
opti.minimize(sumsqr(u))

if MAKE_EXPLICIT:
    opti.solver('fatrop', {'structure_detection': 'auto', "expand":True})
else:
    opti.solver('ipopt', {"expand":True})

sol = opti.solve()

xx_sol = sol.value(x)
uu_sol = sol.value(u)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1)
axs[0].plot(xx_sol[0, :], label='px')
axs[0].plot(xx_sol[1, :], label='py')
axs[0].legend()

axs[1].plot(xx_sol[2, :], label='vx')
axs[1].plot(xx_sol[3, :], label='vy')
axs[1].legend()

axs[2].plot(uu_sol[0, :], label='ux')
axs[2].plot(uu_sol[1, :], label='uy')
axs[2].legend()

plt.show()