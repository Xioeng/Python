import numpy as np
import matplotlib.pyplot as plt


def finiteDiffDerivative(u, dx, dy):
    u_x, u_y = u.copy(), u.copy()
    u_y[1:-1, :] = (u[2:, :] - u[1:-1, :])/(dx)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, 1:-1])/(dy)
    return u_x, u_y

def laplacian(u, dx, dy):
    u_xx, u_yy = u.copy(), u.copy()
    u_yy[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :])/(dx**2)
    u_xx[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2])/(dy**2)
    return u_xx + u_yy 


class AdvectionDiffusion:
    def __init__(self, 
                initial_condition,
                forcing_term,
                velocity_field,
                diffusivity_coef = 0.1):
    
        self.h = initial_condition
        self.f = forcing_term
        self.v = velocity_field   
        self.d = diffusivity_coef

    def solve(self,
              grid_points = (0., 1., 0. ,1.), # Length of the domain in x and y directions (x_min, x_max, y_min, y_max)
              deltas = (0.05, 0.05, 0.05), #Spatial grid spacing and Time step size (dx, dy, dt)
              T = 1.0):
                
        Lx_min, Lx_max, Ly_min, Ly_max = grid_points  
        dx, dy, dt = deltas

        # Initialize grid
        x = np.arange(Lx_min, Lx_max, dx)
        y = np.arange(Ly_min, Ly_max, dy)
        times = np.arange(0., T, dt)
        X, Y = np.meshgrid(x, y)
        u_solutions = np.zeros( (times.size, x.size, y.size) )
        u_solutions[0][1:-1,1:-1] = self.h(X, Y)[1:-1,1:-1]

        for n in range(times.size - 1):
            u_n = u_solutions[n]
            lapl = laplacian(u_n, dx, dy)
            u_x, u_y = finiteDiffDerivative(u_n, dx, dy)
            V_x, V_y = self.v(X, Y, times[n])
            f_n = self.f(X, Y, times[n])
            u_solutions[n + 1] = u_n + dt*( self.d*lapl - V_x*u_x - V_y*u_y + f_n )

        return u_solutions


# Initial condition (Gaussian pulse)
def initial_condition(x,y, center = [.2, .2]):
    return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 0.01)

# Forcing term
def forcing_term(x,y,t):
    val = initial_condition(x,y, [0.1,0.2])
    if (t <= 0.9):
        val *= 1
    else:
        val *= 0
    return val

# Velocity field
def velocity_field(X, Y, t):
    C = np.ones(X.shape)
    return 0.8*C, 0.5*C
# Grid points 
grid_points = (0., 2., 0., 2.)
T = 1.0
deltas = (0.1, 0.1, 0.01)

pde_eq = AdvectionDiffusion(initial_condition, forcing_term, velocity_field)

solutions = pde_eq.solve(grid_points,
                         deltas,
                         T)

# Plot the solution
x = np.arange(grid_points[0], grid_points[1], deltas[0])
y = np.arange(grid_points[2], grid_points[3], deltas[1])

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111)
for n in range(int(T/deltas[2])-1):
    ax.clear()
    im = ax.contourf(X, Y, solutions[n], cmap='viridis', levels=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_title(f'2D Advection-Diffusion Equation Solution {n*deltas[2]}')
    # plt.colorbar(im)
    plt.pause(0.01)
plt.show()
