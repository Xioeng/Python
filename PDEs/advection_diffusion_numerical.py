import numpy as np
import matplotlib.pyplot as plt


def finiteDiffDerivative(u, dx, dy):
    u_x, u_y = u.copy(), u.copy()
    u_y[1:-1, :] = (u[2:, :] - u[:-2, :])/(2*dx)
    u_x[:, 1:-1] = (u[:, 2:] - u[:, :-2])/(2*dy)
    return u_x, u_y

def laplacian(u, dx, dy):
    u_xx, u_yy = u.copy(), u.copy()
    u_yy[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :])
    u_xx[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2])
    # u_yy[0,:] = 0.0; u_yy[-1,:] = 0.0
    # u_xx[:,0] = 0.0; u_xx[:,-1] = 0.0
    return u_xx/(dx**2) + u_yy/(dy**2)


class AdvectionDiffusion:
    def __init__(self, 
                initial_condition,
                forcing_term,
                velocity_field,
                diffusivity_coef = 0.01):
    
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
        laplacians = np.zeros( (times.size, x.size, y.size) )
        u_solutions[0][1:-1,1:-1] = self.h(X, Y)[1:-1,1:-1]

        for n in range(times.size - 1):
            u_n = u_solutions[n]
            laplacians[n] = laplacian(u_n, dx, dy)
            u_x, u_y = finiteDiffDerivative(u_n, dx, dy)
            V_x, V_y = self.v(X, Y, times[n])
            f_n = self.f(X, Y, times[n])
            u_n_plus = u_n + dt*( self.d*laplacians[n] - V_x*u_x - V_y*u_y + f_n )
            u_n_plus[0,:] = u_n_plus[1,:]; u_n_plus[-1,:] = u_n_plus[-2,:]
            u_n_plus[:,0] = u_n_plus[:,1]; u_n_plus[:,-1] = u_n_plus[:,-2]
            u_solutions[n+1] = u_n_plus

        return u_solutions, laplacians


if __name__ == '__main__':
    # Initial condition (Gaussian pulse)
    def pulse(x,y, center = [.2, .2]):
        return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 0.01)
    def initial_condition(x,y, center = [.2, .2]):
        return 0.0*pulse(x,y, center)
    
    # Forcing term
    def forcing_term(x,y,t):
        val = 100*pulse(x,y, [0.8,0.8])
        return val

    # Velocity field
    def velocity_field(X, Y, t):
        C = np.ones(X.shape)
        return -1.0*C, -0.4*C
    diffusivity_constant = 0.01

    # Grid points 
    grid_points = (0., 1., 0., 1.)
    T = 1.0
    deltas = (0.02, 0.02, 0.01)
    pde_eq = AdvectionDiffusion(initial_condition, forcing_term, velocity_field, diffusivity_constant)

    solutions, laplacians = pde_eq.solve(grid_points,
                            deltas,
                            T)

    # Plot the solution
    x = np.arange(grid_points[0], grid_points[1], deltas[0])
    y = np.arange(grid_points[2], grid_points[3], deltas[1])

    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(1,2, figsize = (11,5))
    V_x,V_y = velocity_field(X, Y, 0)
    CFL_number = deltas[2] * (np.max(V_x)/deltas[0] + np.max(V_y)/deltas[1])
    if CFL_number <= 0.5:
        print(f'Stable solution... CFL_number = {CFL_number}')
    else:
        print(f'Unstable solution... CFL_number = {CFL_number}')

    cbar = []
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    cax = make_axes_locatable(axs[0]).append_axes("right", size="5%", pad="2%")
    caxl = make_axes_locatable(axs[1]).append_axes("right", size="5%", pad="2%")

    for n in range(int(T/deltas[2])-1):
        for ax in axs:
            ax.clear()
            cax.clear()
            caxl.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        axs[0].set_title(f'2D Advection-Diffusion Equation Solution {n*deltas[2]}')
        axs[1].set_title(f'Laplacian (only for analysis, no longer needed)')
        im = axs[0].contourf(X, Y, solutions[n], cmap='terrain', levels=10)
        lap = axs[1].imshow(laplacians[n], origin = 'lower')

        cbar = fig.colorbar(im, cax=cax)
        cbarl = fig.colorbar(lap, cax=caxl)
        plt.pause(0.02)
    plt.show()
