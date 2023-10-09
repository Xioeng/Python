import numpy as np
import matplotlib.pyplot as plt

def integrate_forcing_term(forcing, x, y, t, n, v):
    dt = t/n
    sum = np.zeros(x.shape)
    for i in range(n):
        sum += dt*forcing(x + (i*dt - t)*v[0], y + (i*dt - t)*v[1], i*dt)
    return sum

class Advection:
    def __init__(self, initial_condition,
                forcing_term ,
                velocity_field = [1.0, 2.0]):
    
        self.h = initial_condition
        self.f = forcing_term
        self.v = velocity_field
    
    def solve(self,
              grid_points = (0., 10., 0. ,10.), # Length of the domain in x and y directions (x_init, x_end, y_init, y_end)
              deltas = (0.1, 0.1, 0.1), #Spatial grid spacing and Time step size (dx, dy, dt)
              T = 3.0,
              method = 'exact'):
                
        Lx_init, Lx_end, Ly_init, Ly_end = grid_points  
        dx, dy, dt = deltas
        Nx, Ny = int( (Lx_end -Lx_init)/dx ), int( (Ly_end -Ly_init)/dy )
        Nt = int(T/dt) # Number of time steps

        # Initialize grid
        x = np.linspace(Lx_init, Lx_end, Nx)
        y = np.linspace(Ly_init, Ly_end, Ny)
        X, Y = np.meshgrid(x, y)
        
        if method == 'exact':
            u_exact = self.h(X - T*self.v[0], Y - T*self.v[1]) + integrate_forcing_term(self.f, X, Y, T, Nt, self.v )
            return u_exact
        else:
            raise(NotImplementedError)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Initial condition (Gaussian pulse)
    def initial_condition(x,y, center = [2.0, 2.0]):
        return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / .1)

    # Forcing term
    def forcing_term(x,y,t):
        val = initial_condition(x,y, [1,2])
        if (t >= 0.5 and t <= 1.5):
            val *= 1
        else:
            val *= 0
        return val
    
    # Velocity field
    v = [2.0, 1.0]
    # Grid points 
    grid_points = (0., 10., 0. ,10.)
    T = 4.0
    deltas = (0.1, 0.1, 0.1)

    advection_eq = Advection(initial_condition, forcing_term, v)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    times = np.arange(0., T, deltas[2])

    def frame_update(frame_number):
        u_exact  = advection_eq.solve(grid_points, deltas, times[frame_number])
        ax.imshow(u_exact, extent=grid_points, origin='lower', cmap='viridis')
        ax.set_title(f'2D Advection Equation: Exact Solution at t = {times[frame_number]:.2f}. Mass {np.sum(u_exact):.3f}')

    ani = FuncAnimation(fig, frame_update, frames=range(times.size), interval=40)
    ani.save('animation.gif', writer='pillow', fps=10)
    plt.show()



