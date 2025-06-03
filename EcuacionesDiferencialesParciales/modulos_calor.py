from numpy import *
from scipy.sparse import diags # Greate diagonal matrices
from scipy.linalg import solve # Solve linear systems
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D # For 3-d plot
import matplotlib.animation as animation
from matplotlib import cm
newparams = {'figure.figsize': (10.0, 10.0), 'axes.grid': True,
'lines.markersize': 8, 'lines.linewidth': 2,
'font.size': 14}
rcParams.update(newparams)
import sympy as sp 

def plot_heat_solution(x, t, U, txt='Solution'):
    '''
    
    Trazar la solución de la ecuación del calor.
    '''
    fig = figure()
    ax = fig.add_subplot(projection = '3d')
    T, X = meshgrid(t,x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30) # Rotate the figure
    ax.set_ylabel('Position (x)')
    ax.set_xlabel('Time (t)')
    ax.set_zlabel('Temperature (u)')
    title(txt);
    
def solve_heat_equation(f, g0, g1, M, N, tend):
    # Calculate grid spacings
    Dx = 1/M
    Dt = tend/N
    
    # Generate grid points
    x = linspace(0, 1, M+1)
    t = linspace(0, tend, N+1)
    
    # Initialize solution matrix
    U = zeros((M+1, N+1))
    U[:,0] = f(x)
    
    # Calculate Courant number
    r = Dt / Dx**2
    
    # Main loop for solving the heat equation
    for n in range(N):
        U[1:-1, n+1] = U[1:-1, n] + r * (U[2:, n] - 2 * U[1:-1, n] + U[0:-2, n])
        U[0, n+1] = g0(t[n+1])
        U[M, n+1] = g1(t[n+1])
    
    return U,x,t


def plot_mesh_points(N, Nt, points):

    # Discretization
    h = 1 / N
    k = 1 / Nt
    r = k / (h ** 2)
    time_steps = 5  # Can be adjusted as needed

    # Define space and time vectors
    time = arange(0, (time_steps + 0.5) * k, k)
    x = arange(0.0, pi, h)
    X, Y = meshgrid(x, time)

    # Plot mesh grid
    fig = figure(figsize=(8, 6))
    plot(X, Y, 'o-', color="#dadada")  # Mesh grid
    plot(x, 0 * x, 'bo', label='Initial Condition')  # Initial condition u(x,0)=0
    plot(pi * ones(time_steps + 1), time, 'go', label='Boundary Condition')  # Boundary condition u(π,t)=0
    plot(x, 0 * x, 'bo')  # Points on x-axis for initial condition
    plot(0 * time, time, 'go')  # Points on y-axis for boundary condition

    # Plot specified points
    for point in points:
        x,y = point
        plot(x, y, 'ks', label=f'u({x},{y})')

    xlim(-0.01, 3.15)  # Adjust x-axis limits
    ylim(-0.01, 2.5)  # Adjust y-axis limits
    xlabel('x')
    ylabel('time (ms)')
    legend(loc='center left', bbox_to_anchor=(1, 0.5))
    title(r'Mesh $\Omega=\{t≥0,0≤x≤\pi \}$ h= %s, k=%s' % (h, k), fontsize=24, y=1.08)
    grid(True)
    show()

def verify_solution(u_function, differential_equation):

    # Define the variables
    x, t = sp.symbols('x t')

    # Calcula las derivadas parciales
    u_t = sp.diff(u_function, t)
    u_xx = sp.diff(u_function, x, x)

    # substituye ut y uxx en la ecuacion diferencial proporcionada
    diff_eq_result = 4*u_t - u_xx
    simplified_result = sp.simplify(diff_eq_result)

    if simplified_result == 0:
        print("La función u(x, t) si es solución")
    else:
        print("la funcion u(x, t) no es solución")
        print("pues el resultado fue :", simplified_result)

def solve_heat_equation2(f, g0, g1,L, T, Nx, Nt, alpha):
    """
    Resuelve la ecuación del calor utilizando el método de diferencias finitas hacia adelante.

    Parámetros:
    L (float): Longitud del dominio espacial.
    T (float): Tiempo total de simulación.
    Nx (int): Número de puntos de discretización espacial.
    Nt (int): Número de pasos de tiempo.
    alpha (float): Difusividad térmica.

    Retorna:
    X (numpy.ndarray): Malla espacial.
    T (numpy.ndarray): Malla temporal.
    u (numpy.ndarray): Matriz de soluciones.
    """
    # Discretización espacial y temporal
    h = L / (Nx - 1)
    k = T / Nt

    # Inicializar la matriz de solución
    u = zeros((Nx, Nt + 1))

    # Condición inicial
    x = linspace(0, L, Nx)
    t = linspace(0, T, Nt + 1)
    
    u[:, 0] = f(x)

    # Condiciones de frontera
    u[0, :] = g0(t)
    u[-1, :] = g1(t)

    # Método de diferencias finitas hacia adelante
    r = (alpha * k) / h ** 2

    for n in range(Nt):
        for i in range(1, Nx - 1):
            u[i, n + 1] = u[i, n] + r * (u[i - 1, n] - 2 * u[i, n] + u[i + 1, n])

    # Crear malla para graficar
    X, T = meshgrid(x, linspace(0, T, Nt + 1))

    return X, T, u

def plot_heat_solution2(X, T, u):
    """
    Grafica la solución de la ecuación del calor.

    Parámetros:
    X (numpy.ndarray): Malla espacial.
    T (numpy.ndarray): Malla temporal.
    u (numpy.ndarray): Matriz de soluciones.
    """
    fig = figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, u.T, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title('Solución de la ecuación del calor mediante el método de diferencia directa')
    show()
    
def graficar_no_solucion():
    # Definir la función f2
    f2 = lambda x, t: 2 * sin(pi * x / 2) * exp(pi**2 * t / 16) - sin(pi * x) * exp(-pi**2 * t / 4) + 4 * sin(2 * pi * x) * exp(-pi**2 * t)

    # Crear la malla de puntos (x, t)
    L = 4  # Longitud del dominio espacial
    T = 2  # Tiempo total de simulación
    Nx = 50  # Número de puntos de discretización espacial
    Nt = 500  # Número de pasos de tiempo

    x = linspace(0, L, Nx)
    t = linspace(0, T, Nt)
    X, T = meshgrid(x, t)

    # Evaluar f2 en cada punto de la malla
    U = f2(X, T)

    # Graficar la superficie en 3D
    fig = figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, X, U, cmap='viridis')

    # Etiquetas y título
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title('Grafica de la no solución')

    # Mostrar la gráfica
    show()
if __name__ == '__main__':
    None