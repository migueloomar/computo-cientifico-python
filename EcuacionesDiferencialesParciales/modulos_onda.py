import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def solve_wave_equation(f, g, g0, g1, L, T, Nx, Nt, c,k=None,h=None):
    """
    Resuelve la ecuación de onda 1D con condiciones de frontera y condiciones iniciales dadas.

    Parámetros:
    f : función que define la condición inicial u(x, 0)
    g : función que define la condición inicial de la derivada temporal u_t(x, 0)
    g0 : función que define la condición de frontera en x = 0
    g1 : función que define la condición de frontera en x = L
    L : longitud del dominio espacial
    T : tiempo total de simulación
    Nx : número de puntos de discretización espacial
    Nt : número de puntos de discretización temporal
    c : velocidad de la onda
    h: tamaño de paso espacial
    k: tamaño de paso temporal

    Devuelve:
    X : malla de puntos espaciales
    T : malla de puntos temporales
    u : matriz de soluciones
    """
    
    # Discretización espacial y temporal
    if h is  None:  h = L / (Nx - 1)

    if k is  None: k = T / (Nt - 1)
  
    # Inicializar la matriz de solución
    u = np.zeros((Nx, Nt))
    u_t = np.zeros((Nx, Nt))
    
    # Condición inicial
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    
    # Condiciones iniciales
    u[:, 0] = f(x)
    u_t[:, 0] = g(x)
    
    # Condiciones de frontera
    u[0, :] = g0(t)
    u[-1, :] = g1(t)
    
    for n in range(1, Nt):
        # Se usan diferencias centrales
        d2u_dx2 = np.zeros(Nx)
        d2u_dx2[1:-1] = (u[:-2, n-1] - 2*u[1:-1, n-1] + u[2:, n-1]) / h**2
        
        # Actualizar solución usando la ecuación de onda
        u[1:-1, n] = 2*u[1:-1, n-1] - u[1:-1, n-2] + (c*h)**2 * d2u_dx2[1:-1]
    
    # Crear malla para graficar
    X, T = np.meshgrid(x, np.linspace(0, T, Nt))

    return X, T, u

def plot_wave_solution(X, T, u):
    """
    Grafica la solución de la ecuación de la onda.

    Parámetros:
    X (numpy.ndarray): Malla espacial.
    T (numpy.ndarray): Malla temporal.
    u (numpy.ndarray): Matriz de soluciones.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, u.T, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title('cuerda vibrante de la ecuación de onda (u, x, t)')
    plt.show()

def create_solution_dataframe(T, u):
    # Crear un DataFrame para las soluciones en los puntos x2, x3, x4, x5
    data = {
        't_j': T[:, 0],
        'x_2': u[1, :],
        'x_3': u[2, :],
        'x_4': u[3, :],
        'x_5': u[4, :]
    }
    df = pd.DataFrame(data)
    return df