import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from IPython.display import Latex

def linealizacion():
    
    # Define symbols
    x, y, z, d, beta, a, c, b, p, q, s = sp.symbols('x y z d beta a c b p q s')

    # Define el sistema de ecuaciones
    dxdt = s - d*x - (beta*x*y) / (1 + q*z)
    dydt = (beta*x*y) / (1 + q*z) - a*y - p*y*z
    dzdt = c*y - b*z

    # Calculamos la matriz Jacobiana
    J = sp.Matrix([[sp.diff(dxdt, x), sp.diff(dxdt, y), sp.diff(dxdt, z)],
                [sp.diff(dydt, x), sp.diff(dydt, y), sp.diff(dydt, z)],
                [sp.diff(dzdt, x), sp.diff(dzdt, y), sp.diff(dzdt, z)]])

    # Imprimimos la matriz Jacobiana

    display(Latex(r"\begin{align*}  \text{Matriz jacobiana} \end{align*}"))

    jacobiana_latex = sp.latex(J, mat_str='bmatrix', mat_delim='')
    display(Latex(jacobiana_latex))

    # Evaluamos la matriz Jacobiana en un punto de equilibrio arbitrario
    # Por ejemplo, si queremos evaluarla en el punto (x*, y*, z*)

    x_star, y_star, z_star = sp.symbols('x_star y_star z_star')
    J_at_equilibrium = J.subs({x: x_star, y: y_star, z: z_star})

    # Mostramos la parte lineal de la matriz Jacobiana (la matriz A) en latex
    display(Latex(r"\begin{align*}  \text{Parte lineal (matriz A) en el punto de equilibrio} \end{align*}"))
    latex_matrix = sp.latex(J_at_equilibrium, mat_str='bmatrix', mat_delim='')
    display(Latex(latex_matrix))


def eigValsVec():
    # Define symbols
    x, y, z, d, beta, a, c, b, p, q, s = sp.symbols('x y z d beta a c b p q s')

    # Define the system of equations
    dxdt = s - d*x - (beta*x*y) / (1 + q*z)
    dydt = (beta*x*y) / (1 + q*z) - a*y - p*y*z
    dzdt = c*y - b*z

    # Define Jacobian matrix
    J = sp.Matrix([[sp.diff(dxdt, x), sp.diff(dxdt, y), sp.diff(dxdt, z)],
                [sp.diff(dydt, x), sp.diff(dydt, y), sp.diff(dydt, z)],
                [sp.diff(dzdt, x), sp.diff(dzdt, y), sp.diff(dzdt, z)]])

    parameters = {s: 10, d: 0.1, beta: 0.05, q: 0.9, a: 0.1, p: 0.1, c: 0.2, b: 0.1}

    # Evaluate the Jacobian matrix at equilibrium point E_0(s/d, 0, 0)
    E_0 = J.subs({x: s/d, y: 0, z: 0}).subs(parameters)

    display(Latex(r"\begin{align*}  \text{Matriz Jacobiana en el punto de equilibrio} E_0 \end{align*}"))
    latex_matrix = sp.latex(E_0, mat_str='bmatrix', mat_delim='')
    display(Latex(latex_matrix))

    # Calculate eigenvalues
    eigenvalues = E_0.eigenvals()

    display(Latex(r"\begin{align*}  \text{Valores propios de la matriz Jacobiana en el punto de equilibrio} \quad E_0 :\end{align*}"))
    for eigenvalue in eigenvalues:
        display( eigenvalue.evalf())


def spiral():
        # Define parameters
    s = 10
    d = 0.1
    beta = 0.05
    q = 0.9
    a = 0.1
    p = 0.1
    c = 0.2
    b = 0.1
    t_final = 100
    h = 0.01

    # Define ODEs
    def f(x, y, z):
        dxdt = s - d*x - (beta*x*y) / (1 + q*z)
        dydt = (beta*x*y) / (1 + q*z) - a*y - p*y*z
        dzdt = c*y - b*z
        return dxdt, dydt, dzdt

    # Euler's method
    def euler_method(f, x0, y0, z0, t_final, h):
        t = np.arange(0, t_final, h)
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        z = np.zeros_like(t)
        x[0], y[0], z[0] = x0, y0, z0
        for i in range(0, len(t) - 1):
            dxdt, dydt, dzdt = f(x[i], y[i], z[i])
            x[i + 1] = x[i] + h*dxdt
            y[i + 1] = y[i] + h*dydt
            z[i + 1] = z[i] + h*dzdt
        return x, y, z

    # Initial conditions
    x0 = s/d
    y0 = .0000001
    z0 = .0000001

    # Run Euler's method
    x, y, z = euler_method(f, x0, y0, z0, t_final, h)

    # Plot 3D phase space diagram
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x, y, z, label='Espacio fase')
    ax.scatter(x[0], y[0], z[0], color='red', label='Equilibrio en $E_0$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Espacio fase del comportamiento dinámico del equilibrio $E_0$')
    plt.legend()
    plt.show()
    
    
def jacobiana():
    # Define symbols
    x, y, z, d, beta, a, c, b, p, q, s = sp.symbols('x y z d beta a c b p q s')

    # Define the system of equations
    dxdt = s - d*x - (beta*x*y) / (1 + q*z)
    dydt = (beta*x*y) / (1 + q*z) - a*y - p*y*z
    dzdt = c*y - b*z

    # Define Jacobian matrix
    J = sp.Matrix([[sp.diff(dxdt, x), sp.diff(dxdt, y), sp.diff(dxdt, z)],
                [sp.diff(dydt, x), sp.diff(dydt, y), sp.diff(dydt, z)],
                [sp.diff(dzdt, x), sp.diff(dzdt, y), sp.diff(dzdt, z)]])


    display(Latex(r"\begin{align*}  \text{Matriz Jacobiana } J \end{align*}"))
    latex_matrix = sp.latex(J, mat_str='bmatrix', mat_delim='')
    display(Latex(latex_matrix))


def Euler_sol():
    # Define parameters
    s = 160
    d = 0.01
    beta = 0.05
    q = 0.05
    a = 0.1
    p = 0.1
    c = 0.2
    b = 0.1
    t_final = 100
    h = 0.01

    # Define ODEs
    f1 = lambda x, y, z: s - d*x - beta*x*y/(1 + q*z)
    f2 = lambda x, y, z: beta*x*y/(1 + q*z) - a*y - p*y*z
    f3 = lambda x, y, z: c*y - b*z

    # Step size
    h = 0.01
    t = np.arange(0, t_final, h)

    # Initial Conditions
    x0 = 1
    y0 = 1
    z0 = 1

    # Explicit Euler Method
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    z[0] = z0

    for i in range(0, len(t) - 1):
        x[i + 1] = x[i] + h*f1(x[i], y[i], z[i])
        y[i + 1] = y[i] + h*f2(x[i], y[i], z[i])
        z[i + 1] = z[i] + h*f3(x[i], y[i], z[i])

    # Plot solutions
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, label='x(t)')
    plt.plot(t, y, label='y(t)')
    plt.plot(t, z, label='z(t)')
    plt.title('Comportamiento de la solución al aplicar el método de Euler respecto al valor y tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def grafica1():
    # Parámetros
    d = 0.1
    beta = 0.05
    a = 0.1
    c = 0.2
    b = 0.1
    p = 0.1

    # Ecuaciones diferenciales
    def model(t, state, A, q):
        x, y, z = state
        dxdt = A - d*x - beta*x*y/(1 + q*z)
        dydt = beta*x*y/(1 + q*z) - a*y - p*y*z
        dzdt = c*y - b*z
        return [dxdt, dydt, dzdt]

    # Condiciones iniciales
    x0 = 100
    y0 = 10
    z0 = 1
    initial_state = [x0, y0, z0]

    # Valores de A y q
    A_values = [10, 25, 40, 55, 70, 85, 100]
    q_values = np.linspace(0, 1, 100)

    # Gráfica 1
    plt.figure(figsize=(10, 6))
    for A in A_values:
        target_cells = []
        linestyle = '-'  # Por defecto, las líneas serán sólidas
        if A in [25, 55, 85]:  # Si A es 25, 55 o 85, las líneas serán punteadas
            linestyle = '--'
        for q in q_values:
            sol = solve_ivp(model, [0, 100], initial_state, args=(A, q), t_eval=np.linspace(0, 100, 100))
            target_cells.append(sol.y[0][-1])  # Obtenemos el valor final de las células objetivo
        plt.plot(q_values, target_cells, label=f'A = {A}', color='black', linestyle=linestyle)

    plt.xlabel('Tasa de inhibición del virus mediada por CTL (q)')
    plt.ylabel('Numero total de células objetivo')
    plt.legend()
    plt.title('Efecto de la inhibición del virus mediada por CTL sobre el número total de células objetivo')
    plt.grid(True)
    plt.show()
    
def grafica3():
    
        d = 0.1
        beta = 0.05
        a = 0.1
        c = 0.2
        b = 0.1
        p = 0.1

        # Ecuaciones diferenciales
        def model(t, state, A, q):
            x, y, z = state
            dxdt = A - d*x - beta*x*y/(1 + q*z)
            dydt = beta*x*y/(1 + q*z) - a*y - p*y*z
            dzdt = c*y - b*z
            return [dxdt, dydt, dzdt]

        # Condiciones iniciales
        x0 = 100
        y0 = 10
        z0 = 1
        initial_state = [x0, y0, z0]

        # Valores de A y q
        A_values = [10, 25, 40, 55, 70, 85, 100]
        q_values = np.linspace(0, 1, 100)

        plt.figure(figsize=(10, 6))
        for A in A_values:
            ctl_activity = []
            linestyle = '-'  # Por defecto, las líneas serán sólidas
            if A in [25, 55, 85]:  # Si A es 25, 55 o 85, las líneas serán punteadas
                linestyle = '--'
            for q in q_values:
                sol = solve_ivp(model, [0, 100], initial_state, args=(A, q), t_eval=np.linspace(0, 100, 100), rtol=1e-16, atol=1e-10)
                # Calculamos la actividad de los CTL como el número de células inmunes en estado estable
                ctl_activity.append(sol.y[2][-1])
            plt.plot(q_values, ctl_activity, label=f'A = {A}', color='black', linestyle=linestyle)

        plt.xlabel('Tasa de inhibición del virus mediada por CTL (q)')
        plt.ylabel('Actividad CTL')
        plt.legend()
        plt.title('Efecto de la inhibición del virus mediada por CTL en la actividad CTL')
        plt.grid(True)
        plt.show()
        

def grafica2():
    # Gráfica 2
        d = 0.1
        beta = 0.05
        a = 0.1
        c = 0.2
        b = 0.1
        p = 0.1

        # Ecuaciones diferenciales
        def model(t, state, A, q):
            x, y, z = state
            dxdt = A - d*x - beta*x*y/(1 + q*z)
            dydt = beta*x*y/(1 + q*z) - a*y - p*y*z
            dzdt = c*y - b*z
            return [dxdt, dydt, dzdt]

        # Condiciones iniciales
        x0 = 100
        y0 = 10
        z0 = 1
        initial_state = [x0, y0, z0]

        # Valores de A y q
        A_values = [10, 25, 40, 55, 70, 85, 100]
        q_values = np.linspace(0, 1, 100)

        plt.figure(figsize=(10, 6))
        for A in A_values:
            virus_load = []
            linestyle = '-'  # Por defecto, las líneas serán sólidas
            if A in [25, 55, 85]:  # Si A es 25, 55 o 85, las líneas serán punteadas
                linestyle = '--'
            for q in q_values:
                sol = solve_ivp(model, [0, 100], initial_state, args=(A, q), t_eval=np.linspace(0, 100, 100), rtol=1e-20, atol=1e-10)  # Ajuste de tolerancias
                virus_load.append(sol.y[1][-1])  # Obtenemos el valor final de la carga viral
            plt.plot(q_values, virus_load, label=f'A = {A}', color='black', linestyle=linestyle)

        plt.xlabel('Tasa de inhibición del virus mediada por CTL (q)')
        plt.ylabel('Carga de virus')
        plt.legend()
        plt.title('Efecto de la inhibición de virus mediada por CTL sobre la carga de virus')
        plt.grid(True)
        plt.show()
        
def grafica4():
        # Parámetros
    d = 0.1
    beta = 0.05
    a = 0.1
    c = 0.2
    b = 0.1
    p = 0.1
    lambda_value = 40
    q_values = [0.05, 0.5]

    # Función para calcular la solución de las ecuaciones diferenciales
    def model(t, state, q):
        x, y, z = state
        dxdt = lambda_value - d*x - beta*x*y/(1 + q*z)
        dydt = beta*x*y/(1 + q*z) - a*y - p*y*z
        dzdt = c*y - b*z
        return [dxdt, dydt, dzdt]

    # Condiciones iniciales
    initial_conditions = [
        (0.05, 10, 5, 8, '-'),    # Curva normal
        (0.05, 100, 50, 80, ':'), # Curva punteada
        (0.5, 10, 5, 8, '-.'),     # Curva punteada y rayada
        (0.5, 100, 50, 80, '--')   # Curva rayada
    ]

    # Gráfico 4
    plt.figure(figsize=(10, 6))
    for q, x0, y0, z0, linestyle in initial_conditions:
        sol = solve_ivp(model, [0, 200], [x0, y0, z0], args=(q,), t_eval=np.linspace(0, 200, 1000))
        plt.plot(sol.t, sol.y[0], linestyle, label=f'q={q}, x(0)={x0}, y(0)={y0}, z(0)={z0}')

    plt.xlabel('Tiempo (t, escala arbitraria)')
    plt.ylabel('Celula objetivo, x(t)')
    plt.legend()
    plt.title('Serie temporal de células objetivo')
    plt.grid(True)
    plt.show()