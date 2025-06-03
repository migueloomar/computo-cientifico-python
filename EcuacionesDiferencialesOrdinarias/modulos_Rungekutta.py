import numpy as np


def rungeKutta3(f, y0, t, h):
    """Esta función implementa el algoritmo de Runge-Kutta de grado 4 utilizada para resolver Ecuaciones diferenciales de primer orden."""
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        # Apply Runge Kutta Formulas to find next value of y
        k1 = h * f(t[n], y[n])
        k2 = h * f(t[n] + h/2, y[n] + k1/2)
        k3 = h * f(t[n] + h/2, y[n] + k2/2)
        k4 = h * f(t[n] + h, y[n] + k3)

        # Update next value of y
        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4)/6

    return y

def RK4_with_trapezoidal_rule(f, t, y0):
    """"
    Esta variante incluye la implementacion de la regla del trapecio utilizada para resolver el ejercicio 10 del Metodo de Runge-Kutta de grado 4.
    """
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        h = t[n + 1] - t[n]
        
        # Cálculo de la integral con la regla del trapecio
        integral = np.trapz(y[:n+1], dx=h)
        
        # Cálculo de los coeficientes k1, k2, k3, k4
        k1 = f(t[n], y[n], integral)
        k2 = f(t[n] + 0.5 * h, y[n] + 0.5 * k1 * h, integral)
        k3 = f(t[n] + 0.5 * h, y[n] + 0.5 * k2 * h, integral)
        k4 = f(t[n] + h, y[n] + k3 * h, integral)
        
        # Cálculo del siguiente valor de y utilizando los coeficientes k1, k2, k3, k4
        y[n + 1] = y[n] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def rkf45(f, a, b, ya, M, tol):
    """ Esta función implementa el metodo Runge-Kutta-Fehlberg (rkf45) es una forma de  resolver un problema que 
    tiene un critero para determinar en cada momento si estamos utilizando el tamaño de paso h apropiado."""
    
    a2 = 1/4; b2 = 1/4; a3 = 3/8; b3 = 3/32; c3 = 9/32; a4 = 12/13
    b4 = 1932/2197; c4 = -7200/2197; d4 = 7296/2197; a5 = 1
    b5 = 439/216; c5 = -8; d5 = 3680/513; e5 = -845/4104; a6 = 1/2
    b6 = -8/27; c6 = 2; d6 = -3544/2565; e6 = 1859/4104
    f6 = -11/40; r1 = 1/360; r3 = -128/4275; r4 = -2197/75240; r5 = 1/50
    r6 = 2/55; n1 = 25/216; n3 = 1408/2565; n4 = 2197/4104; n5 = -1/5

    big = 1e15
    h = (b - a) / M
    hmin = h / 64
    hmax = 64 * h
    max1 = 200
    Y = [ya]
    T = [a]
    j = 0
    br = b - 0.00001 * abs(b)

    while T[j] < b:
        if T[j] + h > br:
            h = b - T[j]

        k1 = h * f(T[j], Y[j])
        Y2 = Y[j] + b2 * k1
        if big < abs(Y2):
            break

        k2 = h * f(T[j] + a2 * h, Y2)
        Y3 = Y[j] + b3 * k1 + c3 * k2
        if big < abs(Y3):
            break

        k3 = h * f(T[j] + a3 * h, Y3)
        Y4 = Y[j] + b4 * k1 + c4 * k2 + d4 * k3
        if big < abs(Y4):
            break

        k4 = h * f(T[j] + a4 * h, Y4)
        Y5 = Y[j] + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4
        if big < abs(Y5):
            break

        k5 = h * f(T[j] + a5 * h, Y5)
        Y6 = Y[j] + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5
        if big < abs(Y6):
            break

        k6 = h * f(T[j] + a6 * h, Y6)

        err = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)
        ynew = Y[j] + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5

        if err < tol or h < 2 * hmin:
            Y.append(ynew)
            if T[j] + h > br:
                T.append(b)
            else:
                T.append(T[j] + h)
            j += 1

        if err == 0:
            s = 0
        else:
            s = 0.84 * (tol * h / err) ** (0.25)

        if s < 0.75 and h > 2 * hmin:
            h = h / 2
        if s > 1.5 and 2 * h < hmax:
            h = 2 * h

        if big < abs(Y[j]) or max1 == j:
            break

        M = j
        if b > T[j]:
            m = j + 1
        else:
            M = j

    return np.column_stack((T, Y))