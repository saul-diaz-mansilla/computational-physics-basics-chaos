# %% Práctica tema 4. Caos en el péndulo no lineal
# Se recomienda altamente ejecutar el código por secciones,
# ya que hay tiempos de cálculo muy elevados.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft

# Parámetros externos
g = 9.8
l = 9.8
q = .5
F_0 = 1
W = 2/3
T = 2*np.pi/W


# Valores iniciales
x0 = .2
v0 = 0

# Parámetros de la simulación
dt = T/200
tf = 200*T/5/2
t = np.linspace(0,tf,int(tf/dt)+1)
dx0 = 1e-6  # Diferencia en la condición inicial

h = 1

def recta(x,m,n):
    return m*x+n

def EDO_nolineal(x,v,t):
    return -g/l*np.sin(x)-q*v+F_0*np.sin(W*t)

def EDO_parametrica(x,v,t):
    return -q*v-(1+h*np.cos(W*t))*g/l*np.sin(x)+F_0*np.sin(W*t)

def EulerCromer(f,x0,v0,dt,tf):
    x = [x0]
    v = [v0]
    npasos = int(tf/dt)
    for i in range(npasos):
        v.append(v[-1]+f(x[-1],v[-1],i*dt)*dt)
        x.append(x[-1]+v[-1]*dt)
    return [x,v]

def EulerCromer_stop(f,x0,dx0,v0,dt):
    x = [x0]
    v = [v0]
    x2= [x0+dx0]
    v2= [v0]
    i = 0
    while np.abs(x[-1]-x2[-1])>np.exp(-40) and i*dt<=tf:
        v.append(v[-1]+f(x[-1],v[-1],i*dt)*dt)
        x.append(x[-1]+v[-1]*dt)
        v2.append(v2[-1]+f(x2[-1],v2[-1],i*dt)*dt)
        x2.append(x2[-1]+v2[-1]*dt)
        i += 1
    return i*dt
#  and (i*dt>tf/10)*(np.log(np.abs(x1-x2))[int(i)/2:])
def media_periodo(x, T, dt):
    """
    Calcula la media a un periodo de un conjunto de datos dado el
    periodo y el paso de tiempo
    
    Input:
        x (numpy array): Conjunto de datos x(t)
        T (float): Periodo sobre el que promediar
        dt (float): Diferencial de tiempo con el que se discretiza t
        
    Output:
        numpy array: Media a un periodo
    """
    n = len(x)
    x_media = np.zeros(n)
    mitad = T/dt/2    # Promedia sobre T/2 a cada lado
    
    for i in range(n):
        a = max(0, i - int(mitad))  # Evita la frontera
        b = min(n, i + int(mitad) + 1)
        x_media[i] = np.mean(x[a:b])
    
    return x_media

# %% 1.1.1. Comportamiento de dx para distintas fuerzas impulsoras
for F_0 in [.5,1,1.1,1.2,1.3]:

    x0_val = np.linspace(-.5,.5,5)  # Valores de x entre los que promediar
    t_stop = EulerCromer_stop(EDO_nolineal,x0,dx0,v0,dt)
    t = np.linspace(0,t_stop,int(t_stop/dt)+1)
    dx = np.zeros_like(t)

    for x0 in x0_val:
        [x1,_] = EulerCromer(EDO_nolineal,x0,v0,dt,t_stop)      # Cálculo de trayectorias por Euler-Cromer
        [x2,_] = EulerCromer(EDO_nolineal,x0+dx0,v0,dt,t_stop)
        dx += np.abs(np.array(x1)-np.array(x2))
    
    dx = dx/len(dx)                 # Media de las diferencias
    log_dx = np.log(media_periodo(dx,T,dt))     # Promedio a un período para evitar oscilaciones

    parameters, _ = curve_fit(recta, t, log_dx)    # Ajuste a una recta

    plt.plot(t,np.log(dx),label='Sin promediar')
    plt.plot(t,log_dx,label='Promedio a un periodo')
    plt.plot(t,recta(t,parameters[0],parameters[1]),label='Recta de ajuste')
    plt.xlabel('Tiempo (s)')
    plt.ylabel(r"$\log(\Delta\theta)$")
    plt.legend()
    plt.title(r"$\Delta\theta(0)=$"+str(dx0)+r", $F_0=$"+str(F_0))
    plt.show(block=False)

# %% 1.1.2. Transición al caos: exponente de Lyapunov en función de F0
F_val = np.linspace(0.5,1.5,500)
lyapunov = []
lyapunov2= []

for F_0 in F_val:

    x0_val = np.linspace(-.5,.5,5)  # Valores de x entre los que promediar
    t_stop = EulerCromer_stop(EDO_nolineal,x0,dx0,v0,dt)
    t = np.linspace(0,t_stop,int(t_stop/dt)+1)
    dx = np.zeros_like(t)

    for x0 in x0_val:
        [x1,_] = EulerCromer(EDO_nolineal,x0,v0,dt,t_stop)      # Cálculo de trayectorias por Euler-Cromer
        [x2,_] = EulerCromer(EDO_nolineal,x0+dx0,v0,dt,t_stop)
        dx += np.abs(np.array(x1)-np.array(x2))
    
    dx = dx/len(dx)                 # Media de las diferencias
    log_dx = np.log(media_periodo(dx,T,dt))     # Promedio a un período para evitar oscilaciones

    if len(log_dx) > 2:             # Tiene en cuenta casos en los que no se puede iterar
        parameters, _ = curve_fit(recta, t, log_dx)
        lyapunov.append(parameters[0])  # Exponente de Lyapunov para cada F0
    else:
        lyapunov.append(np.nan)  # NaN si no hay puntos suficientes
    lyapunov2.append((log_dx[-1]-log_dx[0])/t[-1])

plt.plot(F_val,lyapunov,label="Método ajuste")
plt.plot(F_val,lyapunov2,label="Método valor medio")
plt.plot(F_val,np.zeros_like(F_val),'r--',label="Transición al caos")
plt.xlabel(r"$F_0\ (s^{-2})$")
plt.ylabel(r"$\lambda\ (s^{-1})$")
plt.title("Exponente de Lyapunov en función de la fuerza impulsora")
plt.legend()
plt.grid()
plt.show(block=False)

# %% 1.2. Sección de Poincaré y atractores del sistema

dt = T/100  # Se toman puntos estocástiacamente
tf = 2000*T
for F_0 in [1.4,1.44,1.465,1.2]:
    x0 = .2
    v0 = 0
    [x,v] = EulerCromer(EDO_nolineal,x0,v0,dt,tf)
    x_f = x[len(x)//2:]
    x = (np.array(x)+np.pi)%(2*np.pi)-np.pi # Se representan los datos módulo 2*pi, entre -pi,pi
    x = x[::100]        # Sólo se representan puntos en los que t = 2*pi*n/W
    v = v[::100]
    x = x[len(x)//2:]   # Se toman los puntos finales para evitar el transitorio
    v = v[len(v)//2:]
    plt.plot(x,v,'.')  # Sección de Poincaré
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2.1,.6)
    plt.xlabel(r"$\theta\ (rad)$")
    plt.ylabel(r"$\omega\ (rad/s)$")
    plt.title(r"$F_0=$"+str(F_0))
    plt.grid()
    plt.show(block=False)
# %% 1.3. Diagrama de bifurcación

dt = T/500  # Se toman puntos estocástiacamente
tf = 2000*T
F_val = np.linspace(1.45,1.48,200)
F_plot = []
x_plot = []
for F_0 in F_val:
    x0 = .2
    [x,v] = EulerCromer(EDO_nolineal,x0,v0,dt,tf)
    x = (np.array(x)+np.pi)%(2*np.pi)-np.pi # Se representan los datos módulo 2*pi, entre -pi,pi
    x = x[::500]        # Sólo se representan puntos en los que t = 2*pi*n/W
    x = x[len(x)//2:]   # Se toman los puntos finales para evitar el transitorio
    for i in range(len(x)):
        F_plot.append(F_0)
        x_plot.append(x[i])

plt.plot(F_plot,x_plot,'k.')  # Graficación
plt.xlabel(r"$F_0$")
plt.ylabel(r"$\theta$")
plt.title("Diagrama de bifurcación")
plt.grid()
plt.show(block=False)

# %% 2.1. Exponente de Lyapunov

h_val = np.linspace(1.5,3,100)
F_0 = .2
lyapunov = []
lyapunov2= []

for h in h_val:

    x0_val = np.linspace(-.5,.5,3)  # Valores de x entre los que promediar
    t_stop = EulerCromer_stop(EDO_parametrica,x0,dx0,v0,dt)
    t = np.linspace(0,t_stop,int(t_stop/dt)+1)
    dx = np.zeros_like(t)

    for x0 in x0_val:
        [x1,_] = EulerCromer(EDO_parametrica,x0,v0,dt,t_stop)      # Cálculo de trayectorias por Euler-Cromer
        [x2,_] = EulerCromer(EDO_parametrica,x0+dx0,v0,dt,t_stop)
        dx += np.abs(np.array(x1)-np.array(x2))
    
    dx = dx/len(dx)                 # Media de las diferencias
    log_dx = np.log(media_periodo(dx,T,dt))     # Promedio a un período para evitar oscilaciones

    if len(log_dx) > 2:             # Tiene en cuenta casos en los que no se puede iterar
        parameters, _ = curve_fit(recta, t, log_dx)
        lyapunov.append(parameters[0])  # Exponente de Lyapunov para cada F0
    else:
        lyapunov.append(np.nan)  # NaN si no hay puntos suficientes
    lyapunov2.append((log_dx[-1]-log_dx[0])/t[-1])

plt.plot(h_val,lyapunov,label="Método ajuste")
plt.plot(h_val,lyapunov2,label="Método valor medio")
plt.plot(h_val,np.zeros_like(h_val),'r--',label="Transición al caos")
plt.xlabel(r"$h$")
plt.ylabel(r"$\lambda\ (s^{-1})$")
plt.title("Exponente de Lyapunov en función de la fuerza impulsora")
plt.legend()
plt.grid()
plt.show(block=False)

# %% 2.2. Secciones de Poincaré

F_0 = .2
W = 2/3
h_val = np.linspace(0,3,10)
for h in h_val:
    T = 2*np.pi/W
    dt = T/100  # Se toman puntos estocástiacamente
    tf = 2000*T
    x0 = .2
    [x,v] = EulerCromer(EDO_parametrica,x0,v0,dt,tf)
    x = (np.array(x)+np.pi)%(2*np.pi)-np.pi # Se representan los datos módulo 2*pi, entre -pi,pi
    x = x[::100]        # Sólo se representan puntos en los que t = 2*pi*n/W
    v = v[::100]
    x = x[len(x)//2:]   # Se toman los puntos finales para evitar el transitorio
    v = v[len(v)//2:]
    plt.plot(x,v,'.')  # Graficación
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-2.6,2.6)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title(r"$h=$"+str(h))
    plt.show(block=False)
# %% 2.3. Péndulo paramétrico: diagrama de bifurcación
W = 2/3
F_0 = .2
h_val = np.linspace(1.5,3,100)
h_plot = []
x_plot = []
for h in h_val:
    T = 2*np.pi/W
    dt = T/100  # Se toman puntos estocástiacamente
    tf = 2000*T 
    x0 = .2
    [x,v] = EulerCromer(EDO_parametrica,x0,v0,dt,tf)
    x = (np.array(x)+np.pi)%(2*np.pi)-np.pi # Se representan los datos módulo 2*pi, entre -pi,pi
    x = x[::100]        # Sólo se representan puntos en los que t = 2*pi*n/W
    x = x[len(x)//2:]   # Se toman los puntos finales para evitar el transitorio
    for i in range(len(x)):
        h_plot.append(h)
        x_plot.append(x[i])

plt.plot(h_plot,x_plot,'k.')  # Graficación
plt.xlabel(r"$h$")
plt.ylabel(r"$\theta$")
plt.title("Diagrama de bifurcación")
plt.grid()
plt.show(block=False)
# %%
