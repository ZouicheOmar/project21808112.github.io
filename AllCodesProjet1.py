import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from math import sqrt

#--------------------------------------------------
# code0.py : Première définitions
#--------------------------------------------------

def u(x): # définition de la fonction u
    return x**4 

def grad1(f, x0, dx = 1.0):
    #Donne le gradient en dimension 1
    return misc.derivative(f, x0, dx, n = 1)

def laplac1(f, x0, dx = 1.0):
    #Donne le Laplacien en dimension 1
    return misc.derivative(f, x0, dx, n = 2)

# On remarque que dans notre cas
	# Le gradient est la dérivée
	# Le Laplacien est la dérivée seconde

#--------------------------------------------------
# code1.py : Graphes
#--------------------------------------------------

fig1 = plt.figure(figsize=(9, 6))

N = 100
x = np.linspace(0, 5, N)
y = u(x)

plt.plot(x,y, label = "$u(x)$")             
plt.title(" $ u(x) $ sur [0,5] ", fontsize = 17)
plt.xlabel('x', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------
# code2.py : Graphes
#--------------------------------------------------

fig2 = plt.figure(figsize = (9, 6))

gradu = np.zeros(N)
laplacu = np.zeros(N)
x = np.linspace(0, 5, N)

for i in range(N):
    laplacu[i] = laplac1(u, x[i])
    gradu[i] = grad1(u, x[i])

plt.plot(x, y, label = 'u(x)')
plt.plot(x, laplacu, label = '$ \Delta u(x) $')    
plt.plot(x, gradu, label = '$\nabla u(x)$')
plt.title("$u$ son Gradient et Laplacien ", fontsize = 17)
plt.xlabel('x', fontsize = 15)
plt.legend()
plt.grid(True)

plt.show()

#--------------------------------------------------
# code3.py : Définitions 
#--------------------------------------------------


def diff_finie1(u, h, a, b):
   
    x = np.arange(a, b, h)                        
    N = np.size(x)
    u_x = np.zeros(N-2)
    
    for i in range(1,N-1):
        u_x[i-1] = (u(x[i+1]) - u(x[i]))/h
    
    return u_x

def diff_finie2(u, h, a, b):
   
    x = np.arange(a, b, h)                        
    N = np.size(x)
    u_xx = np.zeros(N-2)

    for i in range(1,N-1):
        u_xx[i-1] = (u(x[i+1]) - 2*u(x[i]) + u(x[i-1]))/(h**2)
                
    return u_xx

#--------------------------------------------------
# code4.py : Graphes
#--------------------------------------------------

fig3 = plt.figure(figsize = (9, 6))

N = 100
x = np.linspace(0, 5, N)
y = u(x)
u_x = diff_finie1(u, 5/N, 0, 5)
u_xx = diff_finie2(u, 5/N, 0, 5)

plt.plot(x,y, label = 'u(x)')
plt. plot(x[1:N-1], u_x, label = "u'(x)")                 
plt.plot(x[1:N-1], u_xx, label = "u''(x)")
plt.legend()
plt.grid(True)
plt.xlabel('x', fontsize = 15)
plt.title("Visualisation par différences finies", fontsize = 20)


plt.show()

#--------------------------------------------------
# code5.py : Défintions
#--------------------------------------------------


def du(x):
     return 4*x**3


def d2u(x):
	return 12*x**2

#--------------------------------------------------
# code6.py : Code erreur et approximation
#--------------------------------------------------


erreur_Laplacien = np.zeros(99)
erreur_gradient = np.zeros(99)
h = np.zeros(99)

for n in range(3,101):
    h[n-3] = 1/(n-1)
    x = np.arange(0, 5, 1/(n-1))
    k = np.size(x)
    grad = du(x[1:k-1])
    lapla = d2u(x[1:k-1])
    gradh = diff_finie1(u, 1/(n-1), 0, 5)
    laplah = diff_finie2(u, 1/(n-1), 0, 5)
    erreur_Laplacien[n-3] = sqrt(sum((lapla - laplah)**2))             # pour calculer la norme 2 discrète
    erreur_gradient[n-3] = sqrt(sum((grad - gradh)**2))                # pour calculer la norme 2 discrète
    
fig4 = plt.figure(figsize = (9, 6))
plt.plot(h, erreur_Laplacien, label = 'erreur laplacien')
plt.plot(h,h**(3/2), label = " $h^{3/2}$")
plt.legend()
plt.grid(True)
plt.xlabel('pas h', fontsize = 15)
plt.xscale('log')
plt.yscale('log')
plt.title("Erreur du Laplacien numérique (échelle logarithmique)", fontsize = 20)

fig5 = plt.figure(figsize = (9, 6))
plt.plot(h, erreur_gradient, label = "erreur gradient")
plt.plot(h,h**(1/2), label = "h")
plt.legend()
plt.xlabel('pas h', fontsize = 15)
plt.grid(True)
plt.xscale('log')
plt.yscale('log')
plt.title("Erreur du gradient numérique (échelle logarithmique)", fontsize = 20)

plt.show()


#--------------------------------------------------
# code8.py : Amélioriation du derivatives1d.py
#--------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


# NUMERICAL PARAMETERS

#Number of grid points
L=2*3.141592   #2*math.pi
err_Tx = np.zeros(199)
err_Txx = np.zeros(199)
pas = np.zeros(199)

for NX in range(3, 201):
    # Initialisation
    dx = L/(NX-1)  #Grid step (space)
    pas[NX-3] = dx
    x = np.linspace(0.0,L,NX)

    T = np.sin(x)
    Txe = np.cos(x)
    Txxe = -np.sin(x)

    Tx = np.zeros((NX))
    Txx = np.zeros((NX))

    #discretization of the second order derivative (Laplacian)
    for j in range (1, NX-1):
        Tx[j] = (T[j+1]-T[j-1])/(dx*2)
        Txx[j] = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
    
    #Tx and Txx on boundaries
    # use extrapolation in order to have (T,x),xx=0
    #(T,x),xx= (Tx0 -2*Tx1+Tx2) =0
    Tx[0] = 2*Tx[1]-Tx[2]
    #use lower order formula (1st order)
    Tx[NX-1] = (T[NX-2]-T[NX-3])/dx
    Txx[0] = 2*Txx[1]-Txx[2]
    Txx[NX-1] = 2*Txx[NX-2]-Txx[NX-3]
    
    err_Tx[NX-3] = np.sum(abs(Tx-Txe))
    err_Txx[NX-3] = np.sum(abs(Txx-Txxe))

plt.figure(1)
plt.plot(x,T, label = "graphe de sinus")
plt.title(u'Function sinus')
plt.xlabel(u'$x$', fontsize=20)
plt.ylabel(u'$T$', fontsize=26, rotation=90)
plt.legend()

plt.figure(2)
plt.xlabel(u'$x$', fontsize=26)
plt.ylabel(u'$Tx$', fontsize=26, rotation=90)
plt.plot(x,Tx, label='Tx')
plt.plot(x,np.log10(abs(Tx-Txe)), label='Error')
plt.title(u'First Derivative Evaluation (NX = 200)')
plt.legend()

plt.figure(3)
plt.xlabel(u'$x$', fontsize=26)
plt.ylabel(u'$Txx$', fontsize=26, rotation=90)
plt.plot(x,Txx,label='Txx')
plt.plot(x,np.log10(abs(Txx-Txxe)),label='Error')
plt.title(u'Second Derivative Evaluation (NX = 200)')
plt.legend()

plt.figure(4)
plt.plot(pas, err_Tx, label = "err_Tx")
plt.plot(pas, 1/2*pas**(3/2), label = '$h^{3/2}$')
plt.xlabel('pas')
plt.xscale('log')
plt.yscale('log')
plt.title("Courbe d'erreur de la dérivée numérique en fonction du pas")
plt.legend()

plt.figure(5)
plt.plot(pas, err_Txx, label = "err_Txx")
plt.plot(pas, 1/5*pas, label = 'h')
plt.xlabel('pas')
plt.xscale('log')
plt.yscale('log')
plt.title("Courbe d'erreur de la dérivée seconde numérique en fonction du pas")
plt.legend()

plt.show()
