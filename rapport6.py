
# In[14]:


import numpy as np
import matplotlib.pyplot as plt

Fmax = 3   # max heat production source level
V = 0    #flow velocity in range [0, 2 m/s]
K = 0.001  #Reference Diffusion coefficient
L = 1.0    #Domain size
Time = 1   #Integration time 
NX = 100   #Number of grid points
nbctrl=20  #Sampling of V range 
ifre = 1000
Tcible = 1.5
Tref = 1


# Initialisation
x = np.linspace(0.0,1.0, NX)
T = np.ones(NX)
RHS = np.zeros(NX)
F = np.zeros(NX)
cost = np.zeros(nbctrl)
cost1 = np.zeros(nbctrl)
cost2 = np.zeros(nbctrl)
ctrlsave = np.zeros(nbctrl)

# PARAMETRES
dx = L/(NX-1) 

DCNTRL = 0.1
CNTRL = -DCNTRL
gradtarget = 0


for jctrl in range(0,nbctrl):
    CNTRL+= DCNTRL
    ctrlsave[jctrl]=CNTRL
    V = CNTRL
    print('V=',V)
    dt= dx**2/(2*(K+0.5*dx*abs(V)) + abs(V)*dx + Fmax*dx**2)
    NT = int(Time/dt)
    
### MAIN PROGRAM
    
    for j in range(1,NX-1):
        F[j]= np.exp(-1.e10*(x[j]-0.5)**8)*Fmax 
        T[j] = Tref
    T[0]= Tref
        
### MAIN LOOP
    for n in range(0,NT):
        
        for j in range(1,NX-1):
            
            Diffusion = (K+0.5*dx*abs(V))*(T[j-1]-2*T[j] + T[j+1])/(dx**2)
            Advection = V*(T[j+1]-T[j-1])/(2*dx)
            RHS[j] = dt*(Diffusion - Advection + F[j])
            
        for j in range(1,NX-1):
            T[j]+= RHS[j]
        
        T[NX-1] = 2*T[NX-2] - T[NX-3]
        
    j1 = max(0,np.amax(T-Tcible))
    j2 = 0.5*V**2
        
    if (n == NT-1):
        cost[jctrl]= j1 + j2
        cost1[jctrl]= j1
        cost2[jctrl]= j2
        cvtest = np.linalg.norm(RHS)
        print('CV : ', cvtest, ' Coût = ',cost[jctrl])
        plotlabel = "CNTRL = %1.2f" %(ctrlsave[jctrl])
        plt.plot(x,T,label=plotlabel)

        
plt.title("Solution")        
plt.xlabel('x/L')
plt.ylabel('T/Tref')
plt.grid()


# La fonction $J(V)$ qui exprime le coût :

# In[12]:


plt.figure()
plt.plot(ctrlsave,cost,label = 'J(V)=J1(V)+J2(V) et ctrl')
plt.grid()
plt.legend()


# Le front de Pareto : qui nous montre que $min J(V) = J(0.6)$

# In[15]:


plt.figure()
plt.plot(cost1,cost2,label='Front de Pareto')
plt.grid()
plt.legend()


# $V = 0.6$ est donc l'optimum de Pareto, le coût $J$ reste assez faible pour cette valeur $V$.
