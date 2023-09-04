import numpy as np 
import matplotlib.pyplot as plt 
import scipy 
from scipy import stats 
from pymlmc import mlmc_test, mlmc_plot 
import math 

class CallType(object):
    def __init__(self, option, name, M, N, L, Eps):
        self.name   = name
        self.option = option
        self.M      = M 
        self.N      = N 
        self.L      = L 
        self.Eps    = Eps

callTypes = [CallType(1,"European", 2, 50000, 8, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(2, "Asian", 2, 50000, 8, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(3, "Lookback", 2, 50000, 8, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(4, "Digital", 2, 50000, 8, [0.01, 0.02, 0.05, 0.1, 0.2]), 
             CallType(5, "Barrier", 2, 50000, 8, [0.005, 0.01, 0.02, 0.05, 0.1])]

def mlmc_f(l,N,calltype, randn = np.random.randn): 
    K = 100 
    T = 1 
    r = 0.05 
    sig = 0.2 
    B = 0.85*K 
    
    nf = 2**l 
    nc = nf/2 
    
    hf = T/nf 
    hc = T/nc 
    
    sums = np.zeros(6) 
    
    for N1 in range(1, N+1, 10000):
        N2 = min(10000, N-N1+1) 

        X0 = K 
        
        Xf = X0*np.ones(N2)  
        Xc = X0*np.ones(N2) 
        
        Xf0 = X0*np.ones(N2) 
        Xc0 = X0*np.ones(N2) 
        # Si hay errores en la asiática puede ser por esto 
        
        Af = 0.5*hf*Xf
        Ac = 0.5*hc*Xc
        
        Mf = X0*np.ones(N2)  
        Mc = X0*np.ones(N2) 
        
        Bf = 1 
        Bc = 1 
        
        # Hay que tener más cuidado con cómo definimos Xf0 porque hecho así no está bien (misma localización de memoria, cambios a Xf le afectan)
        
        if l ==0: 
            dWf = math.sqrt(hf)*randn(1,N2) 
            
            # Lo que viene abajo no es randn, debería ser la función que toma valores uniformes. Habrá que ver cómo es en python 
            Lf = np.log(np.random.rand(1,N2))
            dIf = math.sqrt(hf/12)*hf*randn(1,N2)
            
            Xf0[:] = Xf
            Xf[:]  = Xf + r*Xf*hf + sig*Xf*dWf + 0.5*(sig**2)*Xf*(dWf**2 - hf) 
            vf = sig*Xf0
            Af[:] = Af + 0.5*hf*Xf + vf*dIf[0,:]
    
            
            Mf[:] = np.minimum(Mf, 0.5*(Xf0 + Xf -np.sqrt((Xf - Xf0)**2 - 2*hf*(vf**2)*Lf)))
            Bf  = Bf*(1-np.exp(-2*np.maximum(0,(Xf0 -B)*(Xf-B) /(hf *(vf**2)))))
        
        else: 
            for n in range(int(nc)):
                dWf = math.sqrt(hf)*randn(2,N2) 
                Lf = np.log(np.random.rand(2,N2))
                dIf = np.sqrt(hf/12)*hf*randn(2,N2) 
                
                # Esto es como lo de for m in range(M) 
                for m in range(2):
                     
                    Xf0[:] = Xf 
                    Xf[:] = Xf + r*Xf*hf + sig*Xf*dWf[m,:] + 0.5*(sig**2)*Xf*(dWf[m,:]**2 - hf) 
                    vf    = sig*Xf0
                    Af[:] = Af + hf*Xf + vf*dIf[m,:]
                    Mf   = np.minimum(Mf, 0.5*(Xf0 + Xf- np.sqrt((Xf - Xf0)**2 - 2*hf*(vf**2)*Lf[m,:])))
                    Bf = Bf*(1-np.exp(-2*np.maximum(0,(Xf0 - B)*(Xf -B)/(hf*(vf**2)))))
                
                dWc = dWf[0,:] + dWf[1,:]
                ddW = dWf[0,:]    - dWf[1,:]
                
                Xc0[:] = Xc
                Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5*(sig**2)*Xc*(dWc**2 - hc)
                        
                
                vc = sig*Xc0 
                
                Ac = Ac + hc*Xc + vc*(np.sum(dIf, axis = 0) + 0.25*hc*ddW) 
                Xc1 = 0.5*(Xc0 + Xc + vc*ddW) 
                
                Mc = np.minimum(Mc, 0.5*(Xc0+Xc1-np.sqrt((Xc1-Xc0)**2 - 2*hf*(vc**2)*Lf[0,:])))
                Mc = np.minimum(Mc, 0.5*(Xc1 + Xc - np.sqrt((Xc-Xc1)**2 - 2*hf*(vc**2)*Lf[1,:])))
                                
                Bc = Bc*(1-np.exp(-2*np.maximum(0,(Xc0 -B)*(Xc1-B)/(hf*(vc**2)))))
                Bc = Bc*(1-np.exp(-2*np.maximum(0,(Xc1-B)*(Xc-B)/(hf*(vc**2)))))
                
            Af = Af-0.5*hf*Xf 
            Ac = Ac - 0.5*hc*Xc 
        
        
        if calltype.option == 1: 
            Pf = np.maximum(0, Xf -K)
            Pc = np.maximum(0, Xc- K) 
        
        elif calltype.option == 2: 
            Pf = np.maximum(0, Af -K) 
            Pc = np.maximum(0, Ac -K)
                         
        elif calltype.option == 3:
            Pf = Xf - Mf 
            Pc = Xc - Mc 
        
        elif calltype.option == 4: 
            if (l==0):
                Pf = K*stats.norm.cdf((Xf0 + r*Xf0*hf -K)/(sig*Xf0*math.sqrt(hf)))
                Pc = Pf 
            else: 
                Pf = K*stats.norm.cdf((Xf0 + r*Xf0*hf -K)/(sig*Xf0*math.sqrt(hf)))
                Pc = K*stats.norm.cdf((Xc0 + r*Xc0*hc + sig*Xc0*dWf[0,:] - K)/(sig*Xc0*math.sqrt(hf)))
        elif calltype.option == 5: 
            Pf = Bf*np.maximum(0,Xf-K) 
            Pc = Bc*np.maximum(0,Xc-K) 
            
        
           
        
        dP = math.exp(-r*T)*(Pf -Pc)
        Pf = math.exp(-r*T)*Pf
        if l==0:
            dP = Pf
        sums += np.array([sum(dP), 
                          sum(dP**2), 
                          sum(dP**3), 
                          sum(dP**4), 
                          sum(Pf), 
                          sum(Pf**2)])       
    cost = N*nf
    return (np.array(sums), cost)

if __name__ =="__main__":
    N0 = 200
    Lmin = 2
    Lmax = 10

    for (i,calltype) in enumerate(callTypes):
        def milstein(l,N):
            return mlmc_f(l,N, calltype) 
        
        filename = "milstein%d.pdf"%(i+1) 
        logfile = open(filename, "w") 
        print('\n ----' + str(calltype.name) + 'Call ---- \n')
        mlmc_test(milstein, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile) 
        del logfile 
        mlmc_plot(filename, nvert = 3) 
        plt.savefig(filename.replace('.pdf', '.eps'), format = 'pdf') 
