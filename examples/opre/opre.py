
import numpy
from math import sqrt, exp
import matplotlib.pyplot as plt
from pymlmc import mlmc_test, mlmc_plot 



class CallType(object):
    def __init__(self, option, name, M, N, L, Eps):
        self.option = option
        self.name  = name
        self.M     = M
        self.N     = N 
        self.L     = L 
        self.Eps   = Eps 
    
callTypes = [CallType(1, "European", 4,  200000, 5, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(2,   "Asian",  4, 200000, 5, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(3, "lookback", 4, 200000, 5, [0.005, 0.01, 0.02, 0.05, 0.1]), 
             CallType(5, "heston European", 4, 200000, 5, [0.005, 0.01, 0.02, 0.05, 0.1])]

#calltypes_2 = [CallType(5, 4, "heston European" 200000, 5, [0.005, 0.01, 0.02, 0.05, 0.1])]

def opre_gbm(l, N, calltype, randn = numpy.random.randn): 
    M = calltype.M

    T   = 1.0 
    r   = 0.05
    sig = 0.2
    K   = 100

    nf = M**l 
    hf = T/nf 

    nc = max(nf/M,1) 
    hc = T/nc

    sums = numpy.zeros(6) 
    
    for N1 in range(1, N+1, 10000):
        if calltype.option <= 4:
        
            N2 = min(10000, N-N1+1) 
            X0 = K 

            Xf = X0*numpy.ones(N2) 
            Xc = X0*numpy.ones(N2)

            Af = 0.5*hf*Xf
            Ac = 0.5*hc*Xc

            Mf = numpy.array(Xf) 
            Mc = numpy.array(Xc) 

            if l == 0:
                dWf = sqrt(hf) * randn(1,N2) 
                Xf[:] = Xf + r*Xf*hf + sig*Xf*dWf
                Af[:] = Af + 0.5*hf*Xf 
                Mf[:] = numpy.minimum(Mf, Xf) 
            
            else: 
                for n in range(int(nc)):
                    dWc = numpy.zeros((1,N2))
                    
                    for m in range(M):
                        dWf = sqrt(hf) *randn(1,N2) 
                        dWc[:] = dWc + dWf 
                        Xf[:] = Xf + r*hf*Xf + sig*Xf*dWf
                        Af[:] = Af + hf*Xf
                        Mf[:] = numpy.minimum(Mf, Xf) 

                    Xc[:] = Xc + r*Xc*hc + sig*Xc*dWc
                    Ac[:] = Ac + hc*Xc
                    Mc[:] = numpy.minimum(Mc, Xc) 
                
                Af[:] = Af - 0.5*hf*Xf
                Ac[:] = Ac - 0.5*hc*Xc
            
            if calltype.option == 1: 
                Pf = numpy.maximum(0, Xf -K)
                Pc = numpy.maximum(0, Xc -K)
            
            elif calltype.option == 2:
                Pf = numpy.maximum(0, Af -K) 
                Pc = numpy.maximum(0, Ac -K) 
            
            elif calltype.option == 3:
                beta = 0.5826
                Pf = Xf - Mf*(1-beta*sig*sqrt(hf))
                Pc = Xc - Mc*(1-beta*sig*sqrt(hc))
            
            elif calltype.option == 4:
                Pf = K*0.5*(numpy.sign(Xf-K) +1) 
                Pc = K*0.5*(numpy.sign(Xc-K) +1) 
            

    
        else:
            N2 = min(10000, N-N1+1) 
            X0 = numpy.array([[K], [0.04]])

            Xf = numpy.vstack((numpy.repeat(X0[0], N2), numpy.repeat(X0[1], N2)))
            Xc = numpy.vstack((numpy.repeat(X0[0], N2), numpy.repeat(X0[1], N2)))

            if l == 0: 
                dWf = sqrt(hf)*randn(2,N2)

                Xf[:] = Xf + mu(Xf,hf)*hf + sig_dW(Xf, dWf, hf) 

            else: 
                for n in range(int(nc)):
                    dWc = numpy.zeros((2,N2))

                    for m in range(M):
                        dWf = sqrt(hf)*randn(2,N2) 

                        dWc[:] = dWc + dWf

                        Xf[:]  = Xf + mu(Xf,hf)*hf + sig_dW(Xf, dWf, hf) 
                    
                    Xc = Xc + mu(Xc,hc)*hc + sig_dW(Xc,dWc, hc) 
        
            Pf = numpy.maximum(0, Xf[0,:] -K) 
            Pc = numpy.maximum(0, Xc[0,:] -K) 

        if l== 0: 
            Pc = 0
        dP = numpy.exp(-r*T)*(Pf - Pc)
        Pf = numpy.exp(-r*T)*Pf

        
        sums += numpy.array([numpy.sum(dP),
                         numpy.sum(dP**2), 
                         numpy.sum(dP**3), 
                         numpy.sum(dP**4), 
                         numpy.sum(Pf), 
                         numpy.sum(Pf**2)])
            
    cost = N*nf 
        
    return (numpy.array(sums), cost) 

# Podríamos poner los argumentos (como la volatilidad inicial y tal) al principio, para que sea más fácil e intuitivo cambiarlo 
def mu(x, h):
    m = numpy.array([0.05*x[0,:], ((1-exp(-5*h))/h)*(0.04 -x[1,:])])
    return m 

def sig_dW(x,dW,h):
    dW[1,:] = -0.5*dW[0,:] + sqrt(0.75)*dW[1,:]

    sigdW = numpy.array([numpy.sqrt(numpy.maximum(0,x[1,:]))*x[0,:]*dW[0,:], exp(-5*h)*0.25*numpy.sqrt(numpy.maximum(0,x[1,:]))*dW[1,:]])

    return sigdW

if __name__ == "__main__": 
    N0 = 1000
    Lmin = 2 
    Lmax = 6
    for (i, calltype) in enumerate(callTypes):
        def opre_l(l,N):
            return opre_gbm(l,N, calltype) 
        
        filename = "opre_gbm%d.txt" % (i+1)
        logfile = open(filename, "w") 
        print('\n ----' + str(calltype.name) + 'Call ---- \n')
        mlmc_test(opre_l, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile) 
        del logfile
        mlmc_plot(filename, nvert = 3) 
        plt.savefig(filename.replace('.pdf', '.eps'), format = 'pdf')
