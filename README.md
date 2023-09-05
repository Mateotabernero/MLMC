# MLMC
## The Multilevel Monte Carlo method

(Aquí iría como todas las matemáticas 


## Examples folder
The "Examples" folder contains examples on how to define the level l estimator and then use it for MLMC

### Opre
The Opre file computes the value of different options using Euler-Maruyama scheme for numerical integration. 

The first four options it prices are European, Asian, Lookback and Digital, under GBM. That is, considering stock prices undergoing the following SDE:

 $$ dS_t = r dt + \sigma dW_t$$
 Where $W_t$ is a Wiener process 

 The fifth option it prices is an Europena option under Heston:

 $$ dS_t   = r S_t dt + \sqrt{\nu_t} S_t dW^{S}_t$$
 $$ d\nu_t = \kappa(\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW^{\nu}_t$$

### Milstein 
Similar to the Opre file, here we compute the value of different optionsu
