# MLMC
## The Multilevel Monte Carlo method

For mathematical background visit https://people.maths.ox.ac.uk/gilesm/files/OPRE_2008.pdf

## Examples folder
The "Examples" folder contains examples on how to define the level l estimator and then use it for MLMC

### Milstein
Here we compute the value of different options (European, Asian, Lookback, Digital and Barrier). We consider stock prices undergo the classical Geometric Brownian motion. That is:

 $$ dS_t = r dt + \sigma dW_t$$
 Where $W_t$ is a Wiener process 

And use MLMC to price the options. For that, we define the level l estimators and then use the functions in the "pymlmc" file to use MLMC and generate plots comparing performance of this method with traditional Monte Carlo

### Opre
The Opre file computes the value of different options using Euler-Maruyama scheme for numerical integration. 

The first four options it prices are European, Asian, Lookback and Digital, under GBM. That is, considering stock prices undergoing the following SDE:

 $$ dS_t = r dt + \sigma dW_t$$

Where $W_t$ is a Wiener process 

The fifth option it prices is an Europen option under Heston file

 $$ dS_t   = r S_t dt + \sqrt{\nu_t} S_t dW^{S}_t$$
 
Where $\nu_t$ is the instant variance. Which undergoes the following CIR process:
 
 $$ d\nu_t = \kappa(\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW^{\nu}_t$$

In this file, we define the level l estimators and then use the functions in the "pymlmc" file to use MLMC and generate plots comparing performance of this method with traditional Monte Carlo 
The four GBM examples were originally coded by P. E. Farell. I commpleted the file with the fifth option, which undergoes a Heston model



## pymlmc folder 

The pymlmc folder was developed by P. E. Farell based on the work of Mike Giles, and includes functions that produce the MLMC estimation when given the low-level routine for level l estimator, which is what we define in the Examples file
