#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
from cycler import cycler

# Set matplotlib defaults to agree with MATLAB code
plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")
# Following option for TeX text seems to not work with EPS figures?
#plt.rc("text", usetex=True)

# NOTE: set clip_on=False in plotting to get consistent style with MATLAB

def mlmc_plot_100(filename):
    """
    Utility to generate MLMC diagnostic plot based on 
    input text file generated by MLMC driver code mlmc_test_100.

    mlmc_plot_100(filename)

    Outputs:
      Matplotlib figure for
        (1) Error (relative to exact or est. mean) as a function of accuracy 
        (2) Normalised error composition (RMS, MC and weak error) as a function of accuracy 
    
    Inputs:
      filename: string, (base of) filename with output from mlmc_test_100 routine
    """

    #
    # read in data
    # 

    # Default file extension is .txt if none supplied
    if not os.path.splitext(filename)[1]:
        file = open(filename + ".txt", "r")
    else:
        file = open(filename, "r")

    # Declare lists for data
    Epss  = []
    data = []

    # Default values for number of samples and file_version
    file_version = 0.8

    # Recognise file version line from the fact that it starts with '*** MLMC file version'
    for line in file:
        if line[0:21] == '*** MLMC file version':
            file_version = float(line[23:30])
            break

    # No need to parse number of samples (=100)
    for _ in range(8):
        line = next(file)

    # Parse exact value
    try:
        value = float(line[13:])
        novalue = False
    except ValueError:
        value = numpy.nan
        novalue = True

    next(file)

    l = -1
    for line in file:
        # Extract accuracy value, eps
        if line[1:4] == 'eps':
            l += 1
            Epss.append(float(line[6:]))
            data.append([])
            # Skip '---------' line
            next(file)
            continue

        # Extract MLMC estimates
        splitline = [float(x) for x in line.split()]
        data[l] += splitline

    data = numpy.array(data)
    # Compute error, if exact answer is not known, estimate by sample mean
    idx = numpy.argmin(Epss)
    if novalue:
        Errs = data - numpy.mean(data[idx,:])
    else:
        Errs = data - value

    ave = numpy.mean(Errs, 1) 
    rms = numpy.sqrt(numpy.mean(Errs**2, 1) - ave**2)

    # 
    # plot figures
    #

    # Fudge to get comparable to default MATLAB fig size
    width_MATLAB = 0.9*8; height_MATLAB = 0.9*6.5;
    plt.figure(figsize=([width_MATLAB, height_MATLAB*0.75]))

    plt.rc('axes', prop_cycle=(cycler('color', ['k'])))

    # Error as a function of accuracy
    plt.subplot(1, 2, 1)
    Eps = numpy.array(Epss)
    plt.loglog(Eps, 3*Eps, '--', label=r'3*Tol', clip_on=False)
    plt.loglog(Eps,   Eps, '-.', label=r'Tol',   clip_on=False)
    for (err, eps) in zip(Errs, Epss):
        plt.loglog([eps]*len(err), numpy.abs(err), 'o', markerfacecolor='none', clip_on=False)
    plt.xlabel(r'accuracy $\varepsilon$')
    plt.ylabel(r'Error')
    plt.legend(loc='upper left', fontsize='medium')

    plt.rc('axes', prop_cycle=(cycler('color', ['k']) *
                               cycler('marker', ['*']) *
                               cycler('linestyle', ['-', '--','-.'])))

    # Normalised error decomposition
    plt.subplot(1, 2, 2)
    plt.semilogx(Eps, numpy.sqrt(rms**2 + ave**2)/Eps, label=r'RMS error', clip_on=False)
    plt.semilogx(Eps, rms/Eps, label=r'MC error', clip_on=False)
    if novalue:
        plt.semilogx(Eps, numpy.abs(ave/Eps), label=r'weak error (est.)', clip_on=False)
    else:
        plt.semilogx(Eps, numpy.abs(ave/Eps), label=r'weak error', clip_on=False)
    plt.xlabel(r'accuracy $\varepsilon$')
    plt.ylabel(r'Error / $\varepsilon$')
    plt.legend(loc='lower left', fontsize='medium')
    axis = plt.axis(); plt.axis([axis[0], axis[1], 0.0, max([axis[3],1.0])])

    # Fix subplot spacing
    plt.subplots_adjust(wspace=0.3)

if __name__ == "__main__":
    import sys

    mlmc_plot(sys.argv[1], nvert=3)
    plt.savefig(sys.argv[1] + ".eps")
