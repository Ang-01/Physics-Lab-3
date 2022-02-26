import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
def line(x,a,b):
    return a*x+b

bc = pd.read_csv("bc.csv")

I_values = {4:['blue','purple'],3.5:['teal','c'],3:['m','r'],2.5:['olive','yellow'],2:['fuchsia','pink']}
print([str(i) for i in I_values])
#i is in mA, need to convert to A
for x in I_values:
    F = 9.81*bc['mass'+str(x)]
    i = bc['i'+str(x)]/1000
    L = bc['L(m)']
    iL = i*L
    error_i = 0.00001
    error_L = 0.0001
    error_iL = iL*np.sqrt((error_i/i)**2+(error_L/L)**2)
    plt.errorbar(F,iL, yerr=error_iL,fmt='o',ecolor='black',color=I_values[x][0],capsize=5, label=str(x)+'Amps')
    popt, pcov = curve_fit(line,F,iL, sigma=error_iL)
    print(popt,pcov)
    print("m =", popt[0], "+/-", pcov[0,0]**0.5)
    print("b =", popt[1], "+/-", pcov[1,1]**0.5)
    xfine = np.arange(2,22,1)
    plt.plot(xfine, line(xfine, popt[0], popt[1]), color=I_values[x][1], label=str(x)+'Amps')
    plt.legend()
    plt.show