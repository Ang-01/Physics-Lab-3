import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
def line(x,a,b):
    return a*x+b

bc = pd.read_csv("bc.csv")

I_values = {4:['indigo','blue'],3.5:['teal','c'],3:['m','r'],2.5:['olive','olive'],2:['fuchsia','pink']}
#i is in mA, need to convert to A
B_values=[]
B_values_error=[]
for x in I_values:
    F = 9.81*bc['mass'+str(x)]/1000
    i = bc['i'+str(x)]
    L = bc['L(m)']
    iL = i*L
    error_i = 0.01
    error_L = 0.0001
    error_iL = iL*np.sqrt((error_i/i)**2+(error_L/L)**2)
    plt.errorbar(F,iL, yerr=error_iL,fmt='o',ecolor='black',color=I_values[x][0],capsize=5, label=str(x)+' Amps')
    popt, pcov = curve_fit(line,F,iL, sigma=error_iL)
    #print(popt,pcov)
    print(f"B_{x}A =", 1/popt[0], "+/-", (pcov[0,0]**0.5)/(popt[0]**2))
    #    print("b =", popt[1], "+/-", pcov[1,1]**0.5)
    B_values+=[1/popt[0]]
    B_values_error+=[(pcov[0,0]**0.5)/(popt[0]**2)]

    xfine = np.arange(0.002,0.022,0.001)
    plt.plot(xfine, line(xfine, popt[0], popt[1]), color=I_values[x][1], label=str(x)+' Amps')
    plt.title("Linear Fit of iL With Respect To Balancing Force")
    plt.xlabel("Balancing Force (N)")
    plt.ylabel("iL (Am)")
    plt.legend()
#    plt.savefig("Graph1.png")
plt.savefig("Graph1.png")
plt.show()

# print(B_values,B_values_error)
# print(list(I_values))
plt.errorbar(list(I_values),B_values, yerr=B_values_error,fmt='o',ecolor='black',color='r',capsize=5)

popt, pcov = curve_fit(line,list(I_values),B_values, sigma=B_values_error)
#print(popt,pcov)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)
xfine = np.arange(2,4,0.01)
plt.plot(xfine, line(xfine, popt[0], popt[1]), color='blue')
plt.title("Linear Fit of B Against I")
plt.xlabel("I (A)")
plt.ylabel("B (T)")
#plt.legend()
plt.savefig("Graph2.png")
plt.show
