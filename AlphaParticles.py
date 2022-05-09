import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import math
from scipy import special

#Bias 0.2
count1 = [155705, 144856, 125788, 105059, 87626, 72818, 60537, 50666, 42364,
          36077, 31143, 26893, 23614, 20265, 18237, 16305, 14439, 13198, 12022,
          10985, 12854, 11817, 10990, 9890, 9479, 8706, 7902, 7322, 6731, 5838,
          5018, 3996, 3024, 2119, 1347, 758, 340, 182, 76, 40, 9, 5, 6, 2, 3]
time1 = [30.11, 30.09, 30.16, 29.95, 30.05, 30.15, 30.12, 30.08, 30.10, 30.08,
         29.94, 30.10, 30.13, 29.95, 30.06, 29.98, 30.00, 30.00, 30.08, 30.13,
         39.95, 39.97, 39.93, 39.88, 40.06, 40.01, 39.98, 40.18, 40.12, 40.40,
         39.98, 40.00, 39.98, 40.16, 40.00, 40.23, 39.97, 40.25, 40.11, 40.08,
         40.03, 40.20, 40.35, 39.95, 40.05]
#Bias 0.4
count2 = [152516, 143112, 150158, 103806, 86403, 72386, 59854, 50395, 42565,
          35963, 30838, 26778, 23648, 20713, 18056, 16429, 14436, 12947, 11459,
          10430, 12971, 11950, 10979, 9924, 9194, 8467, 7587, 7016, 6066, 5080,
          3965, 2917, 1950, 1164, 647, 318, 148, 61, 30, 8, 8, 5, 2, 3]
time2 = [30.06, 30.08, 36.22, 30.03, 29.95, 30.13, 29.95, 30.01, 29.98, 30.11,
         30.06, 30.01, 30.06, 30.08, 30.06, 30.01, 30.02, 30.01, 30.00, 30.01,
         40.05, 40.10, 40.25, 39.98, 40.10, 40.15, 39.99, 40.06, 40.01, 40.13,
         40.10, 40.15, 40.13, 40.03, 39.93, 40.06, 40.15, 39.91, 40.06, 40.06,
         40.21, 40.11, 40.10, 40.10]
#Bias 0.6
count3 = [147891, 140497, 121710, 104455, 86988, 71880, 60264, 50169, 42086,
          36056, 30840, 26887, 23487, 20997, 18183, 16113, 14217, 13060, 11956, 10765,
          12831, 11627, 11049, 9879, 9147, 7995, 6978, 5893, 4643, 3480, 2251,
          1457, 789, 375, 186, 79, 29, 15, 12, 5, 6, 3, 2]
time3 = [29.98, 30.08, 29.96, 30.03, 30.05, 30.01, 30.04, 30.15, 29.95, 30.03,
         30.10, 29.96, 30.00, 30.13, 30.01, 29.98, 30.08, 30.08, 30.04, 30.01,
         39.94, 40.00, 40.00, 40.01, 40.10, 40.06, 40.03, 40.25, 39.98, 40.21,
         40.13, 40.01, 40.06, 40.05, 40.01, 40.01, 40.11, 40.08, 40.06, 39.93,
         40.23, 39.95, 40.13]
#Bias 0.8
count4 = [140558, 133356, 117606, 100442, 84747, 70591, 58807, 49190, 41495,
          35176, 30880, 26237, 23036, 20524, 18292, 16139, 14492, 12683, 11485,
          10922, 12453, 11382, 10055, 8795, 7621, 6154, 4639, 3480, 2180, 1347,
          793, 398, 170, 85, 27, 18, 11, 10, 3, 3]
time4 = [30.10, 30.05, 30.01, 30.08, 30.05, 30.08, 30.08, 30.00, 30.01, 30.00,
         30.15, 29.98, 30.03, 30.05, 30.05, 30.01, 30.19, 30.01, 30.01, 31.25,
         40.08, 40.23, 40.09, 40.03, 40.13, 40.03, 40.05, 40.05, 40.05, 40.07,
         40.08, 40.06, 39.99, 40.03, 39.95, 39.90, 40.03, 40.14, 39.98, 39.90]
#Bias 1.1
count5 = [115687, 112012, 102674, 89788, 76584, 64980, 54332, 45986, 40092,
          33766, 29124, 25283, 22191, 19296, 17019, 14831, 12672, 10839, 9552,
          7239, 7103, 5349, 3748, 2455, 1455, 841, 440, 229, 83, 53, 39, 12,
          17, 4, 2]
time5 = [30.08, 30.08, 30.10, 30.10, 29.91, 30.03, 30.01, 29.93, 30.61, 30.01,
         30.01, 29.98, 30.13, 30.11, 30.00, 30.03, 30.03, 30.01, 31.49, 30.04,
         40.10, 40.08, 40.00, 40.27, 39.91, 39.98, 40.01, 40.11, 40.13, 40.03,
         39.95, 40.21, 40.23, 40.08, 40.05]

unc1 = []
for i in count1:
    unc1.append(np.sqrt(i))
unc2 = []
for i in count2:
    unc2.append(np.sqrt(i))
unc3 = []
for i in count3:
    unc3.append(np.sqrt(i))
unc4 = []
for i in count4:
    unc4.append(np.sqrt(i))
unc5 = []
for i in count5:
    unc5.append(np.sqrt(i))

i = 0
while i < len(count1):
    temp = count1[i]
    count1[i] = temp/time1[i]
    i += 1
i = 0
while i < len(count2):
    temp = count2[i]
    count2[i] = temp/time2[i]
    i += 1
i = 0
while i < len(count3):
    temp = count3[i]
    count3[i] = temp/time3[i]
    i += 1
i = 0
while i < len(count4):
    temp = count4[i]
    count4[i] = temp/time4[i]
    i += 1
i = 0
while i < len(count5):
    temp = count5[i]
    count5[i] = temp/time5[i]
    i += 1

distance1 = []
i = 0
while i < len(count1):
    distance1.append(.4+.08*i)
    i += 1
distance2 = []
i = 0
while i < len(count2):
    distance2.append(.4+.08*i)
    i += 1
distance3 = []
i = 0
while i < len(count3):
    distance3.append(.4+.08*i)
    i += 1
distance4 = []
i = 0
while i < len(count4):
    distance4.append(.4+.08*i)
    i += 1
distance5 = []
i = 0
while i < len(count5):
    distance5.append(.4+.08*i)
    i += 1

i = 0
while i < len(count1):
    unc1[i] = count1[i]*(np.sqrt((1.0)/(count1[i]*time1[i])+((0.2)/(time1[i]))**2.))
    i += 1
i = 0
while i < len(count2):
    temp = unc2[i]
    unc2[i] = count2[i]*(np.sqrt((1.0)/(count2[i]*time2[i])+((0.2)/(time2[i]))**2.))
    i += 1
i = 0
while i < len(count3):
    temp = unc3[i]
    unc3[i] = count3[i]*(np.sqrt((1.0)/(count3[i]*time3[i])+((0.2)/(time3[i]))**2.))
    i += 1
i = 0
while i < len(count4):
    temp = unc4[i]
    unc4[i] = count4[i]*(np.sqrt((1.0)/(count4[i]*time4[i])+((0.2)/(time4[i]))**2.))
    i += 1
i = 0
while i < len(count5):
    temp = unc5[i]
    unc5[i] = count5[i]*(np.sqrt((1.0)/(count5[i]*time5[i])+((0.2)/(time5[i]))**2.))
    i += 1

fig = plt.figure()
plt.errorbar(distance1[4:], count1[4:], yerr=unc1[4:], markersize=1.6, elinewidth=1, fmt='ro', label = 'Threshold: 0.2')
plt.errorbar(distance2[4:], count2[4:], yerr=unc2[4:], markersize=1.6, elinewidth=1, fmt='bo', label = 'Threshold: 0.4')
plt.errorbar(distance3[4:], count3[4:], yerr=unc3[4:], markersize=1.6, elinewidth=1, fmt='go', label = 'Threshold: 0.6')
plt.errorbar(distance4[4:], count4[4:], yerr=unc4[4:], markersize=1.6, elinewidth=1, fmt='yo', label = 'Threshold: 0.8')
plt.errorbar(distance5[4:], count5[4:], yerr=unc5[4:], markersize=1.6, elinewidth=1, fmt='co', label = 'Threshold: 1.1')
plt.xlabel('Distance (cm)')
plt.ylabel('Counting Rate ($s^{-1}$)')
plt.title('Detected Counting Rate of Alpha Particles vs Distance')
plt.legend()
plt.savefig('alphadatafull.png', dpi=300, bbox_inches='tight')
plt.close()

def InverseSquareDiffEvol(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, u1, u2, u3, u4, u5):
    """
    Fits five functions of the form a/(c+x^2) + b, each sharing a common c value, by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x1-x5 : 1-dimensional lists of x-values.
    y1-y5 : 1-dimensional lists of y-values.
    u1-u5 : 1-dimensional lists of uncertainties for each y-value.
        
    Returns
    -------
    a1-a5, b1-b5, c : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    red_chi_sq : float
                 reduced chi squared value of optimal fit as calculated by the sum of squared
                 residuals divided by the degrees of freedom.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    x4 = np.asarray(x4)
    x5 = np.asarray(x5)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y3 = np.asarray(y3)
    y4 = np.asarray(y4)
    y5 = np.asarray(y5)
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    u3 = np.asarray(u3)
    u4 = np.asarray(u4)
    u5 = np.asarray(u5)
    bounds = [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 0.5]]

    def objective(s):
        a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c = np.array_split(s, 11)
        return np.sum(((y1 - (b1+(a1/(c+x1)**2.)))**2.)/(u1**2.)) + np.sum(((y2 - (b2+(a2/(c+x2)**2.)))**2.)/(u2**2.)) + np.sum(((y3 - (b3+(a3/(c+x3)**2.)))**2.)/(u3**2.)) + np.sum(((y4 - (b4+(a4/(c+x4)**2.)))**2.)/(u4**2.)) + np.sum(((y5 - (b5+(a5/(c+x5)**2.)))**2.)/(u5**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x1)+len(x2)+len(x3)+len(x4)+len(x5)-len(s))
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c = np.array_split(s, 11)
    return a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c, red_chi_sq

def InverseSquareEvalFit(x, a, b, c):
    """
    Evaluates the function a/(c+x^2) + b for input parameters a, b and c and
    a 1-D array of values x.
    """
    x = np.asarray(x)
    
    return b+(a/(c+x)**2.)

a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c, red_chi_sq = InverseSquareDiffEvol(distance1[4:13], count1[4:13], distance2[4:13], count2[4:13], distance3[4:13], count3[4:13], distance4[4:13], count4[4:13], distance5[4:13], count5[4:13], unc1[4:13], unc2[4:13], unc3[4:13], unc4[4:13], unc5[4:13])
print(c)
print(a1)
print(b1)
print(a2)
print(b2)
print(a3)
print(b3)
print(a4)
print(b4)
print(a5)
print(b5)
print(red_chi_sq)
chi_sq = 34*red_chi_sq

xx = np.linspace(0.72, 1.36, num=100)
InverseSquareFit1 = InverseSquareEvalFit(xx, a1, b1, c)
InverseSquareFit2 = InverseSquareEvalFit(xx, a2, b2, c)
InverseSquareFit3 = InverseSquareEvalFit(xx, a3, b3, c)
InverseSquareFit4 = InverseSquareEvalFit(xx, a4, b4, c)
InverseSquareFit5 = InverseSquareEvalFit(xx, a5, b5, c)

fig = plt.figure()
plt.errorbar(distance1[4:13], count1[4:13], yerr=unc1[4:13], markersize=1.6, elinewidth=1, fmt='ro', label = 'Threshold: 0.2')
plt.errorbar(distance2[4:13], count2[4:13], yerr=unc2[4:13], markersize=1.6, elinewidth=1, fmt='bo', label = 'Threshold: 0.4')
plt.errorbar(distance3[4:13], count3[4:13], yerr=unc3[4:13], markersize=1.6, elinewidth=1, fmt='go', label = 'Threshold: 0.6')
plt.errorbar(distance4[4:13], count4[4:13], yerr=unc4[4:13], markersize=1.6, elinewidth=1, fmt='yo', label = 'Threshold: 0.8')
plt.errorbar(distance5[4:13], count5[4:13], yerr=unc5[4:13], markersize=1.6, elinewidth=1, fmt='co', label = 'Threshold: 1.1')
plt.plot(xx, InverseSquareFit1, 'r', label = '$2340/(x+0.14)-272$')
plt.plot(xx, InverseSquareFit2, 'b', label = '$2320/(x+0.14)-263$')
plt.plot(xx, InverseSquareFit3, 'g', label = '$2320/(x+0.14)-261$')
plt.plot(xx, InverseSquareFit4, 'y', label = '$2260/(x+0.14)-250$')
plt.plot(xx, InverseSquareFit5, 'c', label = '$2030/(x+0.14)-166$')
plt.xlabel('Distance (cm)')
plt.ylabel('Counting Rate ($s^{-1}$)')
plt.title('Detected Counting Rate of Alpha Particles vs Distance')
plt.legend()
plt.savefig('alphafitclose.png', dpi=300, bbox_inches='tight')
plt.close()

def ErfcDiffEvol(x, y, u):
    """
    Fits a function of the form a*erfc((x-r)/sigma( by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
        
    Returns
    -------
    a, r, sigma : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    red_chi_sq : float
                 reduced chi squared value of optimal fit as calculated by the sum of squared
                 residuals divided by the degrees of freedom.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, 1000]] + [[1, 5]] + [[0.0001, 1]]

    def objective(s):
        a, r, sigma = np.array_split(s, 3)
        return np.sum(((y - a*special.erfc((x-r)/sigma))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    a, r, sigma = np.array_split(s, 3)
    return a, r, sigma, red_chi_sq

def ErfcEvalFit(x, a, r, sigma):
    """
    Evaluates the function a/(c+x^2) + b for input parameters a, b and c and
    a 1-D array of values x.
    """
    x = np.asarray(x)
    
    return a*special.erfc((x-r)/sigma)

a1, r1, sigma1, red_chi_sq1 = ErfcDiffEvol(distance1[29:], count1[29:], unc1[29:])
print(a1)
print(r1)
print(sigma1)
print(red_chi_sq1)
chi_sq1 = 13*red_chi_sq1
a2, r2, sigma2, red_chi_sq2 = ErfcDiffEvol(distance2[28:], count2[28:], unc2[28:])
print(a2)
print(r2)
print(sigma2)
print(red_chi_sq2)
chi_sq2 = 13*red_chi_sq2
a3, r3, sigma3, red_chi_sq3 = ErfcDiffEvol(distance3[25:], count3[25:], unc3[25:])
print(a3)
print(r3)
print(sigma3)
print(red_chi_sq3)
chi_sq3 = 15*red_chi_sq3
a4, r4, sigma4, red_chi_sq4 = ErfcDiffEvol(distance4[22:], count4[22:], unc4[22:])
print(a4)
print(r4)
print(sigma4)
print(red_chi_sq4)
chi_sq4 = 15*red_chi_sq4
a5, r5, sigma5, red_chi_sq5 = ErfcDiffEvol(distance5[17:], count5[17:], unc5[17:])
print(a5)
print(r5)
print(sigma5)
print(red_chi_sq5)
chi_sq5 = 15*red_chi_sq5

xx1 = np.linspace(2.72, 3.92, num=100)
xx2 = np.linspace(2.64, 3.92, num=100)
xx3 = np.linspace(2.40, 3.92, num=100)
xx4 = np.linspace(2.16, 3.92, num=100)
xx5 = np.linspace(1.76, 3.92, num=100)
ErfcFit1 = ErfcEvalFit(xx1, a1, r1, sigma1)
ErfcFit2 = ErfcEvalFit(xx2, a2, r2, sigma2)
ErfcFit3 = ErfcEvalFit(xx3, a3, r3, sigma3)
ErfcFit4 = ErfcEvalFit(xx4, a4, r4, sigma4)
ErfcFit5 = ErfcEvalFit(xx5, a5, r5, sigma5)

fig = plt.figure()
plt.errorbar(distance1[15:], count1[15:], yerr=unc1[15:], markersize=1.6, elinewidth=1, fmt='ro', label = 'Threshold: 0.2')
plt.errorbar(distance2[15:], count2[15:], yerr=unc2[15:], markersize=1.6, elinewidth=1, fmt='bo', label = 'Threshold: 0.4')
plt.errorbar(distance3[15:], count3[15:], yerr=unc3[15:], markersize=1.6, elinewidth=1, fmt='go', label = 'Threshold: 0.6')
plt.errorbar(distance4[15:], count4[15:], yerr=unc4[15:], markersize=1.6, elinewidth=1, fmt='yo', label = 'Threshold: 0.8')
plt.errorbar(distance5[15:], count5[15:], yerr=unc5[15:], markersize=1.6, elinewidth=1, fmt='co', label = 'Threshold: 1.1')
plt.plot(xx1, ErfcFit1, 'r', label = '$91\cdot erfc((x-2.91)/0.32)$')
plt.plot(xx2, ErfcFit2, 'b', label = '$104\cdot erfc((x-2.79)/0.33)$')
plt.plot(xx3, ErfcFit3, 'g', label = '$118\cdot erfc((x-2.63)/0.33)$')
plt.plot(xx4, ErfcFit4, 'y', label = '$148\cdot erfc((x-2.41)/0.36)$')
plt.plot(xx5, ErfcFit5, 'c', label = '$286\cdot erfc((x-1.86)/0.43)$')
plt.xlabel('Distance (cm)')
plt.ylabel('Counting Rate ($s^{-1}$)')
plt.title('Detected Counting Rate of Alpha Particles vs Distance')
plt.legend()
plt.savefig('alphaerfc.png', dpi=300, bbox_inches='tight')
plt.close()

def InverseSquareDiffEvolFixedcParam(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, u1, u2, u3, u4, u5, c):
    """
    Fits five functions of the form a/(c+x^2) + b, each sharing a fixed common c value, by minimizing the chi-square value
    using scipy.optimize.differential_evolution with respect to the other parameters.

    Parameters
    ----------
    x1-x5 : 1-dimensional lists of x-values.
    y1-y5 : 1-dimensional lists of y-values.
    u1-u5 : 1-dimensional lists of uncertainties for each y-value.
    c : fixed input parameter
        
    Returns
    -------
    a1-a5, b1-b5 : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    red_chi_sq : float
                 reduced chi squared value of optimal fit as calculated by the sum of squared
                 residuals divided by the degrees of freedom.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    x4 = np.asarray(x4)
    x5 = np.asarray(x5)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y3 = np.asarray(y3)
    y4 = np.asarray(y4)
    y5 = np.asarray(y5)
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    u3 = np.asarray(u3)
    u4 = np.asarray(u4)
    u5 = np.asarray(u5)
    bounds = [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]] + [[0, 5000]] + [[-1000, 0]]

    def objective(s):
        a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = np.array_split(s, 10)
        return np.sum(((y1 - (b1+(a1/(c+x1)**2.)))**2.)/(u1**2.)) + np.sum(((y2 - (b2+(a2/(c+x2)**2.)))**2.)/(u2**2.)) + np.sum(((y3 - (b3+(a3/(c+x3)**2.)))**2.)/(u3**2.)) + np.sum(((y4 - (b4+(a4/(c+x4)**2.)))**2.)/(u4**2.)) + np.sum(((y5 - (b5+(a5/(c+x5)**2.)))**2.)/(u5**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = np.array_split(s, 10)
    return a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, chi_sq

optChi_Sq = chi_sq
posChi_Sq = chi_sq
negChi_Sq = chi_sq
cdiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    cdiff += 0.01
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, posChi_Sq = InverseSquareDiffEvolFixedcParam(distance1[4:13], count1[4:13], distance2[4:13], count2[4:13], distance3[4:13], count3[4:13], distance4[4:13], count4[4:13], distance5[4:13], count5[4:13], unc1[4:13], unc2[4:13], unc3[4:13], unc4[4:13], unc5[4:13], c+cdiff)
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, negChi_Sq = InverseSquareDiffEvolFixedcParam(distance1[4:13], count1[4:13], distance2[4:13], count2[4:13], distance3[4:13], count3[4:13], distance4[4:13], count4[4:13], distance5[4:13], count5[4:13], unc1[4:13], unc2[4:13], unc3[4:13], unc4[4:13], unc5[4:13], c-cdiff)

def ErfcDiffEvolFixedrParam(x, y, u, r):
    """
    Fits a function of the form a*erfc((x-r)/sigma with a fixed r parameter by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    r : fixed input parameter
        
    Returns
    -------
    a, sigma : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, 1000]] + [[0.0001, 1]]

    def objective(s):
        a, sigma = np.array_split(s, 2)
        return np.sum(((y - a*special.erfc((x-r)/sigma))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a, sigma = np.array_split(s, 2)
    return a, sigma, chi_sq


optChi_Sq1 = chi_sq1
posChi_Sq1 = chi_sq1
negChi_Sq1 = chi_sq1
r1diff = 0
while (((posChi_Sq1-optChi_Sq1) < 1) or ((negChi_Sq1-optChi_Sq1) < 1)):
    r1diff += 0.01
    a, sigma, posChi_Sq1 = ErfcDiffEvolFixedrParam(distance1[29:], count1[29:], unc1[29:], r1+r1diff)
    a, sigma, negChi_Sq1 = ErfcDiffEvolFixedrParam(distance1[29:], count1[29:], unc1[29:], r1-r1diff)
optChi_Sq2 = chi_sq2
posChi_Sq2 = chi_sq2
negChi_Sq2 = chi_sq2
r2diff = 0
while (((posChi_Sq2-optChi_Sq2) < 1) or ((negChi_Sq2-optChi_Sq2) < 1)):
    r2diff += 0.01
    a, sigma, posChi_Sq2 = ErfcDiffEvolFixedrParam(distance2[28:], count2[28:], unc2[28:], r2+r2diff)
    a, sigma, negChi_Sq2 = ErfcDiffEvolFixedrParam(distance2[28:], count2[28:], unc2[28:], r2-r2diff)
optChi_Sq3 = chi_sq3
posChi_Sq3 = chi_sq3
negChi_Sq3 = chi_sq3
r3diff = 0
while (((posChi_Sq3-optChi_Sq3) < 1) or ((negChi_Sq3-optChi_Sq3) < 1)):
    r3diff += 0.01
    a, sigma, posChi_Sq3 = ErfcDiffEvolFixedrParam(distance3[25:], count3[25:], unc3[25:], r3+r3diff)
    a, sigma, negChi_Sq3 = ErfcDiffEvolFixedrParam(distance3[25:], count3[25:], unc3[25:], r3-r3diff)
optChi_Sq4 = chi_sq4
posChi_Sq4 = chi_sq4
negChi_Sq4 = chi_sq4
r4diff = 0
while (((posChi_Sq4-optChi_Sq4) < 1) or ((negChi_Sq4-optChi_Sq4) < 1)):
    r4diff += 0.01
    a, sigma, posChi_Sq4 = ErfcDiffEvolFixedrParam(distance4[22:], count4[22:], unc4[22:], r4+r4diff)
    a, sigma, negChi_Sq4 = ErfcDiffEvolFixedrParam(distance4[22:], count4[22:], unc4[22:], r4-r4diff)
optChi_Sq5 = chi_sq5
posChi_Sq5 = chi_sq5
negChi_Sq5 = chi_sq5
r5diff = 0
while (((posChi_Sq5-optChi_Sq5) < 1) or ((negChi_Sq5-optChi_Sq5) < 1)):
    r5diff += 0.01
    a, sigma, posChi_Sq5 = ErfcDiffEvolFixedrParam(distance5[17:], count5[17:], unc5[17:], r5+r5diff)
    a, sigma, negChi_Sq5 = ErfcDiffEvolFixedrParam(distance5[17:], count5[17:], unc5[17:], r5-r5diff)

def ErfcDiffEvolFixedsParam(x, y, u, sigma):
    """
    Fits a function of the form a*erfc((x-r)/sigma with a fixed sigma parameter by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    sigma : fixed input parameter
        
    Returns
    -------
    a, r : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, 1000]] + [[1, 5]]

    def objective(s):
        a, r = np.array_split(s, 2)
        return np.sum(((y - a*special.erfc((x-r)/sigma))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a, r = np.array_split(s, 2)
    return a, r, chi_sq


optChi_Sq1 = chi_sq1
posChi_Sq1 = chi_sq1
negChi_Sq1 = chi_sq1
s1diff = 0
while (((posChi_Sq1-optChi_Sq1) < 1) or ((negChi_Sq1-optChi_Sq1) < 1)):
    s1diff += 0.01
    a, r, posChi_Sq1 = ErfcDiffEvolFixedsParam(distance1[29:], count1[29:], unc1[29:], sigma1+s1diff)
    a, r, negChi_Sq1 = ErfcDiffEvolFixedsParam(distance1[29:], count1[29:], unc1[29:], sigma1-s1diff)
optChi_Sq2 = chi_sq2
posChi_Sq2 = chi_sq2
negChi_Sq2 = chi_sq2
s2diff = 0
while (((posChi_Sq2-optChi_Sq2) < 1) or ((negChi_Sq2-optChi_Sq2) < 1)):
    s2diff += 0.01
    a, r, posChi_Sq2 = ErfcDiffEvolFixedsParam(distance2[28:], count2[28:], unc2[28:], sigma2+s2diff)
    a, r, negChi_Sq2 = ErfcDiffEvolFixedsParam(distance2[28:], count2[28:], unc2[28:], sigma2-s2diff)
optChi_Sq3 = chi_sq3
posChi_Sq3 = chi_sq3
negChi_Sq3 = chi_sq3
s3diff = 0
while (((posChi_Sq3-optChi_Sq3) < 1) or ((negChi_Sq3-optChi_Sq3) < 1)):
    s3diff += 0.01
    a, r, posChi_Sq3 = ErfcDiffEvolFixedsParam(distance3[25:], count3[25:], unc3[25:], sigma3+s3diff)
    a, r, negChi_Sq3 = ErfcDiffEvolFixedsParam(distance3[25:], count3[25:], unc3[25:], sigma3-s3diff)
optChi_Sq4 = chi_sq4
posChi_Sq4 = chi_sq4
negChi_Sq4 = chi_sq4
s4diff = 0
while (((posChi_Sq4-optChi_Sq4) < 1) or ((negChi_Sq4-optChi_Sq4) < 1)):
    s4diff += 0.01
    a, r, posChi_Sq4 = ErfcDiffEvolFixedsParam(distance4[22:], count4[22:], unc4[22:], sigma4+s4diff)
    a, r, negChi_Sq4 = ErfcDiffEvolFixedsParam(distance4[22:], count4[22:], unc4[22:], sigma4-s4diff)
optChi_Sq5 = chi_sq5
posChi_Sq5 = chi_sq5
negChi_Sq5 = chi_sq5
s5diff = 0
while (((posChi_Sq5-optChi_Sq5) < 1) or ((negChi_Sq5-optChi_Sq5) < 1)):
    s5diff += 0.01
    a, r, posChi_Sq5 = ErfcDiffEvolFixedsParam(distance5[17:], count5[17:], unc5[17:], sigma5+s5diff)
    a, r, negChi_Sq5 = ErfcDiffEvolFixedsParam(distance5[17:], count5[17:], unc5[17:], sigma5-s5diff)

print('r:')
print(r1diff)
print(r2diff)
print(r3diff)
print(r4diff)
print(r5diff)
print('s:')
print(s1diff)
print(s2diff)
print(s3diff)
print(s4diff)
print(s5diff)
print('c')
print(cdiff)

def LinearDiffEvol(x, y, u):
    """
    Fits a linear model by scipy.optimize.differential_evolution

    Parameters
    ----------
    x : (n) array_like
        1-dimensional list of x-values.
    y : (n), array_like
        1-dimensional list of y-values.
    u : (n) array_like
        1-dimensional list of uncertainties for y-values.
    Returns
    -------
    a, b : Optimal parameters for linear model a*x + b
    
    red_chi_sq : float
        reduced chi squared as calculated by the sum of squared residuals divided by # of degrees of freedom
    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-10, 10]] + [[-10, 10]]

    def objective(s):
        a, b = np.split(s, 2)
        return np.sum(((y - a*x-b)**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    a, b = np.split(s, 2)
    return a, b, red_chi_sq

def LinearEvalFit(x, a, b):
    """
    Evaluates the function a*x + b for input parameters a and b and
    a 1-D array of values x.
    """
    x = np.asarray(x)
    
    return a*x+b

bias = [0.2, 0.4, 0.6, 0.8, 1.1]
ranges = [3.05, 2.93, 2.77, 2.55, 2.00]
rangeunc = [.04, .04, .03, .03, .04]
sigmas = [0.32, 0.33, 0.33, 0.36, 0.43]
sigmaunc = [.01, .01, .01, .01, .01]

a_r, b_r, red_chi_sq_r = LinearDiffEvol(bias, ranges, rangeunc)
print(a_r)
print(b_r)
print(red_chi_sq_r)
a_s, b_s, red_chi_sq_s = LinearDiffEvol(bias, sigmas, sigmaunc)
print(a_s)
print(b_s)
print(red_chi_sq_s)

xbias = np.linspace(0.1, 1.2)
LinFit_r = LinearEvalFit(xbias, a_r, b_r)
LinFit_s = LinearEvalFit(xbias, a_s, b_s)

fig = plt.figure()
plt.errorbar(bias, ranges, yerr=rangeunc, markersize=1.6, elinewidth=1, fmt='ro')
plt.plot(xbias, LinFit_r, 'b', label = '$-1.13x+3.38$')
plt.xlabel('Threshold')
plt.ylabel('Mean Range (cm)')
plt.title('Mean Range of Alpha Particles vs Detector Threshold')
plt.legend()
plt.savefig('alpharange.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure()
plt.errorbar(bias, sigmas, yerr=sigmaunc, markersize=1.6, elinewidth=1, fmt='ro')
plt.plot(xbias, LinFit_s, 'b', label = '$0.12x+0.28$')
plt.xlabel('Threshold')
plt.ylabel('Straggling Constant (cm)')
plt.title('Straggling Constant of Alpha Particles vs Detector Threshold')
plt.legend()
plt.savefig('alphastraggling.png', dpi=300, bbox_inches='tight')
plt.close()
