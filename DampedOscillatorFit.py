import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

yData = [55.6, 56.0, 56.6, 57.3, 58.2, 59.4, 60.3, 61.2, 62.4, 63.5, 64.7, 65.5, 66.4, 67.3, 67.7, 68.3, 68.5, 68.8, 68.8,
         68.6, 68.4, 67.9, 67.4, 66.8, 65.9, 65.2, 64.5, 63.7, 63.0, 62.2, 61.5, 61.0, 60.5, 60.1, 59.7, 59.5, 59.5, 59.6,
         59.7, 59.9, 60.5, 60.9, 61.3, 61.8, 62.4, 63.1, 63.5, 64.0, 64.5, 64.9, 65.3, 65.7, 65.9, 66.1, 66.2, 66.2, 66.0,
         65.9, 65.7, 65.4, 65.0, 64.7, 64.3, 63.9, 63.5, 62.7, 62.2, 62.0, 61.7, 61.4, 61.2, 61.1, 61.0, 61.1, 61.2, 61.3,
         61.5, 61.9, 62.2, 62.5, 62.8, 63.2, 63.5, 63.7, 64.1, 64.4, 64.6, 64.8, 64.9, 65.0, 65.0, 65.0, 64.9, 64.9, 64.8,
         64.6, 64.4, 64.2, 64.0, 63.7, 63.5, 63.3, 63.0, 62.8, 62.6, 62.4, 62.3, 62.2, 62.2, 62.1, 62.2, 62.3, 62.4, 62.6,
         62.8, 62.9, 63.2, 63.4, 63.6, 63.8, 64.0, 64.2, 64.5, 64.6, 64.7, 64.6, 64.6, 64.6, 64.6, 64.5, 64.4, 64.3, 64.2,
         64.2, 64.0, 63.8, 63.6, 63.4, 63.3, 63.2, 63.2, 63.1, 63.0, 62.9, 62.9, 62.8, 62.8, 62.8, 62.8, 62.9, 62.9, 63.0,
         63.1, 63.2, 63.3, 63.4, 63.5, 63.6, 63.6, 63.7, 63.7, 63.8, 63.8, 63.8, 63.8, 63.7, 63.7, 63.7, 63.7, 63.7, 63.7,
         63.6, 63.6, 63.5, 63.4, 63.3, 63.2, 63.2, 63.1, 63.0, 62.9, 62.9, 62.9, 62.9, 62.9, 62.9, 62.9, 63.0, 63.0, 63.0,
         63.1, 63.2, 63.2, 63.3, 63.4, 63.5, 63.5, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.6, 63.5,
         63.5, 63.4, 63.3, 63.3, 63.2, 63.2, 63.1, 63.1, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.0, 63.1, 63.1, 63.2,
         63.2, 63.3, 63.3, 63.4]
j = 0
xData = []
for i in yData:
    xData.append(j)
    j += 15

unc = []
for i in yData:
    if i == 55.6:
        unc.append(.1)
    else:
        unc.append(.2)

yDataAngle = []
for i in yData:
    yDataAngle.append((i-55.6)/(2*705.0))

uncAngle = []
for i in yData:
    if i == 55.6:
        uncAngle.append(.1/(2*705.0))
    else:
        uncAngle.append((i-55.6)/(2*705.0)*np.sqrt((.2/(i-55.6))**2+(4.8/705.0)**2))

def DampedOscDiffEvol(x, y, u):
    """
    Fits a damped oscillation of the form a*e^(-gamma*x)*cos(omega*x)+b by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
        
    Returns
    -------
    a, gamma, omega, b : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    red_chi_sq : float
                 reduced chi squared value of optimal fit as calculated by the sum of squared
                 residuals divided by the degrees of freedom.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-10, 0]] + [[0, .01]] + [[0, .1]] + [[0, 100]]

    def objective(s):
        a, gamma, omega, b = np.array_split(s, 4)
        return np.sum(((y - (a*(np.exp(-gamma*x))*(np.cos(omega*x))+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    a, gamma, omega, b = np.split(s, 4)
    return a, gamma, omega, b, red_chi_sq

def EvalFit(x, a, gamma, omega, b):
    """
    Evaluates the function a*e^(-gamma*x)*cos(omega*x)+b for input parameters a, gamma, omega, b, and
    a 1-D array of values x.
    """
    return a*(np.exp(-gamma*x))*(np.cos(omega*x))+b

a, gamma, omega, b, red_chi_sq = DampedOscDiffEvol(xData, yData, unc)
print(a)
print(gamma)
print(omega)
print(b)
print(red_chi_sq)

a2, gamma2, omega2, b2, red_chi_sq2 = DampedOscDiffEvol(xData, yDataAngle, uncAngle)
chi_sq2 = (len(yData)-4)*red_chi_sq2
print(a2)
print(gamma2)
print(omega2)
print(b2)
print(red_chi_sq2)
print(chi_sq2)

xx = np.linspace(0, 3480, num=3480)
yFit = EvalFit(xx, a, gamma, omega, b)
yFitAngle = EvalFit(xx, a2, gamma2, omega2, b2)

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yData, yerr=unc, markersize=1.8, elinewidth=1, fmt='ro')
plt.plot(xx, yFit, 'b', label = '$-7.25e^{-.00109t}\cos{(.0115t)}+63.4$')
plt.title('Damped Oscillation Fit vs Data')
plt.xlabel('Time (s)')
plt.ylabel('Laser Beam Position (cm)')
plt.legend()
plt.savefig('data.png', dpi=300, bbox_inches='tight')
plt.close()

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yDataAngle, yerr=uncAngle, markersize=1.8, elinewidth=1, fmt='ro')
plt.plot(xx, yFitAngle, 'b', label = '$-.00525e^{-.00111t}\cos{(.01149t)}+.00553$')
plt.title('Damped Oscillation Fit vs Angular Deflection Data')
plt.xlabel('Time (s)')
plt.ylabel('Angular Deflection (Radians)')
plt.legend()
plt.savefig('angledata.png', dpi=300, bbox_inches='tight')
plt.close()

def DampedOscDiffEvolFixedaParam(x, y, u, a):
    """
    a is a fixed input parameter.
    
    Fits a damped oscillation of the form a*e^(-gamma*x)*cos(omega*x)+b by minimizing the
    chi-square value using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    a : fixed input parameter.
        
    Returns
    -------
    gamma, omega, b : floats
                         optimal soltution parameters that minimize
                         the chi-square value with the given fixed a value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, .01]] + [[0, .1]] + [[0, 100]]

    def objective(s):
        gamma, omega, b = np.array_split(s, 3)
        return np.sum(((y - (a*(np.exp(-gamma*x))*(np.cos(omega*x))+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    gamma, omega, b = np.split(s, 3)
    return gamma, omega, b, chi_sq

optChi_Sq = chi_sq2
posChi_Sq = chi_sq2
negChi_Sq = chi_sq2
adiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    adiff += 0.00001
    gamma, omega, b, posChi_Sq = DampedOscDiffEvolFixedaParam(xData, yDataAngle, uncAngle, a2+adiff)
    gamma, omega, b, negChi_Sq = DampedOscDiffEvolFixedaParam(xData, yDataAngle, uncAngle, a2-adiff)

def DampedOscDiffEvolFixedgammaParam(x, y, u, gamma):
    """
    gamma is a fixed input parameter.
    
    Fits a damped oscillation of the form a*e^(-gamma*x)*cos(omega*x)+b by minimizing the
    chi-square value using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    gamma : fixed input parameter.
        
    Returns
    -------
    a, omega, b : floats
                         optimal soltution parameters that minimize
                         the chi-square value with the given fixed gamma value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-10, 0]] + [[0, .1]] + [[0, 100]]

    def objective(s):
        a, omega, b = np.array_split(s, 3)
        return np.sum(((y - (a*(np.exp(-gamma*x))*(np.cos(omega*x))+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a, omega, b = np.split(s, 3)
    return a, omega, b, chi_sq

optChi_Sq = chi_sq2
posChi_Sq = chi_sq2
negChi_Sq = chi_sq2
gammadiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    gammadiff += 0.00001
    a, omega, b, posChi_Sq = DampedOscDiffEvolFixedgammaParam(xData, yDataAngle, uncAngle, gamma2+gammadiff)
    a, omega, b, negChi_Sq = DampedOscDiffEvolFixedgammaParam(xData, yDataAngle, uncAngle, gamma2-gammadiff)

def DampedOscDiffEvolFixedomegaParam(x, y, u, omega):
    """
    omega is a fixed input parameter.
    
    Fits a damped oscillation of the form a*e^(-gamma*x)*cos(omega*x)+b by minimizing the
    chi-square value using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    omega : fixed input parameter.
        
    Returns
    -------
    a, gamma, b : floats
                         optimal soltution parameters that minimize
                         the chi-square value with the given fixed omega value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-10, 0]] + [[0, .01]] + [[0, 100]]

    def objective(s):
        a, gamma, b = np.array_split(s, 3)
        return np.sum(((y - (a*(np.exp(-gamma*x))*(np.cos(omega*x))+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a, gamma, b = np.split(s, 3)
    return a, gamma, b, chi_sq

optChi_Sq = chi_sq2
posChi_Sq = chi_sq2
negChi_Sq = chi_sq2
omegadiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    omegadiff += 0.00001
    a, gamma, b, posChi_Sq = DampedOscDiffEvolFixedomegaParam(xData, yDataAngle, uncAngle, omega2+omegadiff)
    a, gamma, b, negChi_Sq = DampedOscDiffEvolFixedomegaParam(xData, yDataAngle, uncAngle, omega2-omegadiff)

def DampedOscDiffEvolFixedbParam(x, y, u, b):
    """
    b is a fixed input parameter.
    
    Fits a damped oscillation of the form a*e^(-gamma*x)*cos(omega*x)+b by minimizing the
    chi-square value using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    b : fixed input parameter.
        
    Returns
    -------
    a, gamma, omega : floats
                         optimal soltution parameters that minimize
                         the chi-square value with the given fixed b value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[-10, 0]] + [[0, .01]] + [[0, .1]]

    def objective(s):
        a, gamma, omega = np.array_split(s, 3)
        return np.sum(((y - (a*(np.exp(-gamma*x))*(np.cos(omega*x))+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    a, gamma, omega = np.split(s, 3)
    return a, gamma, omega, chi_sq

optChi_Sq = chi_sq2
posChi_Sq = chi_sq2
negChi_Sq = chi_sq2
bdiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    bdiff += 0.000001
    a, gamma, omega, posChi_Sq = DampedOscDiffEvolFixedbParam(xData, yDataAngle, uncAngle, b2+bdiff)
    a, gamma, omega, negChi_Sq = DampedOscDiffEvolFixedbParam(xData, yDataAngle, uncAngle, b2-bdiff)

print(adiff)
print(gammadiff)
print(omegadiff)
print(bdiff)
