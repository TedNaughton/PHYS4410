import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

yData = [65.9, 65.8, 65.7, 65.6, 65.3, 65.0, 64.8, 64.5, 64.1, 63.7, 63.3, 62.9,
         62.5, 62.1, 61.5, 61.0]

j = 0
xData = []
for i in yData:
    xData.append(j)
    j += 4

yDataAngle = []
for i in yData:
    yDataAngle.append((i-55.6)/(2*705.0))

uncAngle = []
for i in yData:
    if i == 65.9:
        uncAngle.append((i-55.6)/(2*705.0)*np.sqrt((.1/(i-55.6))**2+(4.8/705.0)**2))
    else:
        uncAngle.append((i-55.6)/(2*705.0)*np.sqrt((.2/(i-55.6))**2+(4.8/705.0)**2))

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yDataAngle, yerr=uncAngle, markersize=1.8, elinewidth=1, fmt='ro')
plt.title('Angular Deflection Data - Initial Acceleration Method')
plt.xlabel('Time (s)')
plt.ylabel('Angular Deflection (Radians)')
plt.legend()
plt.savefig('accelerationdata.png', dpi=300, bbox_inches='tight')
plt.close()

def QuadraticDiffEvol(x, y, u):
    """
    Fits a quadratic function of the form (-a/2)*x^2+b by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
        
    Returns
    -------
    a, b : floats
                         optimal soltution parameters that minimize
                         the chi-square value.
    red_chi_sq : float
                 reduced chi squared value of optimal fit as calculated by the sum of squared
                 residuals divided by the degrees of freedom.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, .00001]] + [[0, .01]]

    def objective(s):
        a, b = np.array_split(s, 2)
        return np.sum(((y - ((-a/2.)*(x**2.)+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    red_chi_sq = objective(s)/(len(x)-len(s))
    a, b = np.split(s, 2)
    return a, b, red_chi_sq

def EvalFit(x, a, b):
    """
    Evaluates the function (-a/2)*x^2+b for input parameters a, and b and
    a 1-D array of values x.
    """
    x = np.asarray(x)
    
    return (-a/2.)*(x**2.)+b

a, b, red_chi_sq = QuadraticDiffEvol(xData, yDataAngle, uncAngle)
chi_sq = (len(yData)-2)*red_chi_sq
print(a)
print(b)
print(red_chi_sq)
print(chi_sq)

xx = np.linspace(0, 60, num=60)
yFitAngle = EvalFit(xx, a, b)
yFitData = EvalFit(xData, a, b)
zeroes = []
for i in xx:
    zeroes.append(0)

#plot of fit vs data
fig = plt.figure()
plt.errorbar(xData, yDataAngle, yerr=uncAngle, markersize=1.8, elinewidth=1, fmt='ro')
plt.plot(xx, yFitAngle, 'b', label = '$(-.00000194/2)x^2+.00717$')
plt.title('Quadratic Fit vs Angular Deflection Data')
plt.xlabel('Time (s)')
plt.ylabel('Angular Deflection (Radians)')
plt.legend()
plt.savefig('quadraticfit.png', dpi=300, bbox_inches='tight')
plt.close()

residuals = np.subtract(yDataAngle, yFitData)

#residual plot
fig = plt.figure()
plt.errorbar(xData, residuals, yerr=uncAngle, fmt='ro')
plt.plot(xx, zeroes, 'b')
plt.title('Residual Plot of Quadratic Fit')
plt.xlabel('Time (s)')
plt.ylabel('Residual')
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
plt.close()

def QuadraticDiffEvolFixedaParam(x, y, u, a):
    """
    a is a fixed input parameter.

    Fits a quadratic function of the form (-a/2)*x^2+b by minimizing the chi-square value
    using scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : 1-dimensional list of x-values.
    y : 1-dimensional list of y-values.
    u : 1-dimensional list of uncertainties for each y-value.
    a : fixed input parameter.
        
    Returns
    -------
    b : float
                 optimal soltution parameters that minimize
                 the chi-square value.
    chi_sq : float
                 chi squared value of optimal fit as calculated by the sum of squared
                 residuals.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    bounds = [[0, .01]]

    def objective(s):
        b = np.array_split(s, 1)
        return np.sum(((y - ((-a/2.)*(x**2.)+b))**2.)/(u**2.))

    result = differential_evolution(objective, bounds)
    print(result)
    s = result['x']
    chi_sq = objective(s)
    b = np.split(s, 1)
    return b, chi_sq

optChi_Sq = chi_sq
posChi_Sq = chi_sq
negChi_Sq = chi_sq
adiff = 0
while (((posChi_Sq-optChi_Sq) < 1) or ((negChi_Sq-optChi_Sq) < 1)):
    adiff += 0.00000001
    b, posChi_Sq = QuadraticDiffEvolFixedaParam(xData, yDataAngle, uncAngle, a+adiff)
    b, negChi_Sq = QuadraticDiffEvolFixedaParam(xData, yDataAngle, uncAngle, a-adiff)

print(adiff)
