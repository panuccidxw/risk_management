'''
Given two assets x and y with their marginal distribution and correlation for return
Find certain joint probability event, like both assets incurring loss
the core idea is to make use of bivariate normal distribution's elegant ability to incorporate correlation
between components
'''

import scipy
from scipy.stats import norm,multivariate_normal
import numpy as np

mean_x, scale_x, df_x = 0.04, 0.15, 8
mean_y, scale_y, df_y = 0.07, 0.17, 5
rho = 0.7

target_x = (0-mean_x)/scale_x
target_y = (0-mean_y)/scale_y

F_x = scipy.stats.t.cdf(target_x, df=df_x)
F_y = scipy.stats.t.cdf(target_y, df=df_y)

print(f"probability that x is less than 0 is {F_x:.4f} ")
print(f"probability that y is less than 0 is {F_y:.4f} ")

# ppf is the inverse function of cdf
Z_x = scipy.stats.norm.ppf(F_x)
Z_y = scipy.stats.norm.ppf(F_y)

mean = [0, 0]
cov = [[1, rho], [rho, 1]]
bvn = multivariate_normal(mean=mean, cov=cov)

# Using a numerical integration approach
from scipy import integrate

# method 1: manual integration with scipy.integrate.dblquad
result1, _ = integrate.dblquad(lambda y, x: bvn.pdf([x, y]), -np.inf, Z_x, -np.inf, Z_y)
print(result1)

# method2: scipy.stats.multivariate_normal.cdf
result2 = bvn.cdf([Z_x, Z_y])  # P(X ≤ Z_x, Y ≤ Z_y)
print(result2)

# method3: use Monte Carlo method to approximate integration, can be inaccurate
def bivariate_normal_cdf(x, y, rho):

    if rho == 0:
        return norm.cdf(x) * norm.cdf(y)
    n_samples = 100000
    samples = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, rho], [rho, 1]],
        size=n_samples)

    result = np.mean((samples[:, 0] <= x) & (samples[:, 1] <= y))
    return result

result3 = bivariate_normal_cdf(Z_x, Z_y, rho)
print(result3)

# method4: use statsmodels built-in
from statsmodels.distributions.copula.api import GaussianCopula
cop = GaussianCopula(rho)
result4 = cop.cdf([F_x, F_y])
print(result4)
