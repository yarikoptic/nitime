#!/usr/bin/env python
"""Bressler GC examples.

Reference: M. Ding, Y. Chen, and S.L. Bressler, "Granger Causality: Basic Theory and Application to Neuroscience," 2008.

"""

import numpy as np
import numpy.testing as npt
from matplotlib import pyplot as plt

from nitime import algorithms as alg
from nitime import granger
from nitime import utils

# For interactive testing
reload(granger)


### Example 1


# Generate noise with defined covariance structure
#   two noise vectors: variance 1, variance 0.7, covariance 0.4

length_of_noise=100000
sigmasq_1 = 1.0
sigmasq_2 = 0.7
cov_12   = 0.4
noise_covariance=np.array([[sigmasq_1, cov_12],[cov_12, sigmasq_2]])


noise_set_covariance = generate_noise_with_set_covariance(noise_covariance, length_of_noise)


# Generate AR time series

npts = 200
sigma = 1 #not used
drop_transients = 0
coefs = np.array([0.9, -0.5])
coefs_x = np.array([0.16, -0.2])
coefs_y = np.array([0.8, -0.5])


#FP STOPPED HERE 

#initialize
x_AR_time_series=np.zeros(200)
y_AR_time_series=np.zeros(200)
x_AR_time_series_all=np.zeros((500,100))
y_AR_time_series_all=np.zeros((500,100))
a=0

while a<500:
    #specify the window of noise to use from the noise vector
    noise_stop=200+(200*a)
    noise_start=noise_stop-200
    epsilon_noise=noise_set_covariance[noise_start:noise_stop,0]
    
    X, v, _ = utils.ar_generator(npts, sigma, coefs, drop_transients, epsilon_noise)
    x_AR_time_series=X
    x_AR_time_series_all[a]=X[100:200]

    eta_noise=noise_set_covariance[noise_start:noise_stop,1]
    X, v, _, _ = ar_generator_2nd_time_series(npts, sigma, coefs_x, coefs_y, drop_transients, eta_noise, x_AR_time_series)
    y_AR_time_series_all[a]=X[100:200]
    
    a=a+1




avg_x_AR_time_series=np.average(x_AR_time_series_all,0)
avg_y_AR_time_series=np.average(y_AR_time_series_all,0)

plt.plot(avg_x_AR_time_series)
plt.plot(avg_y_AR_time_series)


#TESTS:
#specifically test the time-series in this example
avg_sigma_est_x, avg_coeff_est_x = estimate_model_parameters(x_AR_time_series_all, 2)
print "simple_AR_model, sigma:", avg_sigma_est_x
print "correct_coeff:", coefs
print "simple_AR_model, coeff:", avg_coeff_est_x
print " "
avg_sigma_est_y, avg_coeff_est_y = estimate_model_parameters(y_AR_time_series_all, 4)
print "mixed_AR_model, sigma:", avg_sigma_est_y
print "correct_coeff:", coefs_y, coefs_x
print "mixed_AR_model, coeff:", avg_coeff_est_y
print " "

#sigma estimates coming out incorrect???
#coefficients coming out slightly incorrect???


#not sure if working correctly or not, so wrote this test instead
#    uses zero noise and 2nd time-series with constant amplitude
#    for some reason comes off really wrong...

#test with zero noise and 2nd time series constant
sigma_est, coeff_est = test_ar_generator_2nd_time_series_zero_noise_constant_2nd_time_series()
print "zero noise, constant 2nd time-series, sigma:", sigma_est
print "zero noise, constant 2nd time-series, coeff:",coeff_est

# MY GUESS IS THAT CANNOT USE THESE COEFFICIENT ESTIMATORS FOR THESE TYPES OF AR 
#   TIME-SERIES WHERE HAVE TWO SEPARATE INPUTS
#   The model for the autoregressive process takes this convention:
#       s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]
#   BUT THIS TIME-SERIES IS OF THE FORM: s[n] = a1*s[n-1] + a2*r[n-2]


fft_avg_x=np.fft.fft(avg_x_AR_time_series)
fft_avg_y=np.fft.fft(avg_y_AR_time_series)

#take the absolute value to get the magnitude info
abs_avg_x=abs(fft_avg_x)
abs_avg_y=abs(fft_avg_y)

#scale by the number of points so the magnitude does not depend on the length of the signal
abs_avg_x=abs_avg_x/len(abs_avg_x)
abs_avg_y=abs_avg_y/len(abs_avg_y)

#square the signal to convert magnitude to power
abs_avg_x=abs_avg_x**2
abs_avg_y=abs_avg_y**2


# Generate AR(2) time series
#X, v, _ = ar_generator(npts, sigma, coefs, drop_transients)

# Visualize
#plt.figure()
#plt.plot(v)
#plt.title('noise')
#plt.figure()
#plt.plot(X)
#plt.title('AR signal')

# Estimate the model parameters
#sigma_est, coefs_est = alg.yule_AR_est(X, 2, 2*npts, system=True)

#print 'coefs    :', coefs
#print 'coefs est:', coefs_est

#plt.show()
