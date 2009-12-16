"""Granger "causality" support routines."""


import numpy as np
import numpy.testing as npt
from matplotlib import pyplot as plt

from nitime import utils
from nitime import algorithms as alg
#reload(utils)


def generate_gauss_white_noise(npts):
    """Gaussian white noise.

    XXX - incomplete."""

    # Amplitude - should be a parameter
    a = 1.
    # Constant, band-limited amplitudes
    # XXX - no bandlimiting yet
    amp = np.zeros(npts)
    amp.fill(a)
    
    # uniform phases
    phi = np.random.uniform(high=2*np.pi, size=npts)
    # frequency-domain signal
    c = amp*np.exp(1j*phi)
    # time-domain
    n = np.fft.ifft(c)

    # XXX No validation that output is gaussian enough yet
    return n


def generate_noise_with_set_covariance(cov_mat, npts, randgen = np.random.standard_normal):
    """
    # this generates 
    # XXX - incomplete
    """   
    # this decomposes the covariance matrix 
    cov_sqrt=np.linalg.cholesky(cov_mat)

    # this makes random normal values - so 0 mean and variance 1
    
    # Create array of uncorrelated noise
    nseries = len(cov_mat)
    starting_noise = randgen( (npts, nseries))

    # this converts Z into random values with the covariance structure you want
    noise_set_covariance = np.dot(starting_noise, cov_sqrt.T)
    
    # Only return the noise with set covariance
    return noise_set_covariance


def test_gen_set_cov(): 
    length_of_noise=100000
    noise_covariance=np.array([[1.0, 0.4], [0.4, 0.7]])
    noise = generate_noise_with_set_covariance(noise_covariance, length_of_noise)
    # check to make sure noise has structure you want
    mycov = np.cov(noise.T)
    npt.assert_almost_equal(mycov, noise_covariance, 2)


#def ar_generator(N=512, sigma=1., coefs=None, drop_transients=0, v=None):
#    """
#    # this generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
#    # where v(n) is a stationary stochastic process with zero mean
#    # and variance = sigma
#    # this sequence is shown to be estimated well by an order 8 AR system
#    """
#    if coefs is None:
#        coefs = np.array([2.7607, -3.8106, 2.6535, -0.9238])
#    else:
#        coefs = np.asarray(coefs)
#
#    # The number of terms we generate must include the dropped transients, and
#    # then at the end we cut those out of the returned array.
#    N += drop_transients
#
#    # Typically uses just pass sigma in, but optionally they can provide their
#    # own noise vector, case in which we use it
#    if v is None:
#        v = np.random.normal(size=N, scale=sigma**0.5)
#    else:
#        v = v
#        
#    u = np.zeros(N)
#    P = len(coefs)
#    for l in xrange(P):
#        u[l] = v[l] + np.dot(u[:l][::-1], coefs[:l])
#    for l in xrange(P,N):
#        u[l] = v[l] + np.dot(u[l-P:l][::-1], coefs)
#        
#    # Only return the data after the drop_transients terms
#    return u[drop_transients:], v[drop_transients:], coefs



def ar_generator_2nd_time_series(N=512, sigma=1., coefs_x=None, coefs_y=None, drop_transients=0, v=None, x_time_series=None):
    """
    # this generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    # where v(n) is a stationary stochastic process with zero mean
    # and variance = sigma
    # this sequence is shown to be estimated well by an order 8 AR system
    """
    if coefs is None:
        coefs_x = np.array([1.376, -2.1106])
        coefs_y = np.array([2.7607, -3.8106])
    else:
        coefs_x = np.asarray(coefs_x)
        coefs_y = np.asarray(coefs_y)
        
    # The number of terms we generate must include the dropped transients, and
    # then at the end we cut those out of the returned array.
    N += drop_transients

    # Typically uses just pass sigma in, but optionally they can provide their
    # own noise vector, case in which we use it
    if v is None:
        v = np.random.normal(size=N, scale=sigma**0.5)
    else:
        v = v
    
    #x_AR_time_series = x_time_series
    y_AR_time_series = np.zeros(N)
    
    P_x = len(coefs_x)
    P_y = len(coefs_y)
    
    for l in xrange(P_y):
        y_AR_time_series[l] = v[l] + np.dot(y_AR_time_series[:l][::-1], coefs_y[:l]) + np.dot(x_AR_time_series[:l][::-1], coefs_x[:l])

    for l in xrange(P_y,N):
        #print l
        y_AR_time_series[l] = v[l] + np.dot(y_AR_time_series[l-P_y:l][::-1], coefs_y) + np.dot(x_AR_time_series[l-P_x:l][::-1], coefs_x)

        
    # Only return the data after the drop_transients terms
    return y_AR_time_series[drop_transients:], v[drop_transients:], coefs_x, coefs_y


def test_ar_generator_2nd_time_series_zero_noise_constant_2nd_time_series(): 
    npts = 200
    sigma = 1 #not used
    coefs_x = np.array([0.37, -0.4])
    coefs_y = np.array([0.7, -0.5])
    drop_transients = 0
    noise=np.zeros(200)
    x_time_series=np.ones(200)
    
    X, v, _, _ = ar_generator_2nd_time_series(npts, sigma, coefs_x, coefs_y, drop_transients, noise, x_time_series)
    mixed_AR_time_series=X[100:200]

    sigma_est, coefs_est = alg.yule_AR_est(mixed_AR_time_series, 4, npts, system=True)
    return sigma_est, coefs_est




def estimate_model_parameters(time_series, coeffs):
    """
    # time-series: array [number of time-series, length of each time-series]
    # coeffs: number of coeffs to solve for in AR process
    # npts: number of time
    #
    # this loops through as many time series as you have and finds the average sigma & coeff
    # this example have 500 time-series, so est sigma & coeff 500 times and find the average
    # XXX
    """
    
    row, col = np.shape(time_series)
    npts = col
    all_coefs_est=np.zeros((row,coeffs))
    all_sigma_est=np.zeros((row,1))
    a=0
    
    while a<row:
        one_time_series=time_series[a]
        sigma_est, coefs_est = alg.yule_AR_est(one_time_series, coeffs, npts, system=True)
        all_coefs_est[a]=coefs_est
        all_sigma_est[a]=sigma_est
        a=a+1
    
    average_all_coefs_est=np.average(all_coefs_est,0)    
    average_all_sigma_est=np.average(all_sigma_est,0)    
    
    return average_all_sigma_est, average_all_coefs_est
