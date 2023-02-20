import xsimlab as xs
import xarray as xr
import numpy as np
import numba

#----------------

from numba.extending import get_cython_function_address
from numba import cfunc, vectorize, njit
import ctypes

# define the lower regularized incomplete gamma function for use with numba

addr = get_cython_function_address("scipy.special.cython_special", "gammainc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammainc_fn = functype(addr)

@vectorize('float64(float64, float64)')
def vec_gammainc(x, y):
    return gammainc_fn(x, y)

@njit
def gammainc_in_njit(x, y):
    return vec_gammainc(x, y)

# define the upper regularized incomplete gamma function for use with numba

addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaincc_fn = functype(addr)

@vectorize('float64(float64, float64)')
def vec_gammaincc(x, y):
    return gammaincc_fn(x, y)

@njit
def gammaincc_in_njit(x, y):
    return vec_gammaincc(x, y)

# define the gamma function for use with numba

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1gamma")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gamma_fn = functype(addr)

@vectorize('float64(float64)')
def vec_gamma(x):
    return gamma_fn(x)

@njit
def gamma_in_njit(x):
    return vec_gamma(x)

#----------------

def hydroecology(a, ac, lambda0, alpha0, PET, s0):
    '''
    Hydroecological module computing the mean discharge and discharge variability

    In input:
    - a: array of drainage areas
    - ac : array of storm size
    - lambda0 : array of storm frequency
    - alpha0: array of storm depth
    - PET: Potential evapo-transpiration
    - s0: array of dynamic soil storage
    
    In output:
    - mu: mean daily streamflow
    - nu: streamflow variability
    - tau: hydrologic response time scale
    - eta: ratio of drainage area to storm size
    - omega: streamflow ratio
    '''

    tau = (a/13782.539332528866)**0.13267696811579874 #tau is catchment response time
    eta = np.where(a>ac,a/ac,1) # eta is ratio of drainage area to storm size

    pbar = lambda0*alpha0 # mean precipitation rate
    phi = PET/pbar
    sstar = np.where(s0>0, s0 / alpha0, 1) # effective soil depth
    omega = np.where(s0>0,(phi*sstar**(sstar/phi)*np.exp(-sstar)
             /sstar/gamma_in_njit(sstar/phi)/gammainc_in_njit(sstar/phi,sstar)),1)
    
    lambdac = 1-(1-lambda0)**eta
    alphac = lambda0*alpha0/lambdac
    nu = 1/omega/lambdac/tau
    mu = omega*alphac*lambdac
    
    return mu, nu, tau, eta, omega

#----------------

@numba.njit
def solve(h0, m, n, a, kf, mu, nu, epsc, p, b, gamma, dt, length, stack, rec):
    '''
    Variable Discharge SPL model
    Returns incremental change in height over the time step
    
    In input:
    - h0: topograpyh at starts of time step (array of length nnode)
    - m: spl area exponent
    - n: spl slope exponent
    - a: drainage area (array of length nnode)
    - kf erodibility or spl rate parameter
    - mu: mean streamflow (array of length nnode)
    - nu: streamflow variability (array of length nnode)
    - epsc: critical erosion rate
    - p: slope exponent in critical discharge expression
        (=ad; if p=3n/8=> qc decreases with slope,
              if p=n=> qc has no dependence on slope,
    - b: recession exponent (must be 1 or 2)
    - gamma: daily streampower exponent
    - dt: time step
    - length: distance to receiver (array of length nnode)
    - stack: stack order (array of length nnode)
    - rec: receiver node (arrays of length nnode)
    
    In output:
    - dh: incremental topography change over time step
    '''

    h = h0.copy()
    qcstar = np.zeros_like(h)
    tol = 1e-3
    for i in stack:
        if (length[i]>0 and h[i]>h[rec[i]]):
            hr = h[rec[i]] # hr is receiver height
            dh0 = h0[i] - hr # dh0 is difference between node height and its receiver height
            S0 = dh0/length[i] # S0 is initial slope between node and its receiver
            F = dt*kf[i]*mu[i]**m*a[i]**m/dh0*S0**n # F factor
            G = dt*epsc/dh0*S0**p # G factor
            err = tol*2 # initializes error
            z = 1 # initializes z, the normalized/dimensionless height
            iter = 0
            
            while np.abs(err)>tol: # Newton-Raphson iterative loop
                iter=iter+1
                
                qc = (G/F)**(1/gamma)*z**((p-n)/gamma) # qc is normalized critical streamflow
                dxqc = qc*(p-n)/gamma/z # dxqc is derivative of qc with respect to z

                if b==1:
                    # compute mue = mu_epsilon
                    mue = (gamma_in_njit(1/nu[i]+gamma)/gamma_in_njit(1/nu[i])/nu[i]**(-gamma)*
                                   gammaincc_in_njit(1/nu[i]+gamma,qc/nu[i]))
                    # compute lae = lambda_epsilon
                    lae = (gammaincc_in_njit(1/nu[i],qc/nu[i]))
                    # compute dqmue = derivative of mu_epsilon with respect to qc
                    dqmue = (-np.exp(-qc/nu[i])*(qc/nu[i])**(1/nu[i]+gamma-1)/gamma_in_njit(1/nu[i])/nu[i]**(-gamma+1))
                    # compute dqlae = derivative of lambda_epsilon with respect to qc
                    dqlae = (-np.exp(-qc/nu[i])*(qc/nu[i])**(1/nu[i]-1)/gamma_in_njit(1/nu[i])/nu[i])

                elif b==2:
                    # compute mue = mu_epsilon
                    mue = (gamma_in_njit(1/nu[i]+1-gamma)/gamma_in_njit(1/nu[i])/nu[i]**(gamma-1)*
                                   gammainc_in_njit(1/nu[i]+1-gamma,1/qc/nu[i]))
                    # copmute lae = lambda_epsilon
                    lae = (gammainc_in_njit(1/nu[i]+1,1/qc/nu[i]))
                    # compute dqmue = derivative of mu_epsilon with respect to qc
                    dqmue = (-np.exp(-1/qc/nu[i])*(1/qc/nu[i])**(1/nu[i]-gamma+2)/gamma_in_njit(1/nu[i])/nu[i]**(gamma-2))
                    # compute dqlae = derivative of lambda_epsilon with respect to qc
                    dqlae = (-np.exp(-1/qc/nu[i])*nu[i]*(1/qc/nu[i])**(1/nu[i]+2)/gamma_in_njit(1/nu[i]+1))

                dxmue = dqmue*dxqc # dxmue is derivative of mu_epsilon with respect to z
                dxlae = dqlae*dxqc # dxlae is derivative of lambda_epsilon with respect to z

                # compute f(z) the value of the function for which we search the root
                f = z - 1 + F*mue*z**n - G*lae*z**p
                # compute the derivative of f(z) with respect to z at z
                fp = 1 + F*(mue*n*z**(n-1) + dxmue*z**n) - G*(lae*p*z**(p-1) + dxlae*z**p)

                # result is an improved value of z, the root, following Newton-Raphson scheme
                # zk = z - np.where(f>0, f/fp, 0)
                zk = z - f/fp
    
                err = np.abs(z - zk) # computes error
                z = zk # # updates normalized height
            
            h[i] = hr + z*dh0 # updates height once convergence is reached
            qcstar[i] = qc

    return h-h0, qcstar

#----------------

from fastscape.processes import (RasterGrid2D, SurfaceAfterTectonics, SingleFlowRouter, DrainageArea)
import attr

@xs.process
class VariableSPL:
    
    # input variables
    
    k_coef = xs.variable(dims=[(), ('y', 'x')], intent="in",
                         description="Erodibility or SPL rate coefficient",
                         attrs={"units": "m^{2-m}/yrs",
                               "typical range": "1e-6 -> 1e-4"},
                         default=1e-5)
    area_exp = xs.variable(intent="in", description="SPL area exponent",
                           attrs={"units": "",
                                 "typical range": "0.3 -> 0.8"},
                           default=0.45)
    slope_exp = xs.variable(intent='in', description="SPL slope exponent",
                            attrs={"units": "",
                                  "typical range": "0.8->2"},
                            default=1)
    recession_exp = xs.variable(intent='in', description="Recession exponent",
                    attrs={"units": ""}, validator=attr.validators.in_([1, 2]),
                    default=1)
    slope_exp_critical_stream_flow = xs.variable(intent='in',
                        description="Slope exponent in critical stream flow",
                        attrs={"units": "",
                        "typical range": "3n/8 -> n -> 3n/2 (where n is spl slope exponent)"},
                        default=1)
    daily_stream_power_exp = xs.variable(intent='in', description="Daily stream power exponent",
                        attrs={"units": ""},
                        default=0.75)
    threshold_erate = xs.variable(intent='in', description="Threshold erosion rate",
                       attrs={"units": "m/yr",
                             "typical range": "similar to uplift rate"},
                       default=1e-4)
    PET = xs.variable(dims=[(), ('y', 'x')], intent='in',
                      description="Potential Evapo-transpiration",
                      attrs={"units": "mm/day",
                            "typical range": "1.5 -> 5.6"},
                      default=1)
    soil_moisture_capacity = xs.variable(dims=[(), ('y', 'x')], intent='in',
                     description="Dynamic soil moisture capacity",
                     attrs={"units": "mm",
                           "typical range": "1-50"},
                     default=10)
    storm_depth = xs.variable(dims=[(), ('y', 'x')], intent='in',
                         description="Storm depth",
                         attrs={"units": "mm",
                               "typical range": "2 -> 30"},
                         default=10)
    daily_rainfall_frequency = xs.variable(dims=[(), ('y', 'x')], intent='in',
                          description="Mean daily rainfall frequency",
                          attrs={"units": "1/day",
                                "typical range": "0.05 -> 1"},
                          default=0.2)
    storm_size = xs.variable(dims=[(), ('y', 'x')], intent='in',
                     description="Storm size",
                     attrs={"units": "m^2",
                           "typical range": "1e7 -> model area"},
                     default=1e8)
    
    # output variables

    erosion = xs.variable(dims=("y", "x"), groups="erosion", intent="out",
                          description="Incremental erosion",
                          attrs={"units": "m"})
    mean_streamflow = xs.variable(dims=("y", "x"), intent="out",
                     description="Specific mean daily streamflow",
                     attrs={"units": "mm/day"})
    streamflow_variability = xs.variable(dims=("y", "x"), intent="out",
                     description="Streamflow variability index",
                     attrs={"units": ""})
    response_time = xs.variable(dims=("y", "x"), intent="out",
                      description="Catchment response time",
                      attrs={"units": "ds"})
    eta = xs.variable(dims=("y", "x"), intent="out",
                      description="Ratio of drainage area to storm size",
                      attrs={"units": ""})
    streamflow_ratio = xs.variable(dims=("y", "x"), intent="out",
                        description="Streamflow ratio",
                        attrs={"units":""})
    critical_streamflow = xs.variable(dims=("y", "x"), intent="out",
                        description="Critical streamflow",
                        attrs={"units":""})

    # foreign variables
    
    lengths = xs.foreign(SingleFlowRouter, "lengths", intent="in")
    elevation = xs.foreign(SurfaceAfterTectonics, "elevation", intent="in")
    receivers = xs.foreign(SingleFlowRouter, "receivers", intent="in")
    flowacc = xs.foreign(DrainageArea, "flowacc", intent="in")
    stack = xs.foreign(SingleFlowRouter, "stack",intent="in")
    shape = xs.foreign(RasterGrid2D, "shape")
    
    @xs.runtime(args=("step_delta"))
    def run_step(self, dt):
        '''
        compute erosion, mu, nu, tau, eta, omega, phi
        '''
        kf = np.broadcast_to(self.k_coef, self.shape).flatten()
        PET = np.broadcast_to(self.PET, self.shape).flatten()
        s0 = np.broadcast_to(self.soil_moisture_capacity, self.shape).flatten()
        alpha0 = np.broadcast_to(self.storm_depth, self.shape).flatten()
        lambda0 = np.broadcast_to(self.daily_rainfall_frequency, self.shape).flatten()
        ac = np.broadcast_to(self.storm_size, self.shape).flatten()
        area = self.flowacc.flatten()
        
        mu, nu, tau, eta, omega = hydroecology(area,
                                               ac, lambda0,
                                               alpha0, PET, s0)
        
        dh, qc = solve(self.elevation.flatten(),
                   self.area_exp, self.slope_exp,
                   area, kf,
                   mu, nu, self.threshold_erate,
                   self.slope_exp_critical_stream_flow,
                   self.recession_exp, self.daily_stream_power_exp,
                   dt, self.lengths,
                   self.stack, self.receivers)
        
        self.erosion = -dh.reshape(self.shape)
        self.mean_streamflow = mu.reshape(self.shape)
        self.streamflow_variability = nu.reshape(self.shape)
        self.response_time = tau.reshape(self.shape)
        self.eta = eta.reshape(self.shape)
        self.streamflow_ratio = omega.reshape(self.shape)
        self.critical_streamflow = qc.reshape(self.shape)



