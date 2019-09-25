# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:43:47 2019

@author: admin
"""

# normal model
#import package
import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt

#define normal function
def normal_formula(strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac
    #already maturity
    if( texp<0 or vol*np.sqrt(texp)<1e-8 ):
        return disc_fac * np.fmax( cp_sign*(forward-strike), 0 )

    vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)
    d = (forward - strike) / vol_std

    price = disc_fac * (cp_sign * (forward - strike) * ss.norm.cdf(cp_sign * d) + vol_std * ss.norm.pdf(d))
    return price

#define class
class NormalModel:
    
    vol, intr, divr = None, None, None
    
    def __init__(self, vol, intr=0, divr=0):
        self.vol = vol
        self.intr = intr
        self.divr = divr
    
    def price(self, strike, spot, texp, cp_sign=1):
        return normal_formula(strike, spot, self.vol, texp, intr=self.intr, divr=self.divr, cp_sign=cp_sign)
    
    def delta(self, strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
        div_fac = np.exp(-texp*divr)
        disc_fac = np.exp(-texp*intr)
        forward = spot / disc_fac * div_fac
        vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)#??这里应该使用类变量还是成员变量？？？？？？
        d = (forward - strike) / vol_std
        D= cp_sign*ss.norm.cdf(cp_sign * d)##基本上都用到d,在每个函数里面都要写吗？？？
        return D#原来这里写的是零？？？

    def vega(self, strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
        div_fac = np.exp(-texp*divr)
        disc_fac = np.exp(-texp*intr)
        forward = spot / disc_fac * div_fac
        vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)#??这里应该使用类变量还是成员变量？？？？？？
        d = (forward - strike) / vol_std
        V=ss.norm.pdf(d)*np.sqrt(texp)
        return V

    def gamma(self, strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
        div_fac = np.exp(-texp*divr)
        disc_fac = np.exp(-texp*intr)
        forward = spot / disc_fac * div_fac
        vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)#??这里应该使用类变量还是成员变量？？？？？？
        d = (forward - strike) / vol_std
        G=ss.norm.pdf(d)/vol_std
        return G

    def impvol(self, price, strike, spot, texp, cp_sign=1):
        func_impvol = lambda vol: normal_formula(strike, spot, vol, texp, intr=self.intr, divr=self.divr, cp_sign=cp_sign)- price
        IV = sopt.brentq(func_impvol,-100,100)#
        return IV
    