
# coding: utf-8

# # Basket option implementation based on normal model

# In[1]:


import numpy as np
from option_models import basket
from option_models import bsm
from option_models import normal


# In[2]:


### only run this when you changed the class definition
import imp
imp.reload(basket)


# In[3]:


# A trivial test case 1: 
# one asset have 100% weight (the others zero)
# the case should be equivalent to the BSM or Normal model price

spot = np.ones(4) * 100
vol = np.ones(4) * 0.4
weights = np.array([1, 0, 0, 0])
divr = np.zeros(4)
intr = 0
cor_m = 0.5*np.identity(4) + 0.5
texp = 5
strike = 120

print(weights)

np.random.seed(123456)
price_basket = basket.basket_price_mc(strike, spot, vol*spot, weights, texp, cor_m, bsm=False)


# In[4]:


# Compare the price to normal model formula

norm1 = normal.NormalModel(vol=40)
price_norm = norm1.price(strike=120, spot=100, texp=texp, cp_sign=1)
print(price_basket, price_norm)


# In[6]:


# A trivial test case 2
# all assets almost perfectly correlated:
# the case should be equivalent to the BSM or Normal model price

spot = np.ones(4) * 100
vol = np.ones(4) * 0.4
weights = np.ones(4) * 0.25
divr = np.zeros(4)
intr = 0
cor_m = 0.0001*np.identity(4) + 0.9999*np.ones((4,4))
texp = 5
strike = 120

print( cor_m )

np.random.seed(123456)
price_basket = basket.basket_price_mc(strike, spot, vol*spot, weights, texp, cor_m, bsm=False)
print(price_basket, price_norm)


# In[7]:


# A full test set for basket option with exact price

spot = np.ones(4) * 100
vol = np.ones(4) * 0.4
weights = np.ones(4) * 0.25
divr = np.zeros(4)
intr = 0
cor_m = 0.5*np.identity(4) + 0.5
texp = 5
strike = 100
price_exact = 28.0073695


# In[8]:


cor_m


# In[9]:


price_basket = basket.basket_price_mc(strike, spot, vol*spot, weights, texp, cor_m, bsm=False)
print(price_basket, price_exact)


# # [To Do] Basket option implementation based on BSM model
# ## Write the similar test for BSM

# In[9]:


price_basket = basket.basket_price_mc(strike, spot, vol*spot, weights, texp, cor_m, bsm=True)


# In[10]:


# A trivial test case 1: 
# one asset have 100% weight (the others zero)
# the case should be equivalent to the BSM or Normal model price

spot = np.ones(4) * 100
vol = np.ones(4) * 0.4
weights = np.array([1, 0, 0, 0])
divr = np.zeros(4)
intr = 0
cor_m = 0.5*np.identity(4) + 0.5
texp = 5
strike = 120

print(weights)

np.random.seed(123456)
price_basket = basket.basket_price_mc(strike, spot, vol, weights, texp, cor_m, bsm=True)


# In[12]:


# Compare the price to bsm model formula

bsm1 = bsm.BsmModel(vol=0.4)
price_bsm = bsm1.price(strike=120, spot=100, texp=texp, cp_sign=1)
print(price_basket, price_bsm)


# # Spread option implementation based on normal model

# In[13]:


# A full test set for spread option

spot = np.array([100, 96])
vol = np.array([0.2, 0.1])
weights = np.array([1, -1])
divr = np.array([1, 1])*0.05
intr = 0.1
cor_m = np.array([[1, 0.5], [0.5, 1]])
texp = 1
strike = 0
price_exact = 8.5132252


# In[15]:


# MC price based on normal model
# make sure that the prices are similar

np.random.seed(123456)
price_spread = basket.basket_price_mc(strike, spot, vol*spot, weights, texp, cor_m, intr=intr, divr=divr, bsm=False)
print(price_spread, price_exact)


# # Spread option implementation based on BSM model

# In[16]:


# Once the implementation is finished the BSM model price should also work
price_spread = basket.basket_price_mc(
    strike, spot, vol, weights, texp, cor_m, intr=intr, divr=divr, bsm=True)


# In[17]:


# You also test Kirk's approximation
price_kirk = basket.spread_price_kirk(strike, spot, vol, texp, 0.5, intr, divr)
print(price_kirk, price_spread)


# # [To Do] Complete the implementation of basket_price_norm_analytic
# # Compare the MC stdev of BSM basket prices from with and without CV

# In[18]:


# The basket option example from above
spot = np.ones(4) * 100
vol = np.ones(4) * 0.4
weights = np.array([1, 0, 0, 0])
divr = np.zeros(4)
intr = 0
cor_m = 0.5*np.identity(4) + 0.5
texp = 5
strike = 120


# In[22]:


### only run this when you changed the class definition
import imp
imp.reload(basket)


# In[25]:


### Make sure that the analytic normal price is correctly implemented
price=basket.basket_price_norm_analytic(strike, spot, vol*spot, weights, texp, cor_m, intr=intr, divr=divr)
print("analytic normal price is " ,price)


# In[26]:


# Run below about 100 times and get the mean and stdev
### Returns 2 prices, without CV and with CV 
price = []
for i in range(100):
    price_basket = basket.basket_price_mc_cv(strike, spot, vol, weights, texp, cor_m, intr, divr)
    price.append(price_basket)

print("price mean:", np.mean(price, axis=0))
print("price std:", np.std(price, axis=0))

