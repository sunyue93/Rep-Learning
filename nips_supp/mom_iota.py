from scipy.io import loadmat
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# import torch.utils.data
# from torch.utils import data
# from torch.utils.data import DataLoader, TensorDataset
import scipy
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import shelve

N = 10
pf = .3
pt = .2
rf = int(N*pf)
rt = int(N*pt)
iota = 0.01*np.arange(101)
err = np.zeros((101,))
for idx in range(101):
  i = iota[idx]
  bSi = np.concatenate(  (np.ones((rf,)), i * np.ones((N-rf,)))  )
  B = np.concatenate(  (np.ones((rt,)), i * np.ones((N-rt,)))  )
  r1 = np.sum(B*bSi)
  sf = np.sum(bSi)
  st = np.sum(B)
  err[idx] = np.sqrt(sf)*r1 + np.sqrt(st)
  #save
  


plt.plot(iota,err/np.max(err),'b',linewidth = 2)
plt.xlabel(r'$\iota$',fontsize=25)
plt.ylabel(r'$||\hat\mathbf{M} -  \mathbf{M}||$',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.grid(True)
#plt.savefig('align_err_YS2.pdf')
plt.savefig('align_err_YS2.eps', format='eps')
plt.show()
