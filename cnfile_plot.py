#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from inp_file import * 


# In[2]:



rx = []
ry = []
rz = []
ux = []
uy = []
uz = []

file = open("cnfile",'r')
cnt = 0

while True:
    data = file.readline()
    if not data:
        break
    data = data.replace("\n"," ")
    data = data.split(' ')
    rx.append(np.float(data[0]))
    ry.append(np.float(data[1]))
    rz.append(np.float(data[2]))
    ux.append(np.float(data[3]))
    uy.append(np.float(data[4]))
    uz.append(np.float(data[5]))
    cnt = cnt +1 
    
npart = cnt
boxl = np.cbrt(npart/ndens)

file.close()


rx = np.array(rx,dtype=np.float64)
ry = np.array(ry,dtype=np.float64)
rz = np.array(rz,dtype=np.float64)
ux = np.array(ux,dtype=np.float64)
uy = np.array(uy,dtype=np.float64)
uz = np.array(uz,dtype=np.float64)


# In[3]:


from mayavi import mlab
#get_ipython().run_line_magic('gui', 'qt')

mlab.clf()
mlab.quiver3d(rx,ry,rz,ux,uy,uz,mode = '2ddash')
mlab.show()


# In[ ]:




