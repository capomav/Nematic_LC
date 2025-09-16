#!/usr/bin/env python
# coding: utf-8

# Code for setting up the alpha-fcc lattice

# In[1]:


import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import inp_file


#Defining the variables and constants

k = 4
npart = inp_file.npart
uci = round(np.cbrt(npart/k))
ndens = inp_file.ndens
boxl = np.cbrt(npart/ndens,dtype = np.float64)
hboxl = 0.5 * boxl
ucl = boxl/uci
hucl = ucl/2
rroot3 = 1/np.sqrt(3,dtype = np.float64)


rx = np.zeros(npart,dtype=np.float64)
ry = np.zeros(npart,dtype=np.float64)
rz = np.zeros(npart,dtype=np.float64)
ux = np.zeros(npart,dtype=np.float64)
uy = np.zeros(npart,dtype=np.float64)
uz = np.zeros(npart,dtype=np.float64)

rx[0] = 0.0
ry[0] = 0.0
rz[0] = 0.0
ux[0] =  rroot3
uy[0] =  rroot3
uz[0] =  rroot3

rx[1] = hucl
ry[1] = hucl
rz[1] = 0.0
ux[1] =   rroot3
uy[1] =  -rroot3
uz[1] =  -rroot3

rx[2] = 0.0
ry[2] = hucl
rz[2] = hucl
ux[2] =  -rroot3
uy[2] =   rroot3
uz[2] =  -rroot3

rx[3] = hucl
ry[3] = 0.0
rz[3] = hucl
ux[3] =  -rroot3
uy[3] =  -rroot3
uz[3] =   rroot3


def create_afcc(uci,ucl,boxl,rx,ry,rz,ux,uy,uz):
    '''
    create_afcc(uci,ucl,boxl,rx,ry,rz,ux,uy,uz)
    This function produces the alpha fcc lattice with given parameters 
    
    '''
    
    m = 0
    for iz in range(uci):
        for iy in range(uci):
            for ix in range(uci):
                for iref in range(4):
                    rx[iref + m] = rx[iref] + ix * ucl
                    ry[iref + m] = ry[iref] + iy * ucl 
                    rz[iref + m] = rz[iref] + iz * ucl
                    
                    ux[iref + m] = ux[iref]
                    uy[iref + m] = uy[iref]
                    uz[iref + m] = uz[iref]
                m =m + 4
    
    
    for i in range(len(rx)):
        rx[i] = rx[i] - hboxl
        ry[i] = ry[i] - hboxl
        rz[i] = rz[i] - hboxl
    
    
    return rx,ry,rz,ux,uy,uz  


# In[2]:


if __name__ == "__main__" :
    
    rx,ry,rz,ux,uy,uz = create_afcc(uci,ucl,boxl,rx,ry,rz,ux,uy,uz)
    
    file = open("alpha_fcc_{}".format(int(ndens*100)),"w")
    #prop = zip(rx,ry,rz,ux,uy,uz)
    for i in range(len(rx)):
        file.write(f"{rx[i]:.12f} {ry[i]:.12f} {rz[i]:.12f} {ux[i]:.12f} {uy[i]:.12f} {uz[i]:.12f}\n")    
    file.close()    
    
    
    # Plotting
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    #ax.set_title('')
    #plt.xlim(-15,6)
    #plt.ylim(-)
    #ax.scatter(rx[0:],ry[0:],rz[0:],c = 'k')
    ax.quiver(rx,ry,rz,ux,uy,uz,length = 0.7,pivot='middle')
    #plt.savefig("plot.png", dpi = 400)
    plt.show()
    


# In[ ]:


'''%gui qt
mlab.clf()
plot = mlab.quiver3d(rx,ry,rz,ux,uy,uz,mode = '2ddash')
mlab.show_pipeline()
#mlab.show_pipeline()
#mlab.pipeline.vectors.glyph_position = 'center'
#vectors.glyph.glyph_source.glyph_position = 'center'''

