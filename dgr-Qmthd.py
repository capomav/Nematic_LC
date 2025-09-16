#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import struct
from inp_file import ndens,npart
from numba import jit,njit
import time

@njit()
def kdelta(x,y):
    if (x==y):
        return 1
    else:
        return 0
    
@njit()
def nvector(m):
    n = m#np.absolute(m)
    mx = max(n)
    pos = 0
    for i in range(len(n)):
        if(mx==n[i]):
            pos = i
    return pos


@njit()
def dir_gr(npart,ndens,boxl,rx,ry,rz,u):
    
    Q = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            for k in range(len(rx)):
            	Q[i][j]  = Q[i][j] + (3/2 * u[i][k] * u[j][k] - kdelta(i,j)/2) 

    Q = np.divide(Q , int(len(rx)))

    w,v = np.linalg.eigh(Q)

    n_vec_pos = nvector(w)

    n_vect2 = v[int(n_vec_pos)]

    #n_vect2_mag = np.sqrt(n_vect2[0]**2 + n_vect2[1]**2 + n_vect2[2]**2)
    n_vect2_mag = np.linalg.norm(n_vect2)
    
    n_vect2 = n_vect2 / n_vect2_mag
    
    r_per = []
    r_par = []
    
    ct = 0
        
    for i in range(len(rx)):
        for j in range(len(rx)):
            if(i!=j):
                rxij = rx[i]-rx[j]
                rxij-=boxl*round(rxij/boxl)
                ryij = ry[i]-ry[j]
                ryij-=boxl*round(ryij/boxl)
                rzij = rz[i]-rz[j]
                rzij-=boxl*round(rzij/boxl)

                rij = np.sqrt( rxij**2 + ryij**2 + rzij**2 )
                #rij-=boxl*round(rij/boxl)
                
                #r.append(rij)
                #r_par[ct] = abs(rxij*n_vect2[0] +  ryij*n_vect2[1] +  rzij*n_vect2[2])
                #r_per[ct] = np.sqrt(rij **2 - (rxij*n_vect2[0] +  ryij*n_vect2[1] +  rzij*n_vect2[2])**2)
                #ct+=1
                r_par.append(abs(rxij*n_vect2[0] +  ryij*n_vect2[1] +  rzij*n_vect2[2]))
                r_per.append(np.sqrt(rij **2 - (rxij*n_vect2[0] +  ryij*n_vect2[1] +  rzij*n_vect2[2])**2))
    
    return r_par,r_per



####################

t0 = time.time()

boxl = np.cbrt(npart/ndens)

start = int(2e4)
end = start + 50
'''
rx = [0]*npart
ry = [0]*npart
rz = [0]*npart
ux = [0]*npart
uy = [0]*npart
uz = [0]*npart
'''

rx = np.zeros(npart)
ry = np.zeros(npart)
rz = np.zeros(npart)
ux = np.zeros(npart)
uy = np.zeros(npart)
uz = np.zeros(npart)

rmax = boxl/2
dr = boxl/200
tbins = int(rmax/dr)

print(tbins,dr)

bin_fil = open("traj_file","rb")

cnt = 0
totsteps = 0

h_par = np.zeros(tbins)
h_per = np.zeros(tbins)

while True:
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    rx[:] = data[:]    
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    ry[:] = data[:]
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    rz[:] = data[:]
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    ux[:] = data[:]
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    uy[:] = data[:]
    data = bin_fil.read(8*npart)
    if not data:
        break
    data = list(struct.unpack('d'*npart,data))
    uz[:] = data[:]
    
    
    if(cnt%10000==0):
        print(cnt)
    
    if(cnt >= start and cnt<end ):
        u = np.array([ux,uy,uz],dtype=np.float64)
        
        r_par,r_per = dir_gr(npart,ndens,boxl,rx,ry,rz,u)
        
        totsteps+=1
        print(totsteps)
        
        hist_per,edges = np.histogram(r_per,bins=tbins,range=(0,rmax))
        hist_par,edges = np.histogram(r_par,bins=tbins,range=(0,rmax))

        histn_per = np.pi*ndens*boxl*(edges[1:tbins+1]**2-edges[0:tbins]**2)
        histn_par = 2*ndens*boxl**2*(edges[1:tbins+1]-edges[0:tbins])

        h_par = h_par + (hist_par[:]/histn_par[:])
        h_per = h_per + (hist_per[:]/histn_per[:])
        
    cnt+=1
    if(cnt>end):
        break
    

t1 = time.time()
print((t1-t0)/60," mins\n")

g_par = h_par/(npart*totsteps)
g_per = h_per/(npart*totsteps)
rmid = 0.5*(edges[0:-1]+edges[1:])
plt.plot(rmid,g_par,label="par")
plt.plot(rmid,g_per,label="per")
plt.title("Averaged radial distribution function ")
plt.legend()
plt.xlabel("r")
plt.ylabel("g(r)")
plt.savefig("gr-time-avg combined-1",dpi=400,format="svg")

