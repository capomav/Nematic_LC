#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba import njit,jit,float64
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import struct
from inp_file import npart,ndens,temp,traj_interval


@njit()
def kdelta(x,y):
    if (x==y):
        return 1
    else:
        return 0
    
@njit()
def nvector(m):
    n = np.absolute(m)
    mx = max(n)
    pos = 0
    for i in range(len(n)):
        if(mx==n[i]):
            pos = i
    return pos
         
@jit()
def S_bin(Sa,binsize):
    
    bs = binsize
    Sb = [Sa[0]]
    x = [0]
    binavg = 0
    for i in range(len(Sa)):
        binavg = binavg + Sa[i]
        if(i%bs==0):
            Sb.append(binavg/bs)
            x.append((i/bs)*bs + (bs/2) )
            binavg = 0
    return Sb,x

@njit(float64[:](float64[:],float64[:],float64[:]))
def avg_meth(ux,uy,uz):
    
    ux_avg = 0.0
    uy_avg = 0.0
    uz_avg = 0.0
    
    n = int(len(ux))
    
    for i in range(n):
        ux_avg += ux[i]
        uy_avg += uy[i]
        uz_avg += uz[i]
    
    ux_avg = ux_avg/n
    uy_avg = uy_avg/n
    uz_avg = uz_avg/n
    
    n_vect = np.array([ux_avg,uy_avg,uz_avg],dtype=np.float64)
    
    n_vect = np.divide(n_vect,np.sqrt(ux_avg**2 + uy_avg**2 +uz_avg**2 ))
    
    return n_vect
    
@njit(float64[:](float64[:],float64[:],float64[:]))
def Qmeth(ux,uy,uz):
    
    Q = np.zeros((3,3),dtype=np.float64)
    u = [ux,uy,uz]
    
    for i in range(3):
        for j in range(3):
            for k in range(len(ux)):
                Q[i][j]  = Q[i][j] + (3/2 * u[i][k] * u[j][k] - kdelta(i,j)/2) 

    Q = np.divide(Q , int(len(ux)))
    
    w,v = np.linalg.eigh(Q)
    
    n_vec_pos = nvector(w)
    
    n_vect2 = v[int(n_vec_pos)]
    
    #n_vect2_mag = np.sqrt(n_vect2[0]**2 + n_vect2[1]**2 + n_vect2[2]**2)
    
    n_vect2_mag = np.linalg.norm(n_vect2)
    
    n_vect2 = np.divide(n_vect2 , n_vect2_mag) 
    
    return n_vect2
    
@njit()
def calOrder(n_vt,ux,uy,uz):

    S = 0
    for i in range(len(ux)):
        S = S + ( (3/2)*(np.dot(n_vt,np.array([ux[i],uy[i],uz[i]]))**2) -1/2 )
    S = S/int(len(ux))	
    
    return S

'''
rx = [0]*npart
ry = [0]*npart
rz = [0]*npart
ux = [0]*npart
uy = [0]*npart
uz = [0]*npart
'''

rx = np.zeros(npart,dtype=np.float64)
ry = np.zeros(npart,dtype=np.float64)
rz = np.zeros(npart,dtype=np.float64)
ux = np.zeros(npart,dtype=np.float64)
uy = np.zeros(npart,dtype=np.float64)
uz = np.zeros(npart,dtype=np.float64)

S_all = []

S_all2 = []


bin_fil = open("traj_file","rb")

cnt = 0

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
    
    
    n_vect = avg_meth(ux,uy,uz)
    
    S = calOrder(n_vect,ux,uy,uz)
    S_all.append(S)
    
    ###########################################
    
    n_vect2 = Qmeth(ux,uy,uz)

    assert np.linalg.norm(n_vect2) <=1.01 and np.linalg.norm(n_vect2) >=0.990, "not normalized"  
    
    S = calOrder(n_vect2,ux,uy,uz)
    S_all2.append(S)
    
    ############################################
    
    if(cnt%1000 == 0):
    	print(cnt)
    
    cnt+=1


bin_fil.close()

binsize =1000

Sbin,xvl = S_bin(S_all,binsize)
Sbin2,xvl2 = S_bin(S_all2,binsize)

plt.plot(xvl,Sbin,label = 'avg n')
plt.plot(xvl2,Sbin2,label = 'Q method')

plt.legend()
plt.xlabel(f'Time steps (x {traj_interval})')
plt.ylabel('Order Parameter S (P2)')
plt.title(f"Order parameter(binsize:{binsize},dens={ndens},T={temp})")

plt.ylim(-1,1.5)

plt.savefig('CompareS.svg',format='svg',dpi = 350)
plt.show()




