#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from inp_file import * 
import time
from numba import njit,jit,prange,set_num_threads
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import struct
import os
import sys

set_num_threads(5)

np.random.seed(10)

###################
# Calculate chi,chip,chicm,chipcm

chi = (sigmaebys**2 - 1)/(sigmaebys**2 + 1)
chip = (1 - np.float_power(epsilonebys,1/mu))/(1 + np.float_power(epsilonebys,1/mu))
sigma0inv = np.float64(1.0/sigma0)
chicm = (sigmaebyscm**2 - 1) / (sigmaebyscm**2 + 1)
chipcm = 1 - epsilonsbyecm**(-1/mu)
boxl = np.cbrt(npart/ndens)


###################
# Defining the freuently used functions/operations

@njit()
def g(ch,rusum,rudiff,u1u2,rr):
    return ( 1 - ch * (0.5/rr) * (  (rusum**2/(1+ch*u1u2)) + (rudiff**2/(1-ch*u1u2))  ) )

@njit()
def epsilon_bar(u1u2):
    return 1/np.sqrt(1 - (chi* u1u2)**2)

@njit()
def dg_dxi(ch,r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr):
    return (-ch*(1/rr)*( (rusum*(u1xi+u2xi)/(1+ch*u1u2)) +  (rudiff*(u1xi-u2xi)/(1-ch*u1u2)) ))            + ch*r12xi*(1/rr**2)*( (rusum**2/(1+ch*u1u2)) + (rudiff**2/(1 - ch*u1u2))  )

@njit()
def dg_duxi(ch,r12xi,u2xi,rusum,rudiff,u1u2,rr):
    return -ch*0.5*(1/rr)*( r12xi*( (2*rusum/(1+ch*u1u2)) + (2*rudiff/(1-ch*u1u2)) )           + ch*u2xi*( (rudiff**2/(1-ch*u1u2)**2) - (rusum**2/(1+ch*u1u2)**2) ) )

@njit()
def fgbxi(r12xi,rusum,rudiff,u1u2,u1xi,u2xi,r12,rr,R,LJ,dLJ):
    global chi,chip,sigma0inv,mu,nu
    
    return -4 * (epsilon0 *epsilon_bar(u1u2)**nu * mu * g(chip,rusum,rudiff,u1u2,rr)**(mu-1)           * dg_dxi(chip,r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr) * LJ + epsilon0*epsilon_bar(u1u2)**nu           * g(chip,rusum,rudiff,u1u2,rr)**mu * dLJ           * (sigma0inv * (r12xi/r12) + 0.5*(1/g(chi,rusum,rudiff,u1u2,rr)**(1.5))*dg_dxi(chi,r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr) ) )
    
@njit()
def du_duxi(r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr,R,LJ,dLJ):
    global chi,chip,sigma0inv,mu,nu
    
    return -4 * (LJ*epsilon0 * (nu*u2xi*u1u2*epsilon_bar(u1u2)**(nu+2)*g(chip,rusum,rudiff,u1u2,rr)**mu*chi**2            + mu*epsilon_bar(u1u2)**nu * g(chip,rusum,rudiff,u1u2,rr)**(mu-1)*dg_duxi(chip,r12xi,u2xi,rusum,rudiff,u1u2,rr) )            + epsilon0*epsilon_bar(u1u2)**nu * g(chip,rusum,rudiff,u1u2,rr)**mu * dLJ            * (0.5*(1/g(chi,rusum,rudiff,u1u2,rr)**(1.5))*dg_duxi(chi,r12xi,u2xi,rusum,rudiff,u1u2,rr) ) )

@njit()
def GBpot(rusum,rudiff,u1u2,rr,LJ,LJ_cut):
    return 4*epsilon0*epsilon_bar(chi,u1u2)**nu*g(chip,rusum,rudiff,u1u2,rr)**mu*(LJ - LJ_cut)

@njit()
def f_shift(rusum,rudiff,u1u2,rr,R_cut):
    global chi,chip,sigma0inv,mu,nu
    
    return 4*epsilon0*epsilon_bar(u1u2)**nu*(6*(1/R_cut**7) - 12*(1/R_cut**13))*g(chip,rusum,rudiff,u1u2,rr)**mu



###################
# Defining the freuently used functions/operations for colloid-mesogen interactions

@njit()
def Y(k,C1iui):
    return (1-k)/(1-k*(1 - C1iui**2))


@njit()
def sigmacm(sigma0,chicm,C1iui):
    return sigma0*Y(chicm,C1iui)**(-1/2)


@njit()
def epsilon_cm(mu,epsilon0cm,chipcm,C1iui):
    return epsilon0cm * Y(chipcm,C1iui)**mu


@njit()
def funZ(sigma0,E1i,C1i):
    return (1/45)*(sigma0/E1i)**9 - (1/6)*(sigma0/E1i)**3 - (1/40)*(sigma0/C1i)*(sigma0/E1i)**8             + (1/4)*(sigma0/C1i)*(sigma0/E1i)**2


@njit()
def dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi):
    return  (C1ixi/C1i) - (sigma0*chicm*C1iui*(uixi-C1iui*(C1ixi/C1i))*Y(chicm,C1iui)**(-1/2) )/(C1i*(1 - chicm*(1 - C1iui**2)) ) 


@njit()
def depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1ixi,C1iui,uixi):
    return (2*epsilon0cm*mu*chipcm*C1iui*(C1iui*(C1ixi/C1i)-uixi)*Y(chipcm,C1iui)**(mu))/(C1i*(1-chipcm*(1-C1iui**2)) )
    

@njit()
def dfunZ_dCxi(sigma0,chicm,E1i,C1ixi,C1i,C1iui,uixi):
    return (-1/5)*sigma0**9 *(1/E1i)**10 *dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi)             + (1/2)*sigma0**3 *(1/E1i)**4 *dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi)             + (1/40)*sigma0**9 *(1/E1i)**8 *(1/C1i)**3 *C1ixi             + (1/5)*(sigma0/E1i)**9 *(1/C1i)*dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi)             - (1/2)*(sigma0/E1i)**3 *(1/C1i)*dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi)             - (1/4)*(1/E1i)**2 *(sigma0/C1i)**3 *C1ixi


@njit()
def CM_Force(mu,sigma0,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui,uix,uiy,uiz):
    
    funz = funZ(sigma0,E1i,C1i)
    epsiloncm = epsilon_cm(mu,epsilon0cm,chipcm,C1iui)
    
    Fxcm = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1ix,C1iui,uix)*funz            + epsiloncm * dfunZ_dCxi(sigma0,chicm,E1i,C1ix,C1i,C1iui,uix)
    Fycm = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1iy,C1iui,uiy)*funz            + epsiloncm * dfunZ_dCxi(sigma0,chicm,E1i,C1iy,C1i,C1iui,uiy)
    Fzcm = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1iz,C1iui,uiz)*funz            + epsiloncm * dfunZ_dCxi(sigma0,chicm,E1i,C1iz,C1i,C1iui,uiz)
    
    Fxcm,Fycm,Fzcm = -Fxcm,-Fycm,-Fzcm
    
    return Fxcm,Fycm,Fzcm


@njit()
def depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1ixi):
    return -2*(epsilon0cm*mu*chipcm*Y(chipcm,C1iui)**(mu)*C1iui*C1ixi) / ( C1i*(1 - chipcm*(1 - C1iui**2)) )


@njit()
def dE1i_duixi(sigma0,chicm,C1i,C1iui,C1ixi):
    return -1*(sigma0*chicm*C1iui*C1ixi*Y(chicm,C1iui)**(-1/2) ) / ( C1i*(1- chicm*(1 - C1iui**2)) )


@njit()
def dfunZ_duixi(sigma0,chicm,E1i,C1iui,C1i,C1ixi):
    return ( (-1/5)*sigma0**9 *(1/E1i)**10 + (1/2)*sigma0**3 *(1/E1i)**4 + (1/5)*(sigma0/E1i)**9 *(1/C1i)             - (1/2)*(sigma0/E1i)**3 *(1/C1i) )*dE1i_duixi(sigma0,chicm,C1i,C1iui,C1ixi)      
    

@njit()
def CM_Torque(mu,sigma0,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui):
    
    funz = funZ(sigma0,E1i,C1i)
    epsiloncm = epsilon_cm(mu,epsilon0cm,chipcm,C1iui)
    
    Excm = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1ix) * funz            + epsiloncm * dfunZ_duixi(sigma0,chicm,E1i,C1iui,C1i,C1ix)
    Eycm = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1iy) * funz            + epsiloncm * dfunZ_duixi(sigma0,chicm,E1i,C1iui,C1i,C1iy)
    Ezcm = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1iz) * funz            + epsiloncm * dfunZ_duixi(sigma0,chicm,E1i,C1iui,C1i,C1iz)
    
    return -Excm,-Eycm,-Ezcm


@njit()
def dE1i_dCxi_sft(sigma0,chicm,C1ixi,rcut_c,C1iui,uixi):
    return -(sigma0*chicm*C1iui*uixi*Y(chicm,C1iui)**(-1/2) )/(rcut_c*(1 - chicm*(1 - C1iui**2)) ) 


@njit()
def depscm_dCxi_sft(mu,epsilon0cm,chipcm,rcut_c,C1iui,uixi):
    return -(2*epsilon0cm*mu*chipcm*C1iui*uixi*Y(chipcm,C1iui)**(mu))/(rcut_c*(1-chipcm*(1-C1iui**2)) )


@njit()
def dfunZ_dCxi_sft(sigma0,chicm,E1i_c,C1ixi,rcut_c,C1i,C1iui,uixi):
    return ( (-1/5)*sigma0**9 *(1/E1i_c)**10 + (1/2)*sigma0**3 *(1/E1i_c)**4 + (1/5)*(sigma0/E1i_c)**9 *(1/rcut_c)             - (1/2)*(sigma0/E1i_c)**3 *(1/rcut_c) )*dE1i_dCxi(sigma0,chicm,C1ixi,C1i,C1iui,uixi)


@njit()
def funZ_sft(sigma0,E1i_c,rcut_c):
    return (1/45)*(sigma0/E1i_c)**9 - (1/6)*(sigma0/E1i_c)**3 - (1/40)*(sigma0/rcut_c)*(sigma0/E1i_c)**8            + (1/4)*(sigma0/rcut_c)*(sigma0/E1i_c)**2


@njit()
def FCM_shift(mu,sigma0,rcut_c,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui,uix,uiy,uiz):
    
    E1i_c = E1i - C1i + rcut_c
    
    funz_sft = funZ_sft(sigma0,E1i_c,rcut_c)
    epsiloncm = epsilon_cm(mu,epsilon0cm,chipcm,C1iui)
    
    FCM_shft_x = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1ix,C1iui,uix)*funz_sft                  + epsiloncm * dfunZ_dCxi_sft(sigma0,chicm,E1i_c,C1ix,rcut_c,C1i,C1iui,uix)
    FCM_shft_y = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1iy,C1iui,uiy)*funz_sft                  + epsiloncm * dfunZ_dCxi_sft(sigma0,chicm,E1i_c,C1iy,rcut_c,C1i,C1iui,uiy)
    FCM_shft_z = depscm_dCxi(mu,epsilon0cm,chipcm,C1i,C1iz,C1iui,uiz)*funz_sft                  + epsiloncm * dfunZ_dCxi_sft(sigma0,chicm,E1i_c,C1iz,rcut_c,C1i,C1iui,uiz)
    
    return -FCM_shft_x,-FCM_shft_y,-FCM_shft_z


@njit()
def dfunZ_duixi_sft(sigma0,chicm,E1i_c,C1iui,rcut_c,C1ixi):
    return ((-1/5)*sigma0**9 *(1/E1i_c)**10 + (1/2)*sigma0**3 *(1/E1i_c)**4 + (1/5)*(sigma0/E1i_c)**9 *(1/rcut_c)             - (1/2)*(sigma0/E1i_c)**3 *(1/rcut_c) )*dE1i_duixi(sigma0,chicm,rcut_c,C1iui,C1ixi)           


@njit()
def CM_Torque_sft(mu,sigma0,rcut_c,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui):
    
    E1i_c = E1i - C1i + rcut_c
    
    funz_sft = funZ_sft(sigma0,E1i_c,rcut_c)
    epsiloncm = epsilon_cm(mu,epsilon0cm,chipcm,C1iui)
    
    Excm_sft = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1ix) * funz_sft                + epsiloncm * dfunZ_duixi_sft(sigma0,chicm,E1i_c,C1iui,rcut_c,C1ix)
    Eycm_sft = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1iy) * funz_sft                + epsiloncm * dfunZ_duixi_sft(sigma0,chicm,E1i_c,C1iui,rcut_c,C1iy)
    Ezcm_sft = depscm_duixi(mu,epsilon0cm,chipcm,C1i,C1iui,C1iz) * funz_sft                + epsiloncm * dfunZ_duixi_sft(sigma0,chicm,E1i_c,C1iui,rcut_c,C1iz)
    
    return -Excm_sft,-Eycm_sft,-Ezcm_sft


###################
# Functions for saving the trajectories  

def save_config(rx,ry,rz,ux,uy,uz,curts,Cx,Cy,Cz):
    
    curtstep = open(os.path.join(sys.path[0],'currtstep'),'w')
    curtstep.write(f'{curts}\n')
    curtstep.close()
    Ctemp = open(os.path.join(sys.path[0],'ctemp'),'w')
    Ctemp.write(f"{Cx} {Cy} {Cz} ")
    Ctemp.close()
    
    cnfile = open(os.path.join(sys.path[0],'cnfile'),'w')
    
    for i in range(len(rx)):
        cnfile.write(f"{rx[i]:.12f} {ry[i]:.12f} {rz[i]:.12f} {ux[i]:.12f} {uy[i]:.12f} {uz[i]:.12f}\n")  
    
    cnfile.close()


def write_traj(traj_fil,rx,ry,rz,ux,uy,uz):
    
    add = struct.pack('d'*len(rx),*rx)
    traj_fil.write(add)
    add = struct.pack('d'*len(ry),*ry)
    traj_fil.write(add)
    add = struct.pack('d'*len(rz),*rz)
    traj_fil.write(add)
    add = struct.pack('d'*len(ux),*ux)
    traj_fil.write(add)
    add = struct.pack('d'*len(uy),*uy)
    traj_fil.write(add)
    add = struct.pack('d'*len(uz),*uz)
    traj_fil.write(add)

def write_Ctraj(Ctraj_fil,Cx,Cy,Cz):
    Ctraj_fil.write(f"{Cx:.12f} {Cy:.12f} {Cz:.12f}\n")
    return None


#End of funtions    
#################################################################

# Initializing the empty variables for storing particle attributes

# For mesogens
rx = []
ry = []
rz = []
ux = []
uy = []
uz = []


# For Colloids

Cx = 0.0
Cy = 0.0
Cz = 0.0


file = open(os.path.join(sys.path[0],"alpha_fcc"),'r')
cnt = 0

while True:
    data = file.readline()
    if not data:
        break
    data = data.replace("\n"," ")
    data = data.split(' ')
    rx.append(float(data[0]))
    ry.append(float(data[1]))
    rz.append(float(data[2]))
    ux.append(float(data[3]))
    uy.append(float(data[4]))
    uz.append(float(data[5]))
    cnt = cnt +1 
    
npart = cnt

file.close()


rx = np.array(rx,dtype=np.float64)
ry = np.array(ry,dtype=np.float64)
rz = np.array(rz,dtype=np.float64)
ux = np.array(ux,dtype=np.float64)
uy = np.array(uy,dtype=np.float64)
uz = np.array(uz,dtype=np.float64)


# Calculating the frictional coefficients

ecc = np.sqrt(sigmar**2-1)/sigmar                         #sigmar is the aspect ratio
L = np.log((1+ecc)/(1-ecc))

XA = (8/3) * ecc**3 / (-2*ecc + (1 + ecc**2)*L)
YA = (16/3)* ecc**3 / (2*ecc + (3*ecc**2 -1)*L)
XC = (4/3) * ecc**3 * (1 - ecc**2) / (2 * ecc - (1 - ecc**2)*L)  
YC = (4/3) * ecc**3 * (2 - ecc**2) / (-2* ecc + (1 + ecc**2)*L)

zetapara_t = 4.*temp*sigmar*XA/sigma0**2
zetaper_t = 4.*temp*sigmar*YA/sigma0**2
zetapara_r = (4/3)*temp*sigmar**3*XC
zetaper_r = (4/3)*temp*sigmar**3*YC

# Calculating the standard deviation

std_zetapara_t = np.sqrt(2*zetapara_t*temp/dt)
std_zetapara_r = np.sqrt(2*zetapara_r*temp/dt)
std_zetaper_t = np.sqrt(2*zetaper_t*temp/dt)
std_zetaper_r = np.sqrt(2*zetaper_r*temp/dt)


# printing the dt value before starting the main loop

print("dt value is : ",dt)

# Main function is defined here which takes the system one step ahead/simulates on time step

@njit()
def force_cal(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz,i):

    Fxi = 0.0        # Force Accumulating variable for particle no. i
    Fyi = 0.0
    Fzi = 0.0

    Exi = 0.0        # Torque term accumulating variable for particle no. i
    Eyi = 0.0
    Ezi = 0.0

    #####
    
    # Adding force accumulator term if more than single colloid
    
    #####
    
    r1x = rx[i]      # Assigning the iterating particle to dummy variable i.e i th particle 
    r1y = ry[i]
    r1z = rz[i]

    u1x = ux[i]     
    u1y = uy[i]
    u1z = uz[i]

    if(np.isnan(u1x)):
            print('u1x is Nan for step and partice No:',step,i)

    for j in range(int(npart)):

        if(i!=j):

            r2x = rx[j]      # Assigning the iterating particle to dummy variable i.e j th particle 
            r2y = ry[j]
            r2z = rz[j]

            u2x = ux[j]
            u2y = uy[j]
            u2z = uz[j]

            # Now calculating the distances between the two particles (i.e. i th and j th particle)

            r12x = r1x - r2x  
            r12y = r1y - r2y
            r12z = r1z - r2z

            # Applying the Minimum image convention

            if(abs(r12x) != boxl): 
                r12x = r12x - boxl*round(r12x/boxl)    # Minimum Image Convention
            if(abs(r12y) != boxl):     
                r12y = r12y - boxl*round(r12y/boxl)    #
            if(abs(r12z) != boxl): 
                r12z = r12z - boxl*round(r12z/boxl)    #

            # Application of cutoff radius 

            if(r12x<rcut or r12y<rcut or r12z<rcut):           #Componentwise Cutoff

                # Now calculating the euclidean distance between paricles

                rr = r12x**2 + r12y**2 + r12z**2

                if(rr<rcutsq):                                 #Spherical cutoff

                    r12 = np.sqrt(rr)

                    u1u2 = u1x*u2x + u1y*u2y + u1z*u2z

                    ru1 = r12x*u1x + r12y*u1y + r12z*u1z  

                    ru2 = r12x*u2x + r12y*u2y + r12z*u2z

                    rusum = ru1 + ru2

                    rusumsq = rusum*rusum

                    rudiff = ru1 - ru2

                    rudiffsq = rudiff*rudiff

                    R = (r12/sigma0) - (1/np.sqrt(g(chi,rusum,rudiff,u1u2,rr))) + 1.0

                    Rinv = 1/R

                    R6inv = Rinv**6

                    R12inv = R6inv*R6inv

                    LJ = R12inv - R6inv

                    dLJ = 6*R6inv*Rinv - 12*R12inv*Rinv

                    R_cut = (rcut/sigma0) - (1/np.sqrt(g(chi,rusum,rudiff,u1u2,rr)))  + 1.0

                    LJ_cut = (1/R_cut)**12 - (1/R_cut)**6
                    
                    dLJ_cut = 6*(1/R_cut)**7 - 12*(1/R_cut)**13


                    # Calculating and accumulating the total force and torque acting on i th particle 

                    Fxi = Fxi + fgbxi(r12x,rusum,rudiff,u1u2,u1x,u2x,r12,rr,R,LJ,dLJ)                               + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12x/r12) 
                    
                    Fyi = Fyi + fgbxi(r12y,rusum,rudiff,u1u2,u1y,u2y,r12,rr,R,LJ,dLJ)                               + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12y/r12)
                    
                    Fzi = Fzi + fgbxi(r12z,rusum,rudiff,u1u2,u1z,u2z,r12,rr,R,LJ,dLJ)                               + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12z/r12)

                    Exi += du_duxi(r12x,rusum,rudiff,u1u2,u1x,u2x,rr,R,LJ,dLJ) #- du_duxi(r12x,rusum,rudiff,u1u2,u1x,u2x,rr,R_cut,LJ_cut,dLJ_cut)
                    Eyi += du_duxi(r12y,rusum,rudiff,u1u2,u1y,u2y,rr,R,LJ,dLJ) #- du_duxi(r12y,rusum,rudiff,u1u2,u1y,u2y,rr,R_cut,LJ_cut,dLJ_cut)
                    Ezi += du_duxi(r12z,rusum,rudiff,u1u2,u1z,u2z,rr,R,LJ,dLJ) #- du_duxi(r12z,rusum,rudiff,u1u2,u1z,u2z,rr,R_cut,LJ_cut,dLJ_cut)
    
    
    # Calculations for Colloid-mesogen interaction i.e Colloid and ith particle

    C1ix = r1x - Cx
    C1iy = r1y - Cy
    C1iz = r1z - Cz
    
    # Applying Minimum Image Convention
    
    if(abs(C1ix) != boxl): 
        C1ix = C1ix - boxl*round(C1ix/boxl)    # Minimum Image Convention
    if(abs(C1iy) != boxl): 
        C1iy = C1iy - boxl*round(C1iy/boxl)    #
    if(abs(C1iz) != boxl): 
        C1iz = C1iz - boxl*round(C1iz/boxl)    #
    

    C1i = np.sqrt(C1ix**2 + C1iy**2 + C1iz**2)
    d1i = C1i - CRa
    
    if(C1i < rcut_c):

        # Calculating the dot product of distance vector of Colloid-mesogen with oreintation vector of ith particle 
        C1iui = C1ix*u1x + C1iy*u1y + C1iz*u1z
        
        # Converting C1i_dot_ui_cap to C1i_cap_dot_ui_cap
        C1iui = C1iui/C1i
        
        # Calculating the distance term required in the equations
        E1i =  d1i - sigmacm(sigma0,chicm,C1iui) + sigma0

        # Passing the arguments for ith particle to calculate force and torque on LC

        FMxi,FMyi,FMzi = CM_Force(mu,sigma0,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui,u1x,u1y,u1z)
        TMxi,TMyi,TMzi = CM_Torque(mu,sigma0,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui)
         
        
        # Calculating the additional force term for shifted Colloid-mesogen interaction potential
        
        FCM_sftx,FCM_sfty,FCM_sftz = FCM_shift(mu,sigma0,rcut_c,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui,u1x,u1y,u1z)
        
        FMxi -= FCM_sftx
        FMyi -= FCM_sfty
        FMzi -= FCM_sftz
        
        # Calculating the additional torque term for shifted Colloid-mesogen interaction potential
        
        TM_sftx,TM_sfty,TM_sftz = CM_Torque_sft(mu,sigma0,rcut_c,epsilon0cm,chipcm,chicm,E1i,C1ix,C1iy,C1iz,C1i,C1iui)
        
        TMxi -= TM_sftx
        TMyi -= TM_sfty
        TMzi -= TM_sftz
    
    return Fxi,Fyi,Fzi,Exi,Eyi,Ezi,FMxi,FMyi,FMzi,TMxi,TMyi,TMzi


@njit(parallel=True)
def force_cal_i(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz):
    
     # Forces and Torques on mesogen due to mesogen-mesogen interactions
    
    Fx = np.zeros(npart,dtype=np.float64)
    Fy = np.zeros(npart,dtype=np.float64)
    Fz = np.zeros(npart,dtype=np.float64)
    Tx = np.zeros(npart,dtype=np.float64)
    Ty = np.zeros(npart,dtype=np.float64)
    Tz = np.zeros(npart,dtype=np.float64)
    
    Ex = np.zeros(npart,dtype=np.float64)
    Ey = np.zeros(npart,dtype=np.float64)
    Ez = np.zeros(npart,dtype=np.float64)
    
    
    # Forces and Torques on mesogen due to Colloid-Mesogen interactions
    
    FMx = np.zeros(npart,dtype=np.float64)
    FMy = np.zeros(npart,dtype=np.float64)
    FMz = np.zeros(npart,dtype=np.float64)
    TMx = np.zeros(npart,dtype=np.float64)
    TMy = np.zeros(npart,dtype=np.float64)
    TMz = np.zeros(npart,dtype=np.float64)
    
    EMx = np.zeros(npart,dtype=np.float64)
    EMy = np.zeros(npart,dtype=np.float64)
    EMz = np.zeros(npart,dtype=np.float64)
    
    # Forces on Colloid
    
    FCx = 0.0     # It is a list if more than single colloid in the system
    FCy = 0.0
    FCz = 0.0
    
    for i in prange(int(npart)):
        Fx[i],Fy[i],Fz[i],Ex[i],Ey[i],Ez[i],FMx[i],FMy[i],FMz[i],EMx[i],EMy[i],EMz[i] = force_cal(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz,i)
    
    
    for i in range(int(npart)):
        
        #Fx[i] = Fxi
        #Fy[i] = Fyi
        #Fz[i] = Fzi

        #FMx[i] = FMxi
        #FMy[i] = FMyi
        #FMz[i] = FMzi

        Txi = uy[i]*Ez[i] - uz[i]*Ey[i]
        Tyi = uz[i]*Ex[i] - ux[i]*Ez[i]
        Tzi = ux[i]*Ey[i] - uy[i]*Ex[i]

        Tx[i] = Txi
        Ty[i] = Tyi
        Tz[i] = Tzi

        TMx[i] = uy[i]*EMz[i] - uz[i]*EMy[i]
        TMy[i] = uz[i]*EMx[i] - ux[i]*EMz[i]
        TMz[i] = ux[i]*EMy[i] - uy[i]*EMx[i]

        FCx = FCx - FMx[i]
        FCy = FCy - FMy[i]
        FCz = FCz - FMz[i]

        
    return Fx,Fy,Fz,Tx,Ty,Tz,FMx,FMy,FMz,TMx,TMy,TMz,FCx,FCy,FCz


@njit()
def move(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz):
    
    # Calculating forces and torques on all particles (iterating over ith particle)
    Fx,Fy,Fz,Tx,Ty,Tz,FMx,FMy,FMz,TMx,TMy,TMz,FCx,FCy,FCz = force_cal_i(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz)
    
    #np.random.seed(2)
    rannum1 = np.random.normal(loc = 0, scale = std_zetapara_t, size = npart )
    rannum2 = np.random.normal(loc = 0, scale = std_zetaper_t, size = npart )
    rannum3 = np.random.normal(loc = 0, scale = std_zetaper_t, size = npart )
    rannum4 = np.random.normal(loc = 0, scale = std_zetapara_r, size = npart )
    rannum5 = np.random.normal(loc = 0, scale = std_zetaper_r, size = npart )
    rannum6 = np.random.normal(loc = 0, scale = std_zetaper_r, size = npart )
    
    
    for i in range(int(npart)):

        # Calculate the unit vectors in the body axes
        if(ux[i]**2 != 1.):
            
            mag_gamma = np.sqrt(uy[i]**2 + uz[i]**2)

            gamma_x = mag_gamma
            gamma_y = -ux[i]*uy[i]/mag_gamma
            gamma_z = -ux[i]*uz[i]/mag_gamma

            beta_x = gamma_y*uz[i] - gamma_z*uy[i]
            beta_y = gamma_z*ux[i] - gamma_x*uz[i]
            beta_z = gamma_x*uy[i] - gamma_y*ux[i]
            
        else :
            
            gamma_x = 0.    
            gamma_y = 0.
            gamma_z = 1.

            beta_x = 0.
            beta_y = 1.
            beta_z = 0.

        # Calculate the forces in body axes
        
        FGB_ucap = Fx[i]*ux[i] + Fy[i]*uy[i] + Fz[i]*uz[i]
        FGB_gamma = Fx[i]*gamma_x + Fy[i]*gamma_y + Fz[i]*gamma_z
        FGB_beta = Fx[i]*beta_x + Fy[i]*beta_y + Fz[i]*beta_z

        TGB_ucap = Tx[i]*ux[i] + Ty[i]*uy[i] + Tz[i]*uz[i]
        TGB_gamma = Tx[i]*gamma_x + Ty[i]*gamma_y + Tz[i]*gamma_z
        TGB_beta = Tx[i]*beta_x + Ty[i]*beta_y + Tz[i]*beta_z
        
        
        FM_ucap = FMx[i]*ux[i] + FMy[i]*uy[i] + FMz[i]*uz[i] 
        FM_gamma = FMx[i]*gamma_x + FMy[i]*gamma_y + FMz[i]*gamma_z
        FM_beta = FMx[i]*beta_x + FMy[i]*beta_y + FMz[i]*beta_z
        
        TM_ucap = TMx[i]*ux[i] + TMy[i]*uy[i] + TMz[i]*uz[i]
        TM_gamma = TMx[i]*gamma_x + TMy[i]*gamma_y + TMz[i]*gamma_z
        TM_beta = TMx[i]*beta_x + TMy[i]*beta_y + TMz[i]*beta_z
        
        
        FB_para = rannum1[i]
        FB_per1 = rannum2[i]
        FB_per2 = rannum3[i]
        TB_para = rannum4[i]
        TB_per1 = rannum5[i]
        TB_per2 = rannum6[i]
        
        
        # Calculate the velocities in the body axes

        v_ucap = (FGB_ucap + FB_para + FM_ucap)/zetapara_t
        v_gamma = (FGB_gamma + FB_per1 + FM_gamma)/zetaper_t
        v_beta = (FGB_beta + FB_per2 + FM_beta)/zetaper_t

        w_ucap = (TGB_ucap + TB_para + TM_ucap)/zetapara_r
        w_gamma = (TGB_gamma + TB_per1 + TM_gamma)/zetaper_r
        w_beta = (TGB_beta + TB_per2 + TM_beta)/zetaper_r
        
        
        # Calculate the velocities in space axes

        vx = v_ucap*ux[i] + v_gamma*gamma_x + v_beta*beta_x
        vy = v_ucap*uy[i] + v_gamma*gamma_y + v_beta*beta_y
        vz = v_ucap*uz[i] + v_gamma*gamma_z + v_beta*beta_z

        wx = w_ucap*ux[i] + w_gamma*gamma_x + w_beta*beta_x
        wy = w_ucap*uy[i] + w_gamma*gamma_y + w_beta*beta_y
        wz = w_ucap*uz[i] + w_gamma*gamma_z + w_beta*beta_z
        
        # Update the positions of particles
        
        rxi = rx[i] + vx*dt
        ryi = ry[i] + vy*dt
        rzi = rz[i] + vz*dt
        
        # Periodic Boundary Condition 
        
        rx[i] = rxi - boxl*round(rxi/boxl)
        ry[i] = ryi - boxl*round(ryi/boxl)
        rz[i] = rzi - boxl*round(rzi/boxl)
        
        # Calculating the change in orientation
        
        duxi = wy*uz[i] - wz*uy[i]
        duyi = wz*ux[i] - wx*uz[i]
        duzi = wx*uy[i] - wy*ux[i]
        
        # Update the orientation of particles
        
        uxi = ux[i] + duxi * dt
        uyi = uy[i] + duyi * dt 
        uzi = uz[i] + duzi * dt
        
        # Correction in the orientation vectors to unit vectors 

        correction_part = (ux[i]*uxi + uy[i]*uyi + uz[i]*uzi)**2 - (uxi**2 + uyi**2 + uzi**2) + 1.
        if(correction_part < 0):
            
            # Printing out the values of particle and the correction part and respective force, displacement and torque values
            print("Particle no:",i,"\nstep:",step,"\nCorrection:",correction_part,"\n")
            
            dummy = 1/0.0
            #return rx,ry,rz,np.inf*ux,uy,uz,Cx,Cy,Cz
            
            # Using direct normalization if error encountered
            norm = uxi*uxi + uyi*uyi + uzi*uzi
            ux[i] = uxi / norm
            uy[i] = uyi / norm
            uz[i] = uzi / norm
            
        else:    
            lamda = - (ux[i]*uxi + uy[i]*uyi + uz[i]*uzi) + np.sqrt(correction_part) 
            
            ux[i] = uxi + lamda*ux[i]
            uy[i] = uyi + lamda*uy[i]
            uz[i] = uzi + lamda*uz[i]
    
    
    # Calculating the Brownian force on colloid particle 
    
    Czeta = 8*CRa*temp/sigma0**3
    std_Czeta = np.sqrt(2*Czeta*temp/dt)
    
    FCBx = np.random.normal(loc = 0, scale = std_Czeta)
    FCBy = np.random.normal(loc = 0, scale = std_Czeta)
    FCBz = np.random.normal(loc = 0, scale = std_Czeta)
    
    # Calculating the velocities in space axis
    
    vcmx = (FCx + FCBx)/Czeta
    vcmy = (FCy + FCBy)/Czeta
    vcmz = (FCz + FCBz)/Czeta
    
    # Updating the position of Colloid
    
    Cx = Cx + vcmx*dt
    Cy = Cy + vcmy*dt
    Cz = Cz + vcmz*dt
    
    # Periodic Boundary Condition
    
    Cx = Cx - boxl*round(Cx/boxl)
    Cy = Cy - boxl*round(Cy/boxl)
    Cz = Cz - boxl*round(Cz/boxl)
    
    return rx,ry,rz,ux,uy,uz,Cx,Cy,Cz


print("Main Loop starts now\n")

# Timer for the program starts here 

t0 = time.time()

# Variables for counting time
time_counter_move = 0
time_counter_cnfile = 0
time_counter_trajfile = 0

# Creating a file for storing the trajectories of particles

traj_fil = open(os.path.join(sys.path[0],"traj_file"),"wb")
Ctraj_fil = open(os.path.join(sys.path[0],"Ctraj_file"),"w")

for step in range(tstep):
    
    t1 = time.time()
    # Calling the move function which performes simulation for one time step
    rx,ry,rz,ux,uy,uz,Cx,Cy,Cz = move(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,epsilon0cm,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chicm,chip,chipcm,rcut,rcut_c,rcutsq,temp,CRa,rx,ry,rz,ux,uy,uz,Cx,Cy,Cz)
    
    if(np.isnan(ux).any()):
        break
    
    t2 = time.time()
    time_counter_move += (t2-t1)
    
    # Printing the progress of the program/current tstep at regular interval
    if(step%step_interval == 0):
        print("step:",step)
    
    # Saving the configuration of this time step to cnfile for safety measure
    if(step%cn_interval == 0):
        save_config(rx,ry,rz,ux,uy,uz,step,Cx,Cy,Cz)
    
    t3 = time.time()
    time_counter_cnfile += (t3-t2)
    
    # Saving the trajectory at regular interval to binary file
    if(step%traj_interval==0):
        write_traj(traj_fil,rx,ry,rz,ux,uy,uz)
        write_Ctraj(Ctraj_fil,Cx,Cy,Cz)
        
    t4 = time.time()
    time_counter_trajfile += (t4-t3)
    
        
# Closing the trajectory file 
traj_fil.close()
Ctraj_fil.close()


tf = time.time()

# Printing out the total time required for running the program

out = open(os.path.join(sys.path[0],"out.txt"),"w")

#print("Total time : ",tf-t0)
out.write(f"Total time : {tf-t0} seconds\n")
#print("Time taken by move : ",time_counter_move)
out.write(f"Time taken by move : {time_counter_move} seconds\n")
#print("Time taken by cnfile : ",time_counter_cnfile)
out.write(f"Time taken by move : {time_counter_cnfile} seconds\n")
#print("Time taken by trajfile : ",time_counter_trajfile)
out.write(f"Time taken by move : {time_counter_trajfile} seconds\n")
out.write("\nEnd")
#print("\nEnd")
out.close()

#force_cal.parallel_diagnostics(level=1)






