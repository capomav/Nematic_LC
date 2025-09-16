#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from inp_file import * 
import time
from numba import njit,jit,prange,set_num_threads
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import struct
import os
import sys


# Set the total number of cores to be used(need to match with the job script)
set_num_threads(6)

# Calculate chi,chip

chi = (sigmaebys**2 - 1)/(sigmaebys**2 + 1)
chip = (1 - np.float_power(epsilonebys,1/mu))/(1 + np.float_power(epsilonebys,1/mu))
sigma0inv = np.float64(1.0/sigma0)

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
    global chi,chip,sigma0inv
    
    return -4 * (epsilon0 *epsilon_bar(u1u2)**nu * mu * g(chip,rusum,rudiff,u1u2,rr)**(mu-1)           * dg_dxi(chip,r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr) * LJ + epsilon0*epsilon_bar(u1u2)**nu           * g(chip,rusum,rudiff,u1u2,rr)**mu * dLJ           * (sigma0inv * (r12xi/r12) + 0.5*(1/g(chi,rusum,rudiff,u1u2,rr)**(1.5))*dg_dxi(chi,r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr) ) )
    
@njit()
def du_duxi(r12xi,rusum,rudiff,u1u2,u1xi,u2xi,rr,R,LJ,dLJ):
    global chi,chip,sigma0inv
    
    return -4 * (LJ*epsilon0 * (nu*u2xi*u1u2*epsilon_bar(u1u2)**(nu+2)*g(chip,rusum,rudiff,u1u2,rr)**mu*chi**2            + mu*epsilon_bar(u1u2)**nu * g(chip,rusum,rudiff,u1u2,rr)**(mu-1)*dg_duxi(chip,r12xi,u2xi,rusum,rudiff,u1u2,rr) )            + epsilon0*epsilon_bar(u1u2)**nu * g(chip,rusum,rudiff,u1u2,rr)**mu * dLJ            * (0.5*(1/g(chi,rusum,rudiff,u1u2,rr)**(1.5))*dg_duxi(chi,r12xi,u2xi,rusum,rudiff,u1u2,rr) ) )

@njit()
def GBpot(rusum,rudiff,u1u2,rr,LJ,LJ_cut):
    return 4*epsilon0*epsilon_bar(chi,u1u2)**nu*g(chip,rusum,rudiff,u1u2,rr)**mu*(LJ - LJ_cut)

@njit()
def f_shift(rusum,rudiff,u1u2,rr,R_cut):
    global chi,chip,sigma0inv
    
    return 4*epsilon0*epsilon_bar(u1u2)**nu*(6*(1/R_cut**7) - 12*(1/R_cut**13))*g(chip,rusum,rudiff,u1u2,rr)**mu


def save_config(rx,ry,rz,ux,uy,uz):
    
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
    
    

# Initializing the empty variables for storing particle attributes

rx = []
ry = []
rz = []
ux = []
uy = []
uz = []

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
boxl = np.cbrt(npart/ndens)

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
def force_cal(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz,i):

    Fxi = 0.0        # Force Accumulating variable for particle no. i
    Fyi = 0.0
    Fzi = 0.0

    Exi = 0.0        # Torque term accumulating variable for particle no. i
    Eyi = 0.0
    Ezi = 0.0

    r1x = rx[i]      # Assigning the iterating particle to dummy variable i.e i th particle 
    r1y = ry[i]
    r1z = rz[i]

    u1x = ux[i]     
    u1y = uy[i]
    u1z = uz[i]
    
    # Calculating the field strength 

    uhz = np.sqrt(1 - uhx**2 - uhy**2)
    h_dot_u1 = uhx*u1x + uhy*u1y + uhz*u1z
    
    
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
                    

                    # Calculating and accumulating the total force and torque acting on i th particle 

                    Fxi = Fxi + fgbxi(r12x,rusum,rudiff,u1u2,u1x,u2x,r12,rr,R,LJ,dLJ) + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12x/r12) 
                    Fyi = Fyi + fgbxi(r12y,rusum,rudiff,u1u2,u1y,u2y,r12,rr,R,LJ,dLJ) + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12y/r12)
                    Fzi = Fzi + fgbxi(r12z,rusum,rudiff,u1u2,u1z,u2z,r12,rr,R,LJ,dLJ) + f_shift(rusum,rudiff,u1u2,rr,R_cut)*(r12z/r12)
                    
                    if(step<fld_cutoff):
                        Exi += du_duxi(r12x,rusum,rudiff,u1u2,u1x,u2x,rr,R,LJ,dLJ) + h**2*2*h_dot_u1*uhx 
                        Eyi += du_duxi(r12y,rusum,rudiff,u1u2,u1y,u2y,rr,R,LJ,dLJ) + h**2*2*h_dot_u1*uhy
                        Ezi += du_duxi(r12z,rusum,rudiff,u1u2,u1z,u2z,rr,R,LJ,dLJ) + h**2*2*h_dot_u1*uhz
                    else:
                        Exi += du_duxi(r12x,rusum,rudiff,u1u2,u1x,u2x,rr,R,LJ,dLJ)
                        Eyi += du_duxi(r12y,rusum,rudiff,u1u2,u1y,u2y,rr,R,LJ,dLJ)
                        Ezi += du_duxi(r12z,rusum,rudiff,u1u2,u1z,u2z,rr,R,LJ,dLJ)
                    
    
    return Fxi,Fyi,Fzi,Exi,Eyi,Ezi


@njit(parallel=True)
def force_cal_i(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz):
    
    Fx = np.zeros(npart,dtype=np.float64)
    Fy = np.zeros(npart,dtype=np.float64)
    Fz = np.zeros(npart,dtype=np.float64)
    Tx = np.zeros(npart,dtype=np.float64)
    Ty = np.zeros(npart,dtype=np.float64)
    Tz = np.zeros(npart,dtype=np.float64)
    Ex = np.zeros(npart,dtype=np.float64)
    Ey = np.zeros(npart,dtype=np.float64)
    Ez = np.zeros(npart,dtype=np.float64)
    
    for i in prange(int(npart)):
        Fx[i],Fy[i],Fz[i],Ex[i],Ey[i],Ez[i] = force_cal(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz,i)
        
        #Fx[i] = Fxi
        #Fy[i] = Fyi
        #Fz[i] = Fzi

        Tx[i] = uy[i]*Ez[i] - uz[i]*Ey[i]
        Ty[i] = uz[i]*Ex[i] - ux[i]*Ez[i]
        Tz[i] = ux[i]*Ey[i] - uy[i]*Ex[i]

        #Tx[i] = Txi
        #Ty[i] = Tyi
        #Tz[i] = Tzi
    
    return Fx,Fy,Fz,Tx,Ty,Tz



@jit()
def move(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz):
    
    # Calculating forces and torques on all particles (iterating over ith particle)
    Fx,Fy,Fz,Tx,Ty,Tz = force_cal_i(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz)
    
    '''    
    rx,ry,rz,ux,uy,uz = diplace_all(step,boxl,n,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz,Fx,Fy,Fz,Tx,Ty,Tz)
    '''
    
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
        
        
        FB_para = rannum1[i]
        FB_per1 = rannum2[i]
        FB_per2 = rannum3[i]
        TB_para = rannum4[i]
        TB_per1 = rannum5[i]
        TB_per2 = rannum6[i]
        
        
        # Calculate the velocities in the body axes

        v_ucap = (FGB_ucap + FB_para)/zetapara_t
        v_gamma = (FGB_gamma + FB_per1)/zetaper_t
        v_beta = (FGB_beta + FB_per2)/zetaper_t

        w_ucap = (TGB_ucap + TB_para)/zetapara_r
        w_gamma = (TGB_gamma + TB_per1)/zetaper_r
        w_beta = (TGB_beta + TB_per2)/zetaper_r
        
        
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
    
    return rx,ry,rz,ux,uy,uz




print("Main Loop starts now\n")

# Timer for the program starts here 

t0 = time.time()

# Variables for counting time
time_counter_move = 0
time_counter_cnfile = 0
time_counter_trajfile = 0

# Creating a file for storing the trajectories of particles

traj_fil = open(os.path.join(sys.path[0],"traj_file"),"wb")


for step in range(tstep):
    
    t1 = time.time()
    # Calling the move function which performes simulation for one time step
    rx,ry,rz,ux,uy,uz = move(step,boxl,npart,dt,mu,nu,sigma0,epsilon0,sigmaebys,sigma0inv,sigmar,epsilonebys,chi,chip,rcut,rcutsq,temp,rx,ry,rz,ux,uy,uz)
    
    if(np.isnan(ux).any()):
        break
    
    t2 = time.time()
    time_counter_move += (t2-t1)
    
    # Printing the progress of the program/current tstep at regular interval
    if(step%step_interval == 0):
        print("step:",step)
    
    # Saving the configuration of this time step to cnfile for safety measure
    if(step%cn_interval == 0):
        save_config(rx,ry,rz,ux,uy,uz)
    
    t3 = time.time()
    time_counter_cnfile += (t3-t2)
    
    # Saving the trajectory at regular interval to binary file
    if(step%traj_interval==0):
        write_traj(traj_fil,rx,ry,rz,ux,uy,uz)
    
    t4 = time.time()
    time_counter_trajfile += (t4-t3)
    
        
# Closing the trajectory file 
traj_fil.close()

tf = time.time()

# Printing out the total time required for running the program

out = open("out.txt","w")

#print("Total time : ",tf-t0)
out.write(f"Total time : {tf-t0} seconds")
#print("Time taken by move : ",time_counter_move)
out.write(f"Time taken by move : {time_counter_move} seconds")
#print("Time taken by cnfile : ",time_counter_cnfile)
out.write(f"Time taken by move : {time_counter_cnfile} seconds")
#print("Time taken by trajfile : ",time_counter_trajfile)
out.write(f"Time taken by move : {time_counter_trajfile} seconds")
out.write("\nEnd")
#print("\nEnd")
out.close()

#force_cal.parallel_diagnostics(level=1)


# In[ ]:




