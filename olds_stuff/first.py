import math
import pandas as pd
import numpy as np
import copy
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dom_data(N_train,T,min_max):
	xmax,xmin,ymax,ymin,tmax,tmin=min_max

	num_pts=int(N_train/T) # Gives the number of points per time step to be included in training dataset
	data_idx=np.zeros((N_train,3)) # stores the 

	for i in range(len(data_idx)):
		data_idx[i,0:2]=[float(np.random.uniform(xmin,xmax,1)),float(np.random.uniform(ymin,ymax,1))]

	j=0
	for i in range(0,len(data_idx),num_pts):
		data_idx[i:i+num_pts,2:3]=j
		i=i+1
		j=j+0.1

	return data_idx

def load_data(N_train):

	path=r"C:\Users\Vamsi\Downloads\all_pressures"

	uvel=pd.read_csv(path+r"/u_vel.csv")
	uvel=uvel.to_numpy()

	vvel=pd.read_csv(path+r"/v_vel.csv")
	vvel=vvel.to_numpy()

	press=pd.read_csv(path+r"/static_press.csv")
	press=press.to_numpy()

	xy=pd.read_csv(path+r"/xy.csv")
	xy=xy.to_numpy()

	t=pd.read_csv(path+r"/time.csv")
	t=t.to_numpy()

	N=xy.shape[0]
	T=t.shape[0]
	idx1 = np.arange(0, N*T)

	XX = np.tile(xy[:,0:1], (1,T)) # N x T
	YY = np.tile(xy[:,1:2], (1,T)) # N x T
	TT = np.tile(t, (1,N)).T # N x T

	x = XX.flatten()[:,None] # NT x 1
	y = YY.flatten()[:,None] # NT x 1
	t = TT.flatten()[:,None] # NT x 1

	u = uvel.flatten()[:,None] # NT x 1
	v = vvel.flatten()[:,None] # NT x 1
	p = press.flatten()[:,None] # NT x 1


	ax=np.unique(x)
	xmin=min(ax)
	xmax=max(ax)
	ay=np.unique(y)
	ymin=min(ay)
	ymax=max(ay)
	at=np.unique(t)
	tmin=min(at)
	tmax=max(at)



	data1=np.concatenate([x ,y ,t , u , v ,p ],1)

	#======================== domain =================================#
	data2=data1[:,:][data1[:,2]<=20] # Taking data upto time step 20.0
	 
	data3=data2[:,:][data2[:,0]>=xmin] # Taking data greater than xmin

	data4=data3[:,:][data3[:,0]<=xmax] # Taking data less than xmax

	data5=data4[:,:][data4[:,1]>=ymin] # Taking data greater than ymin

	data_domain=data5[:,:][data5[:,1]<=ymax] # Taking data less than ymax

	#======================= initial ==================================#
	data_t0=data_domain[:,:][data_domain[:,2]==0]


	
	batch_size=32

	#idx = np.random.choice(data_domain.shape[0], N_train, replace=False)
	#data_idx=data_domain[idx][:,0:3]
	min_max=[xmax,xmin,ymax,ymin,tmax,tmin]
	data_idx=dom_data(N_train,T,min_max)
	#===================== boundary ======================================#
	bc1_data=data_domain[:,:][data_domain[:,1]==ymax]
	bc2_data=data_domain[:,:][data_domain[:,0]==xmin]
	bc3_data=data_domain[:,:][data_domain[:,0]==xmax]
	bc4_data=data_domain[:,:][data_domain[:,1]==ymin]


	data_sup_b_train = np.concatenate([bc1_data,bc2_data,bc3_data,bc4_data], 0)

	return data_idx,data_t0,data_sup_b_train

N_train=140000
data_idx,data_t0,data_sup_b_train=load_data(N_train)

#===================================================================================================#
'''
domain_batches(batch_size)[i][0][j] will access 'jth' point of 'ith' batch
ic_batches(batch_size)[i][0][j]     will access jth point of ith batch
bc_batches(batch_size)[i][0][j]     will access jth point of ith batch
'''
def summon_batch_domain(batch_size,start):
	''' 
	This function returns a list containing randomnly picked points from the domain
	'''
	data=copy.deepcopy(data_idx)
	points=[]

	for i in range(start,len(data),batch_size): # Something is wrong here

		points.append(data[i:i+batch_size,0:3])
		break # breaks after the first iteration

	return points
def domain_batches(batch_size):
	'''
	This function is used to call the domain data points in batches
	This function returns an array of arrays of shape(438,1) each of the 438 arrays
	will have batch size number of points
	'''
	batches=[]
	num_batches=math.ceil(N_train/batch_size)

	for i in range(0,len(data_idx),batch_size):
		batches.append(summon_batch_domain(batch_size,i))

	return np.asarray(batches,dtype=object) 
def summon_batch_ic(batch_size,start):

	data=copy.deepcopy(data_t0)
	points=[]

	for i in range(start,len(data),batch_size): # Something is wrong here

		points.append(data[i:i+batch_size])
		break # breaks after the first iteration

	return points
def ic_batches(batch_size):
	'''
	This function is used to call the domain data points in batches
	'''
	batches=[]
	num_batches=math.ceil(data_t0.shape[0]/batch_size)

	for i in range(0,len(data_t0),batch_size):
		batches.append(summon_batch_ic(batch_size,i))

	
	return np.asarray(batches,dtype=object) 

def summon_batch_bc(batch_size,bc_con,start):
	if bc_con==1:
		data=copy.deepcopy(data_sup_b_train[0:20000,:])
	elif bc_con==2:
		data=copy.deepcopy(data_sup_b_train[20000:40000,:])
	elif bc_con==3:
		data=copy.deepcopy(data_sup_b_train[40000:60000,:])
	elif bc_con==4:
		data=copy.deepcopy(data_sup_b_train[60000:80000,:])
		
	points=[]

	for i in range(start,len(data),batch_size): # Something is wrong here

		points.append(data[i:i+batch_size,:])
		break # breaks after the first iteration

	return points	
def bc_batches(batch_size,bc_con):

	batches=[]
	num_batches=math.ceil(data_sup_b_train[0:20000,:].shape[0]/batch_size)

	for i in range(0,int(len(data_sup_b_train)/4),batch_size):
		batches.append(summon_batch_bc(batch_size,bc_con,i))

	
	return np.asarray(batches,dtype=object) 



def scatter(data_idx,data_sup_b_train):
	xd=data_idx[13930:14000,0:1]
	yd=data_idx[13930:14000,1:2]

	dom=plt.scatter(xd,yd,c='blue',s=10)


	xb1=data_sup_b_train[0:20000,0:1]
	xb2=data_sup_b_train[20000:40000,0:1]
	xb3=data_sup_b_train[40000:60000,0:1]
	xb4=data_sup_b_train[60000:80000,0:1]

	yb1=data_sup_b_train[0:20000,1:2]
	yb2=data_sup_b_train[20000:40000,1:2]
	yb3=data_sup_b_train[40000:60000,1:2]
	yb4=data_sup_b_train[60000:80000,1:2]

	bc1=plt.scatter(xb1,yb1,c='red',s=10)
	bc2=plt.scatter(xb2,yb2,c='green',s=10)
	bc3=plt.scatter(xb3,yb3,c='black',s=10)
	bc4=plt.scatter(xb4,yb4,c='magenta',s=10)

	plt.legend((dom,bc1,bc2,bc3,bc4),
		('domain','bc1','bc2','bc3','bc4'),loc='lower left',
		ncol=2,fontsize=8)

	plt.show()


scatter(data_idx,data_sup_b_train)
