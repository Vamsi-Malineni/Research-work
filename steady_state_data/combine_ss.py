import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readu(fn):
	return(pd.read_csv(fn,delim_whitespace=0,usecols=[0]))
def readv(fn):
	return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))
def readp(fn):
	return(pd.read_csv(fn,delim_whitespace=0,usecols=[2]))
def readxy(fn):
	return(pd.read_csv(fn,delim_whitespace=0,usecols=[3,4]))

def data_proc(name_path):
	
	name=name_path+'\Steady_state_model.csv'

	# sort the values in ascending order

	bod=pd.read_csv(name)
	bod.sort_values(["Y (m)","X (m)"],axis=0,ascending=[True,True],inplace=True)
	bod.to_csv(name, index=False)


	u=readu(name)
	v=readv(name)
	p=readp(name)
	xy=readxy(name)

	nu=name_path+'/u_vel.csv'
	nv=name_path+'/v_vel.csv'
	np=name_path+'/press.csv'
	nxy=name_path+'/xy.csv'

	u.to_csv(nu,index=False)
	v.to_csv(nv,index=False)
	p.to_csv(np,index=False)
	xy.to_csv(nxy,index=False)

name_path=r"D:\Research work\Simulations and data\Unsteady trials\data"
data_proc(name_path)

uvel=pd.read_csv(name_path+"/u_vel.csv")
uvel=uvel.to_numpy()

vvel=pd.read_csv(name_path+"/v_vel.csv")
vvel=vvel.to_numpy()

press=pd.read_csv(name_path+"/press.csv")
press=press.to_numpy()

xy=pd.read_csv(name_path+"/xy.csv")
xy=xy.to_numpy()


u=np.reshape(uvel,(100,100))
plt.imshow(u)
plt.show()