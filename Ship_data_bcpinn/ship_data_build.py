import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits import mplot3d
import os
import glob

# Building the dataset

finalpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Final data"
xpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel x"
ypath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel y"
zpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel z"
vpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\v_vel x"

def file_names(xpath,ypath,zpath,vpath):

	os.chdir(xpath)
	xall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    xall_filenames.append(str(infile))

	os.chdir(ypath)
	yall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    yall_filenames.append(str(infile))

	os.chdir(zpath)
	zall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    zall_filenames.append(str(infile))

	os.chdir(vpath)
	vall_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    vall_filenames.append(str(infile))

	return xall_filenames,yall_filenames,zall_filenames,vall_filenames

# Extract x,y,z columns from each file in each path and merge them to single file
# extact u from the u_vel x as well and append

# These functions return the u,v, and coordinates from each file
def readu(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))
def readv(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))

def read_coordinates(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[0]))


def build_dataset():
	xall_filenames,yall_filenames,zall_filenames,vall_filenames=file_names(xpath,ypath,zpath,vpath)
	for i in range(len(xall_filenames)):
		print("Building"+str(i)+"th time_step file")
		name='/time_step'+str(i)+".csv"
		# reading each file name per timestep
		file_i=xpath+'/'+str(xall_filenames[i])
		file_j=ypath+'/'+str(yall_filenames[i])
		file_k=zpath+'/'+str(zall_filenames[i])
		file_v=vpath+'/'+str(vall_filenames[i])

		# reading coordinates and converting them to numpy arrays
		x_co=read_coordinates(file_i).to_numpy()
		y_co=read_coordinates(file_j).to_numpy()
		z_co=read_coordinates(file_k).to_numpy()
		u=readu(file_i).to_numpy()
		v=readv(file_v).to_numpy()
		t=np.repeat((0.01*i),u.shape[0])
		t=np.reshape(t,(t.shape[0],1))
		data=np.concatenate([x_co,y_co,z_co,t,u,v],1)
		data_frame=pd.DataFrame(data)
		data_frame.to_csv(finalpath + name,index=False)


#==================================================================================#
#==================================================================================#
#==================================================================================#
def build_reduced_data():
	path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Final data"
	save_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Reduced data2"

	os.chdir(path)
	all_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    all_filenames.append(str(infile))

	for i in range(len(all_filenames)):
		
		print("Building"+str(i)+"th time_step file")
		test_name='/time_step'+str(i)+".csv"
		
		x=pd.read_csv(path+'/'+str(all_filenames[i]))
		x.sort_values(["0","1","2"],axis=0,ascending=[True,True,True],inplace=True)
		x=x.to_numpy()

		x_red=x[:,:][x[:,0]>-10]
		x_red=x_red[:,:][x_red[:,0]<-5]
		x_red=x_red[:,:][x_red[:,1]>0]
		x_red=x_red[:,:][x_red[:,1]<2]
		reduced_data=pd.DataFrame(x_red)
		reduced_data.to_csv(save_path+test_name,index=False)

# build_reduced_data()
#==================================================================================#
#Creating a folder 
#==================================================================================#
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#==================================================================================#
#==================================================================================#
#==================================================================================#
# Taking raw data from reduced data folder
folder_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Reduced data"

# saving to a newly created folder in this path
save_folder_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data"

os.chdir(folder_path)

all_filenames=[]
for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
    all_filenames.append(str(infile))

# print(all_filenames)
def readu(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[4]))
def readv(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[5]))
def readt(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[3]))
def readxyz(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[0,1,2]))

def build_csv_files(all_filenames,save_folder_path):
	name=input("Enter folder name: ")
	name=os.path.join(save_folder_path,name)
	createFolder(name)

	u=name+'/u_vel.csv'
	bd=pd.concat([readu(fn) for fn in all_filenames],axis=1)
	bd.to_csv(u,index=False)

	u=name+'/v_vel.csv'
	bd=pd.concat([readv(fn) for fn in all_filenames],axis=1)
	bd.to_csv(u,index=False)

	u=name+'/time.csv'
	bd=pd.concat([readt(fn) for fn in all_filenames],axis=1)
	bd.to_csv(u,index=False)
	
	u=name+'/xy.csv'
	fname=all_filenames[0]
	bd=readxyz(fname)
	bd.to_csv(u,index=False)

# build_csv_files(all_filenames,save_folder_path)
#==================================================================================#
#==================================================================================#
#==================================================================================#

file_path=save_folder_path+r'\Testing combine'

u=pd.read_csv(file_path+r'\u_vel.csv')
u=u.to_numpy()
print(u.shape)
#==================================================================================#
#==================================================================================#
#==================================================================================#

# path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Reduced data"
# name=r"\time_step0.csv"
# test_name=r"\test.csv"
# x=pd.read_csv(path+name)
# x.sort_values(["0","1","2"],axis=0,ascending=[True,True,True],inplace=True)
# x=x.to_numpy()
# x_red=x[:,:][x[:,0]>-10]
# x_red=x_red[:,:][x_red[:,0]<-5]

# x_red=x_red[:,:][x_red[:,1]>0]
# x_red=x_red[:,:][x_red[:,1]<2]
# reduced_data=pd.DataFrame(x_red)
# reduced_data.to_csv(path+test_name,index=False)


#==================================================================================#
#==================================================================================#
#==================================================================================#
#==================================================================================#


# x=x.to_numpy()
# print(x.shape)
# So now im going to use 5096 data points. per time step.

#==================================================================================#
#==================================================================================#
#==================================================================================#

# print(np.unique(xx[:,0]).shape)
# print(xx[0,0])

# v,i,c=(np.unique(x_red[:,0],return_index=True,return_counts=True))

# # c gives the number of times an unique element appears in the array
# # i gives the indexes of unique elements in the array
# # dup.shape gives the number of duplicates

# dup=v[c>1]

# print("number of duplicates: ",dup.shape[0])
# print(i.shape[0])

#==================================================================================#
#==================================================================================#
#==================================================================================#
# os.chdir(r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Final data")

# all_filenames=[]
# for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
#     all_filenames.append(str(infile))

# duplist=[]
# for i in range(601):
# 	x=pd.read_csv(all_filenames[i])
# 	x=x.to_numpy()
# 	# Extracting the grid points in range (-10,-5)
# 	x_red=x[:,:][x[:,0]>-10]
# 	x_red=x_red[:,:][x_red[:,0]<-5]

# 	v,c=np.unique(x_red[:,0],return_counts=True)
# 	dup=v[c>1]
# 	duplist.append(dup.shape[0])


# print(duplist)
# This is the list of number of duplicates in u_vel x direction alone! 
# Should find the indexes of each of these duplicates in each file and then
# average out the velocities in these files.





