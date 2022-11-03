import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits import mplot3d
import os
import glob

#==================================================================================#
# Utility functions
#==================================================================================#

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

#==================================================================================#
# Build a full dataset
#==================================================================================#

def build_dataset():
	finalpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Final data"
	xpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel x"
	ypath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel y"
	zpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\u_vel z"
	vpath=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\v_vel x"

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
# Build a reduced dataset
#==================================================================================#
def build_reduced_data():
	path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Final data"
	save_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Reduced data3"

	os.chdir(path)
	all_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    all_filenames.append(str(infile))

	for i in range(len(all_filenames)):
		
		print("Building"+str(i)+"th time_step file")
		test_name='/time_step'+str(i)+".csv"
		
		x=pd.read_csv(path+'/'+str(all_filenames[i]))
		x.sort_values(by=["0","1", "2"],axis=0,ascending=[True,True,True],inplace=True)
		x=x.to_numpy()

		x_red=x[:,:][x[:,0]>-10]
		x_red=x_red[:,:][x_red[:,0]<-5]
		x_red=x_red[:,:][x_red[:,1]>0]
		x_red=x_red[:,:][x_red[:,1]<2]

		# making a copy of points about x axis
		temp=np.copy(x_red)
		temp[:,1]=-1*temp[:,1]
		u_final=np.concatenate((x_red,temp),axis=0)
		# converting the numpy array to dataframe
		reduced_data=pd.DataFrame(u_final)
		# saving the dataframe as a csv file
		reduced_data.to_csv(save_path+test_name,index=False)

# Run this to build a reduced dataset
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
# Finding names of files in a folder 
#==================================================================================#
def find_name(folder_path):
	os.chdir(folder_path)

	all_filenames=[]
	for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	    all_filenames.append(str(infile))
	return all_filenames

#==================================================================================#
# Building four csv files : THIS HAS BE DEPRECATED
#==================================================================================#
# Taking raw data from reduced data folder
folder_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data\Reduced data"

# saving to a newly created folder in this path
save_folder_path=r"E:\Vamsi_oe20s302\Ship simulations\Ship_data"


all_filenames=find_name(folder_path)

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


# Run this to build the csv files
# build_csv_files(all_filenames,save_folder_path)

#==================================================================================#
# Building boundary dataset
#==================================================================================#
'''
Each timestep has different number of points on the boundary,
hence combining them as a single csv.
'''
file_path=save_folder_path+r'\Reduced Data'
test_path=save_folder_path+r'\Final Testing combine'

def build_boundary(file_path,test_path):
	files=find_name(file_path)
	for i in range(1,len(files)):
		e=pd.read_csv(file_path+'/'+str(files[i]))
		e=e.to_numpy()
		xmin=e.min(axis=0)[0]
		xmax=e.max(axis=0)[0]
		ymin=e.min(axis=0)[1]
		ymax=e.max(axis=0)[1]

		xmin_limit=xmin+0.05
		xmax_limit=xmax-0.05
		ymin_limit=ymin+0.05
		ymax_limit=ymax-0.05

		bc1=e[:,:][e[:,0]<xmin_limit]
		bc2=e[:,:][e[:,0]>xmax_limit]
		bc3=e[:,:][e[:,1]<ymin_limit]
		bc4=e[:,:][e[:,1]>ymax_limit]
		if i ==1:
			boundary=np.concatenate([bc1,bc2,bc3,bc4],0)
		else:
			boundary=np.append(boundary,np.vstack((bc1,bc2,bc3,bc4)),axis=0)

	boundary_df=pd.DataFrame(boundary)
	name=r'\boundary.csv'
	boundary_df.to_csv(test_path+name,index=False)

# Run this to build the boundary dataset csv file
# build_boundary(file_path,test_path)

#==================================================================================#
# Showing the scatter plots of xy plane
#==================================================================================#
	
def scatter_boundary():
	name=r'\time_step0.csv'

	test_name=file_path+name
	e=pd.read_csv(test_name).to_numpy()
	x=e[:,0]
	y=e[:,1]
	plt.scatter(x,y,c='blue',s=1)

	xmin=e.min(axis=0)[0]
	xmax=e.max(axis=0)[0]
	ymin=e.min(axis=0)[1]
	ymax=e.max(axis=0)[1]

	xmin_limit=xmin+0.05
	xmax_limit=xmax-0.05
	ymin_limit=ymin+0.05
	ymax_limit=ymax-0.05

	bc1=e[:,:][e[:,0]<xmin_limit]
	bc2=e[:,:][e[:,0]>xmax_limit]
	bc3=e[:,:][e[:,1]<ymin_limit]
	bc4=e[:,:][e[:,1]>ymax_limit]

	xb=bc1[:,0]
	yb=bc1[:,1]
	plt.scatter(xb,yb,c='red',s=1)
	xb=bc2[:,0]
	yb=bc2[:,1]
	plt.scatter(xb,yb,c='red',s=1)
	xb=bc3[:,0]
	yb=bc3[:,1]
	plt.scatter(xb,yb,c='red',s=1)
	xb=bc4[:,0]
	yb=bc4[:,1]
	plt.scatter(xb,yb,c='red',s=1)
	plt.show()
	z=e[:,2]
	u=e[:,4]

	# Creating figure
	fig = plt.figure(figsize = (10, 7))
	ax = plt.axes(projection ="3d")
	 
	# Creating plot
	img=ax.scatter3D(x, y,z , c=u,s=1)
	ax.set_xlabel('X', fontweight ='bold')
	ax.set_ylabel('Y', fontweight ='bold')
	ax.set_zlabel('Z', fontweight ='bold')
	fig.colorbar(img)
	# show plot
	plt.show()

# Run this to show the scatter plots
# scatter_boundary()

#==================================================================================#
# Building domain dataset
#==================================================================================#
def build_domain(file_path,test_path):	
	files=find_name(file_path)
	for i in range(1,len(files)):
		e=pd.read_csv(file_path+'/'+str(files[i]))
		e=e.to_numpy()
		xmin=e.min(axis=0)[0]
		xmax=e.max(axis=0)[0]
		ymin=e.min(axis=0)[1]
		ymax=e.max(axis=0)[1]

		xmin_limit=xmin+0.05
		xmax_limit=xmax-0.05
		ymin_limit=ymin+0.05
		ymax_limit=ymax-0.05

		bc1=e[:,:][e[:,0]>xmin_limit]
		bc2=bc1[:,:][bc1[:,0]<xmax_limit]
		bc3=bc2[:,:][bc2[:,1]>ymin_limit]
		domain_temp=bc3[:,:][bc3[:,1]<ymax_limit]
		
		if i==1:
			domain=domain_temp
		else:
			domain=np.append(domain,domain_temp,axis=0)
	domain_df=pd.DataFrame(domain)
	name=r'\domain.csv'
	domain_df.to_csv(test_path+name,index=False)
	
# Run this to build the domain dataset csv file
# build_domain(file_path,test_path)

#==================================================================================#
# Building initial dataset
#==================================================================================#

def build_initial(file_path,test_path):
	files=find_name(file_path)
	initial_file=str(files[0])
	e=pd.read_csv(file_path+'/'+initial_file)
	e=pd.DataFrame(e)
	name=r'\initial.csv'
	e.to_csv(test_path+name,index=False)

# build_initial(file_path,test_path)

#==================================================================================#
# Buidling timesteps for domain
#==================================================================================#
def build_timesteps_domain(test_path):
	name=r'\domain.csv'
	t=pd.read_csv(test_path+name).to_numpy()
	# print(t)

	# Code to extract time steps in a time segment as a list.
	time_steps=[]

	# time_seg=1
	c,v=np.unique(t[:,3],return_counts=True)

	# print(v.shape)
	# print(c)
	# print(v)
	# dt=0.01
	def sum_sizes(sizes):
		return sum(sizes)

	sizes=[]
	ed=[]
	for i in range(len(v)):
		if i ==0:
			time_steps.append(t[0:v[i],:])
			sizes.append(time_steps[i].shape[0])
		else:
			start=sum_sizes(sizes)
			end=start+v[i]
			time_steps.append(t[start:end,:])
			sizes.append(time_steps[i].shape[0])

	return time_steps

# tsd=build_timesteps_domain(test_path)

#==================================================================================#
# Buidling timesteps for boundary 
#==================================================================================#
def build_timesteps_boundary(test_path):
	name=r'\boundary.csv'
	t=pd.read_csv(test_path+name).to_numpy()

	# Code to extract time steps in a time segment as a list.
	time_steps=[]

	c,v=np.unique(t[:,3],return_counts=True)

	def sum_sizes(sizes):
		return sum(sizes)

	sizes=[]
	ed=[]
	for i in range(len(v)):
		if i ==0:
			time_steps.append(t[0:v[i],:])
			sizes.append(time_steps[i].shape[0])
		else:
			start=sum_sizes(sizes)
			end=start+v[i]
			time_steps.append(t[start:end,:])
			sizes.append(time_steps[i].shape[0])

	return time_steps

# tsb= build_timesteps_boundary(test_path)

#==================================================================================#
#==================================================================================#
#==================================================================================#



# name=r'\time_step0.csv'

# e=pd.read_csv(file_path+name).to_numpy()

# xmin=e.min(axis=0)[0]
# xmax=e.max(axis=0)[0]
# ymin=e.min(axis=0)[1]
# ymax=e.max(axis=0)[1]

# xmin_limit=xmin+0.05
# xmax_limit=xmax-0.05
# ymin_limit=ymin+0.05
# ymax_limit=ymax-0.05




# print(do.shape)

# xb=bc2[:,0]
# yb=bc2[:,1]
# plt.scatter(xb,yb,c='blue',s=1)

# print(bc2.shape)
# xb=bc2[:,0]
# yb=bc2[:,1]
# plt.scatter(xb,yb,c='red',s=1)
# plt.show()
#==================================================================================#
#==================================================================================#
#==================================================================================#

# u=pd.read_csv(test_path+r'\u_vel.csv')
# u=u.to_numpy()
# print(u.shape)


# u=pd.read_csv(file_path+r'\v_vel.csv')
# u=u.to_numpy()
# print(u.shape)


#==================================================================================#
# NOTES for tomorrow
#==================================================================================#

'''
The amount of data obtained at each time step is not the same nor is it uniform.
How to deal with this !??
'''
#==================================================================================#
# NOTES
#==================================================================================#
'''
We dont have the need to have same number of data points in each time step
lets stick with minimum number of points in a given time segment,

Its better to create separate files for boundary conditions, initial conditions
and domain.
As in each time step the number of grid points are varying and you dont want 
to end up with weird distribution.

'''




