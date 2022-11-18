import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
import pickle

from mpl_toolkits import mplot3d
import os
import glob

#==================================================================================#
# Utility functions
#==================================================================================#

def file_names(xpath,ypath,zpath):

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

	# os.chdir(vpath)
	# vall_filenames=[]
	# for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
	#     vall_filenames.append(str(infile))

	return xall_filenames,yall_filenames,zall_filenames
	#,vall_filenames

# Extract x,y,z columns from each file in each path and merge them to single file
# extact u from the u_vel x as well and append

# These functions return the u,v, and coordinates from each file
def read_vel(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))

def read_coordinates(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[0]))

#==================================================================================#
# Build a full dataset
#==================================================================================#

def build_dataset():
	finalpath=	r"E:\Vamsi_oe20s302\Original Ship Simulation\Final data"
	xpath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\uvel_x"
	ypath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\vvel_y"
	zpath=    	r"E:\Vamsi_oe20s302\Original Ship Simulation\wvel_z"
	

	xall_filenames,yall_filenames,zall_filenames=file_names(xpath,ypath,zpath)
	for i in range(len(xall_filenames)):
		print("Building"+str(i)+"th time_step file")
		name='/time_step'+str(i)+".csv"
		# reading each file name per timestep
		file_i=xpath+'/'+str(xall_filenames[i])
		file_j=ypath+'/'+str(yall_filenames[i])
		file_k=zpath+'/'+str(zall_filenames[i])

		# reading coordinates and converting them to numpy arrays
		x_co=read_coordinates(file_i).to_numpy()
		y_co=read_coordinates(file_j).to_numpy()
		z_co=read_coordinates(file_k).to_numpy()
		u=read_vel(file_i).to_numpy()
		v=read_vel(file_j).to_numpy()
		w=read_vel(file_k).to_numpy()

		t=np.repeat((0.01*i),u.shape[0])
		t=np.reshape(t,(t.shape[0],1))
		
		data=np.concatenate([x_co,y_co,z_co,t,u,v,w],1)
		data_frame=pd.DataFrame(data)
		data_frame.to_csv(finalpath + name,index=False)

# build_dataset()
#==================================================================================#
# Build a reduced dataset
#==================================================================================#
def build_reduced_data():
	path=r"E:\Vamsi_oe20s302\Original Ship Simulation\Final data"
	save_path=r"E:\Vamsi_oe20s302\Original Ship Simulation\Reduced data"

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

		# This reduces the computational domain  
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
# Building four csv files : THIS HAS BEEN DEPRECATED
#==================================================================================#
# Taking raw data from reduced data folder
folder_path=r"E:\Vamsi_oe20s302\Original Ship Simulation\Reduced data"

# saving to a newly created folder in this path
save_folder_path=r"E:\Vamsi_oe20s302\Original Ship Simulation"


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
	for i in range(len(files)):
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
		
		if i==0:
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
test_path=save_folder_path+r'\Final Testing combine'

def build_timesteps_domain(test_path):
	'''
	This function returns u,v,w values from inside the domain, along with 
	x,y,z,t

	'''
	name=r'\domain.csv'
	t=pd.read_csv(test_path+name).to_numpy()
	# Excluding the t=0 time step from the domain data points
	t=t[:,:][t[:,3]>0]

	# Code to extract time steps in a time segment as a list.
	time_steps=[]

	# time_seg=1
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

# tsd=build_timesteps_domain(test_path)

#==================================================================================#
# Building time steps for collocation points
#==================================================================================#
def build_timesteps_colloc(test_path):
	'''
	This function doesn't return u,v,w values from inside the domain, along with 
	x,y,z,t

	'''
	name=r'\domain.csv'
	t=pd.read_csv(test_path+name).to_numpy()
	# Excluding the t=0 time step from the collocation points
	t=t[:,:][t[:,3]>0]

	# Code to extract time steps in a time segment as a list.
	time_steps=[]

	# time_seg=1
	c,v=np.unique(t[:,3],return_counts=True)

	def sum_sizes(sizes):
		return sum(sizes)

	sizes=[]
	ed=[]
	for i in range(len(v)):
		if i ==0:
			time_steps.append(t[0:v[i],0:4])
			sizes.append(time_steps[i].shape[0])
		else:
			start=sum_sizes(sizes)
			end=start+v[i]
			time_steps.append(t[start:end,0:4])
			sizes.append(time_steps[i].shape[0])

	return time_steps

# tsc=build_timesteps_colloc(test_path)
# print(tsc[0])
#==================================================================================#
# Buidling timesteps for boundary 
#==================================================================================#
def build_timesteps_boundary(test_path):
	name=r'\boundary.csv'
	t=pd.read_csv(test_path+name).to_numpy()
	# Excluding the t=0 time step from the boundary points
	t=t[:,:][t[:,3]>0]
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
# Buidling time segments
#==================================================================================#
def build_timesegments(file):
	
	# Returns a list of arrays containing timesteps
	# as follows [0.01,0.02,....,0.09,0.1] 

	time_segments=[]
	for i in range(0,len(file),10):
		if i == 0:
			temp=file[0:10]
			time_segments.append(np.concatenate(temp,0))
		
		else:
			temp=file[i:i+10]
			time_segments.append(np.concatenate(temp,0))
			
	return time_segments

#==================================================================================#
# Final data loading function
#==================================================================================#
def load_complete_data(test_path):

	tsd=build_timesteps_domain(test_path)
	tsb=build_timesteps_boundary(test_path)
	tsc=build_timesteps_colloc(test_path)
	
	# Sending domain data
	data_domain=build_timesegments(tsd)
	
	# Sending boundary data
	boundary=build_timesegments(tsb)

	# Sending initial timestep
	initial_timestep=pd.read_csv(test_path+r'\initial.csv')
	initial=initial_timestep.to_numpy()

	# Sending collocation points per time segment
	domain=build_timesegments(tsc)

	return domain,initial,boundary,data_domain

#==================================================================================#
# Time segment wise data loading
#==================================================================================#
def load_data(time_segment,test_path):
	d,i,b,dd=load_complete_data(test_path)
	size=dd[time_segment].shape[0]
	per=10
	didx= np.random.choice(size,int((per*size)/100),replace=False)
	data_domain=dd[time_segment][didx,:]
	return d[time_segment],i,b[time_segment],data_domain


#==================================================================================#
# Load_train_data
#==================================================================================#
def load_train_data(test_path,time_step):
	# This function should return collocation points at each time step of the given
	# time segment.

	tsd=build_timesteps_domain(test_path)

	domain=tsd[time_step]
	x_star=domain[:,0].reshape(domain[:,0].shape[0],1)
	y_star=domain[:,1].reshape(domain[:,1].shape[0],1)
	z_star=domain[:,2].reshape(domain[:,2].shape[0],1)
	t_star=domain[:,3].reshape(domain[:,3].shape[0],1)

	u_star=domain[:,4].reshape(domain[:,4].shape[0],1)
	v_star=domain[:,5].reshape(domain[:,5].shape[0],1)
	w_star=domain[:,6].reshape(domain[:,6].shape[0],1)
	
	X_star=[x_star,y_star,z_star,t_star]
	Y_star=[u_star,v_star,w_star]

	return X_star,Y_star

# x,y=load_train_data(test_path,0)

# print(x[3])
#==================================================================================#
# Load Test data
#==================================================================================#
def load_testdata(time_step):
    '''
    This function returns single timestep domain points 

    '''
    test_path=r"E:\Vamsi_oe20s302\Original Ship Simulation\Final Testing combine"
    tsd=build_timesteps_domain(test_path)
    if time_step==0:
    	ts=0
    else:
    	ts=int(time_step*100-1)
    
    domain=tsd[ts]
    x_star=domain[:,0].reshape(domain[:,0].shape[0],1)
    y_star=domain[:,1].reshape(domain[:,1].shape[0],1)
    z_star=domain[:,2].reshape(domain[:,2].shape[0],1)
    t_star=domain[:,3].reshape(domain[:,3].shape[0],1)

    u_star=domain[:,4].reshape(domain[:,4].shape[0],1)
    v_star=domain[:,5].reshape(domain[:,5].shape[0],1)
    w_star=domain[:,6].reshape(domain[:,6].shape[0],1)

    X_star=[x_star,y_star,z_star,t_star]
    Y_star=[u_star,v_star,w_star]

    return X_star,Y_star

#==================================================================================#
# Time_segments and return_indexes function for previous data predictions
#==================================================================================#

def time_segments(time_step):
    dt=0.01
    steps=[]
    for i in np.arange(round((time_step-1),3),round((time_step+dt),3),dt):
        steps.append(round(i,3))

    return steps

def return_indexes(time_step):
    idx=[]
    dt=0.01
    times=[round(i,3) for i in np.arange(0,6+dt,dt)]
    segs=time_segments(time_step)
    print(segs)
    for i in range(len(segs)):
        idx.append(times.index(segs[i]))

    return idx


#==================================================================================#
# Building larger time segments [0.1,0.2,...1],[1.1,1.2,...2]
#==================================================================================#
def build_timesegments_large(tsd):

	time_segs=[]
	temps=[]
	for i in range(10,len(tsd)+10,10):
		time_segs.append(tsd[i-1])

	for i in range(0,len(time_segs)+10,10):
		if i == 0:
			temp_list= time_segs[0:10]
			temp_file=np.concatenate(temp_list,0)
			temps.append(temp_file)
			temp_list=[]
		elif i!=60 :
			temp_list=time_segs[i:i+10]
			temps.append(np.concatenate(temp_list,0))

	return temps

#==================================================================================#
# Final data loading function for larger timesteps
#==================================================================================#
def load_complete_data_larger(test_path):
# This function returns all the points in the timesegments as a list of timesegments

	tsd=build_timesteps_domain(test_path)
	tsb=build_timesteps_boundary(test_path)
	tsc=build_timesteps_colloc(test_path)
	
	# Sending domain data
	data_domain=build_timesegments_large(tsd)
	
	# Sending boundary data
	boundary=build_timesegments_large(tsb)

	# Sending initial timestep
	initial_timestep=pd.read_csv(test_path+r'\initial.csv')
	initial=initial_timestep.to_numpy()

	# Sending collocation points per time segment
	domain=build_timesegments_large(tsc)

	return domain,initial,boundary,data_domain

#==================================================================================#
# Time segment wise data loading for larger timesegments
#==================================================================================#
def load_data_larger(time_segment,test_path):
	d,i,b,dd=load_complete_data_larger(test_path)
	size=dd[time_segment].shape[0]
	per=10
	didx= np.random.choice(size,int((per*size)/100),replace=False)
	data_domain=dd[time_segment][didx,:]
	return d[time_segment],i,b[time_segment],data_domain

# d,i,b,dd=load_data_larger(0,test_path)

# print(np.unique(d[:,3]))
#==================================================================================#
# Load_train_data_larger
#==================================================================================#
def load_train_data_larger(test_path,time_step):
	# This function should return collocation points at each time step of the given
	# time segment.

	tsd=build_timesteps_domain(test_path)

	domain=tsd[time_step]
	x_star=domain[:,0].reshape(domain[:,0].shape[0],1)
	y_star=domain[:,1].reshape(domain[:,1].shape[0],1)
	z_star=domain[:,2].reshape(domain[:,2].shape[0],1)
	t_star=domain[:,3].reshape(domain[:,3].shape[0],1)

	u_star=domain[:,4].reshape(domain[:,4].shape[0],1)
	v_star=domain[:,5].reshape(domain[:,5].shape[0],1)
	w_star=domain[:,6].reshape(domain[:,6].shape[0],1)
	
	X_star=[x_star,y_star,z_star,t_star]
	Y_star=[u_star,v_star,w_star]

	return X_star,Y_star
#==================================================================================#
#
#==================================================================================#
def time_segments_larger(time_step):
    dt=0.1
    steps=[]
    for i in np.arange(round((time_step-1),3),round((time_step+dt),3),dt):
        steps.append(round(i,3))

    return steps

def return_indexes_larger(time_step):
    idx=[]
    dt=0.1
    times=[round(i,3) for i in np.arange(0,6+dt,dt)]
    segs=time_segments_larger(time_step)
    # print(segs)
    for i in range(len(segs)):
        idx.append(times.index(segs[i]))

    return idx


def read_prev_data(fileDr):
    root_path=Path(r"E:\Vamsi_oe20s302\Vamsi\seq2seq learning\Ship_data_results\exp1")
    my_path=root_path/fileDr
    
    with open(my_path,'rb') as f:
        pdata=pickle.load(f)
    print("Previous predictions are loaded from pickle file")
    return pdata


# pdata=read_prev_data('prev_data_predictions_lts_exp1')

# print(np.unique(pdata[:,3]))

d,i,b,dd=load_data(0,test_path)

print(np.unique(d[:,3])[-1])