import os
import glob
import time
import pandas as pd
import numpy as np

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


os.chdir(r"E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data\raw_data")

all_filenames_orig=[]
for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
    all_filenames_orig.append(str(infile))

#======================================================================================#
# Refining the raw data by reordering the data points
#======================================================================================#
source_path=r'E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data'

name1="raw_data_final"
name1=os.path.join(source_path,name1)
createFolder(name1)

for i in range(len(all_filenames_orig)):
    file_i=name1+'/'+str(all_filenames_orig[i])
    bod=pd.read_csv(all_filenames_orig[i])
    bod.sort_values(["Y (m)","X (m)"],axis=0,ascending=[True,True],inplace=True)
    bod.to_csv(file_i,index=False)

#======================================================================================#
# Code for creating a separate folder for storing data thats extracted 
# a new folder is created each time this script is executed.
#======================================================================================#

# path where the files will be saved 
path=r'E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data\trials'   

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#======================================================================================#
# Code for extracting the refined raw data
#======================================================================================#

os.chdir(r"E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data\raw_data_final")

all_filenames=[]
for infile in sorted(glob.glob('*.csv'),key=os.path.getmtime):
    all_filenames.append(str(infile))
#======================================================================================#
# Code for extracting data and presenting it in the form as given by paper.
#======================================================================================#

def readu(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[0]))
def readv(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[1]))
def readp(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[3]))
def readxy(fn):
    return(pd.read_csv(fn,delim_whitespace=0,usecols=[4,5]))

name=input("Enter the name: ")
name=os.path.join(path,name)
createFolder(name)

u=name+'/u_vel.csv'
bd=pd.concat([readu(fn) for fn in all_filenames],axis=1)
bd.to_csv(u,index=False)

u=name+'/v_vel.csv'
bd=pd.concat([readv(fn) for fn in all_filenames],axis=1)
bd.to_csv(u,index=False)

u=name+'/press.csv'
bd=pd.concat([readp(fn) for fn in all_filenames],axis=1)
bd.to_csv(u,index=False)

u=name+'/xy.csv'
fname=all_filenames[0]
bd=readxy(fname)
bd.to_csv(u,index=False)


combined_csv1=pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv1=combined_csv1.drop_duplicates()
a=np.unique((combined_csv1[['Time (s)']]).to_numpy())
a=np.reshape(a,(a.shape[0],1))
combined_csv1=pd.DataFrame(a,columns=['Time'])
u=name+'/time.csv'
combined_csv1.to_csv(u,index=False)
#======================================================================================#

#======================================================================================#
# Code for deleting files the raw_data and raw_data_final
#======================================================================================#
for folder,subfolders,files in os.walk(r'E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data\raw_data'):
    for file in files:
            if file.endswith('.csv'):
                path=os.path.join(folder,file)
                os.remove(path)

for folder,subfolders,files in os.walk(r'E:\Vamsi_oe20s302\Research work\Simulations and data\Unsteady trials\data\raw_data_final'):
    for file in files:
            if file.endswith('.csv'):
                path=os.path.join(folder,file)
                os.remove(path)

#======================================================================================#
#======================================================================================#