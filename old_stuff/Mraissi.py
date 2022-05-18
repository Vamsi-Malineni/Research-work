#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np 
import sciann as sn 
import matplotlib.pyplot as plt 
import scipy.io


# In[2]:


def PrepareData(num_data=5000, random=True):
    
    data = scipy.io.loadmat("E:\Research work\codes\cylinder_nektar_wake.mat")
    
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    # Pick random data.
    if random:
        idx = np.random.choice(N*T, num_data, replace=False)
    else:
        idx = np.arange(0, N*T)
    
    x = XX.flatten()[idx,None] # NT x 1
    y = YY.flatten()[idx,None] # NT x 1
    t = TT.flatten()[idx,None] # NT x 1
    
    u = UU.flatten()[idx,None] # NT x 1
    v = VV.flatten()[idx,None] # NT x 1
    p = PP.flatten()[idx,None] # NT x 1
 
    return (x,y,t,u,v,p)


# In[22]:


x_train, y_train, t_train, u_train, v_train, p_train = PrepareData(5000, random=True)


# In[4]:


print(x_train.shape)


# In[5]:


x = sn.Variable("x", dtype='float64')
y = sn.Variable("y", dtype='float64')
t = sn.Variable("t", dtype='float64')


# In[6]:


P = sn.Functional("P", [x, y, t], 8*[20], 'tanh')
Psi = sn.Functional("Psi", [x, y, t], 8*[20], 'tanh')


# In[7]:


lambda1 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda1")
lambda2 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda2")


# In[8]:


u = sn.diff(Psi, y)
v = -sn.diff(Psi, x)

u_t = sn.diff(u, t)
u_x = sn.diff(u, x)
u_y = sn.diff(u, y)
u_xx = sn.diff(u, x, order=2)
u_yy = sn.diff(u, y, order=2)

v_t = sn.diff(v, t)
v_x = sn.diff(v, x)
v_y = sn.diff(v, y)
v_xx = sn.diff(v, x, order=2)
v_yy = sn.diff(v, y, order=2)

p_x = sn.diff(P, x)
p_y = sn.diff(P, y)


# In[9]:


# Define constraints 
d1 = sn.Data(u)
d2 = sn.Data(v)
d3 = sn.Data(P)

c1 = sn.Tie(-p_x, u_t+lambda1*(u*u_x+v*u_y)-lambda2*(u_xx+u_yy))
c2 = sn.Tie(-p_y, v_t+lambda1*(u*v_x+v*v_y)-lambda2*(v_xx+v_yy))
c3 = sn.Data(u_x + v_y)


# In[10]:


c4 = Psi*0.0


# In[11]:


# Define the optimization model (set of inputs and constraints)
model = sn.SciModel(
    inputs=[x, y, t],
    targets=[d1, d2, d3, c1, c2, c3, c4],
    loss_func="mse",
)


# In[23]:


input_data = [x_train, y_train, t_train]


# In[24]:


data_d1 = u_train
data_d2 = v_train
data_d3 = p_train
data_c1 = 'zeros'
data_c2 = 'zeros'
data_c3 = 'zeros'
data_c4 = 'zeros'
target_data = [data_d1, data_d2, data_d3, data_c1, data_c2, data_c3, data_c4]


# In[25]:


history = model.train(
    x_true=input_data,
    y_true=target_data,
    epochs=100,
    batch_size=100,
    shuffle=True,
    learning_rate=0.001,
    reduce_lr_after=100,
    stop_loss_value=1e-8,
    verbose=1
)


# In[26]:


plt.semilogy(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')


# In[27]:


x_test,y_test,t_test,u_test,v_test,p_test=PrepareData(500,random=False)


# In[31]:


p_pred=P.eval(model,[x_test,y_test,t_test])
print(p_pred.shape)


# In[29]:


ax=x_test
ay=y_test

ax=np.unique(ax)
xmin=min(ax)
dx=ax[2]-ax[1]
xmax=max(ax)+dx

ay=np.unique(ay)
ymin=min(ay)
dy=ay[3]-ay[2]
ymax=max(ay)+dy

x,y=np.meshgrid(np.arange(xmin,xmax,dx),np.arange(ymin,ymax,dy))


# In[30]:


p_pred=np.reshape(p_pred,(5000,200))
p_plot=p_pred[:,50]
p_plot=np.reshape(p_plot,(50,100))
plt.pcolor(x,y,p_plot,cmap='rainbow')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Pressure (at a time instance)')
plt.colorbar()


# In[32]:


p_orig=np.reshape(p_test,(5000,200))
p_plot=p_orig[:,50]
p_plot=np.reshape(p_plot,(50,100))
plt.pcolor(x,y,p_plot,cmap='rainbow')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual Pressure (at a time instance)')
plt.colorbar()


# In[ ]:




