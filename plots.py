
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from read_root import read_data
from generate_ids_tensor import ids_tensor
import matplotlib as mpl


#########################
######## plots ##########
#########################

## call data 
read_data = read_data()
path = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPL_2e8protons.root"
read_data = read_data()
read_data.get_root_entry_str_list(path)
df_x = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fX')
df_y = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fY')
df_z = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fZ')
df_id = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMId')
df_QDC = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMQDC')

## get ids tensor
ids_tensor = ids_tensor()
ids_tensor_12 = ids_tensor.get_ids_tensor()
ids_tensor_16 = ids_tensor.get_ids_tensor((16,32,2))

## 
def get_values(df, ux_uy):
    """
    get values of df in a specific  (or half)
    input: DataFrame and array of index of quarter
    output: DataFrame with values only in one quarter
    """
    df_new = df.copy()
    df_new["ux_uy"] = ux_uy
    df_new = df_new.apply(lambda x: x.loc[x.ux_uy],axis=1)
    return df_new

########################
## get plots of the positions
df_x_null = df_x.fillna(0)
df_y_null = df_y.fillna(0)
df_z_null = df_z.fillna(0)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
# plot hit detector from all events 
idx = 1
# To create a scatter graph
ax.scatter(df_z_null.loc[:,idx], df_y_null.loc[:,idx], df_x_null.loc[:,idx], c=df_QDC.loc[:,idx])
# trun off/on axis
plt.axis('off')
# show the graph
plt.show()
#%%
## plot event
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
idx = 95
ax.scatter(df_x_null.loc[idx,:], df_y_null.loc[idx,:], df_z_null.loc[idx,:], c=df_QDC.loc[idx,:])
plt.axis('on')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.show()

#%%
## plot everything 
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter(df_x, df_y, df_z)
# for i, txt in enumerate(df_z.index):
#     ax.annotate(txt, (df_x.loc[i], df_y.loc[i],df_z.loc[i]))
plt.axis('on')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
#%%
## more plots 
## plot grid of SiPM with color as the count of occurences

## get norm of plots
id_counts = df_id.apply(pd.value_counts,axis=1).sum(axis=0)
id_counts_norm = id_counts / np.sum(id_counts)

id_counts_norm_dict = id_counts_norm.to_dict()
id_counts_dict = id_counts.to_dict()

df_normed = df_id.replace(id_counts_norm_dict)
df_counts = df_id.replace(id_counts_dict)

fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
ax.set_title("Triggered SiPMs, coloring trigger counts")

cmap = plt.cm.viridis
ax.scatter(df_x_null,df_y_null,df_z_null, c=df_counts,cmap=cmap)
ax.set_xlim(120,300)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text(200, 30, 90, "Dim 28*4*2       Dim: 32*8*2 ")
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)
#plt.savefig("3dtesndor.pdf")
plt.show()
#%%
fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
ax.set_title("Triggered SiPMs of 512 events, coloring trigger counts", size=14)
cmap = plt.cm.viridis
ax.scatter(df_x_null,df_y_null,df_z_null, c=df_counts,cmap=cmap)
ax.set_xlim(120,300)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text(150, 30, 90, "Scatterer dim: 28*4*2                   Absorber dim: 32*8*2 ", size=12)
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap),fraction=0.01,pad=0.02)
plt.savefig("3dtesndor.png")
plt.show()

#%%
fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
ax.set_title("Triggered SiPMs of 512 events, coloring trigger counts", size=14)

cmap = plt.cm.viridis
ax.scatter(ids_tensor)
ax.set_xlim(120,300)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text(150, 30, 90, "Scatterer dim: 28*4*2                   Absorber dim: 32*8*2 ", size=12)
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap),fraction=0.01,pad=0.02)
plt.savefig("3dtesndor.png")
plt.show()
#%%

ids_tensor_12[ids_tensor_12 > 0] = 1
ids_tensor_12[ids_tensor_12 == 0] = 0
#%%
fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,16,1)
y = np.arange(-1,1,1)
for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_16[j,i,k] == 1:
                    ax.scatter(x[j],y[k],z[i], c="b")
                else:
                    ax.scatter(x[j],y[k],z[i], c="k")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
ax.scatter(x[j],y[k],z[i], c="b",label="(QDC,Trig)")
ax.scatter(x[j],y[k],z[i],c="k",label="Zeros")
ax.legend(prop={'size':12})
ax.tick_params(left = False, right = False , labelleft = False ,
                 labelbottom = False, bottom = False)
ax.text(-10,1,15,"32*16*2*2 Tensor",size=16)
plt.savefig("32_16_tesndor.png")
plt.show()
#%%
ids_tensor_12[ids_tensor_12 > 0] = 1
ids_tensor_12[ids_tensor_12 == 0] = 0
#%%

fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0,32,1)
x = np.arange(0,12,1)
y = np.arange(-1,1,1)
for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_12[j,i,k] == 1:
                    ax.scatter(x[j],y[k],z[i], c="b")
                else:
                    ax.scatter(x[j],y[k],z[i], c="k")
ax.tick_params(left = False, right = False , labelleft = False ,
                 labelbottom = False, bottom = False)
ax.text(-9,1,13,"32*12*2*2 Tensor",size=16)
plt.savefig("32_12_tesndor.png")
plt.show()