## Event Plot 

#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from src.generate_ids_tensor import ids_tensor
from read_root import read_data

from src.train_target_utils import Detector, DfHelper, FillTensor
from src.generate_train_target import QDCTrig
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

##from mpl_toolkits.mplot3d.art3d import Arrow3D
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
#%%
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
# %%
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)
#%%

detecor = Detector()
helper = DfHelper()
fill_tensor = FillTensor()
qdc_trig = QDCTrig()
detecor.get_xz_idx_shift()
detecor.get_xz_index()
helper.reduced_df_idx()
fill_tensor.get_tensor_idx()
qdc_trig.get_delta_t_df()

#%%
#########################
######## plots ##########
#########################
## call data get_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_dataget_data
#read_data = read_data()
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
output_data = np.load(r"C:\Users\georg\Desktop\master_thesis\target_data_Raster_38e8_ideal_true.npz")
target_data = output_data["arr_0"]
get_data = read_data()
#read_data.get_root_entry_str_list(path)
#%%
# SiPM 
df_x = get_data.get_df_from_root(path=path_root, root_entry_str='SiPMData.fSiPMPosition',pos='fX')
df_y = get_data.get_df_from_root(path=path_root, root_entry_str='SiPMData.fSiPMPosition',pos='fY')
df_z = get_data.get_df_from_root(path=path_root, root_entry_str='SiPMData.fSiPMPosition',pos='fZ')
df_id = get_data.get_df_from_root(path=path_root, root_entry_str='SiPMData.fSiPMId')
df_QDC = get_data.get_df_from_root(path=path_root, root_entry_str='SiPMData.fSiPMQDC')
df_trig = get_data.get_df_from_root(path_root,'SiPMData.fSiPMTriggerTime')
#%%

df_trig = get_data.get_df_from_root(path_root,'SiPMData.fSiPMTriggerTime')
array_trig = get_data.get_array_from_root(path_root,'SiPMData.fSiPMTriggerTime')
#%%
a = qdc_trig.get_delta_t_array(array_trig)
b = qdc_trig.get_delta_t_df(df_trig)

#%%
idx = 200571
x = df_x.loc[idx]
y = df_y.loc[idx]
z = df_z.loc[idx]
#%%#
ax = plt.axes(projection='3d')

ax.scatter(x,y,z, c="k",marker="o",alpha=0.6)
#%%
## get ids tensor
ids_tensor = ids_tensor()
ids_tensor_12 = ids_tensor.get_ids_tensor()
ids_tensor_16 = ids_tensor.get_ids_tensor((16,32,2))

# %%
compton_idx = np.where(target_data == 1)[0]
df_id= df_id.loc[compton_idx]
df_trig = df_trig.loc[compton_idx]
df_QDC = df_QDC.loc[compton_idx]

#%%
df_tensor_idx = df_id.applymap(lambda x: fill_tensor.get_tensor_idx(x,(16,32,2)),na_action="ignore")
#%%
df_tensor_idx_12 = df_id.applymap(lambda x: fill_tensor.get_tensor_idx(x,(12,32,2)),na_action="ignore")

#%%
ids_tensor_16[ids_tensor_16 > 0] = 1
ids_tensor_16[ids_tensor_16 == 0] = 0
ids_tensor_12[ids_tensor_12 > 0] = 1
ids_tensor_12[ids_tensor_12 == 0] = 0
a = np.copy(ids_tensor_16[0:2,:,:])
b = np.copy(ids_tensor_16[4:6,:,:]) 

ids_tensor_16[0:2,:,:] = b
ids_tensor_16[4:6,:,:] = a
#%%
# Define the vertices of the cube
faces = np.array([
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [7, 6, 5, 4],
    [4, 5, 1, 0],
    [1, 5, 6, 2],
    [4, 0, 3, 7]])

# Define the 6 faces of the cube
points = np.array([
    [1.5, -0.5, -1],
    [29.5, -0.5, -1],
    [29.5, 3.5,-1],
    [1.5, 3.5, -1],
    [1.5, -0.5, 0],
    [29.5, -0.5, 0],
    [29.5, 3.5, 0],
    [1.5, 3.5, 0]])

points_2 = np.array([
    [-0.5, 7.5, -1],
    [32, 7.5, -1],
    [32, 15.5,-1],
    [-0.5, 15.5, -1],
    [-0.5, 7.5, 0],
    [32, 7.5, 0],
    [32, 15.5, 0],
    [-0.5, 15.5, 0]])


#%%
#######################################################
############# plot an events with triggers in SiPMs and FIbers
##########################################################
#######################################################

fig = plt.figure(figsize=(30, 12)) 
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
idx = 200571

# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,16,1)
y = np.arange(-1,1,1)

labels_y_1 = np.int16(np.arange(143,157,4))
labels_y_1 = [143,147,151,155]
labels_y_2 = np.int16(np.linspace(255,285.1,8))
labels_y_2 = [255,"",263,"",272,"",280,285]
#labels_y_3 = [175, 195, 215, 235]
labels_y_3 = ["", "", "", ""]

labels_y = np.concatenate([labels_y_1,labels_y_3,labels_y_2])


labels_z = [""]*32
z_labels = np.linspace(-63.4,63.4,32)
for i in range(0,32,3):
    labels_z[i] = str(np.round(z_labels[i],1))


for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_16[j,i,k] == 1:
                    ax.plot(z[i],x[j],y[k], c="k",marker="o",alpha=0.6)
                else:
                    #pass
                    ax.plot(z[i],x[j],y[k], c="grey",marker="o",alpha=0.5)

#ax.set_zlim(-1,0)
ax.arrow3D(-6,-1,-1.5,
            0,11,0.1,
            mutation_scale=30,
            arrowstyle="-|>",
            linestyle='dashed')

ax.arrow3D(15,-5,-1.5,
            16,0,1,
            mutation_scale=30,
            arrowstyle="-|>",
            linestyle='dashed')


plt.title("3D Tensor Representation of the Detectors", fontsize=20)
## scattere box
ax.plot((1.5,1.5),(-0.5,3.5),(0,0),c="k",ls="--")
ax.plot((29.5,29.5),(-0.5,3.5),(-1,-1),c="k",ls="--")
ax.plot((1.5,1.5),(-0.5,3.5),(-1,-1),c="k",ls="--")
ax.plot((29.5,29.5),(-0.5,3.5),(0,0),c="k",ls="--")

ax.plot((1.5,29.5),(-0.5,-0.5),(0,0),c="k",ls="--")
ax.plot((1.5,29.5),(3.5,3.5),(0,0),c="k",ls="--")
ax.plot((1.5,29.5),(-0.5,-0.5),(-1,-1),c="k",ls="--")
ax.plot((1.5,29.5),(3.5,3.5),(-1,-1),c="k",ls="--")

ax.plot((29.5,29.5),(-0.5,-0.5),(0,-1),c="k",ls="--")
ax.plot((29.5,29.5),(3.5,3.5),(0,-1),c="k",ls="--")
ax.plot((1.5,1.5),(-0.5,-0.5),(0,-1),c="k",ls="--")
ax.plot((1.5,1.5),(3.5,3.5),(0,-1),c="k",ls="--")

## Absorber box 
ax.plot((-0.5,-0.5),(7.5,15.5),(0,0),c="k",ls="--")
ax.plot((32,32),(7.5,15.5),(-1,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(7.5,15.5),(-1,-1),c="k",ls="--")
ax.plot((32,32),(7.5,15.5),(0,0),c="k",ls="--")

ax.plot((-0.5,32),(7.5,7.5),(0,0),c="k",ls="--")
ax.plot((-0.5,32),(15.5,15.5),(0,0),c="k",ls="--")
ax.plot((-0.5,32),(7.5,7.5),(-1,-1),c="k",ls="--")
ax.plot((-0.5,32),(15.5,15.5),(-1,-1),c="k",ls="--")

ax.plot((32,32),(7.5,7.5),(0,-1),c="k",ls="--")
ax.plot((32,32),(15.5,15.5),(0,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(7.5,7.5),(0,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(15.5,15.5),(0,-1),c="k",ls="--")
# z = np.arange(0, 32, 1)
# x = np.arange(0,16,1)
# y = np.arange(-1,1,1)
ax.set_zticks(np.arange(-1,1,1), labels=["-51","51"])
ax.set_yticks(np.arange(1,17,1),labels=labels_y)
ax.set_xticks(np.arange(-1,31,1),labels=labels_z)
#z[i],x[j],y[k]
ax.plot(10.44, 2.62,-0.62,"X",markersize=15,alpha=0.8,label="e interaction" )
ax.plot(18.44,12.3, -0.7,"X",markersize=15,alpha=0.8, label ="p interaction" )
ax.plot([10.44,18.44],[2.62,12.3],[-0.62,-0.7],c="k",ls="-")
legend_elements = [
     plt.Line2D([0], [0], marker='o', color='k', alpha=0.5,label='Detector SiPM', markersize=10),
     plt.Line2D([0], [0], marker='o', color='grey', alpha=0.5, label='Empty ', markersize=10),
    plt.Line2D([0], [0], marker='o', color='brown', alpha=0.5, label='Triggered SiPM ', markersize=10),

      plt.Line2D([0], [0], marker='X', alpha=0.8,label='e int', markersize=15),
     plt.Line2D([0], [0], marker='X',  alpha=0.8, c="orange",label='p int ', markersize=15)
]
ax.legend(handles=legend_elements, fontsize=14)

ax.set_zlabel("Y",labelpad=-5)
ax.set_ylabel("X")
ax.set_xlabel("Z")

# (z,x,y)
ax.set_zlim(-1,0)
for indices in range(df_tensor_idx.loc[8].count()):
    ax.plot(df_tensor_idx.loc[idx,indices][1],df_tensor_idx.loc[idx,indices][0],df_tensor_idx.loc[idx,indices][2]-1,'o',markersize=10,color="brown",alpha=1)

#plt.savefig("event_triggered_scatt_abs_fibers.pdf",bbox_inches='tight')
plt.show()

#%%

#%%
######################################################
#####################################################
########## print 3d Tensor of shape 32,16,2,2 with no triggered SiPMs
###################################################
## x in range 143, 157, or 255,285, (0,4), (8,16)
## y in range -51,51 or equivalen -1,0
## z in range -63, 63 or equivalent 0,32

## in the data it is sorted as x,y,z 

fig = plt.figure(figsize=(30, 12)) 
plt.rcParams["font.family"] = "serif"
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,16,1)
y = np.arange(-1,1,1)

ax.tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)
#ax.text(-10,1,15,"32*16*2*2 Tensor",size=18)
ax.text(-6,2,0,"Scatterer",size=20,c="black")
ax.text(-6,12,0,"Absorber",size=20,c="black")
#ax.text(-2,-4,-1.5,"Prompt gamma rays",size=20,c="black")

for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_16[j,i,k] == 1:
                #     print("hello")
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o")
                else:
                #     #pass
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o",alpha=0.5)


ax.set_zlim(-1,0)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='grey', alpha=1,label='Detector SiPM', markersize=8),
    plt.Line2D([0], [0], marker='o', color='grey', alpha=0.4, label='Empty ', markersize=8)
#    plt.Line2D([0], [0], marker='o', color='brown', alpha=0.5, label='Triggered SiPM ', markersize=8)

]
ax.legend(handles=legend_elements, fontsize=16)

#plt.savefig("scattere_absorber_3d_nottriggered.png",bbox_inches='tight')
plt.show()
#%%
#####################################################
####################################################
##### 3d plot with detector cubic shape included and triggered SiPMs
##############################################
fig = plt.figure(figsize=(30, 12)) 
plt.rcParams["font.family"] = "serif"
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,16,1)
y = np.arange(-1,1,1)

ax.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
#ax.text(-10,1,15,"32*16*2*2 Tensor",size=18)
ax.text(-6,2,0,"Scatterer",size=20,c="black")
ax.text(-6,12,0,"Absorber",size=20,c="black")
#ax.text(-2,-4,-1.5,"Prompt gamma rays",size=20,c="black")

# Create a Poly3DCollection and add it to the plot
collection = Poly3DCollection(points[faces], alpha=0.1, facecolor='green')
collection_2 = Poly3DCollection(points_2[faces], alpha=0.1, facecolor='magenta')

ax.add_collection(collection)
ax.add_collection(collection_2)


for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_16[j,i,k] == 1:
                    ax.plot(z[i],x[j],y[k], c="k",marker="o",alpha=0.6)
                else:
                    #pass
                    ax.plot(z[i],x[j],y[k], c="grey",marker="o",alpha=0.5)


ax.set_zlim(-1,0)


legend_elements = [
    plt.Line2D([0], [0], marker='o', color='k', alpha=0.5,label='Detector SiPM', markersize=10),
    plt.Line2D([0], [0], marker='o', color='grey', alpha=0.5, label='Empty ', markersize=10)
]
ax.legend(handles=legend_elements, fontsize=20)

## scattere box
ax.plot((1.5,1.5),(-0.5,3.5),(0,0),c="k",ls="--")
ax.plot((29.5,29.5),(-0.5,3.5),(-1,-1),c="k",ls="--")
ax.plot((1.5,1.5),(-0.5,3.5),(-1,-1),c="k",ls="--")
ax.plot((29.5,29.5),(-0.5,3.5),(0,0),c="k",ls="--")

ax.plot((1.5,29.5),(-0.5,-0.5),(0,0),c="k",ls="--")
ax.plot((1.5,29.5),(3.5,3.5),(0,0),c="k",ls="--")
ax.plot((1.5,29.5),(-0.5,-0.5),(-1,-1),c="k",ls="--")
ax.plot((1.5,29.5),(3.5,3.5),(-1,-1),c="k",ls="--")

ax.plot((29.5,29.5),(-0.5,-0.5),(0,-1),c="k",ls="--")
ax.plot((29.5,29.5),(3.5,3.5),(0,-1),c="k",ls="--")
ax.plot((1.5,1.5),(-0.5,-0.5),(0,-1),c="k",ls="--")
ax.plot((1.5,1.5),(3.5,3.5),(0,-1),c="k",ls="--")

## Absorber box 
ax.plot((-0.5,-0.5),(7.5,15.5),(0,0),c="k",ls="--")
ax.plot((32,32),(7.5,15.5),(-1,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(7.5,15.5),(-1,-1),c="k",ls="--")
ax.plot((32,32),(7.5,15.5),(0,0),c="k",ls="--")

ax.plot((-0.5,32),(7.5,7.5),(0,0),c="k",ls="--")
ax.plot((-0.5,32),(15.5,15.5),(0,0),c="k",ls="--")
ax.plot((-0.5,32),(7.5,7.5),(-1,-1),c="k",ls="--")
ax.plot((-0.5,32),(15.5,15.5),(-1,-1),c="k",ls="--")

ax.plot((32,32),(7.5,7.5),(0,-1),c="k",ls="--")
ax.plot((32,32),(15.5,15.5),(0,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(7.5,7.5),(0,-1),c="k",ls="--")
ax.plot((-0.5,-0.5),(15.5,15.5),(0,-1),c="k",ls="--")

for indices in range(df_tensor_idx.loc[idx].count()):
    ax.plot(df_tensor_idx.loc[idx,indices][1],df_tensor_idx.loc[idx,indices][0],df_tensor_idx.loc[idx,indices][2]-1,'o',markersize=12,color="brown",alpha=1)
plt.title("All Triggered SiPMs of a Single Compton Event",size=20)
#plt.savefig("scattere_absorber_3d_trigger_detectors.png",bbox_inches='tight')
plt.show()



#%%

#####################################################
####################################################
##### 3d plot of triggered SiPM with numberin on triggered SiPMs in order
##############################################


fig = plt.figure(figsize=(30, 12)) 
plt.rcParams["font.family"] = "serif"
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,12,1)
y = np.arange(-1,1,1)

ax.tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)

for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_12[j,i,k] == 1:
                #     print("hello")
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o",)
                else:
                #     #pass
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o",alpha=0.5)

ax.set_zlim(-1,0)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='grey', alpha=1,label='Detector SiPM', markersize=8),
    plt.Line2D([0], [0], marker='o', color='grey', alpha=0.4, label='Empty ', markersize=8),
    plt.Line2D([0], [0], marker='o', color='brown', alpha=0.5, label='Triggered SiPM ', markersize=8)

]
ax.legend(handles=legend_elements, fontsize=16)
idx = 200571
# # Plot the line
for i,indices in enumerate(df_trig.loc[idx].sort_values().dropna().index):
    ax.plot(df_tensor_idx_12.loc[idx,indices][1],df_tensor_idx_12.loc[idx,indices][0],df_tensor_idx_12.loc[idx,indices][2]-1,'o',markersize=12,color="indianred",alpha=1-0.05*i)
    ax.text(df_tensor_idx_12.loc[idx,indices][1],df_tensor_idx_12.loc[idx,indices][0],df_tensor_idx_12.loc[idx,indices][2]-1,i,fontsize=18,c="k")
ax.text(-25,10,0,"32*12*2 Tensor",fontsize=20)
#plt.savefig("scattere_absorber_3d_trigger_12tensor_text.pdf",bbox_inches='tight')
plt.show()
#%%

fig = plt.figure(figsize=(30, 12)) 
plt.rcParams["font.family"] = "serif"
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
z = np.arange(0, 32, 1)
x = np.arange(0,16,1)
y = np.arange(-1,1,1)

ax.tick_params(left = False, right = False , labelleft = False ,
                  labelbottom = False, bottom = False)


for k in (y):
    for j in (x):
        for i in (z):
                if ids_tensor_16[j,i,k] == 1:
                #     print("hello")
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o",)
                else:
                #     #pass
                    ax.plot(z[i],x[j],y[k], c="gray",marker="o",alpha=0.5)

ax.set_zlim(-1,0)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='grey', alpha=1,label='Detector SiPM', markersize=8),
    plt.Line2D([0], [0], marker='o', color='grey', alpha=0.4, label='Empty ', markersize=8),
    plt.Line2D([0], [0], marker='o', color='brown', alpha=0.5, label='Triggered SiPM ', markersize=8)

]
#ax.legend(handles=legend_elements, fontsize=16)
idx = 200571
# # Plot the line
for i,indices in enumerate(df_trig.loc[idx].sort_values().dropna().index):
    print(indices)
    ax.plot(df_tensor_idx.loc[idx,indices][1],df_tensor_idx.loc[idx,indices][0],df_tensor_idx.loc[idx,indices][2]-1,'o',markersize=12,color="indianred",alpha=1-0.05*i)
    ax.text(df_tensor_idx.loc[idx,indices][1],df_tensor_idx.loc[idx,indices][0],df_tensor_idx.loc[idx,indices][2]-1,i,fontsize=16)
#plt.title("Triggered Event in 32*16*2 Tensor",fontsize=20)
ax.text(-20,11,0,"32*16*2 Tensor",fontsize=20)

##plt.savefig("scattere_absorber_3d_trigger_16tensor_text.pdf",bbox_inches='tight')
plt.show()
#%%
