"""
Plot position of source along beam axis 
"""
#%%
import numpy as np 
from read_root import read_data
import matplotlib.pyplot as plt 
#%%
path_root_0mm = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
path_root_5mm = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_BP5mm_4e9protons.root"

get_data = read_data()
df_pos_z_0mm = get_data.get_df_from_root(path_root_0mm,"MCPosition_source",pos="fZ",col_name="Pos Z")  
df_pos_z_5mm = get_data.get_df_from_root(path_root_5mm,"MCPosition_source",pos="fZ",col_name="Pos Z")  

output_data_0mm = np.load(r"C:\Users\georg\Desktop\master_thesis\ideal_targets_raster_ep.npz")
output_data_5mm = np.load(r"C:\Users\georg\Desktop\master_thesis\ideal_targets_raster_BP5mm_ep.npz")

target_data_0mm = output_data_0mm["arr_0"]
target_data_5mm = output_data_5mm["arr_0"]
# %%
array_pos_z_0mm = df_pos_z_0mm.to_numpy().flatten()
array_pos_z_5mm = df_pos_z_5mm.to_numpy().flatten()
array_pos_z_0mm_compton = array_pos_z_0mm[np.bool8(target_data_0mm)]
array_pos_z_5mm_compton = array_pos_z_5mm[np.bool8(target_data_5mm)]

#%%
SMALL_SIZE = 14
MEDIUM_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('font', family="serif")          # controls default text sizes

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsizeplt.hist(array_pos_z_0mm, np.arange(-100,6,0.5),alpha=0.5,color="red",label="All Events")
plt.hist(array_pos_z_0mm, np.arange(-100,6,0.5),alpha=0.5,color="red",label="All Events")
plt.hist(array_pos_z_0mm_compton, np.arange(-100,6,0.5),alpha=0.5,color="blue",label="Compton Events")
plt.xlabel("mm")
plt.ylabel("counts")
plt.legend()
plt.title("Source Pos along Beam Axis with Brag Peak at 0mm")
plt.ylim(0,5000)
plt.xticks([-100, -80, -60, -40, -20, -10, 0.0])
plt.savefig("0mm_BP_source_pos.PNG")

plt.show()
# %%
plt.hist(array_pos_z_5mm, np.arange(-100,10,0.5),alpha=0.5,color="red")
plt.hist(array_pos_z_5mm_compton, np.arange(-100,10,0.5),alpha=0.5,color="blue")
plt.xlabel("mm")
plt.ylabel("counts")
plt.title("Source Pos along Beam Axis with Brag Peak at 5mm")
plt.ylim(0,5000)
plt.xticks([-100, -80, -60, -40, -20, -10, 0.0, 5])

plt.savefig("5mm_BP_source_pos.PNG")
plt.show()
# %%
print(len(array_pos_z_5mm))