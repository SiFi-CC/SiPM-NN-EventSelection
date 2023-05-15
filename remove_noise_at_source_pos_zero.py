"""
remove all noise at point 0  
try to train with this ds to see if this is reason of bias 
then predict with whole DS 
"""
#%% 
import numpy as np 
from read_root import read_data 
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras

#%%
get_data = read_data()
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z")  
output_data = np.load(r"C:\Users\georg\Desktop\master_thesis\ideal_targets_raster_ep.npz")

# %%
target_data = output_data["arr_0"]
compton_events = np.where(target_data == 1)[0]
#%%
new_data = df_pos_z.loc[df_pos_z["Pos Z"] != 0]

target_data_reduced = target_data[new_data.index.to_numpy()]
#%%
class_weights = class_weight.compute_class_weight(class_weight="balanced",classes=[0,1],y=target_data_reduced)
sample_weights_reduced=class_weight.compute_sample_weight("balanced",target_data_reduced) 

np.savez("sample_weight_0mmBP_nonoise.npz",sample_weights_reduced)
np.savez("indices_0mmBP_nonoise.npz",new_data.index.to_numpy())

# %%
