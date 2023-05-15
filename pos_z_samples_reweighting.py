
#%%
"""
 now we will try something new 
 after we have realised that the data at the peak is beign less predicted
 we have plotted the data and saw a slump at the peak because the noise data 
 had higher statistics 
 therefore we have removed some noise from the peak tto balance the data 

"""
#%%
import pandas as pd
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.utils import class_weight
#%%
path = r"C:\Users\georg\Desktop\master_thesis"

training_data_name =  r"\training_data_raster_38e8_bothneg_normed_compton.npz"
target_data_name =  r"\target_data_Raster_38e8_ideal.npz"

 #%%
## import data 
#input_data = np.load(r"D:\master_thesis_data\training_data_raster_38e8_bothneg_normed_compton.npz")
output_data = np.load(path + target_data_name)

#%%

target_data = output_data["arr_0"]
idx_compton = np.where(target_data == 1)[0]
idx_noise = np.where(target_data == 0)[0]
#%%
## import dataframe of primary energy and pos z 
get_data = read_data()
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Z") 
#%%
bars, bins = np.histogram(df_pos_z,np.arange(-100,10,0.1))
bars_compton, bins_compton = np.histogram(df_pos_z.loc[idx_compton], np.arange(-100,10,0.1))
bars_noise, bins_noise = np.histogram(df_pos_z.loc[idx_noise], np.arange(-100,10,0.1))

#%%
plt.plot(bins[:-1], bars_compton/bars_noise,label="Compton/Noise")
plt.plot(bins[:-1], bars_compton/bars,label="Compton/All")

#plt.plot(bins[:-1], bars_noise/bars,label="Noise/All")
plt.vlines(-9,0,2)
plt.hlines(0.38,-100,1)
plt.hlines(0.34,-100,1)

plt.legend() 
plt.ylim(0,0.5)
#plt.xlim(-10,0)
#%%
##choose the bereich where we want to eleminate some data
df_z_reduced_1 = df_pos_z[(df_pos_z["Z"] >= -9 ) & (df_pos_z["Z"] < -6)]
df_z_reduced_2 = df_pos_z[(df_pos_z["Z"] >= -6 ) & (df_pos_z["Z"] < 0)]
#%%
idx_reduced_1 = df_z_reduced_1.index
idx_reduced_2 = df_z_reduced_2.index


reduced_target_1 = target_data[idx_reduced_1.to_numpy()]
reduced_target_2 = target_data[idx_reduced_2.to_numpy()]
#%%
def get_num_drop(reduced_target):
    a = len(np.where(reduced_target == 1)[0]) / 0.385
    return int(np.round(a) )

## see how many points we want to drop 
to_drop_0906 = len(np.where(reduced_target_1 == 0)[0] ) - get_num_drop(reduced_target_1)
to_drop_0600 = len(np.where(reduced_target_2 == 0)[0]) - get_num_drop(reduced_target_2)


print(to_drop_0906)


# %%
indice_to_drop_raw_1 = np.random.choice(np.where(reduced_target_1==0)[0], size=to_drop_0906,replace=False)
indice_to_drop_raw_2 = np.random.choice(np.where(reduced_target_2==0)[0], size=to_drop_0600,replace=False)

#%%
indices_to_drop_1 = df_z_reduced_1.iloc[indice_to_drop_raw_1 ].index.to_numpy()
indices_to_drop_2 = df_z_reduced_2.iloc[indice_to_drop_raw_2 ].index.to_numpy()
indices_to_drop = np.concatenate([indices_to_drop_1, indices_to_drop_2])
print(len(indices_to_drop))

#%%
relevant_indices = df_pos_z.drop(indices_to_drop).index.to_numpy()
# %%
#df_pos_z_new = df_pos_z.loc[relevant_indices].reset_index(drop=True)
df_pos_z_new = df_pos_z.loc[relevant_indices].reset_index(drop=True)
new_targets = target_data[relevant_indices]
idx_target = np.where(new_targets == 1)[0]
idx_zero = np.where(new_targets == 0)[0]
#%%
## now get new ckass weights 
## save new indices 
new_target_data = target_data[relevant_indices]
np.savez(path+r"\new_indices_newtarget.npz", relevant_indices)
## find class weight of the data 
## and assign them to target data loc
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(new_target_data),
                                                 y=new_target_data)


idx_compton_new = np.where(new_target_data == 1)[0]
idx_no_compton_new = np.where(new_target_data == 0)[0]

sample_weights = np.zeros(len(new_target_data))
sample_weights[idx_compton_new] = class_weights[1]
sample_weights[idx_no_compton_new] = class_weights[0]

np.savez(path+r"\sample_weight_ideal_class_new.npz", sample_weights)

#%%
bars, bins = np.histogram(df_pos_z_new,np.arange(-100,10,0.5))
bars_compton, bins_compton = np.histogram(df_pos_z_new.loc[idx_target], np.arange(-100,10,0.5))
bars_noise, bins_noise = np.histogram(df_pos_z_new.loc[idx_zero], np.arange(-100,10,0.5))

#%%
plt.plot(bins[:-1], bars_compton/bars_noise,label="Compton/Noise")
plt.plot(bins[:-1], bars_compton/bars,label="Compton/All")
#plt.plot(bins[:-1], bars_noise/bars,label="Noise/All")
plt.vlines(-9,0,2)
plt.hlines(0.38,-100,1)
plt.hlines(0.34,-100,1)

plt.legend() 
plt.ylim(0,0.5)
#plt.xlim(-10,0)
# %%
## plot the data at the peak 

plt.hist(df_pos_z,np.arange(-100,10,0.5),alpha=0.5)
plt.hist(df_pos_z.loc[idx_compton],np.arange(-100,10,0.5),alpha=0.5)
plt.vlines(-9,0,4000)
plt.ylim(0,4000)
#%%
plt.hist(df_pos_z_new,np.arange(-100,10,0.5),alpha=0.5)
plt.hist(df_pos_z_new.loc[idx_compton_new],np.arange(-100,10,0.5),alpha=0.5)
plt.ylim(0,4000)
plt.vlines(-9,0,4000)
#plt.xlim(-10,0)