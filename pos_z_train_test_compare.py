"""
just some plots to compare the data from train and test sets in the posistion 
doing so to see if there is a difference in the train/test/val data sets
there was not 

"""
#%%
import pandas as pd
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
#from tensorflow import keras
#from sklearn.utils import class_weight
#%%
path = r"path"

# path of root file 
path_root = r"path_root"

# npz training and target data
training_data_name =  "training_data_name"
target_data_name =  "target_data_name"
#%%
## import data 
#input_data = np.load(r"D:\master_thesis_data\training_data_raster_38e8_bothneg_normed_compton.npz")
output_data = np.load(path + target_data_name)
target_data = output_data["arr_0"]
## import dataframe of primary energy and pos z 
get_data = read_data()
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Z")  
#%%
# split the data 
trainset_index  = int(target_data.shape[0]*0.6)
valset_index    = int(target_data.shape[0]*0.8)


target_data_new = target_data[:trainset_index]
df_pos_z_new = df_pos_z.loc[:trainset_index] 

target_data_val = target_data[valset_index:]
df_pos_z_val = df_pos_z.loc[valset_index:].reset_index(drop=True) 

target_data_test = target_data[trainset_index:valset_index]
df_pos_z_test = df_pos_z.loc[trainset_index:valset_index].reset_index(drop=True) 

idx_target = np.where(target_data == 1)[0]
idx_noise = np.where(target_data == 0)[0]

idx_target_new = np.where(target_data_new == 1)[0]
idx_noise_new = np.where(target_data_new == 0)[0]

idx_target_val = np.where(target_data_val == 1)[0]
idx_noise_val = np.where(target_data_val == 0)[0]

idx_target_test = np.where(target_data_test == 1)[0]
idx_noise_test = np.where(target_data_test == 0)[0]

plt.hist(df_pos_z_new,np.arange(-100,10,0.5),alpha=0.5)
plt.hist(df_pos_z.loc[idx_target_new],np.arange(-100,10,0.5),alpha=0.5)

plt.ylim(0,2000)
#%%
plt.hist(df_pos_z_test,np.arange(-100,10,0.5),alpha=0.5)
plt.hist(df_pos_z_test.loc[idx_target_test],np.arange(-100,10,0.5),alpha=0.5)

plt.ylim(0,800)
#%%
plt.hist(df_pos_z_val,np.arange(-100,10,0.5),alpha=0.5)
plt.hist(df_pos_z_val.loc[idx_target_val],np.arange(-100,10,0.5),alpha=0.5)

plt.ylim(0,800)
#%%
bars, bins = np.histogram(df_pos_z.loc[trainset_index:],np.arange(-100,10,0.5))
bars_compton, bins_compton = np.histogram(df_pos_z.loc[idx_target], np.arange(-100,10,0.5))
bars_noise, bins_noise = np.histogram(df_pos_z.loc[idx_noise], np.arange(-100,10,0.5))
#%%
bars_train, bins_train = np.histogram(df_pos_z_new,np.arange(-100,10,0.5))
bars_compton_train, bins_compton_train = np.histogram(df_pos_z_new.loc[idx_target_new], np.arange(-100,10,0.5))
bars_noise_train, bins_noise_train = np.histogram(df_pos_z_new.loc[idx_noise_new], np.arange(-100,10,0.5))
#%%
bars_val, bins_val = np.histogram(df_pos_z_val,np.arange(-100,10,0.5))
bars_compton_val, bins_compton_val = np.histogram(df_pos_z_val.loc[idx_target_val], np.arange(-100,10,0.5))
bars_noise_val, bins_noise_val = np.histogram(df_pos_z_val.loc[idx_noise_val], np.arange(-100,10,0.5))
#%%
bars_test, bins_test = np.histogram(df_pos_z_test,np.arange(-100,10,0.5))
bars_compton_test, bins_compton_test = np.histogram(df_pos_z_test.loc[idx_target_test], np.arange(-100,10,0.5))
bars_noise_test, bins_noise_test = np.histogram(df_pos_z_test.loc[idx_noise_test], np.arange(-100,10,0.5))

#%%
# plt.plot(bins[:-1], bars_compton/bars_noise,label="Compton/Noise All")
# plt.plot(bins[:-1], bars_compton/bars,label="Compton/All All")

plt.plot(bins[:-1], bars_compton_train/bars_noise_train,label="Compton/Noise Train")
plt.plot(bins[:-1], bars_compton_train/bars_train,label="Compton/All Train")

plt.plot(bins[:-1], bars_compton_val/bars_noise_val,label="Compton/Noise Val")
plt.plot(bins[:-1], bars_compton_val/bars_val,label="Compton/All Val")

plt.plot(bins[:-1], bars_compton_test/bars_noise_test,label="Compton/Noise Test")
plt.plot(bins[:-1], bars_compton_test/bars_test,label="Compton/All Test")
#plt.plot(bins[:-1], bars_noise/bars,label="Noise/All")
plt.vlines(-9,0,2)
plt.hlines(0.3865,-100,1)
plt.legend() 
plt.ylim(0,0.5)
#plt.xlim(-10,0.3)