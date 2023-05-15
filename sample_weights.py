#%%
"""
    find class weights for unbalanced training data
    this notebook will find: 
    - normal class weights for all data
    - sample weights weighted according to the energy distribution
    - sample weight according to position distribution 
At the end only normal class weights have been used for training the thesis 
other methods of weights did not have an effect
"""
#%%
import pandas as pd
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.utils import class_weight
#%%
path = r"data_path"

training_data_name =  r"training_data_name"
target_data_name =  r"target_data_name"

 #%%
## import data 
output_data = np.load(path + target_data_name)
target_data = output_data["arr_0"]
#%%
## import dataframe of primary energy and pos z 
get_data = read_data()
path_root = r"path_root"
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Z")  
#%%
## find class weight of the data 
## and assign them to target data loc
## and save them as sample weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(target_data),
                                                 y=target_data)


print(class_weights)
index_compton = np.where(target_data == 1)[0]
index_no_compton = np.where(target_data == 0)[0]

sample_weights = np.zeros(len(target_data))
sample_weights[index_compton] = class_weights[1]
sample_weights[index_no_compton] = class_weights[0]

np.savez(path+"\sample_weight_ideal_class_continuous.npz", sample_weights)
#%%
cls_weight_dict = {0: class_weights[0], 1: class_weights[1]}

class_weight.compute_sample_weight(cls_weight_dict, target_data)
#%%
## with or without cut 

cut = True
if cut == True:
    df_primary_cut = df_primary.loc[((df_primary["energy"] >= 1) & (df_primary["energy"] <= 10))]
    cut_index = df_primary_cut.index.to_numpy()
 #   df_primary_cut = df_primary_cut.reset_index(drop=True)
    target_data_cut = target_data[cut_index]
else: 
    pass

#%%
## first let's  
#df_primary_cut =  df_primary.loc[(df_primary["energy"] >= 1)                                & (df_primary["energy"] <= 10)].reset_index(drop=True)
target_data = output_data["arr_0"]
index_compton = np.where(target_data == 1)[0]
index_no_compton = np.where(target_data == 0)[0]
df_primary_compton = df_primary.loc[index_compton]
df_primary_no_compton = df_primary.loc[index_no_compton]

values, bins = np.histogram(df_primary, bins=np.arange(0,44.5,0.1))
values_no_compton, bins_no_compton = np.histogram(df_primary_no_compton ,bins=np.arange(0,43,0.1))
values_compton, bins_compton = np.histogram(df_primary_compton ,bins=np.arange(0,43,0.1))

#%%
df_primary_sorted = df_primary.sort_values(by="energy").reset_index()
df_primary_compton_sorted = df_primary_compton.sort_values(by="energy").reset_index()
df_primary_no_compton_sorted = df_primary_no_compton.sort_values(by="energy").reset_index()

#%%
bin_index = np.zeros(len(values) + 1)
bin_index[1:] = values 
bin_index = np.cumsum(bin_index)
bin_index = bin_index.astype(int)

#%%
## trying something:
## look at the data at bragg peak 
## use the points at the bragg peak as weights 
bar_z_compton, bin_z_compton = np.histogram(df_pos_z.loc[index_compton],bins=np.arange(-70,1,0.5))

## -4 is the point where we start consider Bragg peak
## 800 is the bar height so we dont take data after Bragg peak drops
bragg_idx = np.where((bar_z_compton>800)&(bin_z_compton[1:]>-4))[0]
bar_z_bragg, bin_z_bragg = bar_z_compton[bragg_idx], bin_z_compton[bragg_idx]

bin_pos = np.digitize(df_pos_z.loc[index_compton],bins=np.arange(-70,1,0.5))
idx_peak_bar = np.where(np.isin(bar_z_compton,bar_z_bragg)==True)
#idx_peak_bar = np.where(np.isin(bar_z_dist,bar_z_bragg_dist)==True)

## find idx of position 
list_idx_peak = []
for idx in idx_peak_bar:
      list_idx_peak.append(np.where(bin_pos == idx)[0])
#%%
#%%
## possibility 1: 
## give all samples weight 1 and focus on events at peak to predict
df_pos_compton = df_pos_z.loc[index_compton]
index_peak_in_dataset = df_pos_compton.loc[(df_pos_compton["Z"] >= -10) & (df_pos_compton["Z"] <= 0)].index.to_numpy()
#%%

new_weights = np.ones(len(df_primary))
new_weights[index_peak_in_dataset] = new_weights[index_peak_in_dataset] * 5.25
#idx = 
#new_weights_cut110 = new_weights[cut_index]
np.savez("sample_weights_allones_bpeak5.npz",new_weights)
#%%
## do the same for cut events 110 
new_weights_cut = np.ones(len(df_primary_cut))
index_peak_in_cut_db = cut_index[np.isin(cut_index, index_peak_in_dataset)]

new_weights[index_peak_in_cut_db] = new_weights[index_peak_in_cut_db] * 3
#idx = 
#new_weights_cut110 = new_weights[cut_index]
np.savez("sample_weights_allones_bpeaktrippled_110cut.npz",new_weights_cut)
#%%
## other possibility bin the data and find sample weights at each region 
## then use it for the sample weights 
df_primary_reduced_idx = df_primary.loc[index_compton]
df_primary_reduced_peak = df_primary_compton.iloc[list_idx_peak[0]]
df_index =  []
class_sample_weights = []
weights_exist = []
for i in range (0,len(bin_index)-1) :
    df_reduced = df_primary_sorted[bin_index[i]: bin_index[i+1]]
    if len(df_reduced != 0): 
        df_index.append(df_reduced["index"])
        target_data_reduced = target_data[df_reduced["index"].to_numpy()]
        try:

            weights = class_weight.compute_class_weight('balanced',
                                                classes=[0,1],
                                                y=target_data_reduced)

            weights = weights # * values[i]/N 
            class_sample_weights.append(weights)
            target_data_one_idx = np.where(target_data_reduced == 1)[0]
            target_data_zero_idx = np.where(target_data_reduced == 0)[0]
            target_one_reduced_df_idx = df_reduced.iloc[target_data_one_idx].loc[:,"index"].values
            target_zero_reduced_df_idx = df_reduced.iloc[target_data_zero_idx].loc[:,"index"].values
            sample_weights[target_zero_reduced_df_idx] = weights[0]
            sample_weights[target_one_reduced_df_idx] = weights[1]

            weights_exist.append(1)
        except:
            print("############ errorrrrr ##########")
            print(i)
            print(bins[i], bins[i+1])
            weights = 0
            # weights = 0.001
            # weights = weights/values[i]
            class_sample_weights.append(weights)
            target_zero_reduced_df_idx = df_reduced.loc[:,"index"].values

            sample_weights[target_zero_reduced_df_idx] = weights

#%%
df_primary_reduced = df_primary.loc[index_compton]
df_primary_reduced_peak = df_primary_reduced.iloc[list_idx_peak[0]]

#%%
peak_values_in44peak = df_primary_reduced_peak.loc[(df_primary_reduced_peak["energy"] > 4.2) & (df_primary_reduced_peak["energy"] < 4.5)].index.to_numpy()
peak_values = df_primary_reduced_peak.index.to_numpy() 
peak_values = peak_values[~np.isin(peak_values,peak_values_in44peak)]
#%%
## we concentrate more on peak values in energy peak of 4.4 MeV 
sample_weights[peak_values_in44peak] = sample_weights[peak_values_in44peak] * 3
sample_weights[peak_values] = sample_weights[peak_values] * 2


#%%
df_primary_reduced_idx = df_primary.loc[index_compton]
df_primary_reduced_peak = df_primary_compton.iloc[list_idx_peak[0]]

#%%
np.savez("sample_weights_raw_bpeakdub.npz",sample_weights)
#%%

## second method to find sample weights 
## manually find it:
## 1- bin the data  
## 2- set 1 for all non compton events 
## 3- then find the weights of the compton events in a bin 
## 4- finally weight it with percentage of events in bin  
df_index = []
target_data_reduced_idx = []
N = len(df_primary)
class_sample_weights = []
weights_exist = []
normed_values = values / N 
sample_weights = np.zeros(len(df_primary_sorted))
for i in range (0,len(bin_index)-1) :
    df_reduced = df_primary_sorted[bin_index[i]: bin_index[i+1]]
    if len(df_reduced != 0): 

        #print("hello")
        df_index.append(df_reduced["index"])
        target_data_reduced = target_data[df_reduced["index"].to_numpy()]
        percentage_data_bin = (1-normed_values[i])
        try:          
            target_data_one_idx = np.where(target_data_reduced == 1)[0]
            target_data_zero_idx = np.where(target_data_reduced == 0)[0]
            if len(target_data_one_idx) == 0:
                raise("Error")
            weight_one = len(target_data_zero_idx) / len(target_data_one_idx)
            print(" ################# ")
            print(f"tot num of data={values[i]},tot num of compton={len(target_data_one_idx)}, tot non compton={len(target_data_zero_idx)}")
            print ("\n weight one before reweighting", weight_one)
            weight_one = weight_one * percentage_data_bin
            weight_zero = 1
            print("\n weight zero before reweighting", weight_zero)
            weight_zero = weight_zero * percentage_data_bin
            print(f"\n after reweighting,\n weight_one={weight_one},\n weight_zero={weight_zero},\n data_percentage={percentage_data_bin}")
            
            target_data_reduced_idx.append([target_data_one_idx, target_data_zero_idx])
            target_one_reduced_df_idx = df_reduced.iloc[target_data_one_idx].loc[:,"index"].values
            target_zero_reduced_df_idx = df_reduced.iloc[target_data_zero_idx].loc[:,"index"].values
            sample_weights[target_zero_reduced_df_idx] = weight_zero
            sample_weights[target_one_reduced_df_idx] = weight_one
            weights_exist.append(1)
        except:
            print("############ errorrrrr ##########")
            print(i)
            print(bins[i], bins[i+1])
            weights = 1 * percentage_data_bin
            # weights = 0.001
            # weights = weights/values[i]
            class_sample_weights.append(weights)
            target_zero_reduced_df_idx = df_reduced.loc[:,"index"].values

            sample_weights[target_zero_reduced_df_idx] = weights
#%%
np.savez("sample_weights_manually_newnorm.npz",sample_weights)
