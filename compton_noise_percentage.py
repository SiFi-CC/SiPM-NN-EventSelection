"""
   Plot histograms for energy and z direcrtion 
   check the follwiong:
   1- check where the energy at the bragg peak come from 
   2- plot energy and position for noise and compton 
   3- the percentage of noise/compton as a function of energy
   4- compare this percentage with the percentage of predicted compton events 
"""
#%%
from read_root import read_data
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.utils import class_weight
import pandas as pd
#%%
##  1- check where the energy at the bragg peak come from 

path_root = r"path_root"
path_training = r"path_training"
path_target = r"path_target"

get_data = read_data()
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z")  
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")

input_data = np.load(path_training)
output_data = np.load(path_target)

Y = output_data["arr_0"]
#%%

Y_compton_idx = np.where(Y == 1)[0]
## get the histograms of source position 
bar_z_compton, bin_z_compton = np.histogram(df_pos_z.loc[Y_compton_idx],bins=np.arange(-70,1,0.5))
#%%
#plt.bar(bin_z_compton[1:],bar_z_compton,color="b",alpha=0.5,label="ideal compton events")
#plt.plot(np.linspace(-70,0,1000),np.ones(1000)*1000,'--')
plt.hist(df_pos_z.loc[Y_compton_idx],bins=np.arange(-70,1,0.5),color="b",alpha=0.5,label="ideal compton events")
plt.plot(np.ones(1000)*-4,np.linspace(0,1000,1000),'--')
plt.legend()
#plt.xlim(-10,)
plt.title("Z direction Ideal Compton")
plt.xlabel("Z direction")
plt.ylabel("Count")
#plt.savefig("Ideal_Compton_z.PNG")
plt.show()

#%%
# -5 is on x axis which value with BP at 0 
# 800 is how high the bars, can be set by looking at prev histogram 
bragg_idx = np.where((bar_z_compton>800)&(bin_z_compton[1:]>-5))[0]

bar_z_bragg, bin_z_bragg = bar_z_compton[bragg_idx], bin_z_compton[bragg_idx]

## bin_pos find index of Z of compton evens in bins 
## find which bins match criteria 
bin_pos = np.digitize(df_pos_z.loc[Y_compton_idx],bins=np.arange(-70,1,0.5))
idx_peak_bar = np.where(np.isin(bar_z_compton,bar_z_bragg)==True)

## find idx of position 
list_idx_peak = []
for idx in idx_peak_bar:
   list_idx_peak.append(np.where(bin_pos == idx)[0])
#%%
## get sample weights at the peak 
sample_weights = np.ones(len(df_pos_z))
sample_weights[list_idx_peak[0]] = 3
## try to give extra sample weight for peak events 
## concentrate on these events during training due to missing BP events 
## Actually it did not make a difference 
np.savez("sample_weights_allones_peak3.npz",sample_weights)


# %%
## plot primary energy at peak 
df_primary_reduced = df_primary.loc[Y_compton_idx].reset_index(drop=True)
df_primary_reduced_peak = df_primary_reduced.loc[list_idx_peak[0]]

plt.hist(df_primary.loc[Y_compton_idx],np.arange(0,18,0.1), color="k",alpha=0.5)
plt.hist(df_primary_reduced_peak,np.arange(0,18,0.1), color="b",alpha=0.5)
plt.yscale("log")

#%%
## 2- plot energy and position for noise and compton 
bar_z, bin_z = np.histogram(df_pos_z,bins=np.arange(-70,1,0.5))

#plt.bar(bin_z[1:],bar_z,color="red",alpha=0.5, label="all events")
#plt.bar(bin_z_compton[1:],bar_z_compton,color="blue",alpha=0.5,label="ideal compton events")
#plt.bar(bin_z_dist[1:],bar_z_dist,color="k",alpha=0.5,label="dist compton events")
#plt.plot(np.linspace(-70,0,1000),np.ones(1000)*800,'--')
plt.hist(df_pos_z.loc[Y_compton_idx],bins=np.arange(-70,1,0.5),color="b",alpha=0.5,label="ideal compton events")
plt.plot(np.ones(1000)*-5,np.linspace(0,1000,1000),'--')
#plt.ylim(-1,4000)
plt.legend()
plt.title("Z direction Compton vs All")
plt.xlabel("Z direction")
plt.ylabel("Count")
#plt.savefig("Z_direction_compton_peak.PNG")
#%%

bar_primary, bin_primary = np.histogram(df_primary, np.arange(0,18,0.1))
bar_primary_compton, bin_primary_compton = np.histogram(df_primary.loc[Y_compton_idx], np.arange(0,18,0.1))

#plt.hist(df_primary,bins=np.arange(0,18,0.1))
#plt.hist(df_primary.loc[Y_compton_idx],bins=np.arange(0,18,0.1))
plt.yscale("log")
plt.bar(bin_primary[:-1],bar_primary,width=0.1,color="red",alpha=0.5, label="all events")
plt.bar(bin_primary_compton[:-1],bar_primary_compton,width=0.1,color="blue",alpha=0.5,label="ideal compton")
plt.hist(df_primary_reduced_peak,np.arange(0,18,0.1),color='darkgreen',alpha=0.5,label="ideal compton at peak")
#plt.plot(np.ones(100)*9.3,np.linspace(0,1000,100),'--',lw=1)
plt.legend()
#plt.xlim(0,0.05)
plt.title("Primary Energy Ideal Compton vs Ideal at Bragg Peak Vs All")
plt.xlabel("MeV")
plt.ylabel("Count")
#plt.savefig("Primary_Energy_PeakvsIdeal.PNG")

#%%

##  3- the percentage of noise/compton as a function of energy

percentage_primary = bar_primary_compton/bar_primary

# %%
percentage_all = len(Y_compton_idx)/len(Y)
plt.plot(np.arange(0,18,0.1)[0:-1],percentage_primary)
plt.plot(np.linspace(0,18,100),np.ones(100)*percentage_all,'--',label="Total Ideal Compton Rate")
#plt.plot(np.ones(100)*9.3,np.linspace(0,0.3,100),'--',lw=1,label="Total Ideal Compton Rate")
plt.title("Percentage of Compton Events as a Function of Energy")
plt.xlabel("MeV")
plt.ylabel("%")
plt.legend()
#plt.savefig("Percentage_of_Compton.PNG")

#%%
bars_peak , bins_peak= np.histogram(df_primary_reduced_peak,np.arange(0,18,0.1))

#%%
percentage_peak_compton = bars_peak/bar_primary_compton 
percentage_peak_all = bars_peak/bar_primary
# %%
plt.plot(np.arange(0,18,0.1)[0:-1],percentage_primary,'--',label="percentage compton/all")
plt.plot(np.arange(0,18,0.1)[0:-1],percentage_peak_compton,'--',label="percentage Peak/Comtpon")

#plt.plot(bins_peak[0:-1],percentage_peak_all,'--',label="percentage peak/all")
#plt.plot(bins_peak[0:-1],percentage_peak_compton,'--',label="percentage peak/compton")
plt.legend()
plt.title("Percentage of Compton Events as a Function of Energy")
plt.xlabel("MeV")
plt.ylabel("%")
plt.legend()
#plt.savefig("Percentage_compton_peak.PNG")

# %%
##   4- compare this percentage with the percentage of predicted compton events 
## load model 
model_path = r"model_path"
model_name =    r"\NN_deep_ideal_comp_bothneg_train_weight"
model = keras.models.load_model(model_path+model_name)
#%%
model.summary()

# %%
input_data = np.load(path_training)
output_data = np.load(path_target)
#%%
input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)
#%%
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
#%%
y_pred = model.predict(X_test)
Y_pred = np.zeros(len(Y_test))
index_pred = np.where(y_pred > 0.6)[0]
Y_pred[index_pred] = 1
#%%
#index_pred = np.where(y_pred > 0.5)[0]
num_predicted = len(Y_test[index_pred])
num_correct = Y_test[index_pred[0]].sum()

index_real = np.where(Y_test == 1)[0]
index_correct = np.where(np.logical_and(Y_pred,Y_test) == True)[0]
# %%
df_primary_test = df_primary.loc[valset_index:].reset_index(drop=True)

Y_pred_idx = np.where(Y_pred == 1)[0]
bar_primary_test, bin_primary_test = np.histogram(df_primary_test, np.arange(0,18,0.1))
bar_primary_compton_test_real, bin_primary_compton_test_real = np.histogram(df_primary_test.loc[(np.where(Y_test == 1)[0])], np.arange(0,18,0.1))

bar_primary_compton_test, bin_primary_compton_test = np.histogram(df_primary_test.loc[Y_pred_idx], np.arange(0,18,0.1))
bar_primary_compton_model_correct, bin_primary_model_correct = np.histogram(df_primary_test.loc[index_correct], np.arange(0,18,0.1))

# %%
percentage_model = bar_primary_compton_test/bar_primary_test
percentage_test = bar_primary_compton_test_real/bar_primary_test
percentage_correct = bar_primary_compton_model_correct/bar_primary_test
# %%
#plt.plot(np.arange(0,18,0.1)[:-1],percentage_model,'--',color="red",label="Percentage predicted compton/all")
plt.plot(np.arange(0,18,0.1)[:-1],percentage_test,'--',label="percentage compton/all")
plt.plot(np.arange(0,18,0.1)[:-1],percentage_correct,'--',label="percentage correct/all")
plt.legend()
#plt.plot(np.arange(0,18,0.1)[0:-1],percentage_primary,'--',label="percentage compton/all")

# %%
plt.bar(bin_primary_compton_test_real[:-1], bar_primary_compton_test_real,width=0.1,alpha=0.5,color="blue")
plt.bar(bin_primary_compton_test[:-1], bar_primary_compton_test,width=0.1,alpha=0.5,color="red")
