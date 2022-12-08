#%%
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
#%%

path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
get_data = read_data()
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z")  
df_e = get_data.get_df_from_root(path_root,"MCEnergy_e",col_name="E MeV")  
df_p = get_data.get_df_from_root(path_root,"MCEnergy_p",col_name="P MeV")  
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")

#%%
input_data = np.load(r"C:\Users\georg\Desktop\master_arbeit\data\training_data_raster_38e8_neg.npz")
output_data = np.load(r"C:\Users\georg\Desktop\master_arbeit\data\target_data_ideal_raster_38e8.npz")
#%%
import uproot
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"

root_entry_str = "MCPosition_source"
col_name = "x"
pos = "fX"
path_event = path_root + r":Events;1" 
with uproot.open(path_event) as file:  
    if pos != None:
        data = file[root_entry_str].arrays()[root_entry_str]
#%%
data["fX"].tolist()

#%%
#%%
input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)
#%%
# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)

#%%
model = keras.models.load_model("third_NN_model1_edited")
print(trainset_index)
print(valset_index)
#%%
model.summary()
#%%
X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
#%%
y_pred = model.predict(X_test)
#%%
index_pred = np.where(y_pred > 0.5)[0]
num_predicted = len(Y_test[index_pred])
num_correct = Y_test[index_pred[0]].sum()

y_pred_real = np.zeros(len(Y_test))
y_pred_real[index_pred] = 1
index_real = np.where(Y_test == 1)[0]
index_correct = np.where(np.logical_and(y_pred_real,Y_test) == True)[0]
#%%
## percentage of compton events
tot_compton = len(Y_test[Y_test == 1])
percentage_compton = (tot_compton/len(Y_test)
)
print(tot_compton, percentage_compton)
#%%
## percentage of comptton events in predictions 
tot_pred_compton = len(y_pred_real[y_pred_real == 1]) 
percentage_pred_compton = tot_pred_compton/len(y_pred_real)

print(tot_pred_compton,percentage_pred_compton)
#%%
## efficiency and purity 
## total number of correctly predicted compton events 
num_correct_compton_pred = Y_test[index_pred].sum()
#%%
efficiency = num_correct_compton_pred/len(Y_test[Y_test==1])
Purity = num_correct_compton_pred/len(index_pred)

print("Efficiency is",  efficiency)
print("Purity is",  Purity)
#%%
Y_test

#%%
## save model
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)
#%%
df_e = df_e.loc[valset_index:].reset_index(drop=True)
df_e_correct = df_e.loc[index_correct]
df_e_pred = df_e.loc[index_pred]
df_e_real = df_e.loc[index_real]
#%%
plt.figure(figsize=(10,5))
plt.hist(df_e_real,bins=500,color="b",label="All Compton")
#plt.hist(df_e_pred, bins=np.arange(0,5,0.005),color="r",label="All Predicted")
plt.hist(df_e_correct, bins=500,color="g",label="Correctly Predicted")
plt.title("Electorn Energy Histograms of Compton Events")
plt.ylabel("Counts")
plt.xlabel("MeV")
plt.legend()
plt.xlim(-1,16)
plt.savefig("Compton_events_of_e_energy_edited_model.png",bbox_inches="tight")

#plt.ylim(-0.01,750)


#%%
df_p = df_p.loc[valset_index:].reset_index(drop=True)
df_p_correct = df_p.loc[index_correct]
df_p_pred = df_p.loc[index_pred]
df_p_real = df_p.loc[index_real]
#%%
len(X_test)
#%%
plt.figure(figsize=(10,5))
plt.hist(df_p_real,bins=500,color="b",label="All Compton")
#plt.hist(df_p_pred, bins=np.arange(0,10,0.01),color="r",label="All Predicted")
plt.hist(df_p_correct, bins=500,color="g",label="Correctly Predicted")
plt.title("Photon Energy Histograms of Compton Events")
plt.legend()
plt.ylabel("Counts")
plt.xlabel("MeV")
plt.xlim(-0.1,10)
#plt.ylim(0,350)
plt.savefig("Compton_events_of_p_energy_edited_model.png",bbox_inches="tight")
#%%
df_pos_z = df_pos_z.loc[valset_index:].reset_index(drop=True)
df_pos_correct = df_pos_z.loc[index_correct]
df_pos_pred = df_pos_z.loc[index_pred]
df_pos_real = df_pos_z.loc[index_real]


#%%
plt.figure(figsize=(10,5))
plt.hist(df_pos_real, color="b",bins=500,label="All Compton")
#plt.hist(df_pos_pred, color="r",bins=np.arange(-70,5,1),label="All Predicted")
plt.hist(df_pos_correct,color="g",bins=500,label="Correctly Predicted")
plt.legend()
plt.ylabel("Counts")
plt.xlabel("mm")
plt.title("Source Pos Z Axis Histograms of Compton Events")
plt.xlim(-70,5)
plt.savefig("Compton_events_source_pos_z_edited_model.png",bbox_inches="tight")
#%%

#plt.ylim(-0.1,100)
#%%
df_primary = df_primary.loc[valset_index:].reset_index(drop=True)
df_primary_correct = df_primary.loc[index_correct]
df_primary_pred = df_primary.loc[index_pred]
df_primary_real = df_primary.loc[index_real]

#%%
plt.figure(figsize=(12,6))
plt.hist(df_primary_real, color="b",bins=500,label="All Compton")
#plt.hist(df_primary_pred, color="r",bins=500,label="All Predicted")
plt.hist(df_primary_correct,color="g",bins=500,label="Correctly Predicted")
plt.legend()
plt.ylabel("Counts")
plt.xlabel("MeV")
plt.title("Primary Energy Histograms of Compton Events")
plt.xlim(0,20)
plt.savefig("Compton_events_primary_energy_edited_model.png",bbox_inches="tight")
#%%
df_primary[df_primary["energy"] > 2]
