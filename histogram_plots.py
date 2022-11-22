#%%
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras

path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
get_data = read_data()
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z")  
df_e = get_data.get_df_from_root(path_root,"MCEnergy_e",col_name="E MeV")  
df_p = get_data.get_df_from_root(path_root,"MCEnergy_p",col_name="P MeV")  

input_data = np.load(r"C:\Users\georg\Desktop\master_arbeit\data\training_data_raster_3838.npz")
output_data = np.load(r"C:\Users\georg\Desktop\master_arbeit\data\target_data_raster_3838.npz")

input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)
print(input_data.shape)
print(output_data.shape)
input_data = np.swapaxes(input_data,3,2)
# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)

model = keras.models.load_model("Second_NN_model1")
print(trainset_index)
print(valset_index)

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
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
df_e = df_e.loc[valset_index:].reset_index(drop=True)
df_e_correct = df_e.loc[index_correct]
df_e_pred = df_e.loc[index_pred]
df_e_real = df_e.loc[index_real]
#%%
plt.hist(df_e_real,bins=np.arange(0,16,0.1),color="g",label="All Compton")
plt.hist(df_e_pred, bins=np.arange(0,16,0.1),color="r",label="All Predicted")
plt.hist(df_e_correct, bins=np.arange(0,16,0.1),color="b",label="Correctly Guessed")
plt.xlim(-0.1,2)
#%%
df_p = df_p.loc[valset_index:].reset_index(drop=True)
df_p_correct = df_p.loc[index_correct]
df_p_pred = df_p.loc[index_pred]
df_p_real = df_p.loc[index_real]

plt.hist(df_p_real,bins=100,color="g",label="All Compton")
plt.hist(df_p_pred, bins=100,color="r",label="All Predicted")
plt.hist(df_p_correct, bins=100,color="b",label="Correctly Guessed")
plt.xlim(-0.1,8)
#%%
df_pos_z = df_pos_z.loc[valset_index:].reset_index(drop=True)
df_pos_correct = df_pos_z.loc[index_correct]
df_pos_pred = df_pos_z.loc[index_pred]
df_pos_real = df_pos_z.loc[index_real]

#%%
plt.hist(df_pos_real, color="b",bins=100,label="All Compton")
plt.hist(df_pos_pred, color="r",bins=100,label="All Predicted")
plt.hist(df_pos_correct,color="g",bins=100,label="Correctly Guessed")
plt.legend()
plt.xlim(-70,1)
#plt.ylim(0,100)
