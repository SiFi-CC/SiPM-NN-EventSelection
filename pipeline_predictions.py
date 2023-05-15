"""
this notebook loads the data from two ragg peaks
loads the regression and classification models 
predicts classification 
from classified true events predicts regression 
save the output to have as an input for reco 
"""
import numpy as np 
import keras

#%%
input_data_BP5mm = np.load(r"C:\Users\georg\Desktop\master_thesis\training_data_bothneg_norm26k_1632_2ch_midempty_5mmBP.npz")
output_data_BP5mm = np.load(r"C:\Users\georg\Desktop\master_thesis\ideal_targets_raster_BP5mm_ep.npz")
#%%
input_data_BP0mm = np.load(r"C:\Users\georg\Desktop\master_thesis\training_data_bothneg_norm26k_1632_2ch_midempty_0mmBP.npz")
output_data_BP0mm = np.load(r"C:\Users\georg\Desktop\master_thesis\ideal_targets_raster_ep.npz")
#%%
# slice data
input_data_BP0mm = input_data_BP0mm["arr_0"]
output_data_BP0mm = output_data_BP0mm["arr_0"]
#%%
input_data_BP5mm = input_data_BP5mm["arr_0"]
output_data_BP5mm = output_data_BP5mm["arr_0"]
#%%
trainset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.6)
trainset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.6)

valset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.8)
valset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.8)

#%%
X_test_BP0mm = input_data_BP0mm[trainset_index_BP0mm:]
X_test_BP5mm = input_data_BP5mm[trainset_index_BP5mm:]

Y_test_BP0mm = output_data_BP0mm[trainset_index_BP0mm:]
Y_test_BP5mm = output_data_BP5mm[trainset_index_BP5mm:]
# Y_test_BP5mm  = Y_test_BP5mm[0:len(X_test_BP0mm)]

#%%
#%%
## load model
model_path = r"C:\Users\georg\Desktop\master_thesis\Models\thesis_models"
model_name = r"\increasing_3003_3216_norm26k_declr_twolayernorm"
best_model =    r"\best_model_fourthmodel_shuffeldBP_step001_layernorm.h5"
model = keras.models.load_model(model_path+model_name+best_model)   
#%%
X_test = np.concatenate([X_test_BP0mm,X_test_BP5mm])
y_pred_BP0mm = model.predict(input_data_BP0mm[:,:,:,:,:2])
y_pred_BP5mm = model.predict(input_data_BP5mm[:,:,:,:,:2])
#%%
## choose threshold and get predictions 
threshold = 0.6
Y_pred_BP0mm = np.zeros(len(input_data_BP0mm))
index_pred_BP0mm = np.where(y_pred_BP0mm > threshold)[0]
Y_pred_BP0mm[index_pred_BP0mm] = 1
#%%
index_pred_BP5mm = np.where(y_pred_BP5mm > threshold)[0]
Y_pred_BP5mm = np.zeros(len(input_data_BP5mm))
Y_pred_BP5mm[index_pred_BP5mm] = 1
#%%
## save classification output sigmoid predictions
np.savez(r"sigmoid_prob_pred_BP0mm",y_pred_BP0mm)
np.savez(r"sigmoid_prob_pred_BP5mm",y_pred_BP5mm)

#%%
## load regression models 
model_path = r"C:\Users\georg\Desktop\master_thesis\Models\thesis_models"
model_folder=  r"\increasing_0404_3216_norm26k_pos_regression_8128layers"
best_model = r"\best_model_pos_regression_shuffeldBP_step001_layernorm.h5"
model_pos = keras.models.load_model(model_path+model_folder+best_model)

model_folder=  r"\increasing_0404_3216_norm26k_energy_regression_8128layers"
best_model = r"\best_model_energy_regression_shuffeldBP_step001_layernorm.h5"
model_energy = keras.models.load_model(model_path+model_folder+best_model)
#%%
#Y_pred_BP0mm = Y_pred_BP0mm[0:len(input_data_BP0mm)]
## make sure they have same size 
## so we can compare between them in the reco

index_BP0mm = np.where(Y_pred_BP0mm == 1)[0]
X_pred_BP0mm = input_data_BP0mm[index_BP0mm] 
#%%
index_BP5mm = np.where(Y_pred_BP5mm == 1)[0] ##+ trainset_index_BP0mm
X_pred_BP5mm = input_data_BP5mm[index_BP5mm] 
X_pred_BP5mm = X_pred_BP5mm[0:len(X_pred_BP0mm)]

#%%
pos_pred_BP0mm = model_pos.predict(X_pred_BP0mm)
energy_pred_BP0mm = model_energy.predict(X_pred_BP0mm)

#%%
energy_pred_BP5mm = model_energy.predict(X_pred_BP5mm)
pos_pred_BP5mm = model_pos.predict(X_pred_BP5mm)
#%%
## save data 
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\pos_pred_pipeline_BP0mm_all.npz",pos_pred_BP0mm)
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\pos_pred_pipeline_BP5mm_all.npz",pos_pred_BP5mm)
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\energy_pred_pipeline_BP0mm_all.npz",energy_pred_BP0mm)
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\energy_pred_pipeline_BP5mm_all.npz",energy_pred_BP5mm)
# %%
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\pos_pred_pipeline_BP5mm_all.npz",pos_pred_BP5mm)
np.savez(r"C:\Users\georg\Desktop\master_arbeit\recp_pipeline_data\energy_pred_pipeline_BP5mm_all.npz",energy_pred_BP5mm)

