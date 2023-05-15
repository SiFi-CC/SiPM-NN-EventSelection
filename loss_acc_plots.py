"""
NB to plot loss and accuracy 
Also look into weighted accuracy

"""

#%%
import pandas as pd
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report
#%%
#input_data = np.load(r"C:\Users\georg\Desktop\master_thesis\training_data_Raster_38e8_bothneg_normed_compton.npz")
input_data = np.load(r"C:\Users\georg\Desktop\master_thesis\training_data_Raster_38e8_bothneg_normed_compton.npz")

output_data = np.load(r"C:\Users\georg\Desktop\master_thesis\target_data_Raster_38e8_ideal.npz")
#%%
input_data_cut = np.load(r"C:\Users\georg\Desktop\master_thesis\training_data_Raster_38e8_bothneg_normed_compton_cut.npz")

output_data_cut = np.load(r"C:\Users\georg\Desktop\master_thesis\target_data_Raster_38e8_ideal_cut.npz")
#%%
output_data_dist = np.load(r"C:\Users\georg\Desktop\master_thesis\target_data_Raster_38e8_dist.npz")

#%%
input_data = input_data['arr_0']#.swapaxes(2,3)
#input_data1 = input_data1['arr_0']#.swapaxes(2,3)

output_data = output_data['arr_0']#.swapaxes(2,3)


dist_target = output_data_dist["arr_0"]
#%%
input_data_cut = input_data_cut['arr_0']
output_data_cut = output_data_cut['arr_0']#.swapaxes(2,3)


#%%
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(output_data),
                                                 y=output_data)
class_weights[0] * (len(output_data[output_data == 0])/len(output_data)) * 2
class_weights[1]
class_weights[1] * (len(output_data[output_data == 1])/len(output_data)) * 2


#%%
# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)
#%%
## load model
model_path = r"C:\Users\georg\Desktop\master_thesis\Models\third_model"
model2_path = r"C:\Users\georg\Desktop\master_thesis\Models\second_model_normed"
model_name_inc =    r"\third_NN_increasing_CNN"
model_name_inc_cut =    r"\third_NN_increasing_CNN_cut"

model_name_dec =    r"\third_NN_decreasing_CNN"
model_name_dec_cut =    r"\third_NN_decreasing_CNN_cut"

model_name2 = r"\NN_deep_ideal_comp_bothneg_classweight_cut"


model_inc = keras.models.load_model(model_path+model_name_inc)
model_dec = keras.models.load_model(model_path+model_name_dec)

model_inc_cut = keras.models.load_model(model_path+model_name_inc_cut)
model_dec_cut = keras.models.load_model(model_path+model_name_dec_cut)

model2 = keras.models.load_model(model2_path+model_name2)

#%%
## call position and energy df 
get_data = read_data()
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
df_primary = get_data.get_df_from_root(path_root,"MCEnergyPrimary", col_name="energy")
df_pos_z = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Z")  

df_pos_z_reduced = df_pos_z[:valset_index].reset_index(drop=True)
df_primary_reduced = df_primary[:valset_index].reset_index(drop=True)
#%%
X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
#%%
y_pred_inc = model_inc.predict(X_test)
y_pred_dec = model_dec.predict(X_test)
#%%
trainset_index_cut  = int(input_data_cut.shape[0]*0.7)
valset_index_cut    = int(input_data_cut.shape[0]*0.8)
X_test_cut = input_data_cut[valset_index_cut:]
Y_test_cut = output_data_cut[valset_index_cut:]
#%%
print(len(Y_test))
#%%
y_pred_inc_cut = model_inc_cut.predict(X_test_cut)
y_pred_dec_cut = model_dec_cut.predict(X_test_cut)
#%%

acc_train = pd.read_csv(model2_path+model_name2+"_acc.csv",header=None)
acc_val = pd.read_csv(model2_path+model_name2+"_val_acc.csv",header=None)

loss_train = pd.read_csv(model2_path+model_name2+"_loss.csv",header=None)
loss_val = pd.read_csv(model2_path+model_name2+"_val_loss.csv",header=None)

plt.plot(acc_train.index,acc_train,label="train")
plt.plot(acc_val.index,acc_val ,label="val")
plt.title("weighted model accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("acc")

plt.savefig("acc_plot_model2_cut.png")
plt.show()

plt.plot(loss_train.index,loss_train,label="train")
plt.plot(loss_val.index ,loss_val,label="val")

plt.title("weighted model loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")

plt.savefig("loss_plot_model2_cut.png")
plt.show()
#%%


#%%

## find where threshold higher than 0.5 and set the labels to 1
index_pred_dec = np.where(y_pred_dec > 0.5)[0]
num_predicted = len(Y_test[index_pred_dec])
num_correct = Y_test[index_pred_dec].sum()

Y_pred_dec = np.zeros(len(Y_test))
#index_pred = np.where(y_pred > 0.5)[0]
Y_pred_dec[index_pred_dec] = 1
#%%
len(output_data_cut[output_data_cut==1])/len(output_data_cut)
#%%
len(np.where(Y_test_cut == y_pred_dec_cut)[0])/len(Y_test_cut)
#%%
len(np.where(Y_test_cut == get_onehot_pred(Y_test_cut,y_pred_dec_cut))[0])/len(Y_test_cut)

#%%
def get_onehot_pred(Y_test, y_pred,threshold=0.5):
    np.where(Y_test)
    index_pred = np.where(y_pred > threshold)[0]
    num_correct = Y_test[index_pred].sum()
    Y_pred = np.zeros(len(Y_test))
    Y_pred[index_pred] = 1
    return Y_pred
#%%
## find index where test set is one and zero and where the predictions are right
index_real_one = np.where(Y_test == 1)[0]
index_real_zero = np.where(Y_test == 0)[0]

index_correct_one = np.where(np.logical_and(Y_pred,Y_test) == True)[0]
index_correct_all = np.where(Y_pred == Y_test)[0]
#%%
index_pred_dec_cut = np.where(y_pred_dec_cut > 0.5)[0]

num_correct_compton_pred = Y_test_cut[index_pred_dec_cut].sum()

efficiency = num_correct_compton_pred/len(Y_test_cut[Y_test_cut==1])
Purity = num_correct_compton_pred/len(index_pred_dec_cut)

print("Efficiency is",  efficiency)
print("Purity is",  Purity)
#%%
Y_dec_pred = get_onehot_pred(Y_test,y_pred_dec)
Y_dec_pred_cut = get_onehot_pred(Y_test_cut,y_pred_dec_cut)
#%%
print(len(Y_dec_pred_cut[Y_dec_pred_cut == 1])/len(Y_dec_pred_cut))
#%%
print(len(output_data_cut))
#%%

#########################################################################
###########################################################################
###########################################################################
## Scores: 

def balanced_acc(Y_test:list,Y_pred:list):
    ## get index where prediction is right
    index_correct_all = np.where(Y_pred == Y_test)[0]
    ## get how many are correct positive/negative predictions
    correct_pos_pred = len(np.where(Y_test[index_correct_all] == 1)[0])
    correc_neg_pred =  len(np.where(Y_test[index_correct_all] == 0)[0])
    tot_pos_pred = len(Y_test[Y_test == 1]) 
    tot_neg_pred = len(Y_test[Y_test == 0]) 
    ## balanced acc: (percentage_pos * 0.5 + precentage_neg * 0.5)
    balanced_acc = (correct_pos_pred/tot_pos_pred + correc_neg_pred/tot_neg_pred) * 0.5  
    return balanced_acc

print(balanced_accuracy_score(Y_test, Y_pred,adjusted=False))
#%%

## confusion matrix 
cm = confusion_matrix(Y_test, Y_pred)

cm_plot = ConfusionMatrixDisplay(cm)
cm_plot.plot()
#%%
## classification report 
cr = classification_report(Y_test, Y_pred)
print(cr)
#%%
output_data_dist = output_data_dist['arr_0']
#%%

#%%
## efficiency and purity

num_correct_compton_pred = Y_test[index_pred].sum()

efficiency = num_correct_compton_pred/len(Y_test[Y_test==1])
Purity = num_correct_compton_pred/len(index_pred)

print("Efficiency is",  efficiency)
print("Purity is",  Purity)
#%%
print(model_path+model_name)
#%%
keras.utils.plot_model(model)

#%%
###############################################################
################# plots #######################################
###############################################################

## to do: plot false positives; 
fp_label = np.where((Y_pred == 1) & (Y_test == 0))[0]
print(len(fp_label))
#%%
Y_test_dist  = dist_target[valset_index:]
#%%
fp_in_dist_target = Y_test_dist[fp_label]
pos_in_dist = np.where(fp_in_dist_target == 1)[0]
print(len(fp_in_dist_target))
print(len(pos_in_dist))
#%%

#%%
plt.figure(figsize=(15,10))
plt.subplot(111)
#plt.hist(df_pos_z,color="r",bins=np.arange(-100,5,0.5),label="All")
plt.hist(df_pos_z_reduced.loc[fp_label],color="g",bins=np.arange(-100,5,0.5),label="FP")
plt.legend()
plt.ylabel("Counts")
plt.title("Source Pos Z Axis Histograms of FP")
plt.xlim(-70,5)
#%%
plt.figure(figsize=(15,10))
plt.subplot(111)
#plt.hist(df_pos_z,color="r",bins=np.arange(-100,5,0.5),label="All")
plt.hist(df_primary_reduced.loc[fp_label],color="g",bins=np.arange(-100,5,0.5),label="FP")
plt.ylabel("Counts")
plt.title("Primary Energy Histograms of FP")
plt.xlim(-1,10)
plt.legend()
plt.show()
#plt.ylim(0,5000)
#plt.subplot(212)

#%%
################ plot all data of pos and primary energy ##############################

index_pos = np.where(output_data == 1)[0]
index_neg = np.where(output_data == 0)[0]


#%%
df_pos_pos = df_pos_z.loc[index_pos]
df_pos_neg = df_pos_z.loc[index_neg]
#%%

bins_z_real, results_z_real = np.histogram(df_pos_z,bins=np.arange(-100,5,0.5))
bins_z_correct, results_z_correct = np.histogram(df_pos_pos, bins=np.arange(-100,5,0.5))

#%%
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.hist(df_pos_z,color="r",bins=np.arange(-100,5,0.5),label="All")
plt.hist(df_pos_neg,color="g",bins=np.arange(-100,5,0.5),label="Noise")
plt.hist(df_pos_pos, color="b",bins=np.arange(-100,5,0.5),label="Compton")
plt.legend()
plt.ylabel("Counts")
plt.title("Source Pos Z Axis Histograms of All Events")
plt.xlim(-70,5)
plt.ylim(0,5000)
plt.subplot(212)
plt.bar(results_z_real[:-1],np.float64(bins_z_correct)/np.float64(bins_z_real),color="#008080")
plt.title("Percentage of Compton from all events")
plt.xlim(-70,5)
plt.ylim(0,1)
plt.xlabel("mm")
#plt.savefig("Compton_events_percentage_source_pos_z_new_weighted_model.png",bbox_inches="tight")
#%%
len(df_pos_z[df_pos_z["Z"] == 0])/len(df_pos_z)
#%%

index_pos = np.where(target == 1)[0]
index_neg = np.where(target == 0)[0]
#%%

#df_primary_reduced = df_primary.loc[valset_index:].reset_index(drop=True)
df_primary_pos = df_primary.loc[index_pos]
df_primary_neg = df_primary.loc[index_neg]

## get bins
bins_primary_real, results_primary_real = np.histogram(df_primary,bins=np.arange(0,40,0.5))
bins_primary_correct, results_primary_correct = np.histogram(df_primary_pos, bins=np.arange(0,40,0.5))

#%%

plt.figure(figsize=(10,5))
plt.subplot(111)
plt.hist(df_primary,color="r",bins=np.arange(0,40,0.05),label="All events")
plt.hist(df_primary_neg,color="g",bins=np.arange(0,40,0.05),label="Noise")
plt.hist(df_primary_pos, color="b",bins=np.arange(0,40,0.05),label="Compton")
plt.plot(np.ones(1000) * 1, np.linspace(0,20000,1000), '--',c='k')
plt.plot(np.ones(1000) * 10, np.linspace(0,20000,1000), '--',c='k')

plt.legend()
plt.ylabel("Counts")
plt.xlabel("MeV")
plt.title("Primary Energy Histograms of Compton Events")
plt.xlim(-0.01,30)
#plt.ylim(0,10)
# plt.subplot(212)
# plt.bar(results_primary_real[:-1],np.float64(bins_primary_correct)/np.float64(bins_primary_real),color="#008080")
# plt.title("Percentage of Compton Events from All events")
# #plt.xlim(-70,5)
# plt.ylim(0,1)
# plt.xlim(-0.5,30)
# plt.xlabel("MeV")
#plt.savefig("Compton_events_percentage_primary_energy_new_weighted.png",bbox_inches="tight")

#%%df_ #= df_primary.loc[index_pos]
## we want to see now where the energies are 
high_energy_idx = df_primary[df_primary["energy"] > 1].index
#%%
he_and_pos = np.isin(high_energy_idx, index_pos)
he_and_neg = np.isin(high_energy_idx, index_neg)
#%%
he_and_neg
#%%
df_z_pos_he = df_pos_z.loc[high_energy_idx[he_and_pos]]
df_z_neg_he = df_pos_z.loc[high_energy_idx[he_and_neg]]
#%%
len(np.where(he_and_neg == True)[0])
#%%
bins_z_real, results_z_real = np.histogram(df_pos_z,bins=np.arange(-100,5,0.5))
bins_z_correct, results_z_correct = np.histogram(df_pos_pos, bins=np.arange(-100,5,0.5))

#%%
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.hist(df_pos_z,color="grey",bins=np.arange(-100,5,0.5),label="All")

plt.hist(df_pos_z.loc[high_energy_idx],color="r",bins=np.arange(-100,5,0.5),label="All High Energy")
plt.hist( df_z_neg_he,color="g",bins=np.arange(-100,5,0.5),label="Noise High Energy")
plt.hist(df_z_pos_he, color="b",bins=np.arange(-100,5,0.5),label="Compton High Energy")
plt.legend()
plt.ylabel("Counts")
plt.title("Source Pos Z Axis Histograms of All Events of high energy")
plt.xlim(-70,5)
plt.ylim(0,5000)
#%%
## to dos: 

## 1- energy cut NN 
#%%
plt.subplot(212)
plt.bar(results_z_real[:-1],np.float64(bins_z_correct)/np.float64(bins_z_real),color="#008080")
plt.title("Percentage of Compton from all events")
plt.xlim(-70,5)
plt.ylim(0,1)
plt.xlabel("mm")
#%%

#%%
## plot model 

keras.utils.plot_model(model,show_shapes=True
                            ,expand_nested=True
                            ,show_layer_names=True
                            ,show_layer_activations=True)
#%%
                            
keras.utils.plot_model(model_inc,show_shapes=True
                            ,expand_nested=True
                            ,show_layer_names=True
                            ,show_layer_activations=True)
#%%
#%%
keras.utils.plot_model(model2,show_shapes=True,expand_nested=True,show_layer_names=True,show_layer_activations=True)
