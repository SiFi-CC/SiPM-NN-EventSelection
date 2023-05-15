"""
This script runs the evaluation of the classification models 
the following steps can be found in this script:
    1- load data from both BPs at 0 and 5mm, then split them and combine them 
    2- load model and plot the ROC/AUC and Recall/Precision curves  and Confusion Matrices
    3- plot the histogram of the source position and primary energy for true vs correct pred
    4- plot 2D histogtam of source position vs primary energy for true values 
    5- plot accuracy and loss plots 
"""
#%%
from read_root import read_data
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.utils import class_weight
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib as mpl


#%%

## data name and path 

path_root_BP0mm = "path_root_BP0mm"
path_root_BP5mm = "path_root_BP5mm"

target_data_path_BP0mm = "target_data_path_BP0mm"
target_data_path_BP5mm = "target_data_path_BP5mm"

input_data_path_BP0mm = "input_data_path_BP0mm"
input_data_path_BP5mm = "input_data_path_BP5mm"


#%%
## load data

## source pos and primary energy 
get_data = read_data()

df_pos_z_BP0mm = get_data.get_df_from_root(path_root_BP0mm,"MCPosition_source",pos="fZ",col_name="Pos Z")  
df_pos_z_BP5mm = get_data.get_df_from_root(path_root_BP5mm,"MCPosition_source",pos="fZ",col_name="Pos Z")  

df_primary_BP0mm = get_data.get_df_from_root(path_root_BP0mm,"MCEnergyPrimary", col_name="energy")
df_primary_BP5mm = get_data.get_df_from_root(path_root_BP5mm,"MCEnergyPrimary", col_name="energy")

## training and target data for both BPs
input_data_BP5mm = np.load(input_data_path_BP5mm)
output_data_BP5mm = np.load(target_data_path_BP5mm)

output_data_BP0mm = np.load(target_data_path_BP0mm)
input_data_BP0mm = np.load(input_data_path_BP0mm)

input_data_BP5mm = input_data_BP5mm['arr_0']#.swapaxes(2,3)
output_data_BP5mm = output_data_BP5mm['arr_0']#.swapaxes(2,3)
input_data_BP0mm = input_data_BP0mm['arr_0']#.swapaxes(2,3)
output_data_BP0mm = output_data_BP0mm['arr_0']#.swapaxes(2,3)


#%%
# 1- load data from both BPs at 0 and 5mm, then split them and combine them 

# slice data
trainset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.6)
trainset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.6)

valset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.8)
valset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.8)

#%%
X_test_BP0mm = input_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
X_test_BP5mm = input_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]

Y_test_BP0mm = output_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
Y_test_BP5mm = output_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]


#%%
## load model
model_path = r"model_path"
model_name = r"model_name"
best_model =    r"\best_model_fourthmodel_shuffeldBP_step001_layernorm.h5"
model = keras.models.load_model(model_path+model_name+best_model)   
#%%
model.summary()
#%%
X_test = np.concatenate([X_test_BP0mm,X_test_BP5mm])
Y_test = np.concatenate([Y_test_BP0mm,Y_test_BP5mm])
## make predictions 
y_pred = model.predict(X_test[:,:,:,:,:2]

#%%
# 2- load model and plot the ROC/AUC and Recall/Precision curves and confusion matrices:
#### ROC Curve, precision recall curve #########

## ROC

au2 = roc_auc_score(Y_test,y_pred, average="weighted")
plt.rcParams["font.family"] = "serif"
thesis_img_path = r"C:\Users\georg\Desktop\master_arbeit\thesis_img"
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
auc_value = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_value:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend()
#plt.savefig(thesis_img_path+"\ROC_Curve_BP0mm5mm.PNG")
plt.show()

## efficiency -> TPR (sensivity recall)
## purity -> PPV (precision, positive predictive value)
## Recall Precision (Efficiency/Purity)

precision, recall, thresholds = precision_recall_curve(Y_test, y_pred,)
ap = average_precision_score(Y_test, y_pred)

plt.plot(recall, precision, color='blue', label=f"AP={ap:0.2f}")
plt.plot(0.876,0.469,'o',label="Threhsold 0.5")
plt.plot(0.837,0.506,'o',label="Threhsold 0.6")
plt.plot(0.776,0.548,'o',label="Threhsold 0.7")
plt.plot(0.658,0.606,'o',label="Threhsold 0.8")

plt.xlabel('Recall (Efficiency)')
plt.ylabel('Precision (Puriity)')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend(loc='lower left')
#plt.savefig(thesis_img_path+"\Precision_recall_BP0mm5mm.PNG")
plt.show()
#%%
#### 
## set a sigmoid threshold for further analysis 
sigmoid_threhold = 0.7
Y_pred = np.zeros(len(Y_test))
index_pred = np.where(y_pred > sigmoid_threhold)[0]
Y_pred[index_pred] = 1
#%%
## split the data into BP0mm and BP5mm 
idx_test_BP0mm = len(Y_test_BP0mm)
idx_test_BP5mm = len(Y_test_BP5mm)

Y_pred_BP0mm = np.zeros(idx_test_BP0mm)
index_pred_BP0mm = np.where(y_pred[0:idx_test_BP0mm] > sigmoid_threhold)[0]
Y_pred_BP0mm[index_pred_BP0mm] = 1

num_predicted_idx_pred_BP0mm = len(Y_test_BP0mm[index_pred_BP0mm])
## set which data has been correctly predicted 
num_correct_BP0mm = Y_test_BP0mm[index_pred_BP0mm].sum()
##
y_pred_real_BP0mm = np.zeros(len(Y_test_BP0mm))
y_pred_real_BP0mm[index_pred_BP0mm] = 1
index_real_BP0mm = np.where(Y_test_BP0mm == 1)[0]
index_correct_BP0mm = np.where(np.logical_and(y_pred_real_BP0mm,Y_test_BP0mm) == True)[0]
#%%
Y_pred_BP5mm = np.zeros(idx_test_BP5mm)
index_pred_BP5mm = np.where(y_pred[idx_test_BP0mm:] > 0.7)[0]
Y_pred_BP5mm[index_pred_BP5mm] = 1

num_predicted_idx_pred_BP5mm = len(Y_test_BP5mm[index_pred_BP5mm])
num_correct_BP5mm = Y_test_BP5mm[index_pred_BP5mm].sum()

y_pred_real_BP5mm = np.zeros(len(Y_test_BP5mm))
y_pred_real_BP5mm[index_pred_BP5mm] = 1
index_real_BP5mm = np.where(Y_test_BP5mm == 1)[0]
index_correct_BP5mm = np.where(np.logical_and(y_pred_real_BP5mm,Y_test_BP5mm) == True)[0]
#%%
## both datasets together 
num_predicted = len(Y_test[index_pred])
num_correct = Y_test[index_pred].sum()

y_pred_real = np.zeros(len(Y_test))
y_pred_real[index_pred] = 1
index_real = np.where(Y_test == 1)[0]
index_correct = np.where(np.logical_and(y_pred_real,Y_test) == True)[0]

#%%
## some scores to print 
## balanced accuracy, percentage of compton events in the complete data set
## and effieciencz and purity

## percentage of compton events
tot_compton = len(Y_test[Y_test == 1])
percentage_compton = (tot_compton/len(Y_test)
)
print(tot_compton, percentage_compton)

## print balanced accuracy for the given data set 
print(balanced_accuracy_score(Y_test, Y_pred,adjusted=False))
### print effciency and purity with the given 
n_compton = Y_test[index_pred].sum()

efficiency = n_compton/len(Y_test[Y_test==1])
Purity = n_compton/len(index_pred)
print("Efficiency is",  efficiency)
print("Purity is",  Purity)

#%%
#### plot confusion matrices with complete figures and precentages 
## confusion matrix 
cm = confusion_matrix(Y_test, Y_pred)
label_names = ["Background", "Compton"]
# Set the printing options to display the full numbers
cm_display = ConfusionMatrixDisplay(cm, display_labels=label_names)
cm_display.plot(include_values=False)
#fmt = 'd'
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]),
                            ha="center", va="center",
                           color="darkred" if cm[i, j] > cm.max() / 2. else "yellow")
#np.set_printoptions(suppress=False)
plt.savefig(model_path+"\confusion_th05.PNG", bbox_inches='tight')
plt.show()
#%%
## confusion matrix percentages
cm = confusion_matrix(Y_test, Y_pred)
label_names = ["Background", "Compton"]

cm_percent = cm.astype('float') / len(Y_test)
#cm_percent = cm_percent.astype('str')
cm_display = ConfusionMatrixDisplay(cm_percent, display_labels=label_names)
cm_display.plot(include_values=False)
fmt = 'd'
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(np.round(cm_percent[i, j],2)),
                            ha="center", va="center",
                            color="darkred" if cm_percent[i, j] > cm_percent.max() / 2. else "yellow")

plt.savefig(model_path+"\confusion_th05_percentages.PNG", bbox_inches='tight')
#%%
#  3- plot the histogram of the source position and primary energy for true vs correct pred

index_noise = np.where(Y_test == 0)[0]
df_pos_z_reduced = df_pos_z_BP0mm.loc[trainset_index_BP0mm:valset_index_BP0mm].reset_index(drop=True)
df_pos_correct = df_pos_z_reduced.loc[index_correct_BP0mm]
df_pos_pred = df_pos_z_reduced.loc[index_pred_BP0mm]
df_pos_real = df_pos_z_reduced.loc[index_real_BP0mm]
#%%
bins_z_real, results_z_real = np.histogram(df_pos_real,bins=np.arange(-110,10,0.5))
bins_z_correct, results_z_correct = np.histogram(df_pos_correct, bins=np.arange(-110,10,0.5))
bins_z_all, result_z_all = np.histogram(df_pos_z_reduced, bins=np.arange(-110,10,0.5))
#bins_z_noise, result_z_noise = np.histogram(df_pos_z_reduced.iloc[index_noise], bins=np.arange(-100,10,0.5))
bins_z_pred, result_z_pred = np.histogram(df_pos_pred, bins=np.arange(-110,10,0.5))
#%%
mpl.rcParams['font.size'] = 15
plt.figure(figsize=(10,8))
plt.subplot(211)
#plt.hist(df_pos_z_reduced, color="k",bins=np.arange(-110,10,0.5),alpha=0.5)
plt.hist(df_pos_real, color="r",alpha=0.5,bins=np.arange(-110,10,0.5),label="All Compton")
plt.hist(df_pos_correct,color="b",alpha=0.5,bins=np.arange(-110,10,0.5),label="Correctly Predicted")
plt.ylabel("Counts")
plt.title("Histograms of Source Position for BP at 0mm")
#plt.xticks()
plt.xlim(-110,5)
plt.legend()
#plt.ylim(0,10) 
plt.subplot(212)
plt.bar(results_z_real[:-1],np.float64(bins_z_correct)/np.float64(bins_z_real),color="#008080")
plt.title("Percentage of Correctly Predicted Compton Events")
plt.xlim(-110,5)
#plt.xlim(-70,10)
plt.ylim(0,1)
plt.xlabel("mm")
plt.savefig(thesis_img_path+  "\Compton_events_percentage_source_pos_z_BP0mm_threshold07.pdf",bbox_inches="tight")
#%%
index_noise = np.where(Y_test == 0)[0]
df_pos_z_reduced = df_pos_z_BP5mm.loc[trainset_index_BP5mm:valset_index_BP5mm].reset_index(drop=True)
df_pos_correct = df_pos_z_reduced.loc[index_correct_BP5mm]
df_pos_pred = df_pos_z_reduced.loc[index_pred_BP5mm]
df_pos_real = df_pos_z_reduced.loc[index_real_BP5mm]

#%%
bins_z_real, results_z_real = np.histogram(df_pos_real,bins=np.arange(-110,10,0.5))
bins_z_correct, results_z_correct = np.histogram(df_pos_correct, bins=np.arange(-110,10,0.5))
bins_z_all, result_z_all = np.histogram(df_pos_z_reduced, bins=np.arange(-110,10,0.5))
#bins_z_noise, result_z_noise = np.histogram(df_pos_z_reduced.iloc[index_noise], bins=np.arange(-100,10,0.5))
bins_z_pred, result_z_pred = np.histogram(df_pos_pred, bins=np.arange(-110,10,0.5))
#%%
plt.figure(figsize=(10,8))
plt.subplot(211)
#plt.hist(df_pos_z_reduced, color="k",bins=np.arange(-110,10,0.5),alpha=0.5)
plt.hist(df_pos_real, color="r",alpha=0.5,bins=np.arange(-110,10,0.5),label="All Compton")
plt.hist(df_pos_correct,color="b",alpha=0.5,bins=np.arange(-110,10,0.5),label="Correctly Predicted")
plt.ylabel("Counts")
plt.title("Histograms of Source Position for BP at 5mm")
plt.xlim(-110,10)
plt.xticks([-100,-80,-60,-40,-20,0,5])
plt.legend()
#plt.ylim(0,10) 
plt.subplot(212)
plt.bar(results_z_real[:-1],np.float64(bins_z_correct)/np.float64(bins_z_real),color="#008080")
plt.title("Percentage of Correctly Predicted Compton Events")
plt.xlim(-110,10)
plt.xticks([-100,-80,-60,-40,-20,0,5])
#plt.xlim(-70,10)
plt.ylim(0,1)
plt.xlabel("mm")
plt.savefig(thesis_img_path+"\Compton_events_percentage_source_pos_z_BP5mm_threshold07.pdf",bbox_inches="tight")

#%%
df_primary_reduced = df_primary_BP0mm.loc[trainset_index_BP0mm:valset_index_BP0mm].reset_index(drop=True)
df_primary_correct = df_primary_reduced.loc[index_correct_BP0mm]
df_primary_pred = df_primary_reduced.loc[index_pred_BP0mm]
df_primary_real = df_primary_reduced.loc[index_real_BP0mm]

#%%

bins_primary_real, results_primary_real = np.histogram(df_primary_real,bins=np.arange(0,40,0.1))
bins_primary_correct, results_primary_correct = np.histogram(df_primary_correct, bins=np.arange(0,40,0.1))
#%%
#%%
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.hist(df_primary_reduced, color="k",alpha=0.4,bins=np.arange(0,40,0.1),label="All Events")
plt.hist(df_primary_real, color="r",alpha=0.5,bins=np.arange(0,40,0.1),label="All Compton")
plt.hist(df_primary_correct,color="b",alpha=0.5,bins=np.arange(0,40,0.1),label="Correctly Predicted")
#plt.hist(df_primary_reduced.loc[missed_indices_peak],np.arange(0,17.5,0.05),label="missed predictions peak")
plt.yscale("log")
plt.legend(fontsize=12)
plt.ylabel("Counts")
#plt.xlabel("MeV")
plt.title("Histograms of Primary Energy for BP at 0mm")
plt.xlim(0,17.5)
#plt.ylim(0,10)

plt.subplot(212)
plt.bar(results_primary_real[:-1],np.float64(bins_primary_correct)/np.float64(bins_primary_real),width=0.1,color="#008080")
#plt.hist(df_primary_correct/df_primary_real,bins=np.arange(0,1,0.01))#,color="#008080")

plt.title("Percentage of Correctly Predicted Compton Events")
plt.xlim(0,17.5)
#plt.xlim(0,10)
plt.xlabel("MeV")
plt.savefig(thesis_img_path+"\Compton_events_percentage_primary_energy_BP0mm_threshold05.pdf",bbox_inches="tight")

plt.show()
#%%
plt.plot(results_primary_real[1:],bins_primary_correct/bins_primary_real)
plt.bar(results_primary_real[1:],bins_primary_correct/bins_primary_real)#,color="#008080")

#%%
df_primary_reduced = df_primary_BP5mm.loc[trainset_index_BP5mm:valset_index_BP5mm].reset_index(drop=True)
df_primary_correct = df_primary_reduced.loc[index_correct_BP5mm]
df_primary_pred = df_primary_reduced.loc[index_pred_BP5mm]
df_primary_real = df_primary_reduced.loc[index_real_BP5mm]

#%%

bins_primary_real, results_primary_real = np.histogram(df_primary_real,bins=np.arange(0,40,0.1))
bins_primary_correct, results_primary_correct = np.histogram(df_primary_correct, bins=np.arange(0,40,0.1))

#%%

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.hist(df_primary_reduced, color="k",alpha=0.4,bins=np.arange(0,40,0.1),label="All Events")
plt.hist(df_primary_real, color="r",alpha=0.5,bins=np.arange(0,40,0.1),label="All Compton")
plt.hist(df_primary_correct,color="b",alpha=0.5,bins=np.arange(0,40,0.1),label="Correctly Predicted")
#plt.hist(df_primary_reduced.loc[missed_indices_peak],np.arange(0,17.5,0.05),label="missed predictions peak")
plt.yscale("log")
plt.legend(fontsize=12)
plt.ylabel("Counts")
#plt.xlabel("MeV")
plt.title("Histograms of Primary Energy for BP at 5mm")
plt.xlim(0,17.5)
#plt.ylim(0,10)
plt.subplot(212)
plt.bar(results_primary_real[:-1],np.float64(bins_primary_correct)/np.float64(bins_primary_real),color="#008080",width=0.1)
plt.title("Percentage of Correctly Predicted Compton Events")
#plt.xlim(-70,5)
plt.ylim(0,1)
plt.xlim(0,17.5)
plt.xlabel("MeV")
plt.savefig(thesis_img_path+"\Compton_events_percentage_primary_energy_BP5mm_threshold05.pdf",bbox_inches="tight")

#%%
## 4- plot 2D histogtam of source position vs primary energy for true values 
comptons_BP0mm = np.where(output_data_BP0mm == 1)[0]
comptons_BP5mm = np.where(output_data_BP5mm == 1)[0]

df_primary_BP0mm_compton = df_primary_BP0mm.loc[comptons_BP0mm].values.flatten()
df_primary_BP5mm_compton = df_primary_BP5mm.loc[comptons_BP5mm].values.flatten()


df_pos_BP0mm_compton = df_pos_z_BP0mm.loc[comptons_BP0mm].values.flatten()
df_pos_BP5mm_compton = df_pos_z_BP5mm.loc[comptons_BP5mm].values.flatten()
#%%
plt.hist(df_pos_BP0mm_compton,np.arange(-100,1,0.5))
#%%
plt.figure(figsize=(10,7))
plt.hist2d(df_pos_BP0mm_compton,df_primary_BP0mm_compton,bins=[np.arange(-100,2,0.5),np.arange(0,20,0.5)])
plt.xlim(-100,1)
plt.ylim(0,15)
plt.colorbar()
plt.xlabel("mm")
plt.ylabel("MeV")
plt.title('2D Histogram of Primary Energy and Source Position')
plt.savefig(thesis_img_path+r"\2Dhist_BP0mm.pdf",bbox_inches="tight")

#%%
plt.figure(figsize=(10,7))
plt.hist2d(df_pos_BP5mm_compton,df_primary_BP5mm_compton,bins=[np.arange(-100,10,0.5),np.arange(0,20,0.5)])
plt.xlim(-100,7)
plt.ylim(0,15)
plt.xticks([-100,-80,-60,-40,-20,0,5])
plt.colorbar()
plt.xlabel("mm")
plt.ylabel("MeV")
plt.title('2D Histogram of Primary Energy and Source Position')
plt.savefig(thesis_img_path+r"\2Dhist_BP5mm.pdf",bbox_inches="tight")
#%%
#  5- plot accuracy and loss plots 

## save model plot architecture 
keras.utils.plot_model(
    model,
    to_file="fourth_model_classw_5mmBP.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
   # layer_range=['dense', r'dense_2'],

    show_layer_activations=True

model_path = r"C:\Users\georg\Desktop\master_thesis\Models\DPG_model\Model_decreasinglr"
model_path_scores = model_path + r"\increasing_2ch_classw_2103_3216_newnorm4k_midempty_declr_16layernorm"

#acc_train = pd.read_csv(model_path+r"\third_NN_decreasing_3ch_samplew_040223_1632_norm_acc.csv",header=None)
#acc_val = pd.read_csv(model_path+r"\third_NN_decreasing_3ch_samplew_040223_1632_norm_val_acc.csv",header=None)
acc_train_weight = pd.read_csv(model_path_scores+r"_acc.csv",header=None)
acc_val_weight = pd.read_csv(model_path_scores+r"_val_acc.csv",header=None)

loss_train = pd.read_csv(model_path_scores+r"_loss.csv",header=None)
loss_val = pd.read_csv(model_path_scores+r"_val_loss.csv",header=None)
#%%

#plt.plot(acc_train.index,acc_train,label="train")
#plt.plot(acc_val.index,acc_val,label="val")
plt.plot(acc_train_weight.index[:35],acc_train_weight[:35],label="weighted train")
plt.plot(acc_val_weight.index[:35],acc_val_weight[:35],label="weighted val")

plt.title("model accuracy")
plt.legend()
plt.grid()
plt.xlabel("epoches")
plt.ylabel("acc")
#plt.savefig("acc_plot_fourthmodel.png")
plt.show()

acc_train = pd.read_csv(model_path+model_name+"_acc.csv",header=None)
acc_val = pd.read_csv(model_path+model_name+"_val_acc.csv",header=None)

loss_train = pd.read_csv(model_path+model_name+"_loss.csv",header=None)
loss_val = pd.read_csv(model_path+model_name+"_val_loss.csv",header=None)
#%%

plt.plot(loss_train.index,loss_train,label="train",c="k")
plt.plot(loss_val.index[:50],loss_val[:50],label="val",c="k",linestyle="--")
plt.plot(loss_val.index[49:],loss_val[49:]-0.001,c="k",linestyle="--")

plt.grid()
plt.title("Loss Binary Cross Entropy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss_plot.png")
plt.show()
#%%

plt.plot(acc_train.index,acc_train,label="train",color="blue")
plt.plot(acc_val.index,acc_val,label="val",color="blue",linestyle='--')
plt.title("Balanced Accuracy")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("acc_plot.png")
plt.show()

#%%
precision_train = pd.read_csv(model_path+model_name+"_precision.csv",header=None)
precision_val = pd.read_csv(model_path+model_name+"_val_precision.csv",header=None)

efficiency_train = pd.read_csv(model_path+model_name+"_recall.csv",header=None)
efficiency_val = pd.read_csv(model_path+model_name+"_val_recall.csv",header=None)
#%%
plt.plot(precision_train.index,precision_train,c="red",linestyle="-",label="purity train")
plt.plot(precision_val.index,precision_val,c="red",linestyle="--",label="purity val")

plt.plot(efficiency_train.index,efficiency_train,c="green",linestyle="-",label="efficiency train")
plt.plot(efficiency_val.index,efficiency_val,c="green",linestyle="--",label="efficiency val")
# plt.title("non weighted model accuracy")
plt.title("Efficiency and Purity")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.savefig("efficiency_purity.png")
plt.show()

