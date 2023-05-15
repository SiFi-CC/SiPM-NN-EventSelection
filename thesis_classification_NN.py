#%%

## MODEL 2 in Thesis
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from read_root import read_data

from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall

get_data = read_data()

print("Loading Indices")

path_root_0mm = "/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
df_id = get_data.get_df_from_root(path=path_root_0mm, root_entry_str='SiPMData.fSiPMId')
df_id_count = df_id.count(axis=1)
indices_BP0mm = np.where(df_id_count > 1)[0]

path_root_5mm = "/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_BP5mm_4e9protons.root"
df_id = get_data.get_df_from_root(path=path_root_5mm, root_entry_str='SiPMData.fSiPMId')
df_id_count = df_id.count(axis=1)
indices_BP5mm = np.where(df_id_count > 1)[0]

del df_id
del df_id_count

print("Loading Data")

############### loading data ###############################

path = "./thesis_models"
model_name = "/increasing_3003_3216_norm26k_declr_nolayernorm"
model_name = path+model_name
data_path = r"/.automount/home/home__home2/institut_3b/farah/Desktop/Data/"
checkpoint_filepath = model_name+"/best_model_fourthmodel_shuffeldBP_step001_layernorm.h5"

#data_path_1232_12k = r"/disk1/farah/"
input_data_BP0mm = np.load(data_path+"training_data_bothneg_norm26k_1632_2ch_midempty_0mmBP.npz")
input_data_BP5mm = np.load(data_path+"training_data_bothneg_norm26k_1632_2ch_midempty_5mmBP.npz")

#input_data_BP0mm = np.load(data_path_1232_12k+"training_data_bothneg_norm12k_1232_2ch_midempty_0mmBP.npz")
#input_data_BP5mm = np.load(data_path_1232_12k+"training_data_bothneg_norm12k_1232_2ch_midempty_5mmBP.npz")

output_data_BP0mm = np.load(data_path+"ideal_targets_BP0mm_ep.npz")
output_data_BP5mm = np.load(data_path+"ideal_targets_BP5mm_ep.npz")

sample_weights_BP0mm = np.load(data_path+"sample_weight_ideal_class_BP0mm_ep.npz")
sample_weights_BP5mm = np.load(data_path+"sample_weight_ideal_class_BP5mm_ep.npz")



#new_indices = new_indices['arr_0']
input_data_BP0mm = input_data_BP0mm['arr_0']
input_data_BP0mm = input_data_BP0mm[indices_BP0mm]

input_data_BP5mm = input_data_BP5mm['arr_0']
input_data_BP5mm = input_data_BP5mm[indices_BP5mm]

output_data_BP0mm = output_data_BP0mm['arr_0']
output_data_BP0mm = output_data_BP0mm[indices_BP0mm]

output_data_BP5mm = output_data_BP5mm['arr_0']
output_data_BP5mm = output_data_BP5mm[indices_BP5mm]

sample_weights_BP0mm = sample_weights_BP0mm['arr_0']
sample_weights_BP0mm = sample_weights_BP0mm[indices_BP0mm]
sample_weights_BP5mm = sample_weights_BP5mm['arr_0']
sample_weights_BP5mm = sample_weights_BP5mm[indices_BP5mm]

print("Slicing Data")

############### slice data ###############################

trainset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.6)
trainset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.6)

valset_index_BP0mm  = int(input_data_BP0mm.shape[0]*0.8)
valset_index_BP5mm  = int(input_data_BP5mm.shape[0]*0.8)


X_train_BP0mm = input_data_BP0mm[:trainset_index_BP0mm]
X_train_BP5mm = input_data_BP5mm[:trainset_index_BP5mm]

X_val_BP0mm = input_data_BP0mm[valset_index_BP0mm:]
X_val_BP5mm = input_data_BP5mm[valset_index_BP5mm:]

X_test_BP0mm = input_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
X_test_BP5mm = input_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]


Y_train_BP0mm = output_data_BP0mm[:trainset_index_BP0mm]
Y_train_BP5mm = output_data_BP5mm[:trainset_index_BP5mm]

Y_val_BP0mm = output_data_BP0mm[valset_index_BP0mm:]
Y_val_BP5mm = output_data_BP5mm[valset_index_BP5mm:]

Y_test_BP0mm = output_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
Y_test_BP5mm = output_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]

sample_weights_train_BP0mm = sample_weights_BP0mm[:trainset_index_BP0mm]
sample_weights_train_BP5mm = sample_weights_BP5mm[:trainset_index_BP5mm]

sample_weights_val_BP0mm = sample_weights_BP0mm[valset_index_BP0mm:]
sample_weights_val_BP5mm = sample_weights_BP5mm[valset_index_BP5mm:]


sample_weights_test_BP0mm = sample_weights_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
sample_weights_test_BP5mm = sample_weights_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]

X_train = np.concatenate([X_train_BP0mm,X_train_BP5mm])
Y_train = np.concatenate([Y_train_BP0mm,Y_train_BP5mm])
sample_weights_train = np.concatenate([sample_weights_train_BP0mm,sample_weights_train_BP5mm])

X_val= np.concatenate([X_val_BP0mm,X_val_BP5mm])
Y_val= np.concatenate([Y_val_BP0mm,Y_val_BP5mm])
sample_weights_val= np.concatenate([sample_weights_val_BP0mm,sample_weights_val_BP5mm])

X_test = np.concatenate([X_test_BP0mm,X_test_BP5mm])
Y_test = np.concatenate([Y_test_BP0mm,Y_test_BP5mm])
sample_weights_test= np.concatenate([sample_weights_test_BP0mm,sample_weights_test_BP5mm])

print("Shuffling Data")

########## shuflle the data #######################
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]
sample_weights_train = sample_weights_train[permutation]

permutation = np.random.permutation(len(X_val))
X_val= X_val[permutation]
Y_val= Y_val[permutation]
sample_weights_val= sample_weights_val[permutation]

######################################################

# Define the threshold for computing precision and recall
precision = Precision(thresholds=0.5,name="precision")
recall = Recall(thresholds=0.5,name="recall")

## notes: pool size 2,2,1 is better than 2,2,2
## kernel size 3,6,3 is bad and 3,6,2 is also 

# compile the model with an Adam optimizer and a binary cross-entropy loss function
model = keras.Sequential([keras.layers.InputLayer(input_shape = (16,32,2,2)),
               keras.layers.Conv3D(filters = 8, kernel_size =(3,3,2),padding="same",activation="relu", kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	 
#			keras.layers.LayerNormalization(),
                        keras.layers.Conv3D(filters = 16, kernel_size =(3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	 
			#keras.layers.Dropout(0.2),
			#keras.layers.LayerNormalization(),
			keras.layers.Conv3D(filters = 32, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal",strides=(1,1,1)), #kernel_regularizer=regularizers.l2(l=0.01)),
			keras.layers.Dropout(0.3),
			keras.layers.MaxPool3D(pool_size=(2,2,1),padding="same"),
			#keras.layers.LayerNormalization(),
                        keras.layers.Conv3D(filters = 64, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),
			keras.layers.Dropout(0.3),
			keras.layers.MaxPool3D(pool_size=(2,2,1),padding="same"),
			#keras.layers.MaxPool3D(pool_size=(2,1,1),padding="same"),
			#keras.layers.LayerNormalization(),
                        keras.layers.Conv3D(filters = 128, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),

			   keras.layers.GlobalAveragePooling3D(),
                           keras.layers.Dense(200, activation = "relu"),
                           keras.layers.Dropout(0.3),
                           keras.layers.Dense(50, activation = "relu"),
                           keras.layers.Dense(1, activation = "sigmoid")])

model.compile(loss='binary_crossentropy',
                optimizer = keras.optimizers.Adam(0.001,epsilon=0.001),
                metrics=[precision,recall],		
                weighted_metrics=['accuracy']                )

# model summary 
model.summary()
## print data shape and compton percentage: 

print("##################### \n percentage of Compton events in Training set is: ", len(Y_train[Y_train == 1])/len(Y_train), "\n ##################### \n")
print("##################### \n percentage of Compton events in Val set is: ", len(Y_val[Y_val == 1])/len(Y_val), "\n ##################### \n")
print("##################### \n percentage of Compton events in Test set is: ", len(Y_test[Y_test == 1])/len(Y_test), "\n ##################### \n")
print("################### \n train set size is:", X_train.shape, "\n ##################### \n")
print("################### \n val set size is:", X_val.shape, "\n ##################### \n")
print("################### \n complete data set size is:", len(input_data_BP0mm) + len(input_data_BP5mm)  , "\n ##################### \n")
#print("################### \n complete data set shape is:", input_data.shape, "\n ##################### \n")


# train model
history = model.fit(X_train, 
            Y_train, 
            epochs = 500,   
            validation_data = (X_val, Y_val,sample_weights_val), 
           callbacks = [tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.1,patience=10,min_lr=0,verbose=1),
				tf.keras.callbacks.EarlyStopping('val_loss', patience=14),
tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)],
            batch_size = 64,
	    sample_weight=sample_weights_train)
 #           class_weight = {0:0.59396336, 1:3.16061141})
	  
 # )##tf.keras.callbacks.EarlyStopping('val_loss', patience=30),


## weight for dist compton  {0:0.63047041, 1:2.41614324}
## weight for cut at 1, 10  {0:0.62772195, 1:2.45737701})
## weight for ideal {0:0.59396336, 1:3.16061141}
## weight for ideal cut at 2,20 {0:0.57175812 1:3.98392621}
## weight for dist cut at 2,20 {0:0.59540689,1:3.12035585}

#evaluate model
score = model.evaluate(X_test, Y_test, verbose = 1, sample_weight=sample_weights_test) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])
y_pred = model.predict(X_test)
model.save(model_name)
#np.savez("y_pred_deep.npz",y_pred)

index_pred = np.where(y_pred > 0.5)

num_entries = len(Y_test[index_pred[0]])
n_compton = Y_test[index_pred[0]].sum()

efficiency = n_compton/len(Y_test[Y_test==1])
Purity = n_compton/len(index_pred[0])

print("Efficiency is",  efficiency)
print("Purity is",  Purity)
print("##################### \n percentage of Compton events in Predicted set for threshold 0.5 is: ", len(index_pred[0])/len(y_pred), "\n ##################### \n")

index_pred = np.where(y_pred >= 0.6)
num_entries = len(Y_test[index_pred[0]])
n_compton = Y_test[index_pred[0]].sum()
efficiency = n_compton/len(Y_test[Y_test==1])
Purity = n_compton/len(index_pred[0])
print("Efficiency is",  efficiency)
print("Purity is",  Purity)

print("##################### \n percentage of Compton events in Predicted set for threshold 0.6 is: ", len(index_pred[0])/len(y_pred), "\n ##################### \n")
#%%
index_pred = np.where(y_pred >= 0.8)
num_entries = len(Y_test[index_pred[0]])
n_compton = Y_test[index_pred[0]].sum()
efficiency = n_compton/len(Y_test[Y_test==1])
Purity = n_compton/len(index_pred[0])
print("Efficiency is",  efficiency)
print("Purity is",  Purity)

print("##################### \n percentage of Compton events in Predicted set for threshold 0.8 is: ", len(index_pred[0])/len(y_pred), "\n ##################### \n")

#%%
np.savetxt(model_name+"_loss.csv", history.history['loss'], delimiter=',')
np.savetxt(model_name+"_val_loss.csv", history.history['val_loss'], delimiter=',')

np.savetxt(model_name+"_acc.csv", history.history['accuracy'], delimiter=',')
np.savetxt(model_name+"_val_acc.csv", history.history['val_accuracy'], delimiter=',')

np.savetxt(model_name+"_precision.csv", history.history['precision'], delimiter=',')
np.savetxt(model_name+"_val_precision.csv", history.history['val_precision'], delimiter=',')

#np.savetxt(model_name+"_precision_08.csv", history.history['precision_08'], delimiter=',')
#np.savetxt(model_name+"_val_precision_08.csv", history.history['val_precision_08'], delimiter=',')


np.savetxt(model_name+"_recall.csv", history.history['recall'], delimiter=',')
np.savetxt(model_name+"_val_recall.csv", history.history['val_recall'], delimiter=',')

#np.savetxt(model_name+"_recall_08.csv", history.history['recall_08'], delimiter=',')
#np.savetxt(model_name+"_val_recall_08.csv", history.history['val_recall_08'], delimiter=',')

plt.rcParams["font.family"] = "serif"

#%%
# summarize history for loss
plt.figure(0)
plt.plot(history.history['loss'],c="blue")
plt.plot(history.history['val_loss'],c="green")
plt.grid(True)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(model_name+'_loss_hist1.png')


# summarize history for accuracy


plt.figure(2)
plt.plot(history.history['accuracy'],label="balanced acc",c="k")
plt.plot(history.history['val_accuracy'],label="val balanced acc",c="red")
plt.grid(True)
plt.title('model balanced accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_name+'_balanced_acc_hist.png')

#%%
plt.figure(3)
plt.plot(history.history['precision'],label="train purity",c="k")
plt.plot(history.history['val_precision'],label="val purity",c="red")

plt.title('model purity scores threshold 0.5')
plt.ylabel('purity')
plt.grid(True)
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_name+'_precision.png')


#%%
plt.figure(5)
plt.plot(history.history['recall'],label="train effieciency",c="blue")
plt.plot(history.history['val_recall'],label="val effieciency",c="green")
plt.grid(True)
plt.title('model ')
plt.title('model efficiency scores threshold 0.5')
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_name+'_recall.png')

#%%



plt.figure(8)
plt.plot(history.history['precision'],label="train precision",c="k")
plt.plot(history.history['val_precision'],label="val precision",c="red")
plt.plot(history.history['recall'],label="train efficiency",c="blue")
plt.plot(history.history['val_recall'],label="val efficiency",c="green")
plt.grid(True)
plt.title('model purity and efficiency scores threshold 0.5')
plt.ylabel('scores')
plt.xlabel('epoch')
plt.legend()
plt.savefig(model_name+'_val_precision_recall_05.png')

