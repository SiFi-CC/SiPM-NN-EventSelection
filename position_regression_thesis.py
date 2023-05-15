import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from read_root import read_data

from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall


############### loading data ###############################

path = "./thesis_models"
model_name = "/increasing_0404_3216_norm12k_pos_regression_8128layers_nolayernorm"
model_name = path+model_name
data_path = r"/.automount/home/home__home2/institut_3b/farah/Desktop/Data/"
checkpoint_filepath = model_name+"/best_model_pos_regression_shuffeldBP_step001_layernorm.h5"

#data_path_1232_12k = r"/disk1/farah/"
input_data_BP0mm = np.load(data_path+"training_data_bothneg_norm12k_1632_2ch_midempty_0mmBP_compton.npz")
input_data_BP5mm = np.load(data_path+"training_data_bothneg_norm12k_1632_2ch_midempty_5mmBP_compton.npz")



pos_e_BP0mm = np.load(data_path+"pos_e_xyz_BP0mm_ep.npz")
pos_p_BP0mm = np.load(data_path+"pos_p_xyz_BP0mm_ep.npz")

pos_e_BP5mm = np.load(data_path+"pos_e_xyz_BP5mm_ep.npz")
pos_p_BP5mm = np.load(data_path+"pos_p_xyz_BP5mm_ep.npz")


input_data_BP0mm = input_data_BP0mm['arr_0']
input_data_BP5mm = input_data_BP5mm['arr_0']

pos_e_BP0mm = pos_e_BP0mm['arr_0']
pos_p_BP0mm = pos_p_BP0mm['arr_0']

pos_e_BP5mm = pos_e_BP5mm['arr_0']
pos_p_BP5mm = pos_p_BP5mm['arr_0']
###############################################################################################
## concatenate posisition 
target_data_BP0mm = np.concatenate([pos_e_BP0mm,pos_p_BP0mm],axis=1)
target_data_BP5mm = np.concatenate([pos_e_BP5mm,pos_p_BP5mm],axis=1)

###############################################################################################


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


Y_train_BP0mm = target_data_BP0mm[:trainset_index_BP0mm]
Y_train_BP5mm = target_data_BP5mm[:trainset_index_BP5mm]

Y_val_BP0mm = target_data_BP0mm[valset_index_BP0mm:]
Y_val_BP5mm = target_data_BP5mm[valset_index_BP5mm:]

Y_test_BP0mm = target_data_BP0mm[trainset_index_BP0mm:valset_index_BP0mm]
Y_test_BP5mm = target_data_BP5mm[trainset_index_BP5mm:valset_index_BP5mm]


X_train = np.concatenate([X_train_BP0mm,X_train_BP5mm])
Y_train = np.concatenate([Y_train_BP0mm,Y_train_BP5mm])

X_val= np.concatenate([X_val_BP0mm,X_val_BP5mm])
Y_val= np.concatenate([Y_val_BP0mm,Y_val_BP5mm])

X_test = np.concatenate([X_test_BP0mm,X_test_BP5mm])
Y_test = np.concatenate([Y_test_BP0mm,Y_test_BP5mm])

print("Shuffling Data")

########## shuflle the data #######################
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]

permutation = np.random.permutation(len(X_val))
X_val= X_val[permutation]
Y_val= Y_val[permutation]

######################################################

model = keras.Sequential([keras.layers.InputLayer(input_shape = (16,32,2,2)),
                         keras.layers.Conv3D(filters = 8, kernel_size =(3,3,2),padding="same",activation="relu", kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	 
                   #     keras.layers.LayerNormalization(),
  
			keras.layers.Conv3D(filters = 16, kernel_size =(3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	
                    			    keras.layers.MaxPool3D(pool_size=(2,2,1),padding="same"),
			#eras.layers.LayerNormalization(),
 
	             #   keras.layers.Dropout(0.2),
			keras.layers.Conv3D(filters = 32, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal",strides=(1,1,1)), #kernel_regularizer=regularizers.l2(l=0.01)),
	            	   # keras.layers.Dropout(0.2),
			 keras.layers.MaxPool3D(pool_size=(2,2,1),padding="same"),
			#eras.layers.LayerNormalization(),
 

                          keras.layers.Conv3D(filters = 64, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),
                       # keras.layers.LayerNormalization(),

		         keras.layers.Conv3D(filters = 128, kernel_size = (3,3,2),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),
			   keras.layers.GlobalAveragePooling3D(),
                       #    keras.layers.Dense(200, activation = "relu"),
                      #     keras.layers.Dropout(0.3),
                        #   keras.layers.Dense(50, activation = "relu"),
                           keras.layers.Dense(6, activation = "linear")])



model.compile(loss='mae',
                optimizer = keras.optimizers.Adam(0.001,epsilon=0.001),
                metrics=['mse'])

## print data shape and compton percentage: 

print("################### \n train set size is:", X_train.shape, "\n ##################### \n")
print("################### \n val set size is:", X_val.shape, "\n ##################### \n")
print("################### \n target set size is:", Y_train.shape, "\n ##################### \n")

model.summary()

# train model
history = model.fit(X_train, 
            Y_train, 
            epochs = 500,   
            validation_data = (X_val, Y_val), 
            callbacks = [tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.1,patience=10,min_lr=0,verbose=1),
				tf.keras.callbacks.EarlyStopping('val_loss', patience=15),
				tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,monitor='val_loss',save_best_only=True,mode='min', verbose=1)],
            batch_size = 64)


#evaluate model
loss = model.evaluate(X_test, Y_test, verbose = 1)#, sample_weight=sample_weights_test) 

print('Test loss:', loss) 
#print('Test r2:', r2_score) 

y_pred = model.predict(X_test)
model.save(model_name)

np.savetxt(model_name+"_loss.csv", history.history['loss'], delimiter=',')
np.savetxt(model_name+"_val_loss.csv", history.history['val_loss'], delimiter=',')

# summarize history for loss
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss MAE')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig(model_name+'_loss_hist.png')

plt.figure(1)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'])
plt.savefig(model_name+'_mse.png')




