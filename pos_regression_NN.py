#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from read_root import read_data
import matplotlib.pyplot as plt
#from sklearn.metrics import r2_score
#from tensorflow_addons.metrics import RSquare

# load data

path = "./pos_regression_BP0mm"
#path_root = "/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"

model_name = "/increasing_NN_regression_eppos_110323_mae_BP0mm_4knorm_ep"
model_name = path+model_name
data_path = r"/.automount/home/home__home2/institut_3b/farah/Desktop/Data/"

## data for regression 

get_data = read_data()

## call data
input_data = np.load(data_path+"training_data_Raster_bothneg_newnorm_1632_2ch_midempty.npz")
output_data = np.load(data_path+"ideal_targets_raster_ep.npz")

#input_data = np.load(data_path+"training_data_raster_bothneg_fullnorm_1632_2ch_midempty_.npz")
#output_data = np.load(data_path+"ideal_targets_raster_BP5mm_ep.npz")

pos_e = np.load(data_path+"pos_e_xyz_BP0mm_ep.npz")
pos_p = np.load(data_path+"pos_p_xyz_BP0mm_ep.npz")

checkpoint_name = path+"/pos_regression_BP0mm_4knorm_210323"

input_data = input_data['arr_0']
output_data = output_data['arr_0']
pos_e = pos_e['arr_0']
pos_p= pos_p['arr_0']
###############################################################################################
## concatenate posisition 
target_data = np.concatenate([pos_e,pos_p],axis=1)
###############################################################################################

## flip input data 
idx_compton = np.where(output_data == 1)[0]
input_data = input_data[idx_compton]
input_data = np.swapaxes(input_data,1,2)
input_data = np.swapaxes(input_data,2,3)
input_data = input_data[:,:,:,:,:2]

print(input_data.shape)
print(output_data.shape)

print(len(target_data))
print(len(input_data))


# slice data
trainset_index  = int(input_data.shape[0]*0.6)
valset_index    = int(input_data.shape[0]*0.8)
X_train = input_data[:trainset_index]
Y_train = target_data[:trainset_index]

## shufftle 
permutation = np.random.permutation(len(X_train))
X_train = X_train[permutation]
Y_train = Y_train[permutation]


X_val   = input_data[valset_index:]
Y_val   = target_data[valset_index:]

X_test  = input_data[trainset_index:valset_index]
Y_test  = target_data[trainset_index:valset_index]

print("Train shape: ",X_train.shape)
print("Target shape: ",Y_train.shape) 

### define R2 metrics

# Custom metric function to calculate R2 score
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def loss_energy_relative(y_true, y_pred):
    loss = tf.square(y_true - y_pred) / y_true
    return tf.reduce_mean(loss)

# define model
## fopr class weights I have trained with other val set (0.2 instead of 0.1)

## notes: pool size 2,2,1 is better than 2,2,2
## kernel size 3,6,3 is bad and 3,6,2 is also 

# compile the model with an Adam optimizer and a binary cross-entropy loss function
model = keras.Sequential([keras.layers.InputLayer(input_shape = (32,2,16,2)),
                         keras.layers.Conv3D(filters = 8, kernel_size =(3,2,3),padding="same",activation="relu", kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	 
                   #     keras.layers.LayerNormalization(),
  
			keras.layers.Conv3D(filters = 16, kernel_size =(3,2,3),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),	
                    			    keras.layers.MaxPool3D(pool_size=(2,1,2),padding="same"),
			keras.layers.LayerNormalization(),
 
	             #   keras.layers.Dropout(0.2),
			keras.layers.Conv3D(filters = 32, kernel_size = (3,2,3),padding="same",activation="relu",kernel_initializer="glorot_normal",strides=(1,1,1)), #kernel_regularizer=regularizers.l2(l=0.01)),
	            	   # keras.layers.Dropout(0.2),
			 keras.layers.MaxPool3D(pool_size=(2,1,2),padding="same"),
			keras.layers.LayerNormalization(),
 

                          keras.layers.Conv3D(filters = 64, kernel_size = (3,2,3),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),
                          keras.layers.LayerNormalization(),

		         keras.layers.Conv3D(filters = 128, kernel_size = (3,2,3),padding="same",activation="relu",kernel_initializer="glorot_normal"),# kernel_regularizer=regularizers.l2(l=0.01)),
			   keras.layers.GlobalAveragePooling3D(),
                       #    keras.layers.Dense(200, activation = "relu"),
                      #     keras.layers.Dropout(0.3),
                        #   keras.layers.Dense(50, activation = "relu"),
                           keras.layers.Dense(6, activation = "linear")])

def huber_loss(delta=0.05):
    def loss(y_true, y_pred):
        error = y_true - y_pred
    ##    print(error)
        condition = tf.abs(error) < delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(condition, squared_loss, linear_loss)
    return loss


model.compile(loss='mae',
                optimizer = keras.optimizers.Adam(0.001,epsilon=0.001),
                metrics=['mse','mae'])
# model summary 
model.summary()

## print data shape and compton percentage: 

print("################### \n train set size is:", X_train.shape, "\n ##################### \n")
print("################### \n val set size is:", X_val.shape, "\n ##################### \n")
print("################### \n target set size is:", Y_train.shape, "\n ##################### \n")


# train model
history = model.fit(X_train, 
            Y_train, 
            epochs = 500,   
            validation_data = (X_val, Y_val), 
            callbacks = [#tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.1,patience=10,min_lr=0,verbose=1),
				tf.keras.callbacks.EarlyStopping('val_loss', patience=15),
				tf.keras.callbacks.ModelCheckpoint(
    checkpoint_name,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1)
],
            batch_size = 64)


## weight for dist compton  {0:0.63047041, 1:2.41614324}
## weight for cut at 1, 10  {0:0.62772195, 1:2.45737701})
## weight for ideal {0:0.59396336, 1:3.16061141}
## weight for ideal cut at 2,20 {0:0.57175812 1:3.98392621}
## weight for dist cut at 2,20 {0:0.59540689,1:3.12035585}

#evaluate model
loss = model.evaluate(X_test, Y_test, verbose = 1)#, sample_weight=sample_weights_test) 

print('Test loss:', loss) 
#print('Test r2:', r2_score) 

y_pred = model.predict(X_test)
model.save(model_name)
#np.savez("y_pred_deep.npz",y_pred)

#%%
np.savetxt(model_name+"_loss.csv", history.history['loss'], delimiter=',')
np.savetxt(model_name+"_val_loss.csv", history.history['val_loss'], delimiter=',')

#np.savetxt(model_name+"_r2.csv", history.history['mse'], delimiter=',')
#np.savetxt(model_name+"_val_r2.csv", history.history['val_mse'], delimiter=',')

#np.savetxt(model_name+"_r2.csv", history.history['mae'], delimiter=',')
#np.savetxt(model_name+"_val_r2.csv", history.history['val_mae'], delimiter=',')

#%%
# summarize history for loss
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(model_name+'_loss_hist1.png')

#plt.figure(1)
#plt.plot(history.history['mse'])
#plt.plot(history.history['val_mse'])
#plt.title('model loss mse')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.savefig(model_name+'_loss_hist1.png')


#plt.figure(2)
#plt.plot(history.history['mae'])
#plt.plot(history.history['val_mae'])
#plt.title('model loss mse')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.savefig(model_name+'_loss_hist1.png')
