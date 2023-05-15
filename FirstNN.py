#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# load data
#with np.load(path) as data:
input_data = np.load(r"D:\master_thesis_data\training_data_raster_38e8_neg.npz")
output_data = np.load(r"D:\master_thesis_data\target_data_ideal_raster_38e8.npz")

input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)



#%%
#%%
# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)
print(trainset_index)
print(valset_index)
X_train = input_data[:trainset_index]
Y_train = output_data[:trainset_index]
X_val   = input_data[trainset_index:valset_index]
Y_val   = output_data[trainset_index:valset_index]
X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
input_shape = X_train.shape
#%%
# define model
model = keras.Sequential([keras.layers.InputLayer(input_shape = (12,32,2,2)),
                            keras.layers.Conv3D(filters = 8, kernel_size = (3,3,1),padding="same",activation="relu"),
                            keras.layers.Conv3D(filters = 16, kernel_size = (3,3,1),padding="same",activation="relu"),
                            keras.layers.Conv3D(filters=32,kernel_size=(3,3,1),activation='relu'),
                            keras.layers.Conv3D(filters=64,kernel_size=(1,1,1),activation='relu'),
			                keras.layers.Flatten(),
			                keras.layers.Dropout(0.4),
                            keras.layers.Dense(1, activation = "sigmoid")])

# compile model
model.summary()
model.compile(loss='binary_crossentropy',
                optimizer = keras.optimizers.Adam(0.001),
                metrics=['acc']
                )

# train model
history = model.fit(X_train, 
            Y_train, 
            epochs = 100,    
            validation_data = (X_val, Y_val), 
            callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=10)],
            batch_size = 64,
            class_weight = {0:1, 1:6.5}
            )
#evaluate model
score = model.evaluate(X_test, Y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)
model.save('third_NN_model1_edited_allneg')
#np.savez("y_pred_3.npz",y_pred)
#%%
index_pred = np.where(y_pred > 0.5)
#%%
num_entries = len(Y_test[index_pred[0]])
n_compton = Y_test[index_pred[0]].sum()
#%%
efficiency = n_compton/len(Y_test[Y_test==1])
Purity = n_compton/len(index_pred[0])
#%%
print("Efficiency is",  efficiency)
print("Purity is",  Purity)

#%%
len(Y_test[Y_test==1])/len(Y_test)
#%%
print(len(index_pred[0])/len(y_pred))
#output_data[index_correct]
#%%
# summarize history for loss
fig = plt.figure(figsize=(15,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_hist3_edited.png',bbox_inches='tight')


# summarize history for accuracy
fig = plt.figure(figsize=(15,10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_hist3_edited.png',bbox_inches="tight")
#%%
np.savetxt("acc_third_model_edited.csv", history.history['acc'])
np.savetxt("val_acc_third_model_edited.csv", history.history['val_acc'])
np.savetxt("loss_acc_third_model_edited.csv", history.history['loss'])
np.savetxt("val_acc_third_model_edited.csv", history.history['val_loss'])