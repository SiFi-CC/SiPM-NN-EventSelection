## event plots 
#%%
import matplotlib.pyplot as plt
import numpy as np
from read_root import read_data
from generate_ids_tensor import ids_tensor
from tensorflow import keras
#%%
"""
to do: 

Event display of ten events from the 4.4 MeV peak
# Plot histogram of sum of all QDC values in one event

"""
#%%
root_data = read_data()
tensor_ids = ids_tensor()
#%%
import time
start_time = time.time()
path = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
root_data.get_root_entry_str_list(path)
df_x = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fX')
df_y = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fY')
df_z = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fZ')
df_id = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMId')
df_QDC = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMQDC')
df_trig = root_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMTriggerTime')
print("it took", time.time()-start_time)

## energy 
df_primary = root_data.get_df_from_root(path,"MCEnergyPrimary", col_name="energy")
df_e = root_data.get_df_from_root(path,"MCEnergy_e", col_name="energy")
df_p = root_data.get_df_from_root(path,"MCEnergy_p", col_name="energy")

#%%
# trigger time delta t
def get_delta_t_df(df_trig):
    """ 
    get df of diff betwenn max time in each event and the rest
    return df_delta
    """
    df_trig["min_time"] = np.min(df_trig,axis=1)
    df_trig = df_trig.apply(lambda x: x-x.min_time,axis=1)
    df_trig = df_trig.drop(columns="min_time")
    return df_trig

## Tensor Data 
input_data = np.load(r"D:\master_thesis_data\training_data_raster_38e8_neg.npz")
output_data = np.load(r"D:\master_thesis_data\target_data_ideal_raster_38e8.npz")

input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)

# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)
#%%
model = keras.models.load_model(r"C:\Users\georg\Desktop\NNModels\second_model\NN_shallow_ideal_comp_neg_train_1")

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
y_pred = model.predict(X_test)
#%%
index_pred = np.where(y_pred > 0.6)[0]
num_predicted = len(Y_test[index_pred])
y_pred_real = np.zeros(len(Y_test))
y_pred_real[index_pred] = 1
index_real = np.where(Y_test == 1)[0]
correct_prediction = np.logical_and(y_pred_real,Y_test)
index_correct = np.where(correct_prediction == True)[0]
#%%
ones_wrong_predicted = np.logical_or(y_pred_real,Y_test)  - Y_test
ones_not_predicted = Y_test - correct_prediction
ones_wrong_predicted_idx = np.where(ones_wrong_predicted == 1)[0]
ones_not_predicted_idx = np.where(ones_not_predicted == 1)[0]
#%%
df_primary_reduced = df_primary.loc[valset_index:].reset_index(drop=True)
df_primary_correct = df_primary_reduced.loc[index_correct]
df_primary_pred = df_primary_reduced.loc[index_pred]
df_primary_real = df_primary_reduced.loc[index_real]
#%%
## check where the data is in the 
primary_peak_events_index = df_primary_reduced[np.isclose(df_primary_reduced["energy"],4.4,0.001)].index
pred_in_peak = primary_peak_events_index[np.isin(primary_peak_events_index,index_correct)]
#%%
wrong_pred_in_peak = primary_peak_events_index[np.isin(primary_peak_events_index,ones_wrong_predicted_idx)]
no_pred_in_peak = primary_peak_events_index[np.isin(primary_peak_events_index,ones_not_predicted_idx)]
df_id_reduced = df_id.loc[valset_index:].reset_index(drop=True)

#%%
## to do: plot 10 events from 4.4 peak 
## First prepare the SiPM plots 
x_axis_dim = (df_x.loc[:10000].to_numpy().flatten())
y_axis_dim = (df_y.loc[:10000].to_numpy().flatten())
z_axis_dim = (df_z.loc[:10000].to_numpy().flatten())

x_axis_dim = x_axis_dim[~np.isnan(x_axis_dim)]
y_axis_dim = y_axis_dim[~np.isnan(y_axis_dim)]
z_axis_dim = z_axis_dim[~np.isnan(z_axis_dim)]
#%%
## see 5 predicted and 5 not predicted 
df_x_reduced = df_x.loc[valset_index:].reset_index(drop=True)
df_y_reduced = df_y.loc[valset_index:].reset_index(drop=True)
df_z_reduced = df_z.loc[valset_index:].reset_index(drop=True)
df_QDC_reduced = df_QDC.loc[valset_index:].reset_index(drop=True)
df_trig_reduced = df_trig.loc[valset_index:].reset_index(drop=True)
df_delta_reduced = get_delta_t_df(df_trig_reduced)

#%%
def new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred):
    for j in range(len(x_values_nopred[i][~np.isnan(x_values_nopred[i])])):
        ax.text(x_values_nopred[i,j]+np.random.randint(-10,10),
            y_values_nopred[i,j]+np.random.randint(-10,10),
            z_values_nopred[i,j]+np.random.randint(-20,20),
            s=tuple((j,np.round(c_values_nopred[i,j]),
                np.round(trig_values_nopred[i,j],2))),fontsize=6)


## prepare data to plot
x_values = df_x_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
y_values = df_y_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
z_values =  df_z_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
c_values = df_QDC_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
trig_values = df_delta_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]


#%%
import matplotlib.colors as colors
import matplotlib as mpl
#import mat
fig = plt.figure(figsize=(20,40))
ax = fig.add_subplot(5, 2, 1, projection='3d')
# plot hit detector from all events 
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn

print(x_values,y_values,z_values,c_values)
print(c_values,trig_values)
print(len(c_values),len(trig_values))
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values[0][~np.isnan(x_values[0])],
           y_values[0][~np.isnan(y_values[0])],
           z_values[0][~np.isnan(z_values[0])],
           'o',
           c=c_values[0][~np.isnan(c_values[0])],
           cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
ax.legend()
             
ax = fig.add_subplot(5, 2, 2, projection='3d')
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values[0][~np.isnan(x_values[0])],
           y_values[0][~np.isnan(y_values[0])],
           z_values[0][~np.isnan(z_values[0])],
           'o',
           c=trig_values[0][~np.isnan(trig_values[0])],
           cmap=cmap,norm=colors.Normalize(0,1),label="Trig")
    
ax.legend()
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)


n_plots = 5
for i in range(2,n_plots+1):
    ax = fig.add_subplot(5, 2, i+i-1, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values[i][~np.isnan(x_values[i])],
            y_values[i][~np.isnan(y_values[i])],
            z_values[i][~np.isnan(z_values[i])],
            'o',
            c=c_values[i][~np.isnan(c_values[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
                
    ax = fig.add_subplot(5, 2, i+i, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values[i][~np.isnan(x_values[i])],
            y_values[i][~np.isnan(y_values[i])],
            z_values[i][~np.isnan(z_values[i])],
            'o',
            c=trig_values[i][~np.isnan(trig_values[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
    new_func(ax, i, x_values, y_values, z_values, c_values, trig_values)        

plt.title("Example of 5 Correctly Predicted Compton Events in the Detectors in Charge and Trigger Times (Each Row is an Event)  ")    
plt.show()
#%%

## prepare data to plot
x_values_nopred = df_x_reduced.iloc[no_pred_in_peak].to_numpy()#.flatten()[:]
y_values_nopred = df_y_reduced.iloc[no_pred_in_peak].to_numpy()#.flatten()[:]
z_values_nopred =  df_z_reduced.iloc[no_pred_in_peak].to_numpy()#.flatten()[:]
c_values_nopred = df_QDC_reduced.iloc[no_pred_in_peak].to_numpy()#.flatten()[:]
trig_values_nopred = df_delta_reduced.iloc[no_pred_in_peak].to_numpy()#.flatten()[:]

#%%
print(trig_values_nopred[i-1][~np.isnan(trig_values_nopred[i-1])],
x_values_nopred[i-1][~np.isnan(x_values_nopred[i-1])],
z_values_nopred[i-1][~np.isnan(z_values_nopred[i-1])],
y_values_nopred[i-1][~np.isnan(y_values_nopred[i-1])],)

#%%

fig = plt.figure(figsize=(20,40))
ax = fig.add_subplot(5, 2, 1, projection='3d')
# plot hit detector from all events 
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn
ax.set_title("Example of 5 not  Predicted Compton Events in the Detectors in Charge and Trigger Times (Each Row is an Event)  ")    

ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_nopred[0][~np.isnan(x_values_nopred[0])],
           y_values_nopred[0][~np.isnan(y_values_nopred[0])],
           z_values_nopred[0][~np.isnan(z_values_nopred[0])],
           'o',
           c=c_values_nopred[0][~np.isnan(c_values_nopred[0])],
           cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
# ax.legend()

             
ax = fig.add_subplot(5, 2, 2, projection='3d')
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_nopred[0][~np.isnan(x_values_nopred[0])],
           y_values_nopred[0][~np.isnan(y_values_nopred[0])],
           z_values_nopred[0][~np.isnan(z_values_nopred[0])],
           'o',
           c=trig_values_nopred[0][~np.isnan(trig_values_nopred[0])],
           cmap=cmap,norm=colors.Normalize(0,1),label="Trig")
    
ax.legend()
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)

#plt.show()

n_plots = 5

for i in range(2,n_plots+1):
    ax = fig.add_subplot(5, 2, i+i-1, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values_nopred[i][~np.isnan(x_values_nopred[i])],
            y_values_nopred[i][~np.isnan(y_values_nopred[i])],
            z_values_nopred[i][~np.isnan(z_values_nopred[i])],
            'o',
            c=c_values_nopred[i][~np.isnan(c_values_nopred[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
                
    ax = fig.add_subplot(5, 2, i+i, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values_nopred[i][~np.isnan(x_values_nopred[i])],
            y_values_nopred[i][~np.isnan(y_values_nopred[i])],
            z_values_nopred[i][~np.isnan(z_values_nopred[i])],
            'o',
            c=trig_values_nopred[i][~np.isnan(trig_values_nopred[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
    new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred)        
        
plt.show()
#%%
## plot wrongly predicted 
## prepare data to plot
x_values_wrongpred = df_x_reduced.iloc[wrong_pred_in_peak].to_numpy()#.flatten()[:]
y_values_wrongpred = df_y_reduced.iloc[wrong_pred_in_peak].to_numpy()#.flatten()[:]
z_values_wrongpred =  df_z_reduced.iloc[wrong_pred_in_peak].to_numpy()#.flatten()[:]
c_values_wrongpred = df_QDC_reduced.iloc[wrong_pred_in_peak].to_numpy()#.flatten()[:]
trig_values_wrongpred = df_delta_reduced.iloc[wrong_pred_in_peak].to_numpy()#.flatten()[:]

#%%
fig = plt.figure(figsize=(20,40))
ax = fig.add_subplot(5, 2, 1, projection='3d')
# plot hit detector from all events wrong_pred_in_peak
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn
ax.set_title("Example of 5 not Wrong Predicted Compton Events in the Detectors in Charge and Trigger Times")    

ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_wrongpred[0][~np.isnan(x_values_wrongpred[0])],
           y_values_wrongpred[0][~np.isnan(y_values_wrongpred[0])],
           z_values_wrongpred[0][~np.isnan(z_values_wrongpred[0])],
           'o',
           c=c_values_wrongpred[0][~np.isnan(c_values_wrongpred[0])],
           cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
             
ax = fig.add_subplot(5, 2, 2, projection='3d')
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_wrongpred[0][~np.isnan(x_values_wrongpred[0])],
           y_values_wrongpred[0][~np.isnan(y_values_wrongpred[0])],
           z_values_wrongpred[0][~np.isnan(z_values_wrongpred[0])],
           'o',
           c=trig_values_wrongpred[0][~np.isnan(trig_values_wrongpred[0])],
           cmap=cmap,norm=colors.Normalize(0,1),label="Trig")
    
ax.legend()
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)

n_plots = 5
for i in range(2,n_plots+1):
    ax = fig.add_subplot(5, 2, i+i-1, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values_wrongpred[i][~np.isnan(x_values_wrongpred[i])],
           y_values_wrongpred[i][~np.isnan(y_values_wrongpred[i])],
           z_values_wrongpred[i][~np.isnan(z_values_wrongpred[i])],
            'o',
            c=c_values_wrongpred[i][~np.isnan(c_values_wrongpred[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
                
    ax = fig.add_subplot(5, 2, i+i, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values_wrongpred[i][~np.isnan(x_values_wrongpred[i])],
           y_values_wrongpred[i][~np.isnan(y_values_wrongpred[i])],
           z_values_wrongpred[i][~np.isnan(z_values_wrongpred[i])],
            'o',
            c=trig_values_wrongpred[i][~np.isnan(trig_values_wrongpred[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
    new_func(ax, i, x_values_wrongpred, y_values_wrongpred, z_values_wrongpred, c_values_wrongpred, trig_values_wrongpred)        

plt.show()

#%%

df_e_reduced = df_e.loc[valset_index:].reset_index(drop=True)
df_e_correct = df_e_reduced.loc[index_correct]
df_e_pred = df_e_reduced.loc[index_pred]
df_e_real = df_e_reduced.loc[index_real]
#%%
df_p_reduced = df_p.loc[valset_index:].reset_index(drop=True)
df_p_correct = df_p_reduced.loc[index_correct]
df_p_pred = df_p_reduced.loc[index_pred]
df_p_real = df_p_reduced.loc[index_real]
#%%
df_int_e = root_data.get_df_from_root(path,root_entry_str="MCInteractions_e")
df_int_p = root_data.get_df_from_root(path,root_entry_str="MCInteractions_p")
#%%
df_int_p_reduced = df_int_p.loc[valset_index:].reset_index(drop=True)
df_int_p_correct = df_int_p_reduced.loc[index_correct]
df_int_p_pred = df_int_p_reduced.loc[index_pred]
df_int_p_real = df_int_p_reduced.loc[index_real]
#%%
df_int_p_reduced.iloc[wrong_pred_in_peak]
#%%
df_int_e_reduced = df_int_e.loc[valset_index:].reset_index(drop=True)
df_int_e_correct = df_int_e_reduced.loc[index_correct]
df_int_e_pred = df_int_e_reduced.loc[index_pred]
df_int_e_real = df_int_e_reduced.loc[index_real]

#%%
df_pos_p_x = root_data.get_df_from_root(path,root_entry_str="MCPosition_p",pos="fX")
df_pos_e_x = root_data.get_df_from_root(path,root_entry_str="MCPosition_e",pos="fX")
#%%
df_pos_p_reduced = df_pos_p_x.loc[valset_index:].reset_index(drop=True)
df_pos_e_reduced = df_pos_e_x.loc[valset_index:].reset_index(drop=True)
#%%
fig = plt.figure(figsize=(40,20))

ax = fig.add_subplot(1, 3, 1)
ax.hist(df_QDC_reduced.loc[output_data[valset_index:] == 1].sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of compton events")
ax = fig.add_subplot(1, 3, 2)
ax.hist(df_QDC_reduced.loc[output_data[valset_index:] == 0].sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of non compton events")
ax = fig.add_subplot(1, 3, 3)
ax.hist(df_QDC_reduced.sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of all events")
plt.show()
#%%
fig = plt.figure(figsize=(40,20))

ax = fig.add_subplot(1, 3, 1)
ax.hist(df_QDC.loc[output_data == 1].sum(axis = 1),bins=10)
ax.set_title("QDC sum hist of compton events")
ax = fig.add_subplot(1, 3, 2)
ax.hist(df_QDC.loc[output_data == 0].sum(axis = 1),bins=10)
ax.set_title("QDC sum hist of non compton events")
ax = fig.add_subplot(1, 3, 3)
ax.hist(df_QDC.sum(axis = 1),bins=10)
ax.set_title("QDC sum hist of all events")
plt.show()
#%%
fig = plt.figure(figsize=(40,20))
ax = fig.add_subplot(1, 2, 1)
ax.hist(df_QDC_reduced.loc[(np.isin(df_QDC_reduced.index,index_correct))].sum(axis = 1),bins=50)
ax.set_title("QDC sum hist of compton events")
ax = fig.add_subplot(1, 2, 2)
ax.hist(df_QDC_reduced.loc[ (np.isin(df_QDC_reduced.index,ones_not_predicted_idx))].sum(axis = 1),bins=50)
ax.set_title("QDC sum hist of not predicted compton events")
plt.show()
#%%
fig = plt.figure(figsize=(40,20))

ax = fig.add_subplot(1, 3, 1)
ax.hist(df_trig.loc[output_data == 1].sum(axis = 1),bins=50)
ax.set_title("Trig sum hist of compton events")
ax = fig.add_subplot(1, 3, 2)
ax.hist(df_trig.loc[output_data == 0].sum(axis = 1),bins=50)
ax.set_title("Trig sum hist of non compton events")
ax = fig.add_subplot(1, 3, 3)
ax.hist(df_trig.sum(axis = 1),bins=50)
ax.set_title("Trig sum hist of all events")
plt.show()
#%%
binned_data = np.histogram(df_QDC.loc[output_data == 1].sum(axis = 1),bins=100)[1]
cs = np.cumsum(binned_data)
bin_idx = np.searchsorted(cs, np.percentile(cs, 90))
#%%
binned_data = np.histogram(df_QDC.sum(axis = 1),bins=100)[1]
#%%
np.percentile(df_QDC_reduced.loc[(np.isin(df_QDC_reduced.index,index_correct))].sum(axis = 1),90)
