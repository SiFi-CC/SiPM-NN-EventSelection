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
input_data = np.load(r"D:\master_thesis_data\training_data_Raster_38e8_bothneg_normed_compton.npz")
output_data = np.load(r"D:\master_thesis_data\target_data_ideal_raster_38e8.npz")

input_data = input_data['arr_0']#.swapaxes(2,3)
output_data = output_data['arr_0']#.swapaxes(2,3)

# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)
#%%

model_path = r"C:\Users\georg\Desktop\NNModels\second_model"
model_name =    r"\NN_shallow_ideal_comp_bothneg_train_1_noweight"
model = keras.models.load_model(model_path+model_name)

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]
y_pred = model.predict(X_test)
#%%
index_pred = np.where(y_pred > 0.5)[0]
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


## prepare data to plot
x_values = df_x_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
y_values = df_y_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
z_values =  df_z_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
c_values = df_QDC_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
trig_values = df_delta_reduced.iloc[pred_in_peak].to_numpy()#.flatten()[:]
#%%


#%%
import matplotlib.colors as colors
import matplotlib as mpl
#import mat
fig = plt.figure(figsize=(40,60))
ax = fig.add_subplot(5, 2, 1, projection='3d')
grid = plt.GridSpec(5, 3)

# plot hit detector from all events 
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn

# print(x_values,y_values,z_values,c_values)
# print(c_values,trig_values)
# print(len(c_values),len(trig_values))

def new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred):
    for j in range(len(x_values_nopred[i][~np.isnan(x_values_nopred[i])])):
        ax.text(x_values_nopred[i,j]+np.random.choice([-10,10,0]),
            y_values_nopred[i,j]+np.random.choice([-5,5,0]),
            z_values_nopred[i,j]+np.random.choice([-5,5,0]),
            s=tuple((np.round(c_values_nopred[i,j]),
                np.round(trig_values_nopred[i,j],2))),fontsize=10)

ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values[0][~np.isnan(x_values[0])],
           y_values[0][~np.isnan(y_values[0])],
           z_values[0][~np.isnan(z_values[0])],
           'o',
           c=c_values[0][~np.isnan(c_values[0])],
           cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")
ax.legend()
ax.set_title("Correctly Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[0]),3)),fontsize=12)    

             
ax = fig.add_subplot(5, 2, 2, projection='3d')
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values[0][~np.isnan(x_values[0])],
           y_values[0][~np.isnan(y_values[0])],
           z_values[0][~np.isnan(z_values[0])],
           'o',
           c=trig_values[0][~np.isnan(trig_values[0])],
           cmap=cmap,norm=colors.Normalize(0,1))
new_func(ax, 0, x_values, y_values, z_values, c_values, trig_values)        

    
ax.legend(["Trig"])
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax)


n_plots = 5
for i in range(2,n_plots+1):
    ax = fig.add_subplot(5, 2, i+i-1, projection='3d')
    ax.set_title("Correctly Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[i]),3)),fontsize=12)    

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
#plt.savefig("Example_of_5_events.png",bbox_inches="tight", dpi=900)      
plt.show()

#%%

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
def new_func_annotate(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred):
    for j in range(len(x_values_nopred[i][~np.isnan(x_values_nopred[i])])):
        ax.annotate(str(tuple((np.round(c_values_nopred[i,j]),
                np.round(trig_values_nopred[i,j],2)))), (x_values_nopred[i,j],y_values_nopred[i,j],z_values_nopred[i,j]),x_values_nopred[i,j]+10,
            y_values_nopred[i,j]+np.random.choice([-10,10]),
            z_values_nopred[i,j]+10)

fig = plt.figure(figsize=(20,40), dpi=900)
ax = fig.add_subplot(5, 2, 1, projection='3d')
# plot hit detector from all events 
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn
#ax.set_title("Example of 5 not Predicted Compton Events in the Detectors in Charge and Trigger Times (Each Row is an Event)  ")    

ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_nopred[0][~np.isnan(x_values_nopred[0])],
           y_values_nopred[0][~np.isnan(y_values_nopred[0])],
           z_values_nopred[0][~np.isnan(z_values_nopred[0])],
           'o',
           c=c_values_nopred[0][~np.isnan(c_values_nopred[0])],
           cmap="YlGn",norm=colors.Normalize(0,1),label="QDC")

ax.legend()
ax.set_title("Not Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values_nopred[0]),3)),fontsize=12)    

             
ax = fig.add_subplot(5, 2, 2, projection='3d')
ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
ax.scatter(x_values_nopred[0][~np.isnan(x_values_nopred[0])],
           y_values_nopred[0][~np.isnan(y_values_nopred[0])],
           z_values_nopred[0][~np.isnan(z_values_nopred[0])],
           'o',
           c=trig_values_nopred[0][~np.isnan(trig_values_nopred[0])],
           cmap=cmap,norm=colors.Normalize(0,1),label="Trig")
new_func(ax, 0, x_values, y_values, z_values, c_values, trig_values)            
ax.legend(["Trig"])
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
    ax.set_title("Not Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values_nopred[0]),3)),fontsize=12)    
            
    ax = fig.add_subplot(5, 2, i+i, projection='3d')
    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    ax.scatter(x_values_nopred[i][~np.isnan(x_values_nopred[i])],
            y_values_nopred[i][~np.isnan(y_values_nopred[i])],
            z_values_nopred[i][~np.isnan(z_values_nopred[i])],
            'o',
            c=trig_values_nopred[i][~np.isnan(trig_values_nopred[i])],
            cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
    new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred)        
plt.savefig("Example_of_5_events_notpredicted.png",bbox_inches="tight")              
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
ax.set_title("Example of 5 Wrong Predicted Compton Events in the Detectors in Charge and Trigger Times")    

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
fig = plt.figure(figsize=(15,7.5))

ax = fig.add_subplot(1, 3, 2)
ax.hist(df_QDC.loc[output_data == 1].sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of compton events")
ax.vlines(15933,ymax=1000,ymin=0,color='k',label="norm value")
ax.set_xlim(-1000,80000)
ax.set_xlabel("# photons")
#ax.set_ylabel("counts")
ax.legend()
ax = fig.add_subplot(1, 3, 3)
ax.hist(df_QDC.loc[output_data == 0].sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of non compton events")
ax.set_xlim(-1000,80000)
ax.set_xlabel("# photons")
#ax.set_ylabel("counts")
ax = fig.add_subplot(1, 3, 1)
ax.hist(df_QDC.sum(axis = 1),bins=100)
ax.set_title("QDC sum hist of all events")
ax.set_xlim(-1000,80000)
ax.set_xlabel("# photons")
ax.set_ylabel("counts")
plt.savefig("QDC_sum_histogram.png",bbox_inches="tight")
plt.show()

#%%
df_delta = get_delta_t_df(df_trig)

#%%
fig = plt.figure(figsize=(15,7.5))

ax = fig.add_subplot(1, 3, 2)
ax.hist(df_delta.loc[output_data == 1].sum(axis = 1),bins=100)
ax.set_title("Trig sum hist of compton events")
ax.set_xlim(-0.1,80)
ax = fig.add_subplot(1, 3, 3)
ax.hist(df_delta.loc[output_data == 0].sum(axis = 1),bins=100)
ax.set_xlim(-0.1,80)
ax.set_title("Trig sum hist of non compton events")
ax = fig.add_subplot(1, 3, 1)
ax.hist(df_delta.sum(axis = 1),bins=100)
ax.set_xlim(-0.1,80)
ax.set_title("Trig sum hist of all events")
plt.savefig("delta_t_sum_histogram.png",bbox_inches="tight")

plt.show()
#%%

def get_nentries_in_event(df_QDC):
    n_entries_array = np.apply_along_axis(lambda x: len(x[~np.isnan(x)]),1,df_QDC)
    return n_entries_array
#%%
n_entries_df = get_nentries_in_event(df_QDC)
# %%
fig = plt.figure(figsize=(15,7.5))

ax = fig.add_subplot(1, 3, 2)
ax.hist(n_entries_df[output_data == 1],bins=20)
ax.set_title("# of Events hist of compton events")
ax.set_xlim(-0.1,40)

#ax.set_xlim(-0.1,80)
ax = fig.add_subplot(1, 3, 3)
ax.hist(n_entries_df[output_data == 0],bins=20)
ax.set_title("# of Events hist of non compton events")
ax = fig.add_subplot(1, 3, 1)
ax.hist(n_entries_df,bins=20)
#ax.set_xlim(-0.1,80)
ax.set_title("# of Events hist of all events")
plt.savefig("nentries_histogram.png",bbox_inches="tight")

plt.show()
#%%

time_diff = np.max(df_delta,axis=1) - 0 
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(1, 3, 2)
ax.hist(time_diff[output_data == 1],bins=50)
#ax.set_title("# of Events hist of compton events")
ax.set_xlim(-0.1,20)

#ax.set_xlim(-0.1,80)
ax = fig.add_subplot(1, 3, 3)
ax.hist(time_diff[output_data == 0],bins=50)
ax.set_xlim(-0.1,20)

#ax.set_title("# of Events hist of non compton events")
ax = fig.add_subplot(1, 3, 1)
ax.hist(time_diff,bins=50)
ax.set_xlim(-0.1,20)

#ax.set_xlim(-0.1,80)
#ax.set_title("# of Events hist of all events")
#plt.savefig("nentries_histogram.png",bbox_inches="tight")

plt.show()
#%%
real_peak_index = valset_index + pred_in_peak
wrong_pred_index = valset_index + wrong_pred_in_peak
no_pred_index = valset_index + no_pred_in_peak

trig_values_sort = np.argsort(trig_values,axis=1)

#%%
import matplotlib.colors as colors
import matplotlib as mpl
#import mat
fig = plt.figure(figsize=(40,60))
ax = fig.add_subplot(5, 1, 1, projection='3d')
grid = plt.GridSpec(5, 3)

# plot hit detector from all events 
idx = pred_in_peak[3]
# To create a scatter graph
cmap = plt.cm.YlGn
real_peak_index = pred_in_peak+valset_index
# print(x_values,y_values,z_values,c_values)
# print(c_values,trig_values)
# print(len(c_values),len(trig_values))

def new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred):
    for j in range(len(x_values_nopred[i][~np.isnan(x_values_nopred[i])])):
        if x_values_nopred[i,j] > 200:
            ax.text(x_values_nopred[i,j] - np.random.choice([10,20]) ,
                y_values_nopred[i,j]+np.random.choice([-10,10]),
                z_values_nopred[i,j]+np.random.choice([-10,10]),
                s=tuple((np.round(c_values_nopred[i,j]),
                    np.round(trig_values_nopred[i,j],2))),fontsize=10)
        else:
            ax.text(x_values_nopred[i,j] + np.random.choice([10,20]) ,
                y_values_nopred[i,j]+np.random.choice([-10,10]),
                z_values_nopred[i,j]+np.random.choice([-10,10]),
                s=tuple((np.round(c_values_nopred[i,j]),
                    np.round(trig_values_nopred[i,j],2))),fontsize=10)


ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightgrey')
ax.scatter(x_values[0][~np.isnan(x_values[0])],
           y_values[0][~np.isnan(y_values[0])],
           z_values[0][~np.isnan(z_values[0])],
           'o', c="red")
      #     c=trig_values[0][~np.isnan(trig_values[0])],
     #      cmap="Reds",norm=colors.Normalize(np.nanmin(trig_values[0]),np.nanmax(trig_values[0])),label="QDC")
#ax.legend()
ax.set_title(f"Correctly Predicted Compton Event # {real_peak_index[0]}  at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[0]),3)),fontsize=12)    

new_func(ax, 0, x_values, y_values, z_values, c_values, trig_values)  
         



n_plots = 5
for i in range(1,n_plots):
    ax = fig.add_subplot(5, 1, i+1, projection='3d')
    #ax.set_title("Correctly Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[i]),3)),fontsize=12)    

    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightgrey')
    ax.scatter(x_values[i][~np.isnan(x_values[i])],
            y_values[i][~np.isnan(y_values[i])],
            z_values[i][~np.isnan(z_values[i])],
           'o', c="red")
          #  c=trig_values[i][~np.isnan(trig_values[i])],
          #  cmap="Reds",norm=colors.Normalize(np.nanmin(trig_values[i]),np.nanmax(trig_values[i])),label="QDC")
    new_func(ax, i, x_values, y_values, z_values, c_values, trig_values)  
    ax.set_title(f"Correctly Predicted Compton Event # {real_peak_index[i]}  at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[0]),3)),fontsize=12)    

                
    # ax = fig.add_subplot(5, 2, i+i, projection='3d')
    # ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    # ax.scatter(x_values[i][~np.isnan(x_values[i])],
    #         y_values[i][~np.isnan(y_values[i])],
    #         z_values[i][~np.isnan(z_values[i])],
    #         'o',
    #         c=trig_values[i][~np.isnan(trig_values[i])],
    #         cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
plt.savefig("Example_of_5_events1.png",bbox_inches="tight", dpi=900)      
plt.show()
# %%
wrong_pred_index = valset_index + wrong_pred_in_peak
print(wrong_pred_index)
#%%
fig = plt.figure(figsize=(40,60))
ax = fig.add_subplot(5, 1, 1, projection='3d')
grid = plt.GridSpec(5, 3)

ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightgrey')
ax.scatter(x_values_nopred[0][~np.isnan(x_values_nopred[0])],
           y_values_nopred[0][~np.isnan(y_values_nopred[0])],
           z_values_nopred[0][~np.isnan(z_values_nopred[0])],
           'o', c="red")
         #  c=trig_values_wrongpred[0][~np.isnan(trig_values_wrongpred[0])],
          # cmap="Reds",norm=colors.Normalize(np.nanmin(trig_values_wrongpred[0]),np.nanmax(trig_values_wrongpred[0])),label="QDC")
#ax.legend()
ax.set_title(f"Not Predicted Compton Event # {no_pred_index[0]}  at the Peak for QDC Sum " + str(np.round(np.nansum(c_values_nopred[0]),3)),fontsize=14)    

new_func(ax, 0, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred)  
         



n_plots = 5
for i in range(1,n_plots):
    ax = fig.add_subplot(5, 1, i+1, projection='3d')
    #ax.set_title("Correctly Predicted Compton Event at the Peak for QDC Sum " + str(np.round(np.nansum(c_values[i]),3)),fontsize=12)    

    ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightgrey')
    ax.scatter(x_values_nopred[i][~np.isnan(x_values_nopred[i])],
           y_values_nopred[i][~np.isnan(y_values_nopred[i])],
           z_values_nopred[i][~np.isnan(z_values_nopred[i])],
            'o', c="red")
        #    c=trig_values_wrongpred[i][~np.isnan(trig_values_wrongpred[i])],
        #    cmap="Reds",norm=colors.Normalize(np.nanmin(trig_values_wrongpred[i]),np.nanmax(trig_values_wrongpred[i])),label="QDC")
    new_func(ax, i, x_values_nopred, y_values_nopred, z_values_nopred, c_values_nopred, trig_values_nopred)  
    ax.set_title(f"Not Predicted Compton Event # {no_pred_index[i]}  at the Peak for QDC Sum " + str(np.round(np.nansum(c_values_nopred[i]),3)),fontsize=14)    

                
    # ax = fig.add_subplot(5, 2, i+i, projection='3d')
    # ax.plot(x_axis_dim, y_axis_dim, z_axis_dim, 'o',c='lightblue')
    # ax.scatter(x_values[i][~np.isnan(x_values[i])],
    #         y_values[i][~np.isnan(y_values[i])],
    #         z_values[i][~np.isnan(z_values[i])],
    #         'o',
    #         c=trig_values[i][~np.isnan(trig_values[i])],
    #         cmap="YlGn",norm=colors.Normalize(0,1),label="Trig")
plt.savefig("Example_of_5_events_nopredicted1.png",bbox_inches="tight", dpi=900)      
plt.show()
#%%
trig_values_wrongpred[0][~np.isnan(trig_values_wrongpred[0])]
#%%
x_values_wrongpred[0][~np.isnan(x_values_wrongpred[0])]