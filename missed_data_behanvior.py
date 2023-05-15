"""
- Check Norming the data 
"""
#%%
from read_root import read_data
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

#%%
path_root = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"

get_data = read_data()
#%%
df_qdc = get_data.get_array_from_root(path_root,'SiPMData.fSiPMQDC')
df_pos_z = get_data.get_array_from_root(path_root,"MCPosition_source",pos="fZ")  
df_trig = get_data.get_df_from_root(path_root,'SiPMData.fSiPMTriggerTime')
df_ID = get_data.get_df_from_root(path_root,'SiPMData.fSiPMId')

array_int_p = get_data.get_array_from_root(path_root, root_entry_str="MCInteractions_p")[trainset_index:valset_index]
array_int_e = get_data.get_array_from_root(path_root, root_entry_str="MCInteractions_e")[trainset_index:valset_index]
array_pos_p_x = get_data.get_array_from_root(path_root,root_entry_str="MCPosition_p",pos="fX")[trainset_index:valset_index]
array_pos_e_x = get_data.get_array_from_root(path_root,root_entry_str="MCPosition_e",pos="fX")[trainset_index:valset_index]
#%%
df_trig.loc[trainset_index:valset_index].reset_index(drop=True)

#%%
output_data = np.load(r"C:\Users\georg\Desktop\master_thesis\target_data_Raster_38e8_ideal_true.npz")
missed_data = np.load(r"C:\Users\georg\Desktop\master_thesis\mised_indices_peak_testset.npz")
output_data = output_data["arr_0"]
missed_data = missed_data["arr_0"]
#%%
# slice data
trainset_index  = int(output_data.shape[0]*0.6)
valset_index    = int(output_data.shape[0]*0.8)
compton_idx = np.where(output_data == 1)[0]
#%%
output_data = output_data[trainset_index:valset_index]
#%%
array_qdc = np.array(df_qdc)[trainset_index:valset_index]
array_pos = np.array(df_pos_z)[trainset_index:valset_index]
output_data = output_data[trainset_index:valset_index]
df_trig = df_trig.loc[trainset_index:valset_index].reset_index(drop=True)
df_ID = df_ID.loc[trainset_index:valset_index].reset_index(drop=True)
df_trig_diff = df_trig.apply(lambda x: x - np.min(x),axis=1)

df_pos_e = pd.DataFrame(array_pos_e_x)
df_pos_p = pd.DataFrame(array_pos_p_x)
df_int_p = pd.DataFrame(array_int_p)
df_int_e = pd.DataFrame(array_int_e)
#%%
df_pos_p.loc[1977,0:20]
#%%
df_int_p.loc[1977,0:20]
#%%
df_int_e.loc[1977,0:20]
#%%
df_pos_e.loc[1977,0:20]
#%%
df_ID.loc[2057,0:20]
#%%
df_pos_e.loc[142344]
#%%
## interesting events (from test set): 
# 142344
# 142146
# 3367
# 3424
# 3651
# 3830
# 3943
## pe
# 2080 
# 2184
# 3652
##
#%%
## check whether mutliple data is triggered 
def are_both_in_same_detector(id_1, id_2):
    #((id_1 < 112) and  (id_2>367 and id_2 <480)) or ((id_2 < 112) and  (id_1>367 and id_1 <480)) or
    if  (id_1<112 and id_2<112) or ((id_2>367 and id_2 <480) and (id_1>367 and id_1 <480)) or ((id_1 < 112) and  (id_2>367 and id_2 <480)) or ((id_2 < 112) and  (id_1>367 and id_1 <480)): 
        return False

    #((id_1 >=112 and id_1<368) and (id_2 >= 480)) or ( and (id_1 >= 480)) or
    elif  ((id_1 >=112 and id_1<368) and (id_2 >=112 and id_2<368)) or (id_1 >= 480 and (id_2 >= 480)) or ((id_1 >=112 and id_1<368) and (id_2 >= 480)) or ((id_2 >=112 and id_2<368) and (id_1 >= 480)) :
        return False
    ## lx ly
    else:
        return True
#%%
are_both_in_same_detector(498,23)
#%%
smallest_indices = np.array(len(missed_data))
#smallest_two_idx = np.argsort(arr)[:2]
smallest_two_idx_list = [np.argsort(x)[:2] for x in df_trig_diff.to_numpy()]

#%%
both_in_detector = []
for index,row in df_ID.iterrows():
    #print(row[smallest_two_idx_list[index][0]])
    id_1,id_2 = row[smallest_two_idx_list[index][0]],row[smallest_two_idx_list[index][1]]
    both_in_detector.append(are_both_in_same_detector(id_1,id_2))
    
#%%
both_in_detector = np.array(both_in_detector)
#%%
compton_idx = np.where(output_data == 1)[0]
both_in_detector_compton = both_in_detector[compton_idx]

len(both_in_detector[both_in_detector==False])/len(both_in_detector)
#%%
len(both_in_detector_compton[both_in_detector_compton==False])/len(both_in_detector_compton)
#%%
both_in_detector_missed = both_in_detector[missed_data]

len(both_in_detector_missed[both_in_detector_missed==False])/len(both_in_detector_missed)
#%%
smallest_two_idx_list[-1]
#%%
df_trig_diff
#%%
np.argsort(df_trig_diff.loc[1])[:2]
#%%
df_trig_diff.loc[missed_data[10:20]]
#%%
ID_numpy = df_ID.to_numpy()
#%%
np.where(df_ID == 0)
#%%
df_ID.loc[294]
#%%
df_ID.loc[3943]
# %%
unzipped_list_missed = list(itertools.chain(* array_qdc[missed_data]))
unzipped_list = list(itertools.chain(* array_qdc))
unzipped_list_compton = list(itertools.chain(* array_qdc[np.where(output_data==1)[0]]))

#%%
unzipped_list_normed = [np.array(x)/3000 for x in array_qdc[missed_data]]
#unzipped_list_normed = list(itertools.chain(*unzipped_list_normed))
#%%
mean_unzipped = np.mean(unzipped_list)
std_unzipped = np.std(unzipped_list)

#%%
bars, bins = np.histogram(unzipped_list,bins=np.arange(0,10000,0.5))
bars_normed, bins_normed = np.histogram(unzipped_list_normed,bins=np.arange(0,1000,0.01))
# %%
plt.bar(bins[0:-1],bars)
#%%
plt.bar(bins_normed[1:21],bars_normed[0:20])
plt.xlim(0,10)
plt.show()
#%%1
## normed by dividing over all 
##plt.hist(unzipped_list,bins=100,alpha=0.5)
plt.hist(unzipped_list_compton,bins=100,alpha=0.5)
plt.show()
#%%
plt.hist(unzipped_list_missed,bins=100,alpha=0.5)
plt.show()
#%%
plt.hist(array_qdc[missed_data])
#%%
## normed by gaussian 
plt.hist((unzipped_list)/std_unzipped,bins=1000)
plt.xlim(0,5)

#%%
compton_idx = np.where(output_data == 1)[0]
# %%
qdc_array_compton = [array_qdc[i] for i in compton_idx]
pos_array_compton = [array_pos[i] for i in compton_idx]

unzipped_list_compton = list(itertools.chain(*qdc_array_compton))
#unzipped_list_compton_z = list(itertools.chain(*pos_array_compton))
#%%
plt.hist(pos_array_compton,np.arange(-100,10,0.1))
plt.vlines(-6,0,200,color="k")
#%%
pos_array_compton = np.array(pos_array_compton)
idx_peak = np.where(pos_array_compton > -10)[0]
plt.hist(pos_array_compton[idx_peak],np.arange(-10,1,0.1))
#%%
## find data at peak 
idx_peak_original = np.where(((array_pos > -10).flatten() & (output_data == 1)) == True)[0]
primary_peak = [array_qdc[i] for i in idx_peak_original]
primary_peak_unzipped = list(itertools.chain(*primary_peak))

#%%
plt.hist(np.array(primary_peak_unzipped)/1500,bins=1000)
plt.show()
plt.hist(np.array(unzipped_list_compton)/1500,bins=1000)
plt.show()
plt.hist(np.array(unzipped_list)/1500,bins=1000)
plt.show()
###################################################
###################################################
###################################################
#%%
## now look at the qdc sums 
qdc_sums = np.array([np.sum(x) for x in array_qdc])
qdc_compton_sums = np.array([np.sum(x) for x in qdc_array_compton])
qdc_peak_sums = np.array([np.sum(x) for x in primary_peak])
#%%
print(len(qdc_compton_sums))
#%%
output_data == 1
#%%
#plt.hist(qdc_sums,bins=500,alpha=0.5,label="all")
plt.hist(qdc_compton_sums,bins=500,alpha=0.5,label="compton")
plt.hist(qdc_peak_sums,bins=100,alpha=0.5,label="peak")
plt.hist(qdc_sums[missed_data],bins=100,alpha=0.5,label="missed prediction at peak")

plt.xlim(0,30000)
plt.legend()
plt.show()
#%%
## now look at the qdc counts 
qdc_counts = np.array([len(x) for x in array_qdc])
qdc_compton_counts = np.array([len(x) for x in qdc_array_compton])
qdc_peak_counts = np.array([len(x) for x in primary_peak])
#%%
plt.hist(qdc_peak_counts,np.arange(0,20,1))
plt.show()
#%%
plt.hist(qdc_counts[missed_data],np.arange(0,20,1))
plt.show()
#%%
plt.hist(qdc_compton_counts,np.arange(0,20,1))
plt.show()
#%%
mean = np.mean(qdc_sums)
std = np.std(qdc_sums)
plt.hist((qdc_sums-mean)/std,bins=10000)
#%%
cdf = np.cumsum(bars)*0.5
plt.bar(bins[0:-1],cdf)

#%%
bars_compton, bins_compton = np.histogram(unzipped_list_compton,bins=np.arange(0,10000,0.5),normed=True)
#%%
plt.bar(bins[0:-1],bars)
plt.bar(bins_compton[0:-1],bars_compton,alph=0.)
#%%
mean_unzipped_list = np.mean(unzipped_list)
std_unzipped_list = np.std(unzipped_list)
#%%
qdc_array_normed = [(np.array(array) - mean_unzipped_list)/std_unzipped_list for array in df_qdc]
#%%
unzipped_list_normed = list(itertools.chain(*qdc_array_normed))
#%%
bars_normed, bins_normed = np.histogram(unzipped_list_normed,bins=np.arange(-3,5,0.5),normed=True)
#%%
plt.bar(bins_normed[0:-1],bars_normed)
# %%

trainset_index  = int(output_data.shape[0]*0.6)
valset_index    = int(output_data.shape[0]*0.8)
#%%
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=[0,1],
                                                 y=output_data)
class_weights[0] * (len(output_data[output_data == 0])/len(output_data)) * 2
class_weights[1]
class_weights[1] * (len(output_data[output_data == 1])/len(output_data)) * 2


#%%
# slice data
(output_data[output_data==1])