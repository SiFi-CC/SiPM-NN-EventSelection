"""
-Read and process SIPM signals
-After Reading Save as Parquet
-This is for Test Data 
-Should be generalized for all training data 
-Use Uproot instead of ROOT 

"""
#%%
import matplotlib.pyplot as plt 
from matplotlib import projections
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import uproot 
#%%
#####################################################
### first part: read files with uproot ##############
### write them to pandas df #########################

with uproot.open(r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\ExampleDataFile.root:Events;1") as file:
    event_type = file['MCSimulatedEventType'].arrays()["MCSimulatedEventType"].tolist()
    compton_pos = file['MCComptonPosition'].arrays()['MCComptonPosition'].tolist()
    source_pos = file['MCPosition_source'].arrays()['MCPosition_source'].tolist()

    e_primary = file['MCEnergyPrimary'].arrays()['MCEnergyPrimary'].tolist()
    energy_e = file['MCEnergy_e'].arrays()['MCEnergy_e'].tolist()
    interaction_e = file['MCInteractions_e'].arrays()['MCInteractions_e'].tolist()
    energy_p = file['MCEnergy_p'].arrays()['MCEnergy_p'].tolist()
    interaction_p = file['MCInteractions_p'].arrays()['MCInteractions_p'].tolist()
    ###########################################################
    ## sipm data 
    trig = file['SiPMData.fSiPMTriggerTime'].arrays()['SiPMData.fSiPMTriggerTime'].tolist()
    IDs = file['SiPMData.fSiPMId'].arrays()['SiPMData.fSiPMId'].tolist()
    bits = file['SiPMData.fBits'].arrays()['SiPMData.fBits'].tolist()
    QDC = file['SiPMData.fSiPMQDC'].arrays()['SiPMData.fSiPMQDC'].tolist()
    f_x = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition']['fX'].tolist()
    f_y = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition']['fY'].tolist()
    f_z = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition']['fZ'].tolist()
##
## extract scatterer and absorber positions: 
with uproot.open(r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\ExampleDataFile.root:Setup;1") as file:  
    scatter_pos = file["ScattererPosition"].arrays()["ScattererPosition"].tolist()
    absorber_pos = file["AbsorberPosition"].arrays()["AbsorberPosition"].tolist()
    absorber_thick_x = file["AbsorberThickness_x"].arrays()["AbsorberThickness_x"].tolist()
    absorber_thick_y = file["AbsorberThickness_y"].arrays()["AbsorberThickness_y"].tolist()
    absorber_thick_z = file["AbsorberThickness_z"].arrays()["AbsorberThickness_z"].tolist()

    scatter_thick_x = file["ScattererThickness_x"].arrays()["ScattererThickness_x"].tolist()
    scatter_thick_y = file["ScattererThickness_y"].arrays()["ScattererThickness_y"].tolist()
    scatter_thick_z = file["ScattererThickness_z"].arrays()["ScattererThickness_z"].tolist()
#%%
def get_pos_array(pos_dict): 
    pos_dict = list(pos_dict[0].values())
    pos_dict = np.array(pos_dict)
    return pos_dict

## get absorber position 
scatterer_thick = np.concatenate([scatter_thick_x, scatter_thick_y, scatter_thick_z])
absorber_thick = np.concatenate([absorber_thick_x,absorber_thick_y,absorber_thick_z])


scatter_pos_array = get_pos_array(scatter_pos)
print(scatter_pos_array)
absorber_pos_array = get_pos_array(absorber_pos)
detector_pos = np.concatenate([[scatter_pos_array],[absorber_pos_array]])

#%%
##get arrays as dataframe 

df_comp_pos = pd.DataFrame(data=compton_pos)
df_src_pos = pd.DataFrame(data=source_pos)
df_x = pd.DataFrame(data=f_x)
df_y = pd.DataFrame(data=f_y)
df_z = pd.DataFrame(data=f_z)
df_trig = pd.DataFrame(data=trig)
df_QDC = pd.DataFrame(data=QDC)
df_id = pd.DataFrame(IDs)
df_bits = pd.DataFrame(data=bits)

## documenting them from jonas notes 
""" 
 event type encoding: 
 NOTDEFINED=-1,
 ALLEVENTS=0,ALLCOINCIDENCES=1,REALCOINCIDENCESONLY=2,
 RANDOMCOINCIDENCESONLY=3,SINGLEMODULEONLY=4,REALCOINCIDENCEPILEUP=5,
 RANDOMCOINCIDENCEPILEUP=6 
"""
df_type = pd.DataFrame(data=event_type,columns=["type"])

""" 
  if the Compton effect was the first interaction in the detector MCEnergy_e !=0
  it is the energy of the primary photon that underwent the compton effect
  when it is emitted.
  If this condition is wrong it is the sum of all primary particles interacting in the detector in this event 
"""
df_primary = pd.DataFrame(data=e_primary,columns=["energy"])
"""
 energy of the electron accelerated in the Compton effect
 only !=0 if Compton effect was first interaction in the detector
"""
df_e = pd.DataFrame(data=energy_e,columns=["energy"])
df_int_e = pd.DataFrame(data=interaction_e)
"""
 kinetic energy of the photon undergoing the Compton effect -
 only !=0 if Compton effect was first interaction in the detector
"""
df_p = pd.DataFrame(data=energy_p,columns=["kinetic"])
df_int_p = pd.DataFrame(data=interaction_p)

"""
from awal thesis, conditions for compton events
if e_energy != 0 (condition for compton event)
and e, p go through second detector 
(condition for complete compton event = 
Compton event + both e and p go through a second interation with two condition listed )
i.e: 0 < p interaction < 10 
and 10 <= e interaction < 20 
        
        # complete Compton event= h
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
"""


#######################################################################
#######################################################################
#######################################################################
#### second part: find the idx for all detector and reshape ###########
#### first research pos for ids and idx of all detectors ##############
#### then use what we found to get the index from ids directly ########
#### then use this to automate this process ###########################
#### return: API to get the index in 3dtensor for each event ##########
#### return: API to automate this process for all events ############## 
#######################################################################

## dimensions: absorber: 32*8*2, scatterer: 28*4*2, z*x*y
## pos: scatterer=[150,0,0], absorber=[270,0,0]
## pos_dim: scatterer= [14,100,110], absorber=[30,100,126] 
## pos_dim_real: scatterer=[[-143,157]], {-51,51},[-55,60]],
##               absorber=[[255,285],{-51,51},[-63,64]]
#######################################################################
## first lets check how the indexing work
## get index of pos of  x>/<200 y>/<0, z>/<0 (upper/lower)   

def get_df_half_idx(df, u_l, limit):
    """
    get idx of tirgered SiPM in "upper" or "lower"
    input: Dataframe, u_l str: "upper" or "lower"
           limit x->200 y->0 z->0          
    return pd.Series of list of indices per index 
    """
    if u_l == "upper":
        df_new = df.apply(lambda x:np.where(x > limit)[0],axis=1)
    elif u_l == "lower":
        df_new = df.apply(lambda x:np.where(x < limit)[0],axis=1)
    else:
        raise ValueError(r"Enter \"upper\" or \"lower\"")
    return df_new

def get_df_qrt_idx(pos1, pos2):
    """
    get idx of tirgered SiPM in two interesected halfs
    input Dataframe of two half indices
    return pd.Series of list of indices per index 
    """
    df_ux_uy = pd.DataFrame(data=[pos1, pos2]).T
    ux_uy = df_ux_uy.apply(lambda x:np.intersect1d(x[0],x[1]),axis=1)
    return ux_uy

def get_values(df, ux_uy):
    """
    get values of df in a specific  (or half)
    input: DataFrame and array of index of quarter
    output: DataFrame with values only in one quarter
    """
    df_new = df.copy()
    df_new["ux_uy"] = ux_uy
    df_new = df_new.apply(lambda x: x.loc[x.ux_uy],axis=1)
    return df_new

## kind of inefficient 
## but not so important, just to check 
def get_unique_values(df, ux_uy, ux_ly, lx_uy, lx_ly):
    """
    find all unique values in a dataframe per quarter
    input: Dataframe, idx of quarter 
    return: tuple unique values of df in all for directions
    """
    def get_clean(array):
        array = array.explode().unique()
        array = array[array > -10000000]
        return np.sort(array)
    
    df_new = df.copy()
    df_new["ux_uy"] = ux_uy
    df_new["ux_ly"] = ux_ly
    df_new["lx_uy"] = lx_uy
    df_new["lx_ly"] = lx_ly
    xval_ux_uy = df_new.apply(lambda x: x.loc[x.ux_uy].unique(),axis=1)
    xval_ux_ly = df_new.apply(lambda x: x.loc[x.ux_ly].unique(),axis=1)
    xval_lx_ly = df_new.apply(lambda x: x.loc[x.lx_ly].unique(),axis=1)
    xval_lx_uy = df_new.apply(lambda x: x.loc[x.lx_uy].unique(),axis=1)
    xval_ux_uy = get_clean(xval_ux_uy)
    xval_ux_ly = get_clean(xval_ux_ly)
    xval_lx_ly = get_clean(xval_lx_ly)
    xval_lx_uy = get_clean(xval_lx_uy)
    return xval_ux_uy, xval_ux_ly, xval_lx_uy, xval_lx_ly

pos_upper_x = get_df_half_idx(df_x, "upper", 200)
pos_lower_x = get_df_half_idx(df_x, "lower", 200)

pos_upper_y = get_df_half_idx(df_y, "upper", 0)
pos_lower_y = get_df_half_idx(df_y, "lower", 0)

pos_upper_z = get_df_half_idx(df_z, "upper", 0)
pos_lower_z = get_df_half_idx(df_z, "lower", 0)


ux_uy = get_df_qrt_idx(pos_upper_x,pos_upper_y)
ux_ly = get_df_qrt_idx(pos_upper_x,pos_lower_y)
lx_uy = get_df_qrt_idx(pos_lower_x,pos_upper_y)
lx_ly = get_df_qrt_idx(pos_lower_x,pos_lower_y)

yval_ux_uy, yval_ux_ly, yval_lx_uy, yval_lx_ly = get_unique_values(df_y, ux_uy, ux_ly, lx_uy, lx_ly)
xval_ux_uy, xval_ux_ly, xval_lx_uy, xval_lx_ly = get_unique_values(df_x, ux_uy, ux_ly, lx_uy, lx_ly)
zval_ux_uy, zval_ux_ly, zval_lx_uy, zval_lx_ly = get_unique_values(df_z, ux_uy, ux_ly, lx_uy, lx_ly)
id_ux_uy, id_ux_ly, id_lx_uy, id_lx_ly = get_unique_values(df_id, ux_uy, ux_ly, lx_uy, lx_ly)

df_x_lx_uy = get_values(df_x,lx_uy)
df_z_lx_uy = get_values(df_z,lx_uy)
df_id_lx_uy = get_values(df_id,lx_uy)


#%%
######################################################################
## after checking the data we can see that there is an order for the ids
## the ids of the SiPMs are ordered as: lxuy, uxuy, lxly, uxly
## dimension (28*4),(32*8),(28*4),(32*8) (scat, abs, scat,abs)
## with the zeroth SiPM ID starting with negative z 
#####################################################################
#### we will use now the info from the data exploration 
#### we know how the ids are ordered now 
#### so we can assign ids directly into correct pos of 3d tensor
#### output pos (32(8*4)*16(8*2)*2)
#### create API to assign ids directly into desired shape
#### there are two possible shape 32*16*2 and 32*12*tensor
#### API to assign QDC, trigger time pairs to 3d tensor
#########################################################
#### scatterer dim (28(7*4)*4*2), absorber pos (32(8*4)*4*2)
#### 32(8*4) z axis means 8 Arrays with 4 SiPMs
#### scatt z=28 should be reshaped to 32 (first and last two rows zeros)
#### scatt x=4 should be reshaped to 8 (first and last two  rows zeros)
#### final shape of x will be 16 but last 8 are for absorber
#### absorber z=32 keep
#### absorber x=8 -> shift z values by 8 so they are aligned 
#### with shape 32*16*2 in x direction  

##########################################
## to do: 
## 1- create 3d tensors of shape 32*16*2 done 
## 2- reshpe the ux detector so it's size 32*16*2 done
## 3- add the QDC to the right tensor  done
## 4- merge two tensors 3d of delta and qdc
## 5- (maybe?) norm QDC
## 6-   
def get_xz_index(id,shape,shift_x,shift_z):
    """
    get idx from SiPM id
    for 32*8 or 28*4 shapes
    input: id:int, input_shape:tuple(x,z),shift_x:int, shift_z:int
    return tuple of index in x,z 
    """
    if shape[0] == 8 and shape[1] == 32: 
        idx_x1 = np.floor(id/32)
        idx_z1 = id - (idx_x1 * 32)
    if shape[0] == 4 and shape[1] == 28:
        idx_x1 = np.floor(id/28) 
        idx_z1 = id - (idx_x1 * 28)
    idx_x1 += shift_x
    idx_z1 += shift_z
    return idx_x1, idx_z1 

def get_xz_idx_shift(output_shape,padding):
    """  
    get shift of idx of x,z with pad for 2 diff. output shpaes
    input: output_shape 16*32 or 12*32, and possible zero pading
    return: array to shifts in x,z directions
    """
    if output_shape[0] == 16: 
        to_add_x_4 = 2  # add 2 zeros from one side, 2 added to other auto.
        to_add_x_8 = 8  # add 8 zeros from one side
        to_add_z_28 = 2 # 2 zeros from one side 
        to_add_z_32 = 0 # no need to add -> output 32*16*2
    elif output_shape[0] == 12:
        to_add_x_4 = 0  # no need to add
        to_add_x_8 = 4 # add 4 zeros
        to_add_z_28 = 2 # 2 zerso from one side 
        to_add_z_32 = 0 # no need to add -> output 32*12*2
    else:
        raise ValueError("Shape must be 16*32 or 12*32")
    array = np.array([to_add_x_4, to_add_x_8,to_add_z_28,to_add_z_32])
    array = array + [padding,(padding*3),padding,padding]
    #print(array)
    return array


def get_tensor_idx(id,output_shape,padding=0):
    """
    get idx of a tensor of output_shape+padding*2
    if padding=0 no zeros  
    input: array of sipm ids, output_shape 
    return: idx of id in output_shape+padding*2 shape tensor   
    """ 
    shifts = get_xz_idx_shift(output_shape,padding)
    ## lx uy
    if id < 112: 
        idx_x, idx_z = get_xz_index(id,[4,28],shifts[0],shifts[2])
        idx_y = 1
    ## ux uy 
    ## shifted x
    elif id >=112 and id<368:
        id = id - 112
        idx_x, idx_z = get_xz_index(id,[8,32],shifts[1],shifts[3])
        idx_y = 1
    ## lx ly
    elif id>367 and id <480:
        id = id - 368
        idx_x, idx_z = get_xz_index(id,[4,28],shifts[0],shifts[2])
        idx_y = 0
    ## ux_ly
    else:
        id = id - 480
        idx_x, idx_z = get_xz_index(id,[8,32],shifts[1],shifts[3])
        idx_y = 0
    idx_array = np.array((idx_x,idx_z,idx_y),dtype=int)
    return idx_array

def get_ids(a, b, shape):
    """
    get sorted array of ids
    from a to b in input shape
    input: a and b int, desired shape
    output: array of int 
    """
    array = np.arange(a,b).reshape(shape)
    return array

def get_shifted_array(ids_sorted,output_shape):
    """
    get array of size 28*4 shifter by 2
    first/last two rows/cols of zeros added 
    input: array of size 28*4
    output: array of size 32*8  
    """
    ids = np.zeros(output_shape)
    x_input_shape, z_input_shape = ids_sorted.shape
    x_start = int(np.ceil((output_shape[0]  - x_input_shape) / 2))
    x_end = output_shape[0] - x_start
    z_start = int(np.ceil((output_shape[1]  - z_input_shape) / 2))
    z_end = output_shape[1] - z_start
    ids[x_start:x_end,z_start:z_end] = ids_sorted
    return ids

def get_ids_tensor(output_shape,padding=0):
    """
    get tensor of ids in given output shape 
    output shape is output_shape + padding * 2
    input output_shape (tuple) and padding int 
    output 3d tensor in shape (32+padding*2,12or16+padding*2,2) 
    """
    output_shape_z = output_shape[1] + padding * 2
    if output_shape[0] == 12:
            shape_x_1 = 4 + padding * 2
            shape_x_2 = 8 + padding * 2
    elif output_shape[0] == 16:
            shape_x_1 = 8 + padding * 2
            shape_x_2 = 8 + padding * 2
    else: 
        raise ValueError("enter output shapes of 12*32 or 16*32")
    #print(shape_x_1, shape_x_2)
    lx_uy_ids = get_ids(0,112,(4,28))
    lx_uy_ids_shifted = get_shifted_array(lx_uy_ids, (shape_x_1,output_shape_z))
    ux_uy_ids = get_ids(112,368,(8,32))
    ux_uy_ids_shifted = get_shifted_array(ux_uy_ids, (shape_x_2,output_shape_z))
    lx_ly_ids = get_ids(368,480,(4,28))
    lx_ly_ids_shifted = get_shifted_array(lx_ly_ids, (shape_x_1,output_shape_z))
    ux_ly_ids = get_ids(480,736,(8,32)) 
    ux_ly_ids_shifted = get_shifted_array(ux_ly_ids, (shape_x_2,output_shape_z))
    uy = np.concatenate((lx_uy_ids_shifted, ux_uy_ids_shifted),axis=0)
    ly = np.concatenate((lx_ly_ids_shifted, ux_ly_ids_shifted),axis=0)
    tensor = np.stack([ly,uy], axis =2)
    return tensor

def get_delta_t_df(df_trig):
    """ 
    get df of diff betwenn max time in each event and  
    
    """
    df_trig["max_time"] = df_trig.apply(np.max,axis=1)
    df_trig = df_trig.apply(lambda x: x.max_time-x,axis=1)
    df_trig = df_trig.drop(columns="max_time")
    return df_trig

def get_nentries_in_event(df_QDC=df_QDC):
    """  
    get number of entries in each event 
    input: df of events in row and enties in columns
    output number of non nan values in each row 
    """
    n_entries_array = df_QDC.apply(lambda x: len(x[~x.isnull()]),axis=1)
    return n_entries_array
    
def get_tensor_filled(df_QDC,shape,n_entries_list, df_tensor_idx,full_tensor):
    """
    get a filled tensor of QDC or trigger at respected positions
    input: df of values, df of tensor_idx, event that needs to be filled
    output: tensor of shape 32*16*2 
    """
    event = df_QDC.name
    n_entries = n_entries_list[event]
    qdc_array = df_QDC[0:n_entries]
    idx_array = np.stack(df_tensor_idx[0:n_entries],axis=1)
    full_tensor[(idx_array[0],idx_array[1],idx_array[2])] = qdc_array
    return full_tensor 
#%%
x = np.stack(df_tensor_idx_16.loc[0,0:6],axis=1)
x[0:6]
#%%
entries = get_nentries_in_event()
trig_tensor_16 = df_QDC.apply(lambda x: get_tensor_filled(x,(16,32,2),entries,df_tensor_idx_16.loc[x.name]),axis=1)

#%%
def get_merged_tensor(tensor1, tensor2):
    """
    merge tensors of QDC and time trigger 
    input two array of sizer 32*16*2 
    output one array of size 32*16*2*2
    """

    full_tensor = np.stack([tensor1, tensor2],axis=-1)
    return full_tensor 
#%%

## get idx in 3d tensor of each id in ids dataframe
df_tensor_idx = df_id.applymap(lambda x:get_tensor_idx(x,(12,32),0),na_action="ignore")
df_tensor_idx = df_tensor_idx.astype
## now fill qdc values in right positions 
df_delta = get_delta_t_df(df_trig)
df_delta
##
df_tensor_idx_16 =  df_id.applymap(lambda x:get_tensor_idx(x,(16,32),0),na_action="ignore")
df_tensor_idx_12 =  df_id.applymap(lambda x:get_tensor_idx(x,(12,32),0),na_action="ignore")
##
#%%
df_QDC.loc[512:] = df_QDC
df_id.loc[512:] = df_id
#%%
import time 
start_time = time.time()
n_entries_list = get_nentries_in_event()
#df_QDC2

df_tensor_idx_16 =  df_id.applymap(lambda x:get_tensor_idx(x,(16,32),0),na_action="ignore")
trig_tensor_16 = df_delta.apply(lambda x: get_tensor_filled(x,(16,32,2),entries, df_tensor_idx_16.loc[x.name]),axis=1)
qdc_tensor_16 = df_QDC.apply(lambda x: get_tensor_filled(x,(16,32,2),entries, df_tensor_idx_16.loc[x.name]),axis=1)
#qdc_tensor_12 = df_QDC.apply(lambda x: get_tensor_filled(x,(12,32,2),df_tensor_idx_12),axis=1)
end_time = time.time()
print("it took ",(time.time()-start_time))
#%%
array_test = np.zeros((32*12*2)).reshape(12,32,2)
idx = df_tensor_idx_16.loc[0].dropna()
#%%
np.stack(idx)
#%%
array_test[idx]
#%%
trig_tensor_16 = df_delta.apply(lambda x: get_tensor_filled(x,(16,32,2),df_tensor_idx_16.loc[x.name]),axis=1)

#%%
trig_tensor_16 = df_delta.apply(lambda x: df_tensor_idx_16.loc[x.name],axis=1)
#%%
df_tensor_idx_16.loc[0]
#%%
## test that
def test_ids_match():
        events = np.arange(0,512,1)
        dimensions = ((16,32),(12,32))
        j = 0
        for shape in dimensions:
            ids_tensor = get_ids_tensor(shape,0)
            df_tensor_idx = df_id.applymap(lambda x:get_tensor_idx(x,shape,0),na_action="ignore")
   #         qdc_tensor = df_QDC.apply(lambda x: get_tensor_filled(x,(12,32,2)),axis=1)

            for pos in events:     
                test_array=df_tensor_idx.loc[pos]
                n_entries = len(test_array[~test_array.isnull()]) 
                for i,v in enumerate(test_array):
                    try:
                        x = int(v[0])
                        z = int(v[1])
                        y = int(v[2])
                        if ids_tensor[x,z,y] == df_id.loc[pos,i] and i == n_entries - 1:
                            print(ids_tensor[x,z,y] ,df_id.loc[pos,i])
                            j += 1
                    except:
                            pass
            print(f"################## j is equal to {j} for dim {shape} #################")
            j = 0
def test_qdc_match():
        events = np.arange(0,512,1)
        dimensions = ((16,32),(12,32))
        j = 0
        for shape in dimensions:
            ids_tensor = get_ids_tensor(shape,0)
            df_tensor_idx = df_id.applymap(lambda x:get_tensor_idx(x,shape,0),na_action="ignore")
            shape_full = shape + (2,)
            print(shape_full)
            qdc_tensor = df_QDC.apply(lambda x: get_tensor_filled(x,shape_full,df_tensor_idx),axis=1)
            for pos in events:     
                test_array=df_tensor_idx.loc[pos]
                n_entries = len(test_array[~test_array.isnull()]) 

                for i,v in enumerate(test_array):
                    try:
                        x = int(v[0])
                        z = int(v[1])
                        y = int(v[2])
                        #print(qdc_tensor.loc[pos][x,z,y], df_QDC.loc[pos,i] )
                        if qdc_tensor.loc[pos][x,z,y] == df_QDC.loc[pos,i] and i == n_entries - 1:
                            j += 1
                    except:
                            pass

            print(f"################## j is equal to {j} for dim {shape} #################")
            j = 0
                #    except:
        #        return f"Error in columns {i} of event {pos}"

test_qdc_match()
#%%
df_tensor_idx = df_id.applymap(lambda x:get_tensor_idx(x,(16,32),0),na_action="ignore")
qdc_tensor = df_QDC.apply(lambda x: get_tensor_filled(x,(16,32,2)),axis=1)
#%%

#%%
#########################
######## plots ##########
#########################

########################
## get plots of the positions
df_x_null = df_x.fillna(0)
df_y_null = df_y.fillna(0)
df_z_null = df_z.fillna(0)
#%%
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
# plot hit detector from all events 
idx = 1
# To create a scatter graph
ax.scatter(df_z_null.loc[:,idx], df_y_null.loc[:,idx], df_x_null.loc[:,idx], c=df_QDC.loc[:,idx])
# trun off/on axis
plt.axis('off')
# show the graph
plt.show()
#%%
## plot event
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
idx = 95
ax.scatter(df_x_null.loc[idx,:], df_y_null.loc[idx,:], df_z_null.loc[idx,:], c=df_QDC.loc[idx,:])
plt.axis('on')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.show()

#%%
## plot everything 
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter(df_x, df_y, df_z)
# for i, txt in enumerate(df_z.index):
#     ax.annotate(txt, (df_x.loc[i], df_y.loc[i],df_z.loc[i]))
plt.axis('on')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
#%%
## more plots 
## plot grid of SiPM with color as the count of occurences
id_counts = df_id.apply(pd.value_counts,axis=1).sum(axis=0)
id_counts_norm = id_counts / np.sum(id_counts)

id_counts_norm_dict = id_counts_norm.to_dict()
id_counts_dict = id_counts.to_dict()
print(id_counts)

df_normed = df_id.replace(id_counts_norm_dict)
df_counts = df_id.replace(id_counts_dict)

df_x_null = df_x.fillna(0)
df_y_null = df_y.fillna(0)
df_z_null = df_z.fillna(0)
########################################################
import matplotlib as mpl

fig = plt.figure(figsize=(30, 10)) 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
ax.set_title("Triggered SiPMs")
ax.scatter(df_x_null,df_y_null,df_z_null, c=df_counts.fillna(0),cmap="summer")
ax.set_xlim(120,300)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text(200, 30, 90, "Dim 28*4*2       Dim: 32*8*2 ")
#fig.colorbar(mpl.cm.ScalarMappable(cmap="summer"), ax=ax)
plt.savefig("3dtesndor.pdf")
plt.show()
#%%
## plot quarter
#plot everything 
fig = plt.figure(figsize=(10, 10)) 
# Generating a 3D sine wave
ax = plt.axes()
ax.scatter(get_values(df_x,ux_uy), get_values(df_z,ux_uy), c=get_values(df_counts,ux_uy),cmap="Greens")
plt.plot()
#%%
df_id.apply(pd.value_counts,axis=1)
#%%218
ids = id_counts.index.to_numpy()
ids_full = np.array()
#%%
#print(id_lx_uy)