"""
-Read and process SIPM signals
-Read Data from root using read_root 
-Initialize Training Data and input Features 
-Save data as npz
"""
#%%
import numpy as np
import pandas as pd
from read_root import read_data

def get_delta_t_df(df_trig):
    """ 
    get df of diff betwenn max time in each event and the rest
    return df_delta
    """
    df_trig["min_time"] = np.min(df_trig,axis=1)
    df_trig = df_trig.apply(lambda x: x-x.min_time,axis=1)
    df_trig = df_trig.drop(columns="min_time")
    return df_trig

def normed_qdc_df(df_qdc):
    """"
    get df of normalized df qdc
    we got this number from the 90% of all data points 
    """
    df_qdc = df_qdc.apply(lambda x: x/26474.8,axis=1)
    return df_qdc
def _get_xz_index(id,shape,shift_x,shift_z):
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

def _get_xz_idx_shift(output_shape,padding):
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
    return array

def get_tensor_idx(id,output_shape,padding=0):
    """
    get idx of a tensor of output_shape+padding*2
    if padding=0 no zeros  
    input: array of sipm ids, output_shape 
    return: idx of id in output_shape+padding*2 shape tensor   
    """    
    shifts = _get_xz_idx_shift(output_shape,padding)
    ## lx uy
    if id < 112: 
        idx_x, idx_z = _get_xz_index(id,[4,28],shifts[0],shifts[2])
        idx_y = 1
    ## ux uy 
    ## shifted x
    elif id >=112 and id<368:
        id = id - 112
        idx_x, idx_z = _get_xz_index(id,[8,32],shifts[1],shifts[3])
        idx_y = 1
    ## lx ly
    elif id>367 and id <480:
        id = id - 368
        idx_x, idx_z = _get_xz_index(id,[4,28],shifts[0],shifts[2])
        idx_y = 0
    ## ux_ly
    else:
        id = id - 480
        idx_x, idx_z = _get_xz_index(id,[8,32],shifts[1],shifts[3])
        idx_y = 0
    idx_array = np.array((idx_x,idx_z,idx_y),dtype=int)
    return idx_array

def get_4dtensor_filled(df_qdc_array,df_trig_array,shape, df_tensor_idx,neg,bothneg):
    """
    get a filled tensor of QDC or trigger at respected positions
    neg and bothneg set true if we want to substitue untriggered SiPMs with -1
    can be done for Trigger (neg) or Trigger and QDC (bothneg)
    input: df of values, df of tensor_idx, event that needs to be filled
    output: tensor of shape 32*16*2 
    """
    full_tensor = np.zeros((shape))
    if neg == True:
        full_tensor[:,:,:,1] -= 1
    if bothneg == True:
        full_tensor[:,:,:,1] -= 1
        full_tensor[:,:,:,0] -= 1

    n_entries = len(df_qdc_array[~np.isnan(df_qdc_array)])      
    if n_entries == 0:
        return full_tensor
    qdc_array = df_qdc_array[0:n_entries].to_numpy()
    trig_array = df_trig_array[0:n_entries].to_numpy()
    qdc_trig_array = np.stack([qdc_array,trig_array],axis=-1)
    idx_array = np.stack(df_tensor_idx[0:n_entries],axis=1)
    full_tensor[(idx_array[0],idx_array[1],idx_array[2])] = qdc_trig_array
    return full_tensor 

def get_3dtensor_filled(df_qdc_array,shape, df_tensor_idx):
    """
    get a filled tensor of QDC or trigger at respected positions
    input: df of values, df of tensor_idx, event that needs to be filled
    output: tensor of shape 32*16*2 
    """
    full_tensor = np.zeros((shape))
    n_entries = len(df_qdc_array[~np.isnan(df_qdc_array)])
    if n_entries == 0:
        return full_tensor
    qdc_array = df_qdc_array[0:n_entries].to_numpy()
    idx_array = np.stack(df_tensor_idx[0:n_entries],axis=1)
    full_tensor[(idx_array[0],idx_array[1],idx_array[2])] = qdc_array
    return full_tensor 

def get_merged_tensor(tensor1, tensor2):
    """
    merge tensors of QDC and time trigger 
    input two array of sizer 32*16*2 
    output one array of size 32*16*2*2
    """

    full_tensor = np.stack([tensor1, tensor2],axis=-1)
    return full_tensor

def get_nentries_in_event(df_QDC):
    """  
    get number of entries in each event 
    input: df of events in row and enties in columns
    output number of non nan values in each row 
    """
    n_entries_array = np.apply_along_axis(lambda x: len(x[~np.isnan(x)]),1,df_QDC)
    return n_entries_array


def generate_training_data(path,shape=(12,32,2,2),qdc="Original",neg=True,bothneg=False):
    """
    Generate Training Data as 4d Tensor 
    input: Path of Root file, desired output shape, qdc Normed or not, 
            neg or bothneg to set untriggered SiPMs with -1 (neg only for trigger times, both neg for trig and QDC)
    output: 4d array of shape 12*32*2*2 
    """
   # read_data = read_data()
    df_ID = read_data.get_df_from_root(path,'SiPMData.fSiPMId')
    if qdc == "Original":
        df_QDC = read_data.get_df_from_root(path,'SiPMData.fSiPMQDC')
    elif qdc=="Normed":
        df_QDC = read_data.get_df_from_root(path,'SiPMData.fSiPMQDC')
        df_QDC = normed_qdc_df(df_QDC)
    else:
        raise(ValueError("Enter QDC form Normed or Original"))
    df_trig = read_data.get_df_from_root(path,'SiPMData.fSiPMTriggerTime')
    df_delta = get_delta_t_df(df_trig)
    df_tensor_idx = df_ID.applymap(lambda x: get_tensor_idx(x,shape[0:3],padding=0),na_action="ignore")
    full_tensor = np.array([get_4dtensor_filled(df_QDC.loc[i],df_delta.loc[i],shape,df_tensor_idx.loc[i],neg,bothneg) for i in range(len(df_QDC))])
    return full_tensor
    
def save_training_data(path,output,shape=(12,32,2,2),qdc="Original",neg=True,bothneg=False):
    tensor = generate_training_data(path,shape,qdc,neg,bothneg)
    np.savez(output,tensor)
    return None
    
def _is_points_in_scatterer_x(x, start_x, end_x):
    condition = ((start_x <= x).any() and (end_x >= x).any())
    if condition:
        return 1
    else: 
        return 0

def _is_points_in_absorber_x(x, start_x, end_x):

    condition = ((start_x <= x).any() and (end_x >= x).any())
    if condition:
        return 1
    else: 
        return 0

def get_compton_events(df_e):
    """ get normal compton events 
        condition that electron was induced
        rerturn pd series of energy 
    """
    compton_events = np.zeros(len(df_e))
    df_e[df_e["energy"] != 0]
    compton_events[np.where(df_e["energy"] != 0)[0]] = 1
    return compton_events


def get_complete_compton_targets(df_int_e,df_int_p,df_type,compton_events):
    """
    get complete compton events as 1 and 0s for all events 
    first conditioin that there was second interaction
    second condition that its compton event 
    input df_int_e and df_int_p and int_e (condition for compton event)
    output 1d numpy array
    """

    ## get e and p conditions
    # complete Compton event= Compton event + both e and p go through second interaction
    # 0 < p interaction < 10
    # 10 <= e interaction < 20
    e_int_len = get_nentries_in_event(df_int_e)
    p_int_len = get_nentries_in_event(df_int_p)
    condition_1 = e_int_len >= 1
    condition_2 = p_int_len >= 2
    condition_3 = ((df_int_p.loc[:,1:] > 0) & (df_int_p.loc[:,1:] < 10)).any(axis=1)
    condition_4 = (df_int_e.loc[:,0] >= 10) &(df_int_e.loc[:,0] < 20)  
    type2 = np.zeros(len(df_type))
    type2[df_type[0] == 2] = 1
    final =((condition_1 & condition_2) & (condition_3 & condition_4) & np.logical_and(compton_events, type2))
    final = final.astype(int)
    return final


def get_ideal_targets(df_pos_p_x,df_pos_e_x,df_int_p,complete_compton_events,detector_start,detector_end):
    df_reduced_p_x = df_pos_p_x.loc[complete_compton_events[complete_compton_events != 0].index].reset_index(drop=True)    
    df_reduced_e_x = df_pos_e_x.loc[complete_compton_events[complete_compton_events != 0].index]
    ## get first interaction pos for p
    df_reduced_p = df_int_p.loc[complete_compton_events[complete_compton_events != 0].index]
    ## for e we already know it's at pos 0 
    p_index_1 = np.apply_along_axis(lambda x: np.where(((x[1:]>0) & (x[1:]<10)) == True)[0][0] ,1, df_reduced_p)
    ## find all cases where photon posistion is not at 1 after first interaction
    ## +2 is because we use +1 from [1:] and +1 for undroped index
    pos_p_x_notnull_event = np.where(p_index_1 != 0)[0] 
    pos_p_x_notnull_int = p_index_1[p_index_1 != 0] + 1
    pos_p_x_null = np.where(p_index_1 == 0)[0] 
    ## get the pos of events not at the first pos
    test_1 = df_reduced_p_x.iloc[pos_p_x_notnull_event]
    a = []
    for i in range (len(test_1)):
        a.append(test_1.iloc[i,pos_p_x_notnull_int[i]]) 
    ## initialise pos of first electron interaction
    e_x = df_reduced_e_x.loc[:,0]
    ## initialise pos of first photon interaction
    p_x = np.zeros(len(df_reduced_p_x))
    p_x[test_1.index] = a
    p_x[pos_p_x_null] = df_reduced_p_x.iloc[pos_p_x_null,2]
    p_x = pd.Series(p_x,index=e_x.index)
    ## check pos of e and p in absorber and scatterer
    e_scat = np.logical_and(e_x >= detector_start[0],e_x <= detector_end[0]) 
    e_abs = np.logical_and(e_x >= detector_start[1],e_x <= detector_end[1]) 
    p_scat = np.logical_and(p_x >= detector_start[0],p_x <= detector_end[0]) 
    p_abs = np.logical_and(p_x >= detector_start[1],p_x <= detector_end[1]) 
    ## conditions that first interaction in 
    final_pe = np.logical_and(p_abs, e_scat).astype(int)
    final_ep = np.logical_and(e_abs, p_scat).astype(int)
    final_target = np.zeros(len(df_pos_p_x))
    final_target[final_pe[final_pe != 0].index] = final_pe[final_pe != 0].astype(int)
    final_target[final_ep[final_ep != 0].index] = final_ep[final_ep != 0].astype(int)
    return final_target



def get_distributed_targets(compton_targets, df_pos_p_x,start_x,end_x):
    """
    first condition: check if a photon interaction is both in scatt and abs 
    scond condition: complete compton event (CCE)
    note: in CCE we check whether there was a second interaction 
    so it suffies to check whether x is either in abs or scat
    """
    abs_numpy = np.apply_along_axis(_is_points_in_absorber_x,1 ,df_pos_p_x.to_numpy(),start_x[0],end_x[0])
    scatt_numpy = np.apply_along_axis(_is_points_in_scatterer_x,1 ,df_pos_p_x.to_numpy(),start_x[1],end_x[1])
    first_condition = (abs_numpy) & (scatt_numpy)
    second_condition = compton_targets
    final_condition = np.logical_and((first_condition) , (second_condition))
    final_condition = final_condition.astype(int)
    return final_condition 


def generate_target_data(path):
    df_e = read_data.get_df_from_root(path,root_entry_str="MCEnergy_e", col_name="energy")
    df_int_e = read_data.get_df_from_root(path,root_entry_str="MCInteractions_e")
    df_int_p = read_data.get_df_from_root(path,root_entry_str="MCInteractions_p")
    df_pos_p_x = read_data.get_df_from_root(path,root_entry_str="MCPosition_p",pos="fX")
    df_pos_e_x = read_data.get_df_from_root(path,root_entry_str="MCPosition_e",pos="fX")
    df_type = read_data.get_df_from_root(input_path, "MCSimulatedEventType")
    compton_events = get_compton_events(df_e)
    complete_compton_events = get_complete_compton_targets(df_int_e,df_int_p,df_type,compton_events)
    detector_start = [142.8, 254.8] 
    detector_end = [157.2, 285.2]
    targets =  get_ideal_targets(df_pos_p_x,df_pos_e_x,df_int_p,complete_compton_events,detector_start,detector_end)
    return targets
    
def save_target_data(path,output):
    tensor = generate_target_data(path)
    np.savez(output,tensor)
    return None

#%%
read_data = read_data()
input_path = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
output_path = r"C:\Users\georg\Desktop\master_arbeit\data\target_data_ideal_raster_38e8.npz"


#%%
df_e = read_data.get_df_from_root(input_path,root_entry_str="MCEnergy_e", col_name="energy")
df_int_e = read_data.get_df_from_root(input_path,root_entry_str="MCInteractions_e")
df_int_p = read_data.get_df_from_root(input_path,root_entry_str="MCInteractions_p")
df_pos_p_x = read_data.get_df_from_root(input_path,root_entry_str="MCPosition_p",pos="fX")
df_pos_e_x = read_data.get_df_from_root(input_path,root_entry_str="MCPosition_e",pos="fX")
df_type = read_data.get_df_from_root(input_path, "MCSimulatedEventType")
#%%
compton_events = get_compton_events(df_e)
complete_compton_events = get_complete_compton_targets(df_int_e,df_int_p,df_type,compton_events)

#%%
detector_start = [142.8, 254.8] 
detector_end = [157.2, 285.2]
#%%
final_ideal_targets = get_ideal_targets(df_pos_p_x,df_pos_e_x,df_int_p,complete_compton_events,detector_start,detector_end)
final_dist_targets = get_distributed_targets(complete_compton_events,df_pos_p_x,detector_start,detector_end)
#%%
df_int_p.loc[50][0:10]
df_int_e.loc[50]
df_pos_p_x.loc[50][0:10]
df_pos_e_x.loc[50]

