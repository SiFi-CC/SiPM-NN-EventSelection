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
    df_trig["max_time"] = np.max(df_trig,axis=1)
    df_trig = df_trig.apply(lambda x: x.max_time-x,axis=1)
    df_trig = df_trig.drop(columns="max_time")
    return df_trig


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

def get_4dtensor_filled(df_qdc_array,df_trig_array,shape, df_tensor_idx):
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


def generate_training_data(path,shape=(12,32,2,2)):
   # read_data = read_data()
    df_ID = read_data.get_df_from_root(path,'SiPMData.fSiPMId')
    df_QDC = read_data.get_df_from_root(path,'SiPMData.fSiPMQDC')
    df_trig = read_data.get_df_from_root(path,'SiPMData.fSiPMTriggerTime')
    df_delta = get_delta_t_df(df_trig)
    n_entries_array = get_nentries_in_event(df_QDC)
    df_tensor_idx = df_ID.applymap(lambda x: get_tensor_idx(x,shape[0:3],padding=0),na_action="ignore")
    full_tensor = np.array([get_4dtensor_filled(df_QDC.loc[i],df_delta.loc[i],shape,df_tensor_idx.loc[i]) for i in range(len(df_QDC))])
    return full_tensor
    
def save_training_data(path,output,shape=(12,32,2,2)):
    tensor = generate_training_data(path,shape)
    np.savez(output,tensor)
    return None
    
## condition for complete compton event: 
## for electron goes through a second interaction: 10 <= e interaction < 20 
## for electron secondary is the first interaction (since it's second triggered particle after electron)
## this is why we use  pos 0 
## for photon: 0 < p interaction < 10 
## and since any other hit after first is second, thereby condition [1:]

def _p_complete_comp_cond(interaction_array):
        """ check if a point was inside """
        condition_1 = ((interaction_array[1:-1] > 0 ) & (interaction_array[1:-1] < 10))
        condition_1 = condition_1.any()
        if (condition_1):
            return 1#, idx
        else:
            return 0#, 0 
def _e_complete_comp_cond(interaction_array):
    condition_1 = ((interaction_array[0] > 10 ) & (interaction_array[0] < 20))
    if (condition_1):# and (condition_2)):
        return 1
    else:
        return 0 

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
    df_e["compton"] = 0
    df_e.loc[ df_e.loc[:,"energy"] > 0, "compton"] = 1
    return df_e["compton"]

def get_complete_compton_targets(df_int_e,df_int_p,compton_events):
    """
    get complete compton events as 1 and 0s for all events 
    first conditioin that there was second interaction
    second condition that its compton event 
    input df_int_e and df_int_p and int_e (condition for compton event)
    output 1d numpy array
    """

    df_int_e["n_second_int"] = get_nentries_in_event(df_int_e)
    df_int_p["n_second_int"] = get_nentries_in_event(df_int_p)
    df_e_reduced = df_int_e[df_int_e["n_second_int"] != 0]
    ## get e and p conditions
    e_cond = df_e_reduced.apply(_e_complete_comp_cond,axis=1)
    df_p_reduced = df_int_p[df_int_p["n_second_int"] > 1]
    p_cond = df_p_reduced.apply(_p_complete_comp_cond,axis=1)
    ## e and p condition for a complete compton event
    all_condition = (compton_events & (e_cond & p_cond))
    target_zeros_ones = np.zeros((len(df_int_e)))
    target_zeros_ones[all_condition.index] = all_condition
    return target_zeros_ones


def get_targets(compton_targets, df_pos_p_x,start_x,end_x):
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
    df_e = read_data.get_df_from_root(path,root_entry_str="MCEnergyPrimary", col_name="energy")
    df_int_e = read_data.get_df_from_root(path,root_entry_str="MCInteractions_e")
    df_int_p = read_data.get_df_from_root(path,root_entry_str="MCInteractions_p")
    df_pos_p_x = read_data.get_df_from_root(path,root_entry_str="MCPosition_p",pos="fX")
    detector_pos_array, detector_thick_array = read_data.get_detector_geometry(path) 
    compton_events = get_compton_events(df_e)
    complete_compton_events = get_complete_compton_targets(df_int_e,df_int_p,compton_events)
    scatterer_start_x = detector_pos_array[0,0] - detector_thick_array[0,0] / 2 - 0.1
    scatterer_end_x = detector_pos_array[0,0] + detector_thick_array[0,0] / 2 + 0.1
    absorber_start_x = detector_pos_array[1,0] - detector_thick_array[1,0] / 2 - 0.1
    absorber_end_x = detector_pos_array[1,0] + detector_thick_array[1,0] / 2 + 0.1
    detector_start = [scatterer_start_x,absorber_start_x]
    detector_end = [scatterer_end_x,absorber_end_x]
    targets = get_targets(complete_compton_events,df_pos_p_x,detector_start,detector_end)
    return targets

def save_target_data(path,output,shape=(12,32,2,2)):
    tensor = generate_target_data(path)
    np.savez(output,tensor)
    return None
    
if __name__ == "__main__":
    read_data = read_data()
    input_path = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
    output_path = r""
    output_name_data = "target_data_Raster_3838.npz"
    output_name_targets = "training_data_Raster_3838.npz"
    output_name_data = output_path + output_name_data
    output_name_target = output_path + output_name_targets
    save_training_data(input_path,output_name_data)
    save_target_data(input_path,output_name_target)

