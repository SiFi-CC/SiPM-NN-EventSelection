#%%
import numpy as np
import pandas as pd 
import uproot
from read_root import read_data


## basic data exploration of the SiPMs structures
## find SiPMs in different halves or quarters 
## pos: scatterer=[150,0,0], absorber=[270,0,0]
## pos_dim: scatterer= [14,100,110], absorber=[30,100,126] 
## pos_dim_real: scatterer=[[-143,157]], {-51,51},[-55,60]],
##               absorber=[[255,285],{-51,51},[-63,64]]
## first lets check how the indexing work
## get index of pos of  x>/<200 y>/<0, z>/<0 (upper/lower)   
## this part only used to see how SiPMs are listed  
## also used to test the data later 

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
        array = array[~pd.isnull(array)]
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



def _get_ids(a, b, shape):
    """
    get sorted array of ids
    from a to b in input shape
    input: a and b int, desired shape
    output: array of int 
    """
    array = np.arange(a,b).reshape(shape)
    return array

def _get_shifted_array(ids_sorted,output_shape):
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
    lx_uy_ids = _get_ids(0,112,(4,28))
    lx_uy_ids_shifted = _get_shifted_array(lx_uy_ids, (shape_x_1,output_shape_z))
    ux_uy_ids = _get_ids(112,368,(8,32))
    ux_uy_ids_shifted = _get_shifted_array(ux_uy_ids, (shape_x_2,output_shape_z))
    lx_ly_ids = _get_ids(368,480,(4,28))
    lx_ly_ids_shifted = _get_shifted_array(lx_ly_ids, (shape_x_1,output_shape_z))
    ux_ly_ids = _get_ids(480,736,(8,32)) 
    ux_ly_ids_shifted = _get_shifted_array(ux_ly_ids, (shape_x_2,output_shape_z))
    uy = np.concatenate((lx_uy_ids_shifted, ux_uy_ids_shifted),axis=0)
    ly = np.concatenate((lx_ly_ids_shifted, ux_ly_ids_shifted),axis=0)
    tensor = np.stack([ly,uy], axis =2)
    return tensor

if __name__ == "__main__":
    path = r"C:\Users\georg\Desktop\master_arbeit\SiPMNNNewGeometry\FinalDetectorVersion_RasterCoupling_OPL_2e8protons.root"
    read_data = read_data()
    read_data.get_root_entry_str_list(path)
    ## call data 
    df_x = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fX')
    df_y = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fY')
    df_z = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMPosition',pos='fZ')
    df_id = read_data.get_df_from_root(path=path, root_entry_str='SiPMData.fSiPMId')
    ## check to which quarter and half it belongs and get index
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
    ## get values of x,y,z in each quarter in the data 
    ## we do so to see how the ids are counted in the sipms
    yval_ux_uy, yval_ux_ly, yval_lx_uy, yval_lx_ly = get_unique_values(df_y, ux_uy, ux_ly, lx_uy, lx_ly)
    xval_ux_uy, xval_ux_ly, xval_lx_uy, xval_lx_ly = get_unique_values(df_x, ux_uy, ux_ly, lx_uy, lx_ly)
    zval_ux_uy, zval_ux_ly, zval_lx_uy, zval_lx_ly = get_unique_values(df_z, ux_uy, ux_ly, lx_uy, lx_ly)
    id_ux_uy, id_ux_ly, id_lx_uy, id_lx_ly = get_unique_values(df_id, ux_uy, ux_ly, lx_uy, lx_ly)
    df_x_lx_uy = get_values(df_x,lx_uy)
    df_z_lx_uy = get_values(df_z,lx_uy)
    df_id_lx_uy = get_values(df_id,lx_uy)
