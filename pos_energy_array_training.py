"""
Regression Data:  
- Discover Data and Prepare Input   
- from the index of position in the function:
     generate_tensor_from_root.generate_ideal_target_data()
     the index of the second interaction pos of e and p are generated
- also save energy for regression 

"""
#%%
import numpy as np 
from read_root import read_data
import matplotlib.pyplot as plt

#%%
def get_pos_array(pos_p_x,pos_p_y,pos_p_z,idx_pos_p,compton_idx):
     pos_p_x = np.array(pos_p_x,dtype=object)[compton_idx]
     pos_p_z = np.array(pos_p_z,dtype=object)[compton_idx]
     pos_p_y = np.array(pos_p_y,dtype=object)[compton_idx]
     idx_pos_p = np.int16(idx_pos_p)[compton_idx]
     pos_vector = np.zeros((len(pos_p_x),3))
     for i, idx in enumerate(idx_pos_p):
     #     print(pos_p_x[i,idx]) 
          pos_vector[i,:] = np.array([pos_p_x[i][idx], pos_p_y[i][idx], pos_p_z[i][idx]])
     return pos_vector
#%%
## get e, p positions
path_root = r"insert_root_path"
get_data = read_data()
pos_p_x = get_data.get_array_from_root(path_root,"MCPosition_p",pos="fX")
pos_p_y = get_data.get_array_from_root(path_root,"MCPosition_p",pos="fY")
pos_p_z = get_data.get_array_from_root(path_root,"MCPosition_p",pos="fZ")

pos_e_x = get_data.get_array_from_root(path_root,"MCPosition_e",pos="fX")
pos_e_y = get_data.get_array_from_root(path_root,"MCPosition_e",pos="fY")
pos_e_z = get_data.get_array_from_root(path_root,"MCPosition_e",pos="fZ")
#%%
## get e, p energy: 
get_data = read_data()
energy_e = get_data.get_array_from_root(path_root,"MCEnergy_e")
energy_p = get_data.get_array_from_root(path_root,"MCEnergy_p")
#%%
## load indices
path_index_e = r"insert_path_of_index_electron_pos"
path_index_p = r"insert_path_of_index_photon_pos"
path_target_data = r"insert_path_of_target_data"

idx_pos_p = np.load(path_index_e)
idx_pos_e = np.load(path_index_p)
output_data = np.load(path_target_data)


#%%
idx_pos_p = idx_pos_p['arr_0']
idx_pos_e = idx_pos_e['arr_0']
output_data = output_data["arr_0"]
compton_idx = np.where(output_data == 1)[0]
#%%
pos_array_p = get_pos_array(pos_p_x,pos_p_y,pos_p_z,idx_pos_p,compton_idx)
pos_array_e = get_pos_array(pos_e_x,pos_e_y,pos_e_z, idx_pos_e,compton_idx)
pos_array_ep = np.concatenate([pos_array_e,pos_array_p],axis=1)


#%%
save_path = r"test_path"
save_name_pos = r"test_name"
save_name_energy = r"test_name"

np.savez(save_path+save_name_pos,pos_array_ep)
np.savez(save_path+save_name_energy,pos_array_ep)
