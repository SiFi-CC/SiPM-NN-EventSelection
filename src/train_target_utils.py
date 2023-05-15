

import numpy as np
import pandas as pd

class Detector:
    
    def __init__(self,shape=(16,32,2,2),padding=0):
        self.shape = shape
        self.padding = padding
    
        
    def is_point_in_scatterer_x(self, x: np.array, start_x: float, end_x: float):
        """check if event x dimesnion inside scattere 

        Args:
            x (np.array): array of entries in an event
            start_x (float): start point
            end_x (float): end point

        Returns:
            int: 1 or 0 if any point in event in scatterer
        """
        condition = ((start_x <= x) and (end_x >= x))
        if condition:
            return 1
        else: 
            return 0

    def is_point_in_absorber_x(self, x, start_x, end_x):
        """check if event x dimesnion inside absorber 

        Args:
            x (np.array): array of entries in an event
            start_x (float): start point
            end_x (float): end point

        Returns:
            int: 1 or 0 if any point in event in absorber
        """
        condition = ((start_x <= x) and (end_x >= x))
        if condition:
            return 1
        else: 
            return 0

    def get_xz_index(self,  id, reduced_shape, shift_x, shift_z):
        """
        get idx from SiPM id
        for 32*8 or 28*4 shapes
        input: id:int, input_shape:tuple(x,z),shift_x:int, shift_z:int
        return tuple of index in x,z 
        """
        if reduced_shape[0] == 8 and reduced_shape[1] == 32: 
            idx_x1 = np.floor(id/32)
            idx_z1 = id - (idx_x1 * 32)
            idx_x1 += shift_x
            idx_z1 += shift_z
            return idx_x1, idx_z1 
        elif reduced_shape[0] == 4 and reduced_shape[1] == 28:
            idx_x1 = np.floor(id/28) 
            idx_z1 = id - (idx_x1 * 28)
            idx_x1 += shift_x
            idx_z1 += shift_z
            return idx_x1, idx_z1 

    def get_xz_idx_shift(self,shape=(16,32,2,2),padding=0):
        """  
        get shift of idx of x,z with pad for 2 diff. output shpaes
        input: output_shape 16*32 or 12*32, and possible zero pading
        return: array to shifts in x,z directions
        """
        self.shape = shape
        self.padding = padding
        if self.shape[0] == 16: 
            to_add_x_4 = 0  # add 2 zeros from one side, 2 added to other auto.
            to_add_x_8 = 8  # add 8 zeros from one side
            to_add_z_28 = 2 # 2 zeros from one side 
            to_add_z_32 = 0 # no need to add -> output 32*16*2
        elif self.shape[0] == 12:
            to_add_x_4 = 0  # no need to add
            to_add_x_8 = 4 # add 4 zeros
            to_add_z_28 = 2 # 2 zerso from one side 
            to_add_z_32 = 0 # no need to add -> output 32*12*2
        else:
            raise ValueError("Shape must be 16*32 or 12*32")
        array = np.array([to_add_x_4, to_add_x_8,to_add_z_28,to_add_z_32])
        array = array + [self.padding, (self.padding*3), self.padding, self.padding]
        return array
class DfHelper:

    def __init__(self):
        pass
        
    def reduced_df_idx(self, array_primary: np.array, min_cut: float, max_cut: float) -> np.array:
        """Get index of primary energy dataframe at the energy cut

        Args:
            array_primary (pd.DataFrame): primary energy array
            min_cut (float): minimum energy cut
            max_cut (float): maximum energy cut

        Returns:
            numpy array: array of indices of energy cut
        """
        indices = np.where((array_primary >= min_cut) & (array_primary <= max_cut))[0]
        return indices

    def get_nentries_in_event_df(self, df_QDC: pd.DataFrame) -> np.array:
        """  
        Get number of entries in each event 

        Args:
            df_QDC (pd.DataFrame): dataframe of events in row and entries in columns

        Returns:
            np.array: number of non-NaN values in each row 
        """
        n_entries_array = df_QDC.count(axis=1)
        return n_entries_array

    def get_nentries_in_event_array(self, array_QDC: np.array) -> np.array:
        """  
        Get number of entries in each event 

        Args:
            array_QDC (np.array): 2D numpy array of events in row and entries in columns

        Returns:
            np.array: number of non-NaN values in each row 
        """
        n_entries_array = np.array([len(sub_arrays) for sub_arrays in array_QDC])
        return n_entries_array


class FillTensor:
    def __init__(self,shape=(16,32,2,2),neg=False,bothneg=True,flag_channel=False,padding=0):
        self.shape = np.array(shape)
        self.detector = Detector()
        self.df_help = DfHelper()
        self.bothneg = bothneg
        self.neg = neg
        self.flag_channel = flag_channel
        self.padding = padding
      #  self.df_qdc_array = self.df_qdc_array
        if self.flag_channel:
            self.shape[3] += 1
        if self.padding != 0:
            self.shape[0] += self.padding*4
            self.shape[1] += self.padding*2
        self.full_tensor = np.zeros(self.shape)

        if self.neg:
            self.full_tensor[:, :, :, 1:] -= 1
        if bothneg:
            self.full_tensor[:, :, :, :] -= 1
   #     pass


    def get_4dtensor_filled(self, df_qdc_array, df_trig_array,shape,tensor_idx, neg,bothneg,flag_channel,padding):

        """
        get a filled tensor of QDC or trigger at respected positions
        neg and bothneg set true if we want to substitue untriggered SiPMs with -1
        can be done for Trigger (neg) or Trigger and QDC (bothneg)
        input: df of values, df of tensor_idx, event that needs to be filled
        output: tensor of shape 32*16*2 
        """
        self.shape = shape    
        self.neg = neg
        self.bothneg = bothneg 
        self.flag_channel = flag_channel
        self.padding = padding   
        df_qdc_array = np.array(df_qdc_array)
        n_entries = len(df_qdc_array[~np.isnan(df_qdc_array)])
        if n_entries == 0:
            return self.full_tensor
        qdc_array = df_qdc_array[0:n_entries]
        trig_array = df_trig_array[0:n_entries]
        if self.flag_channel:
            flag_array = np.ones(n_entries)
            qdc_trig_array = np.stack([qdc_array, trig_array, flag_array], axis=-1)
        else:
            qdc_trig_array = np.stack([qdc_array, trig_array], axis=-1)
        idx_array = np.stack(tensor_idx[0:n_entries], axis=1)
        self.full_tensor[(idx_array[0], idx_array[1], idx_array[2])] = qdc_trig_array
        return self.full_tensor

    def get_tensor_idx(self, id, output_shape):
        """
        get idx of a tensor of output_shape+padding*2
        if padding=0 no zeros  
        input: array of sipm ids, output_shape 
        return: idx of id in output_shape+padding*2 shape tensor   
        """    
        shifts = self.detector.get_xz_idx_shift(output_shape, self.padding)
        if id < 112:
            idx_x, idx_z = self.detector.get_xz_index(id, [4, 28], shifts[0], shifts[2])
            idx_y = 1
        elif 112 <= id < 368:
            id = id - 112
            idx_x, idx_z = self.detector.get_xz_index(id, [8, 32], shifts[1], shifts[3])
            idx_y = 1
        elif 367 < id < 480:
            id = id - 368
            idx_x, idx_z = self.detector.get_xz_index(id, [4, 28], shifts[0], shifts[2])
            idx_y = 0
        else:
            id = id - 480
            idx_x, idx_z = self.detector.get_xz_index(id, [8, 32], shifts[1], shifts[3])
            idx_y = 0
        idx_array = np.array((idx_x, idx_z, idx_y), dtype=int)
        return idx_array