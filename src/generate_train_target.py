import numpy as np 
import pandas as pd 
from src.train_target_utils import Detector, DfHelper, FillTensor

class TargetData:
    def __init__(self):
        self.detector = Detector()
        pass
    def get_compton_events(self,array_e):
        """ get normal compton events 
            condition that electron was induced
            rerturn pd series of energy 
        """
        compton_events = np.zeros(len(array_e))
        array_e[array_e["energy"] != 0]
        compton_events[np.where(array_e["energy"] != 0)[0]] = 1
        return compton_events

    def get_complete_compton_targets(self,array_int_e,array_int_p,array_type,compton_events):
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
        e_int_len = np.array([len(sub_arrays) for sub_arrays in array_int_e])
        p_int_len = np.array([len(sub_arrays) for sub_arrays in array_int_p])
        def third_condition(array_int_p):
            condition = np.zeros(len(array_int_p))
            for idx, subarrays in enumerate(array_int_p):
                subarrays = np.array(subarrays[1:])
                if ((subarrays > 0) & (subarrays < 10) ).any():
                    condition[idx] = 1
            return np.int16(condition)
        def fourth_condition(array_int_e):
            condition = np.zeros(len(array_int_e))
            for idx, subarrays in enumerate(array_int_e):
                if len(subarrays) == 0:
                    pass
                else: 
                    first_e_int = subarrays[0]
                    if ((first_e_int >= 10) & (first_e_int < 20)):
                        condition[idx] = 1
            return np.int16(condition)
        condition_1 = e_int_len >= 1        
        condition_2 = p_int_len >= 2
        condition_3 = third_condition(array_int_p)
        condition_4 = fourth_condition(array_int_e) 

        type2 = np.zeros(len(array_type))
        type2[np.array(array_type) == 2] = 1

        final =((condition_1 & condition_2) & (condition_3 & condition_4) & np.logical_and(compton_events, type2))
        final = final.astype(int)
        return final
    def get_ideal_targets(self,array_pos_p_x:np.array,
                        array_pos_e_x:np.array,
                        array_int_p:np.array,
                        array_int_e:np.array,
                        complete_compton_targets:np.array):

        """this function gets the ideal Compton targets
            condition of ideal target: 
            1- a complete compton target
            2- when first e interaction in scatterer second p in absorber
            3- an interaction in absorber then scatterer is also possible
                but eliminated here so the regression NN is not skewed during training 
            
            It also saves the interaction positions index so it can be used 
            to extract all xyz posittions 
        
        Parameters
        ----------
        array_pos_p_x : np.array
            electron position in detector in x direction
        array_pos_e_x : np.array
            electron position in detector in x direction
        array_int_p : np.array
            photon interacton
        array_int_e : np.array
            electron interacton
        complete_compton_targets : np.array
            Complete Compton targets

        Returns
        ----------
        ideal_target: np.array 
            binary values of Compton events and backgroun
        pos_p_idx: np.array
            position of photon interaction idx at Compton events
            it has same length as ideal target, background filled with 0
        pos_e_idx: np.array
            position of electron interaction idx at Compton events
            it has same length as ideal target, background filled with 0
        """
        ideal_target = np.zeros(len(array_pos_e_x))
        compton_targets_idx = np.where(complete_compton_targets != 0)[0]
        first_e_pos = np.zeros(len(array_pos_e_x))
        first_p_pos = np.zeros(len(array_pos_e_x))
        pos_p_idx = np.zeros(len(array_int_e))
        pos_e_idx = np.zeros(len(array_int_p))
        for compton_idx in compton_targets_idx:
            int_p_event = array_int_p[compton_idx]
            pos_p_event = array_pos_p_x[compton_idx]
            int_e_event = array_int_e[compton_idx]
            pos_e_event = array_pos_e_x[compton_idx]
            ## first photon interaction
            for idx in range(1,len(int_p_event)):
                if 0 < int_p_event[idx] < 10:
                    first_p_pos[compton_idx] = pos_p_event[idx]
                    break 
            ## first electron interaction
            for idx in range(0,len(int_e_event)):
                if 10 <= int_e_event[idx] < 20:
                    first_e_pos[compton_idx] = pos_e_event[idx]
                    break 
            ## check is a point in scatterer and absorber 
            cond_1 = self.detector.is_point_in_scatterer_x(first_e_pos[compton_idx],142.8,157.2)
            cond_2 = self.detector.is_point_in_absorber_x(first_p_pos[compton_idx],254.8,285.2)
        #  cond_3 = _is_point_in_scatterer_x(first_p_pos[compton_idx],142.8,157.2)
        #  cond_4 = _is_point_in_absorber_x(first_e_pos[compton_idx],254.8,285.2)
            if cond_1 and cond_2:
                pos_e_idx[compton_idx] = idx
                pos_p_idx[compton_idx] = idx
                ideal_target[compton_idx] = 1
            else:
                pass
        return ideal_target, pos_p_idx, pos_e_idx


class TrainingData:
    def __init__(self):
        self.fill_tensor = FillTensor()
        pass

        
    def get_training_data(self, df_ID, array_QDC, array_trig, shape=(16,32,2,2), neg=False, bothneg=True, pad=0, flag_channel=False):
        df_tensor_idx = df_ID.applymap(lambda x: self.fill_tensor.get_tensor_idx(x, shape[0:3]), na_action="ignore")
        full_tensor = np.array([self.fill_tensor.get_4dtensor_filled(array_QDC[i], array_trig[i], shape, df_tensor_idx.loc[i], neg, bothneg, flag_channel, pad) for i in range(len(array_QDC))])
        return full_tensor

class QDCTrig:
    def __init__(self) -> None:
        pass

    def get_delta_t_array(self,trig_array):
        """ 
        get df of diff betwenn max time in each event and the rest
        return df_delta
        """
        for sublist in trig_array:
            min_time = np.min(sublist,initial= 10000000000)
            for i in range(len(sublist)):
                sublist[i] = sublist[i] - min_time            
        return trig_array 

    def get_delta_t_df(self,df_trig):
        """ 
        get df of diff betwenn max time in each event and the rest
        return df_delta
        """
        df_delta = df_trig.apply(lambda x: x - x.min(), axis=1)
        return df_delta 

    def normed_qdc_df(self,qdc_array,norm_value):
        """"
        get df of normalized df qdc
        we got this number from the 90% of all data points 
        """
        for i in range(len(qdc_array)):
                 qdc_array[i] = [x / norm_value for x in qdc_array[i]]
        return qdc_array

