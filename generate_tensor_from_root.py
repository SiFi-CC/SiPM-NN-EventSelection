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
from src.generate_train_target import TargetData, TrainingData, QDCTrig
from src.train_target_utils import DfHelper, Detector

#%%

class generate_tensor_from_root:
    def __init__(self) -> None:
        self.qdc_trig = QDCTrig()
        self.training_data = TrainingData()
        self.target_data = TargetData()
        self.df_helper = DfHelper()
        self.detector = Detector()
        self.get_data = read_data()
        pass

    def generate_ideal_target_data(self, path:str,cut=False, min_cut=1.0,max_cut=10.0):
        """pipeline to generate ideal targets 
            insert path of the root file 
            it is possible to set an energy cut from primary energy 

        Parameters
        ----------
        path : str
            root file path
        cut : bool, optional
            set a primary energy cut, by default False
        min_cut : float, optional
            set lower limit for energy cut, by default 1.0
        max_cut : float, optional
            set upper limit for energy cut, by default 10.0

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
        df_e = self.get_data.get_df_from_root(path,root_entry_str="MCEnergy_e", col_name="energy")
        array_int_p = self.get_data.get_array_from_root(path, root_entry_str="MCInteractions_p")
        array_int_e = self.get_data.get_array_from_root(path, root_entry_str="MCInteractions_e")
        array_pos_p_x = self.get_data.get_array_from_root(path,root_entry_str="MCPosition_p",pos="fX")
        array_pos_e_x = self.get_data.get_array_from_root(path,root_entry_str="MCPosition_e",pos="fX")
        array_type = self.get_data.get_array_from_root(path, root_entry_str="MCSimulatedEventType")
        if cut == True: 
            df_primary = self.get_data.get_array_from_root(path,"MCEnergyPrimary", col_name="energy")
            reduced_idx = self.df_helper.reduced_df_idx(df_primary,min_cut,max_cut)
            df_e = df_e.loc[reduced_idx].reset_index(drop=True)
            array_int_e = array_int_e[reduced_idx]
            array_int_p = array_int_p[reduced_idx]
            array_pos_p_x = array_pos_p_x[reduced_idx]
            array_pos_e_x = array_pos_e_x[reduced_idx]
            array_type = array_type[reduced_idx]
        elif cut == False: 
            pass
        else: 
            raise(ValueError("Energy cut must be True or False"))
        compton_events = self.target_data.get_compton_events(df_e)
        del df_e
        complete_compton_events = self.target_data.get_complete_compton_targets(array_int_e,array_int_p,array_type,compton_events)
        del array_type
        del compton_events
        ideal_target, pos_p_idx, pos_e_idx =  self.target_data.get_ideal_targets(array_pos_p_x,array_pos_e_x,array_int_p,array_int_e,complete_compton_events)
        return ideal_target, pos_p_idx, pos_e_idx

    def generate_complete_target_data(self,path:str,cut=False, min_cut=1,max_cut=10):
        """pipeline to generate complete compton targets 
            insert path of the root file 
            it is possible to set an energy cut from primary energy 

        Parameters
        ----------
        path : str
            root file path
        cut : bool, optional
            set a primary energy cut, by default False
        min_cut : float, optional
            set lower limit for energy cut, by default 1.0
        max_cut : float, optional
            set upper limit for energy cut, by default 10.0

        Returns
        -------
        ideal_target: np.array 
            binary values of Compton events and background
        """
        df_e = self.get_data.get_df_from_root(path,root_entry_str="MCEnergy_e", col_name="energy")
        df_int_e = self.get_data.get_array_from_root(path,root_entry_str="MCInteractions_e")
        df_int_p = self.get_data.get_array_from_root(path,root_entry_str="MCInteractions_p")
        df_type = self.get_data.get_array_from_root(path, "MCSimulatedEventType")
        if cut == True: 
            df_primary = self.get_data.get_array_from_root(path,"MCEnergyPrimary", col_name="energy")
            reduced_idx = self.df_helper.reduced_df_idx(df_primary,min_cut,max_cut)
            df_e = df_e[reduced_idx]
            df_int_e = df_int_e[reduced_idx]
            df_int_p = df_int_p[reduced_idx]
            df_type = df_type[reduced_idx]
        elif cut == False: 
            pass
        else: 
            raise(ValueError("Energy cut must be True or False"))
        compton_events = self.target_data.get_compton_events(df_e)
        targets = self.target_data.get_complete_compton_targets(df_int_e,df_int_p,df_type,compton_events)
        return targets


    def generate_training_data(self,path,shape=(16,32,2,2),qdc="Normed",norm_value=26000,neg=False,bothneg=True, pad=0,cut=False, min_cut=1,max_cut=10,flag_channel=False,compton=False, compton_path=r"\path"):
        """generates training data with 2 channels according to input shape given
           several differnt options for 
           shapes, QDC norm, -1 values, padding, energy cuts,
           flag channel, and compton (to generate data only with compton values)

        Parameters
        ----------
        path : str
            path of ROOT file 
        shape : tuple, optional
            either (16,32,2,2) or (12,32,2,2), by default (16,32,2,2)
        qdc : str, optional
            either "Normed" or "Original" to norm the QDC channel or not, by default "Normed"
        norm_value : int, optional
            value of the QDC norm, same value norm to all QDC, by default 26000
        neg : bool, optional
            if True insert -1 to non triggered SiPMs in trigger times channel, by default False
        bothneg : bool, optional
            if True insert -1 to non triggered SiPMs in both channels, by default True
        pad : int, optional
            adds a 0 padding with the given width, by default 0
        cut : bool, optional
            if True add an energy cut from the primary energy, by default False
        min_cut : int, optional
            minimum value of the cut applied only if cut is true, by default 1
        max_cut : int, optional
            maximum value of the cut applied only if cut is true, by default 10
        flag_channel : bool, optional
            adds a third channel to flag the triggered SiPMs with 1, else -1, by default False
        compton : bool, optional
            if True generates the data only for the Compton events relevant for the regression network, by default Flase
        compton_path : str, optional
            path of the npz file of the target data used only if compton is true, by default r"\path"

        Returns
        -------
        NArray
            5D tensor of shape (events, 3D shape, channels)
        """
    
        df_ID = self.get_data.get_df_from_root(path,'SiPMData.fSiPMId')
        array_QDC = self.get_data.get_array_from_root(path,'SiPMData.fSiPMQDC')
        array_trig = self.get_data.get_array_from_root(path,'SiPMData.fSiPMTriggerTime')
        array_QDC = np.array(array_QDC)   
        array_trig = np.array(array_trig)   
        if compton == True:
            print("compton is true")
            compton_array = np.load(compton_path)
            compton_array = compton_array["arr_0"]
            print("compton is loaded")
            print("first 10 events ", compton_array[0:10])
            compton_idx = np.where(compton_array == 1)[0]
            df_ID = df_ID.loc[compton_idx].reset_index(drop=True)
            array_QDC = array_QDC[compton_idx]
            array_trig = array_trig[compton_idx]
            print("dfs reduced")
            print("QDC before norm ater idx", array_QDC)
            print("trig before norm ater idx", array_trig)
        if qdc=="Normed":
            array_QDC = self.qdc_trig.normed_qdc_df(array_QDC,norm_value)
            print("QDC after norm ater idx", array_QDC)
        array_trig = self.qdc_trig.get_delta_t_array(array_trig)
        if cut == True: 
            array_primary = self.get_data.get_array_from_root(path,"MCEnergyPrimary")
            array_primary = np.array(array_primary)
            if compton == True: 
                array_primary = array_primary[compton_array]
            reduced_idx = self.df_helper.reduced_df_idx(array_primary,min_cut,max_cut)
            df_ID = df_ID.loc[reduced_idx].reset_index(drop=True)
            array_QDC = array_QDC[reduced_idx].reset_index(drop=True)
            df_delta = df_delta[reduced_idx].reset_index(drop=True)
        elif cut == False: 
            pass
        else: 
            raise(ValueError("Energy cut must be True or False"))
        Training_tensor = self.training_data.get_training_data(df_ID,array_QDC,array_trig,shape,neg,bothneg,flag_channel)
        return Training_tensor

    def save_training_data(self,path,output,shape=(16,32,2,2),qdc="Normed",norm_value=100000, neg=False,bothneg=True,padding=0,cut=False, min_cut=1,max_cut=10,flag_channel=False,compton=False, compton_path="\path"):
        """saving the training data

        Parameters
        ----------
                Parameters
        ----------
        path : str
            path of ROOT file 
        output : str
            output file name 
        shape : tuple, optional
            either (16,32,2,2) or (12,32,2,2), by default (16,32,2,2)
        qdc : str, optional
            either "Normed" or "Original" to norm the QDC channel or not, by default "Normed"
        norm_value : int, optional
            value of the QDC norm, same value norm to all QDC, by default 26000
        neg : bool, optional
            if True insert -1 to non triggered SiPMs in trigger times channel, by default False
        bothneg : bool, optional
            if True insert -1 to non triggered SiPMs in both channels, by default True
        pad : int, optional
            adds a 0 padding with the given width, by default 0
        cut : bool, optional
            if True add an energy cut from the primary energy, by default False
        min_cut : int, optional
            minimum value of the cut applied only if cut is true, by default 1
        max_cut : int, optional
            maximum value of the cut applied only if cut is true, by default 10
        flag_channel : bool, optional
            adds a third channel to flag the triggered SiPMs with 1, else -1, by default False
        compton : bool, optional
            if True generates the data only for the Compton events relevant for the regression network, by default Flase
        compton_path : str, optional
            path of the npz file of the target data used only if compton is true, by default r"\path"

        Returns
        -------
        None
            saves the training data with the name given in the output
        """
        tensor = self.generate_training_data(path,shape,qdc=qdc,norm_value=norm_value,neg=neg,bothneg=bothneg,pad=padding,cut=cut, min_cut=min_cut,max_cut=max_cut,flag_channel=flag_channel,compton=compton,compton_path=compton_path)
        np.savez(output,tensor)
        return None

        
        
    def save_target_data(self,path,output,type="ideal",cut=False, min_cut=1,max_cut=10):
        if type == "ideal":
            tensor = self.generate_ideal_target_data(path,cut=cut, min_cut=1,max_cut=10)
        elif type == "complete":
            tensor = self.generate_complete_target_data(path,cut=cut, min_cut=1,max_cut=10)
        else: 
            raise(ValueError("target type must be: ideal or complete"))
        np.savez(output,tensor)
        return None


 #%%
#if __name__ == "__main__":
#    get_data = read_data()
##    input_path_BP0mm = r"C:\Users\georg\Desktop\master_thesis\FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
 #   input_path_BP5mm = r"C:\Users\georg\Desktop\master_thesis\FinalDetectorVersion_RasterCoupling_OPM_BP5mm_4e9protons.root"
 #   output_path = r"C:\Users\georg\Desktop\master_thesis"
 #   target_data_BP0mm = output_path + r"\ideal_targets_raster_ep.npz"
 #   target_data_BP5mm = output_path + r"\ideal_targets_raster_BP5mm_ep.npz"
 #   output_name_BP0mm = output_path+r"\training_data_bothneg_norm26k_1232_2ch_midempty_0mmBP_compton.npz"
 #   output_name_BP5mm = output_path+r"\training_data_bothneg_norm26k_1232_2ch_midempty_5mmBP_compton.npz"
   # save_training_data(input_path_BP0mm,output_name_BP0mm,shape=(16,32,2,2),qdc="Normed",norm_value=26474,neg=False,bothneg=True,padding=0,cut=False, min_cut=2,max_cut=20,flag_channel=False,compton=True,compton_path=target_data_BP0mm)
   #save_training_data(input_path_BP5mm,output_name_BP5mm,shape=(16,32,2,2),qdc="Normed",norm_value=26474,neg=False,bothneg=True,padding=0,cut=False, min_cut=2,max_cut=20,flag_channel=False,compton=True,compton_path=target_data_BP5mm)
 #   print("DONE Training Data Data")


# %%
#save_target_data(input_path_BP0mm,"test",type="ideal")
# %%
#target = np.load("test.npz")
# %%
#target = target["arr_0"]
# %%
#len(target[0][target[0] == 1])/len(target[0])
# %%
#print(target)