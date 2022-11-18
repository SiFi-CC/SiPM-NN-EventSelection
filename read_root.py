
import numpy as np 
import pandas as pd
import uproot 

class read_data:

    """
    dimensions: absorber: 32*8*2, scatterer: 28*4*2, z*x*y
    pos: scatterer=[150,0,0], absorber=[270,0,0]
    pos_dim: scatterer= [14,100,110], absorber=[30,100,126] 
    pos_dim_real: scatterer=[[-143,157]], {-51,51},[-55,60]],
                absorber=[[255,285],{-51,51},[-63,64]]

    col name: name of df col, optional
    pos: for pos p_x, p_y, p_z must enter pos fX, fY, fZ 
    """
    def __init__(self,path="",root_entry_str="",col_name="",pos=""):
        self.root_entry_str = root_entry_str
        self.path = path
        self.col_name = col_name
        self.pos = pos

    def set_path(self,path):
        self.path = path
    
    def get_path(self):
        return self.path 

    def _detector_pos(self,scatter_pos, absorber_pos):
        def _get_pos_array(pos_dict): 
            pos_dict = list(pos_dict[0].values())
            pos_dict = np.array(pos_dict)
            return pos_dict
        scatter_pos_array = _get_pos_array(scatter_pos)
        absorber_pos_array = _get_pos_array(absorber_pos)
        detector_pos = np.concatenate([[scatter_pos_array],[absorber_pos_array]])
        return detector_pos

    def _scatterer_thick(self,scatter_thick_x, scatter_thick_y, scatter_thick_z):
        scatterer_thick = np.concatenate([scatter_thick_x, scatter_thick_y, scatter_thick_z])
        return scatterer_thick

    def _absorber_thick(self,absorber_thick_x,absorber_thick_y,absorber_thick_z):
        absorber_thick = np.concatenate([absorber_thick_x,absorber_thick_y,absorber_thick_z])
        return absorber_thick    

    def _detector_thick(self,scatterer_thick,absorber_thick):

        detector_thick = np.concatenate([[scatterer_thick],[absorber_thick]])
        return detector_thick

    def _absorber_thick(self,absorber_thick_x,absorber_thick_y,absorber_thick_z):
            absorber_thick = np.concatenate([absorber_thick_x,absorber_thick_y,absorber_thick_z])
            return absorber_thick 


    def get_detector_geometry(self,path):
        self.path = path
        path_setup =  fr"{self.path}" + r":Setup;1" 
        with uproot.open(path_setup) as file:  
            scatter_pos = file["ScattererPosition"].arrays()["ScattererPosition"].tolist()
            absorber_pos = file["AbsorberPosition"].arrays()["AbsorberPosition"].tolist()
            absorber_thick_x = file["AbsorberThickness_x"].arrays()["AbsorberThickness_x"].tolist()
            absorber_thick_y = file["AbsorberThickness_y"].arrays()["AbsorberThickness_y"].tolist()
            absorber_thick_z = file["AbsorberThickness_z"].arrays()["AbsorberThickness_z"].tolist()
            scatter_thick_x = file["ScattererThickness_x"].arrays()["ScattererThickness_x"].tolist()
            scatter_thick_y = file["ScattererThickness_y"].arrays()["ScattererThickness_y"].tolist()
            scatter_thick_z = file["ScattererThickness_z"].arrays()["ScattererThickness_z"].tolist()
            scatterer_thick_array = self._scatterer_thick(scatter_thick_x,scatter_thick_y,scatter_thick_z)
            absorber_thick_array = self._absorber_thick(absorber_thick_x,absorber_thick_y,absorber_thick_z)
            detector_thick_array = self._detector_thick(scatterer_thick_array,absorber_thick_array)
            detector_pos_array = self._detector_pos(scatter_pos,absorber_pos)
            return detector_pos_array, detector_thick_array

        
    def get_array_from_root(self,path,root_entry_str,pos=0):
        """
        create array from specified root branch
        input: path, root_entry_str (too see all str call get_root_entry_str_list)
        return pandas df
        """
        self.path = fr"{path}"
        self.root_entry_str = root_entry_str
        self.pos = pos
        path_event = fr"{self.path}" + r":Events;1" 
        with uproot.open(path_event) as file:   
            if pos == None:
                data = file[root_entry_str].arrays()[root_entry_str].tolist()
            else:
                data = file[root_entry_str].arrays()[root_entry_str][pos].tolist()       
        return data

    def get_setup_from_root(self,path,root_entry_str):
        """
        create array from specified root branch of root setup
        input: path, root_entry_str (too see all str call get_root_entry_str_list)
        return pandas df
        """
        self.path = fr"{path}"
        self.root_entry_str = root_entry_str
        path_setup =  fr"{path}" + r":Setup;1" 
        with uproot.open(path_setup) as file:   
            data = file[root_entry_str].arrays()[root_entry_str].tolist()
        return data
    
    def get_root_entry_str_list(self,path):
        """
        get all possible branches from root file Events 
        return list of keys
        """
        self.path = path
        path_event = self.path + r":Events;1" 
        with uproot.open(path_event) as file:   
            data = [file.keys()]
        return data

    def get_df_from_root(self,path,root_entry_str, col_name=None, pos=None):
        """
        create pandas df from specified root branch
        input: path, root_entry_str (too see all str call get_root_entry_str_list)
                col_name of df optional, pos (fX,fY,fZ) if calling a position
        return pandas df
        """
        self.path = r'\b' + re.escape(path) 
        self.root_entry_str = root_entry_str
        self.col_name = col_name
        self.pos = pos
        path_event = path + r":Events;1" 
        with uproot.open(path_event) as file:  
            if pos == None:
                data = file[root_entry_str].arrays()[root_entry_str].tolist()
            else:
                data = file[root_entry_str].arrays()[root_entry_str][pos].tolist()
        if col_name == None:
            df = pd.DataFrame(data=data)
            return df
        else: 
            df = pd.DataFrame(data=data,columns=[col_name])
            return df
    # source_pos = file['MCPosition_source'].arrays()['MCPosition_source'].tolist()
    # primary_energy = file['MCEnergyPrimary'].arrays()['MCEnergyPrimary'].tolist()
    # energy_e = file['MCEnergy_e'].arrays()['MCEnergy_e'].tolist()
    # interaction_e = file['MCInteractions_e'].arrays()['MCInteractions_e'].tolist()
    # energy_p = file['MCEnergy_p'].arrays()['MCEnergy_p'].tolist()
    # interaction_p = file['MCInteractions_p'].arrays()['MCInteractions_p'].tolist()
    #pos_p_x = file["MCPosition_p"].arrays()["MCPosition_p"]['fX'].tolist()
    #pos_e_x = file["MCPosition_e"].arrays()["MCPosition_e"]['fX'].tolist()
    # pos_p = file["MCPosition_p"].arrays()["MCPosition_p"].tolist()
    # pos_e = file["MCPosition_e"].arrays()["MCPosition_e"].tolist()

    # ## sipm data 
# #bits = file['SiPMData.fBits'].arrays()['SiPMData.fBits'].tolist()
# f_x = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition'][].tolist()
# f_y = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition']['fY'].tolist()
# f_z = file['SiPMData.fSiPMPosition'].arrays()['SiPMData.fSiPMPosition']['fZ'].tolist()
