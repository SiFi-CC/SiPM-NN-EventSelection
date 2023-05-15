# SiPM-NN-EventSelection

Apart from the neural network training scripts and data analysis there are two main scripts for the backend: 
  - read_root which reads from the root and generates either an array or pandas dataframe 
  - generate_tensor_from_root which generates training or target data from a root file and outputs a 3D tensor or 1D target and pos indices

```python
generate_and_save = generate_tensor_from_root()
output_path = "path"
path_root = "insert path root" 

compton_path_BP0mm = r"target_data_path"
######################## generate training data #########################
## there are different parameters such as setting an energy cut or setting the qdc norm and norm value 
## seting the compton as true and gving the target data path generate tensor only for Compton events 
## also using neg and bothneg we can decide if -1 for non triggered SiPMs are set for trigger time or both channels 
## it is also possible to add a third channel that sets 1 for triggered SiPM if flag_channel = True 
## it is also possible to set the shape to (16,32,2,2) or (12,32,2,2), by default (16,32,2,2)
a = generate_and_save.generate_ideal_target_data(path_root,compton=True,compton_path=compton_path_BP0mm)
generate_and_save.save_training_data(path_root ,output=output_path+r"\training_data_1632_BP0mm_12k_compton" ,norm_value=1259,compton=True,compton_path=compton_path_BP0mm)

######################## generate target data #########################
## if cut=true it is possible to includ primary energy cut in target data 
## there are two types "ideal" and "complete" for either ideal targets or complete targets 
b = generate_and_save.generate_ideal_target_data(path_root ,output=output_path+r"\target_path")
generate_and_save.save_target_data(path_root ,output=output_path+r"\target_path", type="ideal")

#################################################################
######################## read target from root #######
# we can either get the array or the pandas data frame of a root leaf 
# all we have to set is the root_entry_str in the case of position entry such as MCPosition_source also pos of fZ or fX or fY should be set 
# to see all entries see call get_data.get_root_entry_str_list()

path_root = "insert path root" 

get_data = read_data()
df_pos_z_0mm = get_data.get_df_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z") 
array_pos_z_0mm = get_data.get_array_from_root(path_root,"MCPosition_source",pos="fZ",col_name="Pos Z")   

print(get_data.get_root_entry_str_list(path_root))
```

Data analysis of the classification NN can be found in classification_model_evaluation script 
Data analysis of the regression NN can be foudn in regression_model_evaluation
