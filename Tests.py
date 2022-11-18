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



