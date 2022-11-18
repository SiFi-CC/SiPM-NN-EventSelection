import numpy as np 

"""
This class generate a tensor with the geometry of the setup with Ids on the right spots
it generate tensor in different shape (default 12,32,2) 
it's useful in ploting and testing 
"""
class ids_tensor():

    def __init__(self,output_shape=(12,32,2),padding=0):
        self.output_shape = output_shape
        self.padding = padding

    def _get_ids(self,a, b, shape):
        """
        get sorted array of ids
        from a to b in input shape
        input: a and b int, desired shape
        output: array of int 
        """
        array = np.arange(a,b).reshape(shape)
        return array

    def _get_shifted_array(self,ids_sorted,output_shape):
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

    def get_ids_tensor(self,output_shape=(12,32,2),padding=0):
        """
        get tensor of ids in given output shape 
        output shape is output_shape + padding * 2
        input output_shape (tuple) and padding int 
        output 3d tensor in shape (32+padding*2,12or16+padding*2,2) 
        """
        self.output_shape = output_shape
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
        lx_uy_ids = self._get_ids(0,112,(4,28))
        lx_uy_ids_shifted = self._get_shifted_array(lx_uy_ids, (shape_x_1,output_shape_z))
        ux_uy_ids = self._get_ids(112,368,(8,32))
        ux_uy_ids_shifted = self._get_shifted_array(ux_uy_ids, (shape_x_2,output_shape_z))
        lx_ly_ids = self._get_ids(368,480,(4,28))
        lx_ly_ids_shifted = self._get_shifted_array(lx_ly_ids, (shape_x_1,output_shape_z))
        ux_ly_ids = self._get_ids(480,736,(8,32)) 
        ux_ly_ids_shifted = self._get_shifted_array(ux_ly_ids, (shape_x_2,output_shape_z))
        uy = np.concatenate((lx_uy_ids_shifted, ux_uy_ids_shifted),axis=0)
        ly = np.concatenate((lx_ly_ids_shifted, ux_ly_ids_shifted),axis=0)
        tensor = np.stack([ly,uy], axis =2)
        return tensor