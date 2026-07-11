import numpy as np
import math 
from Base import *

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_weights = None, kernel_biases = None, stride=(1,1), padding=(1,1), seed = None):
        self.inputs = None
        self.outputs = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rng = np.random.default_rng(seed)
        self.parameters = []
        if kernel_weights is None:
            (kernel_height, kernel_width) = kernel_size
            self.kernel_weights = Parameter(self.rng.normal(loc=0.0, scale=0.1, size=(out_channels, kernel_height, kernel_width, in_channels)).astype(np.float32), None)
        else:
            self.kernel_weights = Parameter(np.asarray(kernel_weights, dtype=np.float32), None)
        if kernel_biases is None:
            self.kernel_biases = Parameter(self.rng.normal(loc=0.0, scale=0.1, size=(out_channels)).astype(np.float32), None)
        else:
            self.kernel_biases = Parameter(np.asarray(kernel_biases, dtype=np.float32), None)
        self.parameters.append(self.kernel_weights)
        self.parameters.append(self.kernel_biases)    

    def im2col(self):
        (self.batch_size, self.input_height, self.input_width, _) = self.inputs.shape
        (kernel_height, kernel_width) = self.kernel_size
        (pad_height, pad_width) = self.padding
        (stride_height, stride_width) = self.stride
        self.padded_inputs = np.pad(self.inputs, ((0,0), (pad_height, pad_height), (pad_width, pad_width), (0,0)), mode="constant")
        self.output_height = math.floor((self.input_height + (2 * pad_height) - kernel_height)/stride_height) + 1
        self.output_width = math.floor((self.input_width + (2 * pad_width) - kernel_width)/stride_width) + 1
        patches = []
        for image in self.padded_inputs:
            for y in range(self.output_height):
                for x in range(self.output_width):
                    input_y = y * stride_height
                    input_x = x * stride_width
                    patch = image[input_y:input_y+kernel_height, input_x:input_x+kernel_width, :]
                    patches.append(patch)
        self.patch_matrix = np.reshape(np.asarray(patches), (len(patches),-1))
        return self.patch_matrix
        
    def col2im(self, dX_col):
        (kernel_height, kernel_width) = self.kernel_size
        (pad_height, pad_width) = self.padding
        (stride_height, stride_width) = self.stride 
        dX_padded = np.zeros(self.padded_inputs.shape, dtype=self.padded_inputs.dtype)
        patch_index = 0
        for b in range(self.batch_size):
            for y in range(self.output_height):
                for x in range(self.output_width):
                    input_y = y * stride_height
                    input_x = x * stride_width
                    # convert back flattened patch
                    patch = dX_col[patch_index].reshape(kernel_height, kernel_width, self.in_channels)
                    # ADD patches since they overlap
                    dX_padded[b, input_y: input_y+kernel_height, input_x: input_x+kernel_width, :] += patch
                    patch_index += 1
        h_end = -pad_height if pad_height > 0 else None
        w_end = -pad_width if pad_width > 0 else None
        dX = dX_padded[:, pad_height:h_end, pad_width:w_end, :]
        return dX
            
    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.im2col()
        kernel_weights = self.kernel_weights.value
        kernel_biases = self.kernel_biases.value
        kernel_weight_matrix = np.reshape(kernel_weights, (self.out_channels,-1))
        output_matrix = self.patch_matrix @ kernel_weight_matrix.T + kernel_biases
        outputs = np.reshape(output_matrix, (self.batch_size, self.output_height, self.output_width, self.out_channels))
        self.outputs = outputs
        return outputs
    
    def backward(self, output_gradient):
        (kernel_height, kernel_width) = self.kernel_size
        kernel_weights = self.kernel_weights.value
        kernel_weight_matrix = np.reshape(kernel_weights, (self.out_channels,-1))
        output_gradient_matrix = np.reshape(output_gradient, (-1, self.out_channels))
        dW_matrix = output_gradient_matrix.T @ self.patch_matrix
        # sum by columns (add all the values in each out channel)
        db = np.sum(output_gradient_matrix, axis=0, dtype=np.float32)
        dX_col = output_gradient_matrix @ kernel_weight_matrix
        self.kernel_weights.gradient = np.reshape(dW_matrix, (self.out_channels, kernel_height, kernel_width, self.in_channels))
        self.kernel_biases.gradient = db
        return self.col2im(dX_col)


            


