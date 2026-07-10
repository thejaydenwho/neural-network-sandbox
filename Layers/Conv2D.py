import numpy as np
import math 
from Base import *

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_weights = None, kernel_biases = None, stride=(1,1), padding=(1,1)):
        self.inputs = None
        self.outputs = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.parameters = []
        if kernel_weights is None:
            (kernel_height, kernel_width) = kernel_size
            self.kernel_weights = Parameter(0.1 * np.random.randn(out_channels, kernel_height, kernel_width, in_channels), None)
        else:
            self.kernel_weights = Parameter(kernel_weights, None)
        if kernel_biases is None:
            self.kernel_biases = Parameter(0.1 * np.random.randn(out_channels), None)
        else:
            self.kernel_biases = kernel_biases
        self.parameters.append(self.kernel_weights)
        self.parameters.append(self.kernel_biases)        

    def forward(self, inputs):
        self.inputs = inputs
        (batch_size, input_height, input_width, _) = inputs.shape
        (kernel_height, kernel_width) = self.kernel_size
        (pad_height, pad_width) = self.padding
        (stride_height, stride_width) = self.stride 
        padded_inputs = np.pad(inputs, ((0,0), (pad_height, pad_height), (pad_width, pad_width), (0,0)), mode="constant")
        output_height = math.floor((input_height + (2 * pad_height) - kernel_height)/stride_height) + 1
        output_width = math.floor((input_width + (2 * pad_width) - kernel_width)/stride_width) + 1
        kernel_weights = self.kernel_weights.value
        kernel_biases = self.kernel_biases.value
        outputs = np.zeros((batch_size, output_height, output_width, self.out_channels))
        image_index = 0
        for image in padded_inputs:
            kernel_index = 0
            for kernel_weight, kernel_bias in zip(kernel_weights, kernel_biases):
                for y in range(output_height):
                    for x in range(output_width):
                        input_y = y * stride_height
                        input_x = x * stride_width
                        patch = image[input_y:input_y+kernel_height, input_x:input_x+kernel_width, :]
                        value = np.sum(patch * kernel_weight) + kernel_bias
                        outputs[image_index, y, x, kernel_index] = value
                kernel_index += 1
            image_index += 1
        self.outputs = outputs
        return outputs
    
    def backward(self, output_gradient):
        (batch_size, input_height, input_width, _) = self.inputs.shape
        (kernel_height, kernel_width) = self.kernel_size
        (pad_height, pad_width) = self.padding
        (stride_height, stride_width) = self.stride 
        padded_inputs = np.pad(self.inputs, ((0,0), (pad_height, pad_height), (pad_width, pad_width), (0,0)), mode="constant")
        output_height = math.floor((input_height + (2 * pad_height) - kernel_height)/stride_height) + 1
        output_width = math.floor((input_width + (2 * pad_width) - kernel_width)/stride_width) + 1
        kernel_weights = self.kernel_weights.value
        kernel_biases = self.kernel_biases.value
        dX_padded = np.zeros_like(padded_inputs)
        dW = np.zeros_like(self.kernel_weights.value)
        db = np.zeros_like(self.kernel_biases.value)
        dX_padded = np.zeros_like(padded_inputs)
        for image_index, image in enumerate(padded_inputs):
            for kernel_index, (kernel_weight, kernel_bias) in enumerate(zip(kernel_weights, kernel_biases)):
                for y in range(output_height):
                    for x in range(output_width):
                        input_y = y * stride_height
                        input_x = x * stride_width
                        patch = image[input_y:input_y+kernel_height, input_x:input_x+kernel_width, :]
                        gradient = output_gradient[image_index, y, x, kernel_index]
                        dX_padded[image_index, input_y:input_y+kernel_height, input_x:input_x+kernel_width,:] += kernel_weight * gradient
                        dW[kernel_index] += patch * gradient
                        db[kernel_index] += gradient
        dX = dX_padded[:, pad_height:-pad_height, pad_width:-pad_width, :]
        self.kernel_weights.gradient = dW
        self.kernel_biases.gradient = db
        return dX

        
        

            


