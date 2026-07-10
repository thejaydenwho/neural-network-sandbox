import numpy as np
from Base import *

class MaxPool2D(Layer):
    def __init__(self, pool_size = (2,2), stride = (2,2) ):
        self.inputs = None
        self.outputs = None
        self.parameters = []
    
    def forward(self, inputs):
        self.inputs = inputs

        batch_size, input_height, input_width, channels = inputs.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        outputs = np.zeros((batch_size, output_height, output_width, channels))

        # store where the maximum happened
        self.max_indices = np.zeros(
            (batch_size, output_height, output_width, channels, 2),
            dtype=int
        )

        for b in range(batch_size):
            for c in range(channels):
                for y in range(output_height):
                    for x in range(output_width):

                        input_y = y * stride_height
                        input_x = x * stride_width

                        patch = inputs[
                            b,
                            input_y:input_y+pool_height,
                            input_x:input_x+pool_width,
                            c
                        ]

                        max_index = np.unravel_index(
                            np.argmax(patch),
                            patch.shape
                        )

                        outputs[b,y,x,c] = patch[max_index]

                        self.max_indices[b,y,x,c] = [
                            input_y + max_index[0],
                            input_x + max_index[1]
                        ]

        return outputs

    def backward(self, output_gradient):

        batch_size, input_height, input_width, channels = self.inputs.shape

        input_gradient = np.zeros_like(self.inputs)

        batch_size, output_height, output_width, channels = output_gradient.shape

        for b in range(batch_size):
            for c in range(channels):
                for y in range(output_height):
                    for x in range(output_width):

                        max_y, max_x = self.max_indices[b,y,x,c]

                        input_gradient[
                            b,
                            max_y,
                            max_x,
                            c
                        ] += output_gradient[b,y,x,c]

        return input_gradient