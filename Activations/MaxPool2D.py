import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from Base import *

class MaxPool2D(Layer):
    def __init__(self, pool_size=(2,2), stride=(2,2)):
        self.inputs = None
        self.outputs = None
        self.parameters = []

        self.pool_size = pool_size
        self.stride = stride

        self.max_indices = None

    def forward(self, inputs):
        # keep float32
        inputs = inputs.astype(np.float32, copy=False)

        self.inputs = inputs

        batch_size, input_height, input_width, channels = inputs.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        # Create sliding windows
        windows = sliding_window_view(
            inputs,
            (pool_height, pool_width),
            axis=(1,2)
        )

        # windows shape:
        # (batch, out_h_raw, out_w_raw, channels, pool_h, pool_w)

        # Apply stride
        windows = windows[
            :,
            ::stride_height,
            ::stride_width,
            :,
            :,
            :
        ]

        output_height = windows.shape[1]
        output_width = windows.shape[2]

        # Flatten pooling region
        flat_windows = windows.reshape(
            batch_size,
            output_height,
            output_width,
            channels,
            pool_height * pool_width
        )

        # Save where maximum occurred
        self.max_indices = np.argmax(
            flat_windows,
            axis=-1
        ).astype(np.int32)

        # Max pooling
        outputs = np.max(
            flat_windows,
            axis=-1
        )

        self.outputs = outputs.astype(np.float32)

        return self.outputs


    def backward(self, output_gradient):

        output_gradient = output_gradient.astype(np.float32, copy=False)

        batch_size, input_height, input_width, channels = self.inputs.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        output_height = output_gradient.shape[1]
        output_width = output_gradient.shape[2]

        input_gradient = np.zeros_like(
            self.inputs,
            dtype=np.float32
        )

        # Convert flattened max index back to coordinates
        max_y = self.max_indices // pool_width
        max_x = self.max_indices % pool_width


        # Scatter gradients back
        for b in range(batch_size):
            for y in range(output_height):
                for x in range(output_width):
                    for c in range(channels):

                        iy = y * stride_height + max_y[b,y,x,c]
                        ix = x * stride_width + max_x[b,y,x,c]

                        input_gradient[
                            b,
                            iy,
                            ix,
                            c
                        ] += output_gradient[b,y,x,c]

        return input_gradient.astype(np.float32)