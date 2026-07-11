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

        output_gradient = output_gradient.astype(
            np.float32,
            copy=False
        )

        batch_size, _, _, channels = self.inputs.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        output_height = output_gradient.shape[1]
        output_width = output_gradient.shape[2]


        input_gradient = np.zeros_like(
            self.inputs,
            dtype=np.float32
        )


        # max position inside pooling window
        max_y = self.max_indices // pool_width
        max_x = self.max_indices % pool_width


        # absolute coordinates
        y_base = (
            np.arange(output_height)[None,:,None,None]
            * stride_height
        )

        x_base = (
            np.arange(output_width)[None,None,:,None]
            * stride_width
        )


        iy = y_base + max_y
        ix = x_base + max_x


        b = np.arange(batch_size)[:,None,None,None]
        c = np.arange(channels)[None,None,None,:]


        np.add.at(
            input_gradient,
            (b, iy, ix, c),
            output_gradient
        )


        return input_gradient