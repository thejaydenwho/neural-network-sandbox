import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view
from Base import *


class Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernel_weights=None,
        kernel_biases=None,
        stride=(1,1),
        padding=(1,1),
        seed=None
    ):
        self.inputs = None
        self.outputs = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.rng = np.random.default_rng(seed)

        self.parameters = []

        kh, kw = kernel_size

        if kernel_weights is None:
            self.kernel_weights = Parameter(
                self.rng.normal(
                    0,
                    0.1,
                    size=(out_channels, kh, kw, in_channels)
                ).astype(np.float32),
                None
            )
        else:
            self.kernel_weights = Parameter(
                np.asarray(kernel_weights, dtype=np.float32),
                None
            )

        if kernel_biases is None:
            self.kernel_biases = Parameter(
                self.rng.normal(
                    0,
                    0.1,
                    size=(out_channels,)
                ).astype(np.float32),
                None
            )
        else:
            self.kernel_biases = Parameter(
                np.asarray(kernel_biases, dtype=np.float32),
                None
            )

        self.parameters.append(self.kernel_weights)
        self.parameters.append(self.kernel_biases)



    def im2col(self):

        batch, height, width, channels = self.inputs.shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride


        padded = np.pad(
            self.inputs,
            (
                (0,0),
                (ph,ph),
                (pw,pw),
                (0,0)
            ),
            mode="constant"
        ).astype(np.float32)


        self.padded_inputs = padded


        windows = sliding_window_view(
            padded,
            (kh,kw),
            axis=(1,2)
        )


        # Apply stride
        windows = windows[
            :,
            ::sh,
            ::sw,
            :,
            :,
            :
        ]


        self.output_height = windows.shape[1]
        self.output_width = windows.shape[2]


        # 
        # before:
        # (batch, out_h, out_w, channels, kh, kw)
        #
        # after:
        # (batch, out_h, out_w, kh, kw, channels)
        #

        windows = windows.transpose(
            0,1,2,4,5,3
        )


        self.patch_matrix = windows.reshape(
            -1,
            kh * kw * channels
        ).astype(np.float32)


        return self.patch_matrix
    
    def col2im(self, dX_col):

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        dX_padded = np.zeros_like(
            self.padded_inputs,
            dtype=np.float32
        )

        # number of patches
        N = self.batch_size * self.output_height * self.output_width

        # reshape patches back
        patches = dX_col.reshape(
            self.batch_size,
            self.output_height,
            self.output_width,
            kh,
            kw,
            self.in_channels
        )

        # Create starting coordinates of each patch
        y_positions = (
            np.arange(self.output_height) * sh
        )

        x_positions = (
            np.arange(self.output_width) * sw
        )


        for c in range(self.in_channels):

            for ky in range(kh):

                for kx in range(kw):

                    ys = y_positions + ky
                    xs = x_positions + kx

                    dX_padded[
                        :,
                        ys[:,None],
                        xs[None,:],
                        c
                    ] += patches[
                        :,
                        :,
                        :,
                        ky,
                        kx,
                        c
                    ]


        h_end = -ph if ph > 0 else None
        w_end = -pw if pw > 0 else None

        return dX_padded[
            :,
            ph:h_end,
            pw:w_end,
            :
        ]



    def forward(self, inputs):

        self.inputs = np.asarray(
            inputs,
            dtype=np.float32
        )

        self.batch_size = self.inputs.shape[0]

        self.im2col()


        W = self.kernel_weights.value.reshape(
            self.out_channels,
            -1
        )


        output = (
            self.patch_matrix @ W.T
            + self.kernel_biases.value
        )


        self.outputs = output.reshape(
            self.batch_size,
            self.output_height,
            self.output_width,
            self.out_channels
        ).astype(np.float32)


        return self.outputs



    def backward(self, output_gradient):

        kh, kw = self.kernel_size


        output_gradient = np.asarray(
            output_gradient,
            dtype=np.float32
        )


        dY = output_gradient.reshape(
            -1,
            self.out_channels
        )


        W = self.kernel_weights.value.reshape(
            self.out_channels,
            -1
        )


        # weight gradient
        dW = dY.T @ self.patch_matrix


        # bias gradient
        db = np.sum(
            dY,
            axis=0,
            dtype=np.float32
        )


        # input gradient
        dX_col = dY @ W


        self.kernel_weights.gradient = dW.reshape(
            self.out_channels,
            kh,
            kw,
            self.in_channels
        ).astype(np.float32)


        self.kernel_biases.gradient = db.astype(
            np.float32
        )


        return self.col2im(
            dX_col
        )

            


