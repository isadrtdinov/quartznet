from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, num_cells=5, kernel_size=33, in_channels=256,
                 out_channels=256, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()

        self.num_cells = num_cells
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # padding = 'same' for stride = 1, dilation = 1
        self.padding = (self.kernel_size - 1) // 2

        self.norm_layer = nn.BatchNorm1d if norm_layer is None else norm_layer
        self.activation = nn.ReLU if activation is None else activation

        def build_cell(in_channels, out_channels):
            return nn.Sequential(
                # 1D Depthwise Convolution
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=self.kernel_size, padding=self.padding,
                          groups=in_channels),
                # Pointwise Convolution
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1),
                # Normalization
                self.norm_layer(num_features=out_channels),
            )
 
        self.cells = [build_cell(self.in_channels, self.out_channels)]
        for i in range(1, self.num_cells):
            self.cells.append(self.activation())
            self.cells.append(build_cell(self.out_channels, self.out_channels))
        self.cells = nn.Sequential(*self.cells)

        # Skip connection
        self.residual = nn.Sequential(
            # Pointwise Convolution
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
            # Normalization
            self.norm_layer(num_features=out_channels),
        )
        self.activation = self.activation()

    def forward(self, inputs):
        outputs = inputs
        outputs = self.cells(outputs)
        outputs = self.activation(outputs + self.residual(inputs))
        return outputs


class QuartzNet(nn.Module):
    def __init__(self, num_labels, num_blocks=5, num_cells=5, num_mels=128,
                 input_kernel=33, input_channels=256,
                 head_kernel=87, head_channels=512,
                 block_kernels=(33, 39, 51, 63, 75),
                 block_channels=(256, 256, 256, 512, 512),
                 norm_layer=None, activation=None):
        super(QuartzNet, self).__init__()

        self.norm_layer = nn.BatchNorm1d if norm_layer is None else norm_layer
        self.activation = nn.ReLU if activation is None else activation
        self.num_blocks = num_blocks
        self.num_cells = num_cells
        self.num_labels = num_labels

        # padding to reduce time frames (T) -> (T / 2)
        input_padding = (input_kernel - 1) // 2
        self.input = nn.Sequential(
            # C1 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=num_mels, out_channels=num_mels,
                      kernel_size=input_kernel, stride=2,
                      padding=input_padding, groups=num_mels),
            nn.Conv1d(in_channels=num_mels, out_channels=input_channels,
                      kernel_size=1),
            self.norm_layer(num_features=input_channels),
            self.activation()
        )

        in_channels = input_channels
        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append(
                BasicBlock(num_cells=self.num_cells, kernel_size=block_kernels[i],
                           in_channels=in_channels, out_channels=block_channels[i],
                           norm_layer=self.norm_layer, activation=self.activation)
            )
            in_channels = block_channels[i]
        self.blocks = nn.Sequential(*self.blocks)

        # padding = 'same' for stride = 1, dilation = 2
        head_padding = head_kernel - 1
        self.head = nn.Sequential(
            # C2 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=head_kernel, dilation=2, 
                      padding=head_padding, groups=in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=head_channels,
                      kernel_size=1),
            self.norm_layer(num_features=head_channels),
            self.activation(),
            # C3 Block: Conv-BN-ReLU
            nn.Conv1d(in_channels=head_channels, out_channels=2 * head_channels,
                      kernel_size=1),
            self.norm_layer(num_features=2 * head_channels),
            self.activation(),
            # C4 Block: Pointwise Convolution
            nn.Conv1d(in_channels=2 * head_channels, out_channels=num_labels,
                      kernel_size=1)
        )

    def forward(self, inputs):
        outputs = self.input(inputs)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.head(outputs)
        return outputs


def quartznet(num_labels, params):
    return QuartzNet(num_labels=num_labels, num_mels=params['num_mels'],
                     num_blocks=params['num_blocks'], num_cells=params['num_cells'],
                     input_kernel=params['input_kernel'], input_channels=params['input_channels'],
                     head_kernel=params['head_kernel'], head_channels=params['head_channels'],
                     block_kernels=params['block_kernels'], block_channels=params['block_channels'])

