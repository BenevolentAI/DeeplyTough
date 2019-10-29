import torch
import torch.nn as nn
import torch.nn.functional as nnf
from se3cnn.blocks import GatedBlock


class VoxelNetwork(nn.Module):
    """ Network for 3D voxel patch classification. """

    def __init__(self, config, nfeat):
        """
        Build the netwrok in a flexible way based on `config` string, which contains sequence
        of comma-delimited layer definiton tokens layer_arg1_arg2_... See README.md for examples.

        :param config:
        :param nfeat: Number of input channels
        """
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, nfeat, 1, 1, 1))
        self.register_buffer('scaler_std', torch.ones(1, nfeat, 1, 1, 1))

        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')

            if conf[0]=='b':  #Batch norm;
                self.add_module(str(d), nn.BatchNorm3d(nfeat))
            elif conf[0]=='r':  #ReLU
                self.add_module(str(d), nn.ReLU(True))

            elif conf[0]=='m':  #Max pooling
                kernel_size = int(conf[1])
                self.add_module(str(d), nn.MaxPool3d(kernel_size))
            elif conf[0]=='a':  #Avg pooling
                kernel_size = int(conf[1])
                self.add_module(str(d), nn.AvgPool3d(kernel_size))

            elif conf[0]=='c':  #3D convolution         args: output feat, kernel size, padding, stride
                nfeato = int(conf[1])
                kernel_size = int(conf[2])
                padding = int(conf[3]) if len(conf)>3 else 0
                stride = int(conf[4]) if len(conf)>4 else 1
                self.add_module(str(d), nn.Conv3d(nfeat, nfeato, kernel_size, stride, padding))
                nfeat = nfeato

            elif conf[0]=='se': # SE(3)-covariant block   args: output feat, mult1, mult2, mult3, kernel size, padding, stride, bnnorm, smoothing
                nfeato = int(conf[1])
                mult1 = int(conf[2])
                mult2 = int(conf[3])
                mult3 = int(conf[4])
                kernel_size = int(conf[5])
                padding = int(conf[6]) if len(conf)>6 else 0
                stride = int(conf[7]) if len(conf)>7 else 1
                normalization = conf[8] if len(conf)>8 else None
                smooth = bool(int(conf[9])) if len(conf)>9 else False

                if isinstance(nfeat, int):
                    nfeat = (nfeat,)
                    nfeato = (nfeato, mult1, mult2, mult3)
                    activation = (None, nnf.sigmoid)
                elif mult1 <= 0:
                    nfeato = (nfeato,)
                    activation = None
                else:
                    nfeato = (nfeato, mult1, mult2, mult3)
                    activation = (nnf.relu, nnf.sigmoid)

                conv = GatedBlock(nfeat, nfeato, size=kernel_size, padding=padding, stride=stride, activation=activation, normalization=normalization, smooth_stride=smooth)
                self.add_module(str(d), conv)

                if mult1 <= 0:
                    nfeato = nfeato[0]
                nfeat = nfeato

            else:
                raise NotImplementedError('Unknown module: ' + conf[0])

        self.nfeato = nfeat

    def set_input_scaler(self, scaler):
        """
        Sets scaling of inputs.

        :param scaler:
        :return:
        """
        self.scaler_mean.copy_(torch.Tensor(scaler.mean_).view(1, scaler.mean_.size, 1, 1, 1))
        self.scaler_std.copy_(torch.Tensor(scaler.scale_).view(1, scaler.scale_.size, 1, 1, 1))

    def forward(self, input):
        input = (input - self.scaler_mean) / self.scaler_std
        for key, module in self._modules.items():
            input = module(input)
        return input


def create_model(args, dataset, device):
    """ Creates a model """
    model = VoxelNetwork(args.model_config, dataset.num_channels)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)
    return model.to(device)