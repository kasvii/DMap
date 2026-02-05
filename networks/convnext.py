import torch
from torch import nn, Tensor
import torchvision.models as tvm
from einops import rearrange

torch.hub.set_dir('/scratch/cvlab/home/ren/code/hub/checkpoints/')

class ConvNeXtExtractor(nn.Module):
    def __init__(self, n_stages=3, ave=False):
        super().__init__()
        convnext = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(nn.Sequential(convnext.features[i], convnext.features[i+1]))

        #print(len(self.stages))
        self.stages = self.stages[:n_stages]
    def forward(self, images):

        features = []
        for stage in self.stages:
            images = stage(images)
            #print(images.shape)
            features.append(images)
        #sys.exit()

        return features


class ConvNeXtExtractorCustom(nn.Module):
    def __init__(self, in_channel=3, n_stages=3, ave=False):
        super().__init__()
        convnext = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        self.stages = nn.ModuleList()
        for i in range(0, len(convnext.features), 2):
            # group together each downsampling + processing stage
            self.stages.append(nn.Sequential(convnext.features[i], convnext.features[i+1]))


        self.stages[0][0][0] = torch.nn.Conv2d(in_channel, 96, kernel_size=(4, 4), stride=(4, 4))

        #print(len(self.stages))
        self.stages = self.stages[:n_stages]
    def forward(self, images):

        features = []
        for stage in self.stages:
            images = stage(images)
            #print(images.shape)
            features.append(images)
        #sys.exit()

        return features



class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
        gaussian=False,
        #iters=1
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)
        #self.iters = iters

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        if gaussian:
            self.activation = GaussianActivation()
        else:
            self.activation = torch.nn.LeakyReLU()

    def forward(self, input):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, ?]
        """

        #batch_size, n_dim = input.shape

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                #x_mid = x.clone()
                x = torch.cat([x, input], -1)
            #print(l, x.shape)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        delta_x = x
        return delta_x

class MLP_v2(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
        gaussian_norm=False,
        #iters=1
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)
        #self.iters = iters

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ModuleList([GaussianActivation(normalized=gaussian_norm) for i in range(depth)])
        #self.activation = GaussianActivation(normalized=gaussian_norm)


    def forward(self, input):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, ?]
        """

        #batch_size, n_dim = input.shape

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                #x_mid = x.clone()
                x = torch.cat([x, input], -1)
            #print(l, x.shape)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation[l](x)
                #x = self.activation(x)

        delta_x = x
        return delta_x

