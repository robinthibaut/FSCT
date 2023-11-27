import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate

__all__ = ["SAModule", "GlobalSAModule", "FPModule", "Net", "MLP"]


class SAModule(torch.nn.Module):
    """
    Set Abstraction (SA) Module using PointNet++ layers.

    Args:
        ratio (float): Sampling ratio for FPS (Farthest Point Sampling).
        r (float): Radius of local region to be considered for each point.
        NN (torch.nn.Module): Neural Network to process local region features.
    """

    def __init__(self, ratio, r, NN):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(NN)

    def forward(self, x, pos, batch):
        # Use FPS to downsample points based on the ratio
        idx = fps(pos, batch, ratio=self.ratio)

        # Find neighbors within a radius r for each point
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)

        # Apply PointNet convolution using local region features
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global Set Abstraction Module for generating global features.

    Args:
        NN (torch.nn.Module): Neural Network to process combined features.
    """

    def __init__(self, NN):
        super(GlobalSAModule, self).__init__()
        self.NN = NN

    def forward(self, x, pos, batch):
        # Combine features and position vectors
        x = self.NN(torch.cat([x, pos], dim=1))

        # Apply global max pooling to get a global feature vector
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    """
    Create a Multi-Layer Perceptron (MLP).

    Args:
        channels (list): List of channel sizes.
        batch_norm (bool): Whether to include batch normalization.

    Returns:
        torch.nn.Sequential: The MLP model.
    """
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ]
    )


class FPModule(torch.nn.Module):
    """
    Feature Propagation (FP) Module to interpolate and propagate features.

    Args:
        k (int): Number of nearest neighbors to use for interpolation.
        NN (torch.nn.Module): Neural Network to process interpolated features.
    """

    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = NN

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # Interpolate features from downsampled points to original resolution
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        # Concatenate with skip-connection features
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        # Apply neural network to the combined features
        x = self.NN(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    """
    Neural Network architecture for point cloud processing.

    Args:
        num_classes (int): Number of classes for classification.
    """

    def __init__(self, num_classes):
        super(Net, self).__init__()
        # Define Set Abstraction layers
        self.sa1_module = SAModule(0.1, 0.2, MLP([3, 128, 256, 512]))
        self.sa2_module = SAModule(0.05, 0.4, MLP([512 + 3, 512, 1024, 1024]))
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 1024, 2048, 2048]))

        self.fp3_module = FPModule(1, MLP([3072, 1024, 1024]))
        self.fp2_module = FPModule(3, MLP([1536, 1024, 1024]))
        self.fp1_module = FPModule(3, MLP([1024, 1024, 1024]))

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, num_classes, 1)
        self.drop1 = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(1024)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = x.unsqueeze(dim=0)
        x = x.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x
