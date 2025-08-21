"""
This file contains the fully convolutional model.
"""
from typing import Any, Dict, List, Tuple
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SegModel(nn.Module):
    """
    Fully Convolutional Network with Pyramid Scene Parsing (PSP) for semantic segmentation with multi-class binary mask outputs.
    """
    def __init__(self, CatDict: Dict[str, Any]) -> None:
        """
        Initalize the model.
        """
        super(SegModel, self).__init__()
        self.Encoder = models.resnet101(pretrained=True)
        # ---------------------------------------------------------------------
        self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]
        self.PSPLayers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2048,1024, kernel_size=3, stride=1, padding=1, bias=True))
            for _ in self.PSPScales
        ])
        # ---------------------------------------------------------------------
        self.PSPSqueeze = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # ---------------------------------------------------------------------
        self.SkipConnections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU()),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU())
        ])
        # ---------------------------------------------------------------------
        self.SqueezeUpsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256 + 512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU())
        ])
        # ---------------------------------------------------------------------
        self.OutLayersList = nn.ModuleList()
        self.OutLayersDict: Dict[str, nn.Module] = {}
        for nm in CatDict:
            self.OutLayersDict[nm] = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1, bias=False)
            self.OutLayersList.append(self.OutLayersDict[nm])

    # ---------------------------------------------------------------------
    def _normalize_images(self, Images: np.ndarray, TrainMode: bool, UseGPU: bool) -> torch.Tensor:
        """
        Normalizes a batch of images using predefined RGB mean and standard deviation values.
        Args:
            Images (np.ndarray): Input images as a NumPy array of shape (batch_size, height, width, channels).
            TrainMode (bool): If True, uses FloatTensor for training; otherwise, uses HalfTensor for inference.
            UseGPU (bool): If True, moves the tensor and the model to GPU; otherwise, keeps them on CPU.

        Returns:
            torch.Tensor: Normalized images as a PyTorch tensor with shape (batch_size, channels, height, width).

        Note:
            - The RGB mean and standard deviation values are hardcoded as [123.68, 116.779, 103.939] and [65, 65, 65], respectively.
            - The input images are expected to be in NHWC format and are transposed to NCHW format.
            - The returned tensor is not set to require gradients.
        """
        RGBMean, RGBStd = [123.68, 116.779, 103.939], [65, 65, 65]
        InpImages = torch.from_numpy(Images.astype(float)).transpose(2, 3).transpose(1, 2).float()
        InpImages = torch.autograd.Variable(InpImages, requires_grad=False)

        if UseGPU:
            InpImages = InpImages.cuda()
            self.cuda()
        else:
            self.cpu()
            InpImages = InpImages.cpu()

        for i in range(len(RGBMean)):
            InpImages[:, i, :, :] = (InpImages[:, i, :, :] - RGBMean[i]) / RGBStd[i]

        return InpImages

    # ---------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encodes the input tensor using the Encoder module and collects skip connection features.

        Args:
            x (torch.Tensor): Input tensor to be encoded.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - The encoded output tensor after passing through all Encoder layers.
                - A list of tensors representing features from intermediate layers (for skip connections).
        """
        SkipConFeatures = []
        x = self.Encoder.conv1(x)
        x = self.Encoder.bn1(x)
        x = self.Encoder.relu(x)
        x = self.Encoder.maxpool(x)

        x = self.Encoder.layer1(x); SkipConFeatures.append(x)
        x = self.Encoder.layer2(x); SkipConFeatures.append(x)
        x = self.Encoder.layer3(x); SkipConFeatures.append(x)
        x = self.Encoder.layer4(x)

        return x, SkipConFeatures

    # ---------------------------------------------------------------------
    def _psp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Pyramid Scene Parsing (PSP) pooling to the input tensor.
        This method performs multi-scale pooling using a set of predefined scales and layers.
        For each scale, the input tensor is resized, processed by a corresponding layer, and
        then resized back to the original spatial dimensions. The outputs from all scales are
        concatenated and passed through a squeeze layer to produce the final feature tensor.
        Args:
            x (torch.Tensor): Input feature tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Output tensor after PSP pooling and feature squeezing.
        """

        PSPSize = (x.shape[2], x.shape[3])
        PSPFeatures = []
        for i, PSPLayer in enumerate(self.PSPLayers):
            NewSize = (np.array(PSPSize) * self.PSPScales[i]).astype(np.int32)
            NewSize = np.maximum(NewSize, 1)
            y = F.interpolate(x, tuple(NewSize), mode='bilinear')
            y = PSPLayer(y)
            y = F.interpolate(y, PSPSize, mode='bilinear')
            PSPFeatures.append(y)
        return self.PSPSqueeze(torch.cat(PSPFeatures, dim=1))

    # ---------------------------------------------------------------------
    def _decode(self, x: torch.Tensor, SkipConFeatures: List[torch.Tensor]) -> torch.Tensor:
        """
        Decodes the input tensor by sequentially applying skip connections and upsampling.
        Args:
            x (torch.Tensor): The input tensor to be decoded, typically the output from the encoder.
            SkipConFeatures (List[torch.Tensor]): A list of feature tensors from the encoder for skip connections.
        Returns:
            torch.Tensor: The decoded output tensor after applying skip connections and upsampling.
        """

        for i in range(len(self.SkipConnections)):
            sp = (SkipConFeatures[-1 - i].shape[2], SkipConFeatures[-1 - i].shape[3])
            x = F.interpolate(x, size=sp, mode='bilinear')
            x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1 - i]), x), dim=1)
            x = self.SqueezeUpsample[i](x)
        return x

    # ---------------------------------------------------------------------
    def _predict(self, x: torch.Tensor, InpImages: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs prediction using the output layers on the input tensor and returns probability and label dictionaries.
        Args:
            x (torch.Tensor): The input feature tensor to be passed through the output layers.
            InpImages (torch.Tensor): The input images tensor, used to determine the output spatial size for interpolation.
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                - OutProbDict: A dictionary mapping layer names to their corresponding probability tensors (after softmax).
                - OutLbDict: A dictionary mapping layer names to their corresponding label tensors (argmax over classes).
        """

        OutProbDict, OutLbDict = {}, {}
        for nm in self.OutLayersDict:
            l = self.OutLayersDict[nm](x)
            l = F.interpolate(l, size=InpImages.shape[2:4], mode='bilinear')
            Prob = F.softmax(l, dim=1)
            _, Labels = l.max(1)
            OutProbDict[nm] = Prob
            OutLbDict[nm] = Labels
        return OutProbDict, OutLbDict

    # ---------------------------------------------------------------------
    def forward(self, Images: np.ndarray, UseGPU: bool = True, TrainMode: bool = True, FreezeBatchNormStatistics: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            Images: Input images as numpy array.
            UseGPU: Whether to run on CUDA GPU.
            TrainMode: Whether to use training or inference mode.
            FreezeBatchNormStatistics: Whether to freeze BatchNorm statistics.
        
        Returns:
            OutProbDict: Probability maps per class.
            OutLbDict: Predicted labels per pixel per class.
        """
        if FreezeBatchNormStatistics:
            self.eval()

        InpImages = self._normalize_images(Images, TrainMode, UseGPU)
        x, SkipConFeatures = self._encode(InpImages)
        x = self._psp(x)
        x = self._decode(x, SkipConFeatures)
        return self._predict(x, InpImages)