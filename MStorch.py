import torch
import torch.nn as nn
import torch.fft as fft
import math
from util import device, getWavelength, getFreqGrid, getKernel, toCPU
import matplotlib.pyplot as plt

class Propagate(nn.Module):
    '''differentiable propagation with angular spectrum method'''
    def __init__(self, kernel, mask, eps, potential=None) -> None:
        super().__init__()
        #kernel: angular spectrum kernel
        self.kernel = kernel

        #mask: the mask of the object
        #TODO: different mask for each layer/dynamic mask
        self.mask = mask

        #eps: the threshold for the object
        self.eps = eps

        #can initialize with a potential
        if potential is None:
            # self.object = nn.Parameter(1 - torch.rand_like(kernel, device=device)*0.05)          
            
            self.object = nn.Parameter(torch.ones_like(kernel, device=device))
        else:
            self.object = nn.Parameter(potential.detach().clone())
        
        
    def forward(self, probe, propagate=True):
        if propagate:
            return fft.ifft2(fft.fft2(self.object*probe)*self.kernel)
        else:
            return self.object*probe


class MultiSlice(nn.Module):
    '''differentiable multi slice propagation with angular spectrum method'''
    def __init__(self, mask, eps, nSlice, zStep, kV, pixelSize, potential=None) -> None:
        super().__init__()
        self.mask = mask
        self.mask.requires_grad = False
        self.eps = eps
        nx, ny = mask.shape        
        self.nSlice = nSlice
        self.zStep = zStep
        self.V = kV*1e3
        
        wavelength = getWavelength(self.V)
        kxx, kyy = getFreqGrid(nx, ny, pixelSize)
        self.kernel = getKernel(kxx, kyy, wavelength, zStep)
        self.kernel.requires_grad = False

        if potential is None:
            self.slices = nn.Sequential(*[Propagate(self.kernel, self.mask, self.eps) for i in range(nSlice)])
        else:
            self.slices = nn.Sequential(*[Propagate(self.kernel, self.mask, self.eps, potential[i]) for i in range(nSlice)])

    def forward(self, probe):
        exitWave = probe
        
        for i in range(self.nSlice-1):
            exitWave = self.slices[i](exitWave)
           
        #don't propagate for the last slice
        exitWave = self.slices[-1](exitWave, propagate=False)
        # return the exit wave in real space
        return exitWave


#define loss functions
def complexMSELoss(input, target):
    return torch.mean(torch.abs(input-target)**2)


