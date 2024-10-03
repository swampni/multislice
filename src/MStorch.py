import torch
import torch.nn as nn
import torch.fft as fft
import math
from src.util import device, getWavelength, getFreqGrid, getKernel, toCPU, SimulationCell
import matplotlib.pyplot as plt

class Propagate(nn.Module):
    '''differentiable propagation with fresnel kernel'''
    def __init__(self, kernel, potential=None) -> None:
        super().__init__()
        # frrsenl propagation kernel
        self.kernel = kernel        
        
        if potential is None:
            # initialize with a random phase object            
            self.object = nn.Parameter(torch.ones_like(kernel, device=device))
        else:
            # initialize with a know potential (phase grating) 
            self.object = nn.Parameter(potential.detach().clone())        
        
    def forward(self, probe, propagate=True):
        if propagate:
            return fft.ifft2(fft.fft2(self.object*probe)*self.kernel)
        else:
            return self.object*probe


class MultiSlice(nn.Module):
    '''differentiable multi slice propagation with fresnel kernel'''
    def __init__(self, simulationCell, zStep, nSlice, kV, potential=None, tilt=None) -> None:
        super().__init__()        
        
        self.cell = simulationCell
        self.zStep = zStep
        self.nSlice = nSlice
        self.V = kV*1e3

        if tilt is None:
            self.tilt = torch.tensor([0.0, 0.0], device=device)
        else:
            self.tilt = torch.tensor([tilt[0]*self.cell.reciprocalx, tilt[1]*self.cell.reciprocaly], device=device)

        

        self.wavelength = getWavelength(self.V)
        self.kxx, self.kyy = getFreqGrid(self.cell.nx, self.cell.ny, self.cell.pixelSizeX, self.cell.pixelSizeY)
        self.kernel = getKernel(self.kxx, self.kyy, self.wavelength, self.zStep, self.tilt)
        self.kernel.requires_grad = False

        if potential is None:
            self.slices = nn.Sequential(*[Propagate(self.kernel) for i in range(nSlice)])
        else:
            if len(potential) == nSlice:                
                self.slices = nn.Sequential(*[Propagate(self.kernel, potential[i]) for i in range(nSlice)])
            elif nSlice % len(potential) == 0:
                self.slices = nn.Sequential(*[Propagate(self.kernel, potential[i%len(potential)]) for i in range(nSlice)])
            else:
                raise ValueError('phase grate must have the same length as nSlice or be a factor of nSlice')

    def forward(self, probe):
        exitWave = probe
        
        for i in range(self.nSlice-1):
            exitWave = self.slices[i](exitWave)
           
        #don't propagate for the last slice
        exitWave = self.slices[-1](exitWave, propagate=False)
        # return the diffraction pattern intensity
        return torch.abs(torch.fft.fftshift(torch.fft.fft2(exitWave)))**2
    
    def setKernel(self, tilt):
        self.tilt = torch.tensor([tilt[0]*self.cell.reciprocalx, tilt[1]*self.cell.reciprocaly], device=device)
        self.kernel = getKernel(self.kxx, self.kyy, self.wavelength, self.zStep, self.tilt)
        for i in range(self.nSlice):
            self.slices[i].kernel = self.kernel
        

class LARBED(MultiSlice):
    '''Large-Angle Rocking Beam Electron Diffraction (LARBED)) simulation'''
    def __init__(self, simulationCell, zStep, nSlice, kV, nTilt, tiltStep, potential=None, tilt=None) -> None:
        super().__init__(simulationCell, zStep, nSlice, kV, potential, tilt)
        self.nTilt = nTilt
        self.tiltStep = tiltStep

    def forward(self, probe):
        x = torch.linspace(-self.cell.cellx / 2, self.cell.cellx / 2, self.cell.nx)
        y = torch.linspace(-self.cell.celly / 2, self.cell.celly / 2, self.cell.nx)
        X, Y = torch.meshgrid(x, y)
        i = torch.arange(-self.nTilt, self.nTilt+1)        
        I, J = torch.meshgrid(i, i)
        res = torch.zeros_like(I)
        for i,j in zip(I.flatten(), J.flatten()):
            self.setKernel([i*self.tiltStep, j*self.tiltStep])            
            # temp = probe*torch.exp(2j*torch.pi*(X*self.cell.reciprocalx*i*self.tiltStep*+Y*self.cell.reciprocaly*j*self.tiltStep)).to(device)
            # temp = temp/torch.abs(temp)
            exitWave = super().forward(probe)
            res[i+self.nTilt,j+self.nTilt] = torch.sum(exitWave[506:518,466:478])
        return res


#define loss functions
def complexMSELoss(input, target):
    return torch.mean(torch.abs(input-target)**2)


