import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from scipy.signal import find_peaks
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
            # phase grating
            if len(potential) == nSlice:                
                self.slices = nn.Sequential(*[Propagate(self.kernel, potential[i]) for i in range(nSlice)])
            # repeated potential
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
    def __init__(self, simulationCell, zStep, nSlice, kV, nTilt, tiltStep, beams, potential=None, tilt=None) -> None:
        super().__init__(simulationCell, zStep, nSlice, kV, potential, tilt)
        self.nTilt = nTilt
        self.tiltStep = tiltStep
        self.beams = beams

    def findLattice(self, threshold=1e10):
        probe = torch.ones(self.cell.nx, self.cell.ny, dtype=torch.complex64, device=device)
        tilt = self.tilt.detach().clone()
        self.setKernel(torch.tensor([0.0, 0.0], device=device))
        dp_cpu = toCPU(super().forward(probe))
        peaks, _ = find_peaks(dp_cpu.flatten(), height=threshold)
        peak_coords = np.unravel_index(peaks, dp_cpu.shape)
        self.center = np.array(dp_cpu.shape)//2
        dist = np.sum((np.array(peak_coords).T - self.center)**2, axis=1)
        min_idx = np.argmin(dist)
        dist = np.sqrt(np.sum((np.array(peak_coords).T - np.array(peak_coords).T[min_idx])**2, axis=1))
        min_idx = np.argsort(dist)[1:3]
        self.vector1 = np.array([peak_coords[0][min_idx[0]] - self.center[0], peak_coords[1][min_idx[0]] - self.center[1]])
        self.vector2 = np.array([peak_coords[0][min_idx[1]] - self.center[0], peak_coords[1][min_idx[1]] - self.center[1]])
        print(f'vector1(red): {self.vector1}\nself.vector2(blue): {self.vector2}')
        # Create a grid of points using the two vectors
        grid_points = []
        for beam in self.beams:
            point = beam[0] * self.vector1 + beam[1] * self.vector2
            grid_points.append(point)

        grid_points = np.array(grid_points) + self.center
        self.mask = torch.zeros((self.cell.nx, self.cell.ny), dtype=torch.int64, device=device)
        # plot the grid points on the diffraction pattern with rectangles
        fig, ax = plt.subplots(dpi=300)
        ax.imshow(dp_cpu, cmap='gray', vmax=threshold)
        for idx, point in enumerate(grid_points):
            ax.add_patch(plt.Rectangle((point[1] -5, point[0] -5), 10, 10, edgecolor='white', facecolor='none'))
            self.mask[point[0]-5:point[0]+5, point[1]-5:point[1]+5] = idx+1
        # plot vector 1 and vector 2
        ax.arrow(self.center[1], self.center[0], self.vector1[1], self.vector1[0], head_width=10, head_length=10, fc='r', ec='r')
        ax.arrow(self.center[1], self.center[0], self.vector2[1], self.vector2[0], head_width=10, head_length=10, fc='r', ec='b')
        plt.show()
        self.setKernel(tilt)
    
    def setIndices(self, v1, v2):
        self.indices = np.array([beam[0]*v1 + beam[1]*v2 for beam in self.beams])

    def forward(self, probe):
        # x = torch.linspace(-self.cell.cellx / 2, self.cell.cellx / 2, self.cell.nx)
        # y = torch.linspace(-self.cell.celly / 2, self.cell.celly / 2, self.cell.nx)
        # X, Y = torch.meshgrid(x, y)
        i = torch.arange(-self.nTilt, self.nTilt+1)        
        I, J = torch.meshgrid(i, i)
        res = torch.zeros((len(self.beams)+1, 2*self.nTilt+1, 2*self.nTilt+1), dtype=torch.float32, device=device)

        for i,j in zip(I.flatten(), J.flatten()):
            self.setKernel([i*self.tiltStep, j*self.tiltStep])            
            # temp = probe*torch.exp(2j*torch.pi*(X*self.cell.reciprocalx*i*self.tiltStep*+Y*self.cell.reciprocaly*j*self.tiltStep)).to(device)
            # temp = temp/torch.abs(temp)
            exitWave = super().forward(probe)
            res[:,i+self.nTilt,j+self.nTilt].scatter_add_(dim=0, index=self.mask.flatten(), src=exitWave.flatten())
        return res[1:,:,:]

    

#define loss functions
def complexMSELoss(input, target):
    return torch.mean(torch.abs(input-target)**2)


