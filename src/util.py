import numpy as np
import matplotlib.pyplot as plt
import torch
from math import pi, sqrt, cos
from dataclasses import dataclass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device}")



#physical constants and kernels
def getWavelength(V):
    '''get the wavelength (Angstrom) of the electron with given accelerating voltage (V) '''
    return 12.2643/sqrt(V*(1+0.978476*1e-6*V))

def getFreqGrid(nx, ny, pixelSizeX, pixelSizeY):
    '''get the frequency grid for angular spectrum kernel'''
    kx = torch.fft.fftfreq(nx, d=pixelSizeY, device=device)
    ky = torch.fft.fftfreq(ny, d=pixelSizeX, device=device)
    kxx, kyy = torch.meshgrid(kx, ky)
    return kxx, kyy

#angular spectrum method modified from https://github.com/mdw771/adorym/blob/master/adorym/propagate.py
# def getKernel(kxx, kyy, wavelength, zStep, tilt):
#     '''get the kernel for angular spectrum method'''
    
#     quad = 1 - wavelength**2*(kxx*(kxx+2*tilt[0])+(kyy+2*tilt[1])*kyy)
#     quad_inner = torch.clip(quad, 0, None)
#     kernel = torch.exp(1j*2*pi*zStep/wavelength*torch.sqrt(quad_inner))
#     kernel[quad<=0] = 0
#     return kernel

def getKernel(kxx, kyy, wavelength, zStep, tilt):
    '''get the kernel for fresnel propagation method'''
    sg = -wavelength/2*((kxx+2*tilt[0])*kxx+(kyy+2*tilt[1])*kyy)                       
    theta = 2*pi*zStep*sg
    kernel = torch.exp(1j*theta)
    
    kernel[kxx**2+kyy**2>16/25*(kxx.max())**2] = 0
    return kernel

#display and plotting
def plotMetric(engine, metric, ylabel, gpu=False):
    '''plot the metric'''
    if gpu:
        metric = array2CPU(metric)
    fig, ax = plt.subplots()
    ax.plot(metric)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('iteration')
    ax.set_ylim(0, max(metric))
    ax.set_xlim(0, len(metric))
    itmap = []
    for idx, config in enumerate(engine.configs):
        type, it = config.realConstraint
        itmap+= [idx]*it
    x = np.arange(len(itmap))
    for i in range(len(engine.configs)):
        ax.fill_between(x, 0, max(metric), where = np.array(itmap)<=i, alpha=0.3)



def display(engine, gpu=False, xlim=None, ylim=None, grange=None, Grange=None):
    '''display the diffraction pattern and the real space image before and after constraint'''
    if gpu:
        g_r = toCPU(engine.g_r)
        G_k = toCPU(engine.G_k)
        gp_r = toCPU(engine.gp_r)
        Gp_k = toCPU(engine.Gp_k)
    else:
        g_r = engine.g_r
        G_k = engine.G_k
        gp_r = engine.gp_r
        Gp_k = engine.Gp_k
    
    if xlim is None:
        xlim = (engine.nx/2-400, engine.nx/2+400)
    if ylim is None:
        ylim = (engine.ny/2+400, engine.ny/2-400)

    

    fig, ax = plt.subplots(2,2, figsize=(6,6), dpi=200)
    ax[0,0].imshow(np.real(g_r), cmap='gray')
    ax[0,0].set_title('g_r')
    ax[0,0].set_xlim(*xlim)
    ax[0,0].set_ylim(*ylim)

    ax[0,1].imshow(np.abs(G_k), vmin=0)
    ax[0,1].set_title('G_k')
    ax[1,0].imshow(np.real(gp_r), cmap='gray')
    ax[1,0].set_title('gp_r')
    ax[1,0].set_xlim(*xlim)
    ax[1,0].set_ylim(*ylim)

    ax[1,1].imshow(np.abs(Gp_k), vmin=0)
    ax[1,1].set_title('Gp_k')

    if grange:
        ax[0,0].get_images()[0].set_clim(*grange)
        ax[1,0].get_images()[0].set_clim(*grange)
    if Grange:
        ax[0,1].get_images()[0].set_clim(*Grange)
        ax[1,1].get_images()[0].set_clim(*Grange)

    plt.tight_layout()


def animation(engine,savepath, interval, dpi, gpu=False):
    pass
#gnerate by chatgpt3.5
# from matplotlib.animation import FuncAnimation

# # Step 2: Prepare Your Images (Replace these with your actual image frames)
# # np.abs(engine.history) = []  # List of image frames

# # Step 4: Create Animation Function
# def animate(frame):
#     plt.clf()  # Clear previous plot
#     plt.imshow(np.real(engine.history)[frame])  # Plot the current frame
#     plt.xlim(512,1536)
#     plt.ylim(1536,512)
#     plt.title(f'Frame {frame}')  # Set title (optional)

# # Step 5: Generate the Video
# output_video_path = 'cds.gif'  # Change to your desired output path
# frame_interval = 3  # Time interval between frames in milliseconds

# # Create the animation
# fig = plt.figure(dpi=200)
# animation = FuncAnimation(fig, animate, frames=len(np.real(engine.history)), interval=frame_interval)

# # Save the animation as a video
# animation.save(output_video_path, writer='pillow')  # Requires ffmpeg installed

# plt.show()  # Display the animation (optional)

#GPU CPU memory transfer
def toGPU(arr):
    '''transfer data from numpy array to GPU'''
    if issubclass(arr.dtype.type,np.floating):
        return torch.from_numpy(arr).float().to(device)
    else:
        return torch.from_numpy(arr).to(device)

def toCPU(arr):
    '''convert a torch tensor to numpy array'''
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    else:
        raise TypeError('input must be a torch tensor')

def array2CPU(arr):
    '''convert a sequence of tenosr on GPU to numpy array'''
    if isinstance(arr, list):
        return np.array([toCPU(i) for i in arr])
    shape = arr.shape
    return np.array([toCPU(i) for i in arr.flatten()]).reshape(shape)

@dataclass
class SimulationCell:
    '''a class to define the simulation cell'''
    nx: int # number of pixels in x direction
    ny: int # number of pixels in y direction
    cellx: float # size of x direction of computation cell in Angstrom
    celly: float # size of y direction of computation cell in Angstrom
    gamma: float # angle between x and y axis in degrees


    def __post_init__(self):        
        self.pixelSizeX = self.cellx/self.nx
        self.pixelSizeY = self.celly/self.ny
        self.cosine = cos(self.gamma*pi/180)
        self.volume = self.cellx*self.celly*sqrt(1 - self.cosine**2)
        self.reciprocalx = self.celly/self.volume
        self.reciprocaly = self.cellx/self.volume


        
        


