import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.MStorch import MultiSlice, LARBED
from src.util import toCPU, toGPU, device, SimulationCell
import sys
sys.path.append(r'C:\Users\hcni2\Box\ZuoLab\tool_scripts')
from wfRead import readPhaseGrating
import os

from pathlib import Path

ms_files = Path('MD').rglob('relaxed*.ms')

for ms_file in list(ms_files)[8:11]:

    print(ms_file)

    os.system(f'zmult64 {ms_file} -output 1')


    si_pg = readPhaseGrating(r'temp.pg', (1024, 1024), 200)
    simCell = SimulationCell(1024, 1024, 108.6, 107.5088, 90.0000)
    probe = torch.ones((1024,1024), dtype=torch.complex64, device=device)
    i,j = np.mgrid[-3:4,-3:4]
    beams = tuple(zip(i.flatten(), j.flatten()))
    larbed = LARBED(simCell, zStep=7.6792/4, nSlice=200, kV=300, nTilt=30, tiltStep=4, potential=toGPU(si_pg), tilt=(0,0), beams=beams).to(device)


    larbed.findLattice(threshold=1e10)
    larbed.setIndices(np.array([1,-1,-1]), np.array([1,-1,1]))

    with torch.no_grad():
        sim = larbed(probe)
    sim = toCPU(sim)


    parent_dir = ms_file.parent.name
    last_part = ms_file.stem
    path_object = Path(f'{parent_dir}/{last_part}.npy')
    path_indices = Path(f'{parent_dir}/{last_part}_indices.npy')
    
    print(r"C:\Users\hcni2\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_20241204"/path_object)
    np.save(r"C:\Users\hcni2\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_20241204"/path_object, sim)

    
    np.save(r"C:\Users\hcni2\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_20241204"/path_indices, larbed.indices)

    del larbed
    del probe
    torch.cuda.empty_cache()