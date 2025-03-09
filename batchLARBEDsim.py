import numpy as np
import torch
# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.MStorch import MultiSlice, LARBED
from src.util import toCPU, toGPU, device, SimulationCell
import sys
from wfRead import readPhaseGrating
import os

from pathlib import Path

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

ms_files = list(Path(r'MD\Si_V_strain_20241209-1\manual_add_V').rglob('Si*.ms'))
# ms_files = Path('MD').rglob('perfect_relax_final.ms')
print(ms_files)
print('*'*10)
for ms_file in ms_files:

    print(ms_file)

    parent_dir = ms_file.parent.name
    last_part = ms_file.stem
    path_object = Path(f'{parent_dir}/{last_part}.npy')
    path_indices = Path(f'{parent_dir}/{last_part}_indices.npy')

    # os.system(f'zmult64 {ms_file} -output 1')

     # Create the corresponding folder if it doesn't exist
    output_pg = Path(rf'..\simulation\MD\Si_V_strain_20241209-1\manual_add_V/{ms_file.parent.name}/{ms_file.stem}.pg')
    # output_folder = output_pg.parent
    # output_folder.mkdir(parents=True, exist_ok=True)
    # Move and rename temp.pg to the corresponding folder
    # os.rename('temp.pg', output_pg)
    si_pg = readPhaseGrating(output_pg, (1024, 1024), 200)
    # # si_pg = readPhaseGrating('temp.pg', (1024, 1024), 200)
    # # si_pg = np.repeat(si_pg, 100, axis=0)
    # # si_pg = np.fromfile(r'test\Si\Si110_20x14x50.img', dtype=np.complex64, offset=8).reshape(-1,1024,1024)
    simCell = SimulationCell(1024, 1024, 103.59351, 100.5407, 90.0000)
    probe = torch.ones((1024,1024), dtype=torch.complex64, device=device)
    i,j = np.mgrid[-3:4,-3:4]
    beams = tuple(zip(i.flatten(), j.flatten()))
    larbed = LARBED(simCell, zStep=1.933475, nSlice=200, kV=300, nTilt=30, tiltStep=4, potential=toGPU(si_pg), tilt=(0,0), beams=beams).to(device)


    larbed.findLattice(threshold=1e10)
    larbed.setIndices(np.array([1,-1,-1]), np.array([1,-1,1]))

    with torch.no_grad():
        sim = larbed(probe)
    sim = toCPU(sim)


    
    
    print(r"C:\Users\Mark\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_strain_20241209-1\manual_add_V"/path_object)
    # np.save(r"C:\Users\Mark\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_strain_20241209-1\manual_add_V"/path_object, sim)

    
    # np.save(r"C:\Users\Mark\Box\ZuoLab\active dopant\LARBED\simulation\MD\Si_V_strain_20241209-1\manual_add_V"/path_indices, larbed.indices)
    torch.cuda.empty_cache()
    # del larbed
    # del probe
    