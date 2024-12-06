import numpy as np
from pathlib import Path


def MDtoDAT(filename):    
    with open(filename, 'r') as f:
        lines = [next(f).strip('\n') for _ in range(8)]
    
    natoms = int(lines[3])
    dim = lines[5:]
    dim = [float(i.split()[1]) for i in dim]
    
    data = np.loadtxt(filename, skiprows=9)
    data = data[:,1:]

    data[data[:,1]<0,1] += dim[0]
    data[:,1] /= dim[0]

    data[data[:,2]<0,2] += dim[1]
    data[:,2] /= dim[1]

    data[data[:,3]<0,3] += dim[2]
    data[:,3] /= dim[2]

    output = np.ones((natoms, 5))
    output[:, :4] = data

    
    if data.shape[0] != natoms:
        print('Error: natoms not match')
        return
    
    header = '''{} {} 90.0000 {}
    {}'''.format(dim[0], dim[1], dim[2], natoms)
    output_filename = filename.with_suffix('.dat')
    np.savetxt(output_filename, output, delimiter='\t', header=header, comments='', fmt='%d\t%.7f\t%.7f\t%.7f\t%d')

    with open('110_P.ms', 'r') as ms_file:
        ms_lines = ms_file.readlines()
    
    ms_lines[4] = f'{output_filename}\n'
    ms_lines[15] = f'-0.000001 {dim[2]} 200 1\n'
    
    with open(filename.with_suffix('.ms'), 'w') as ms_file:
        ms_file.writelines(ms_lines)
    



if __name__ == "__main__":
    txt_files = Path('MD').rglob('relaxed*.txt')
    for txt_file in txt_files:
        # MDtoDAT(txt_file)
        
        parent_dir = txt_file.parent.name
        last_part = txt_file.stem
        path_object = Path(f'{parent_dir}/{last_part}')
        print(path_object.with_suffix('.npy'))
    