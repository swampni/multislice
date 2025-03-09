'''

function [wfStack,fileHeader] = wfRead(filename,imgCrop,totSlice)
% filename = '';
% imgCrop = [4096,4096]; % crop the center of the image. no crop if omitted
% totSlice defines the total outputs to read, equal to maximum when omitted

fileID = fopen(filename);
fread(fileID,1,'int'); % begin
fileHeader.meshx = fread(fileID,1,'int');
fileHeader.meshy = fread(fileID,1,'int');
fileHeader.iform = fread(fileID,1,'int16');
fileHeader.ibyte = fread(fileID,1,'int16');
fileHeader.mout = fread(fileID,1,'int');
fileHeader.deltas = fread(fileID,1,'real*4');
fileHeader.deltab = fread(fileID,1,'real*4');
fileHeader.nss = fread(fileID,1,'int');
fileHeader.nbs = fread(fileID,1,'int');
fileHeader.mth = fread(fileID,1,'int');
fread(fileID,1,'int'); % end
fread(fileID,1,'int'); % begin
fileHeader.wavel = fread(fileID,1,'real*4');
fileHeader.cell2d = fread(fileID,3,'real*4');
fileHeader.cell2r = fread(fileID,3,'real*4');
fread(fileID,1,'int'); % end
mesh = [fileHeader.meshx,fileHeader.meshy];
if nargin < 3 || isequal(totSlice,0)
    totSlice = fileHeader.mout;
end
if nargin < 2 || isequal(imgCrop,0)
	imgCrop = mesh;
end
wfStack = single(complex(zeros(imgCrop(1),imgCrop(2),totSlice)));
for i = 1:totSlice
    fread(fileID,1,'int'); % begin
    currWf = fread(fileID,mesh(1)*mesh(2)*2,'*real*4');
    currWfReal = reshape(currWf(1:2:end),mesh);
    currWfImg = reshape(currWf(2:2:end),mesh);
    currWfComp = complex(currWfReal,currWfImg);
    wfStack(:,:,i) = currWfComp(mesh(1)/2-imgCrop(1)/2+1:mesh(1)/2+imgCrop(1)/2,...
                              mesh(2)/2-imgCrop(2)/2+1:mesh(2)/2+imgCrop(2)/2);
    fread(fileID,1,'int'); % end
end
fclose(fileID);
end

'''


# based on the matlab code above, the following is a python implementation

import numpy as np
import struct
import os

def wfRead(filename,imgCrop=None,totSlice=None):
    fileID = open(filename, 'rb')
    # begin
    struct.unpack('i', fileID.read(4))
    fileHeader = {}
    fileHeader['meshx'] = struct.unpack('i', fileID.read(4))[0]
    fileHeader['meshy'] = struct.unpack('i', fileID.read(4))[0]
    fileHeader['iform'] = struct.unpack('h', fileID.read(2))[0]
    fileHeader['ibyte'] = struct.unpack('h', fileID.read(2))[0]
    fileHeader['mout'] = struct.unpack('i', fileID.read(4))[0]
    fileHeader['deltas'] = struct.unpack('f', fileID.read(4))[0]
    fileHeader['deltab'] = struct.unpack('f', fileID.read(4))[0]
    fileHeader['nss'] = struct.unpack('i', fileID.read(4))[0]
    fileHeader['nbs'] = struct.unpack('i', fileID.read(4))[0]
    fileHeader['mth'] = struct.unpack('i', fileID.read(4))[0]
    # end
    struct.unpack('i', fileID.read(4))
    # begin
    fileHeader['wavel'] = struct.unpack('f', fileID.read(4))[0]
    fileHeader['cell2d'] = struct.unpack('3f', fileID.read(4*3))
    fileHeader['cell2r'] = struct.unpack('3f', fileID.read(4*3))
    # end
    mesh = [fileHeader['meshx'],fileHeader['meshy']]
    if totSlice is None or totSlice == 0:
        totSlice = fileHeader['mout']
    if imgCrop is None or imgCrop == 0:
        imgCrop = mesh
    wfStack = np.zeros((totSlice, imgCrop[0],imgCrop[1]),dtype=np.complex64)
    for i in range(totSlice):
        # begin
        struct.unpack('i', fileID.read(4))
        currWf = np.frombuffer(fileID.read(4*mesh[0]*mesh[1]*2),dtype=np.complex64).reshape(mesh).T
        wfStack[i,:,:] = currWf[mesh[0]//2-imgCrop[0]//2:mesh[0]//2+imgCrop[0]//2, mesh[1]//2-imgCrop[1]//2:mesh[1]//2+imgCrop[1]//2]
        # end
        struct.unpack('i', fileID.read(4))
    fileID.close()
    return wfStack, fileHeader



def readPhaseGrating(pgFileName, mesh, nSlice):
    with open(pgFileName, 'rb') as pgFid:
        np.fromfile(pgFid, dtype=np.int32, count=1)  # begin
        isbulk = np.fromfile(pgFid, dtype=np.int32, count=1)[0]
        isl = np.fromfile(pgFid, dtype=np.int32, count=1)[0]
        count = mesh[0] * mesh[1]
        temp = []
        for i in range(nSlice):
            temp.append(np.fromfile(pgFid, dtype=np.complex64, count=count).reshape(mesh))
            pgFid.read(16)
        Pg = np.array(temp)
        # PgReal = temp[0::2].reshape((mesh[0], mesh[1], nSlice))
        # PgImg = temp[1::2].reshape((mesh[0], mesh[1], nSlice))
        # Pg = PgReal + 1j * PgImg
    
    return Pg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    filename = r'C:\Users\hcni2\Box\ZuoLab\active dopant\LARBED\multislice\temp.pg'
    pg = readPhaseGrating(filename, [1024, 1024], 200)
    plt.imshow(np.abs(pg[0, :, :]), vmax=1)
    # wfStack, fileHeader = wfRead(filename)
    # print(wfStack.shape)
    # print(fileHeader)
    # print(wfStack.dtype)
    # plt.imshow(np.abs((wfStack[0,:,:])), vmax=1, vmin=0)
    plt.show()