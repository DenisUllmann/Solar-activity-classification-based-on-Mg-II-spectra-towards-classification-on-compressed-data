import matplotlib.pyplot as plt
import numpy as np
import tqdm
import irisreader.data.mg2k_centroids as centroid
from irisreader import observation
from irisreader.utils import download
import os
os.chdir( "." )
np.seterr(divide='ignore', invalid='ignore')
#load the observation
download( "20131014_164320_3860258246", target_directory="." )
obs = observation( "20131014_164320_3860258246" )

print(obs.sji[0].line_info)
alldata = obs.raster("Mg II k")[:,:,:].clip(min=0)
timeindex, yindex, spectralindex = alldata.shape
print(timeindex, yindex, spectralindex)
centroidarray=[]
#trasform to centroid data
for i in tqdm.trange(timeindex):
    temp = []
    if 0 < i < timeindex:
        for j in range (yindex):
            if 0 < j < yindex:
                if sum(alldata[i,j,:]) < 0.1:
                    temp.append(-10)
                else:
                    temp.append(centroid.assign_mg2k_centroids(alldata[i, j].reshape([1,]+list(alldata[i, j].shape)))[0])
        centroidarray.append(temp)
Z = np.asarray(centroidarray)
Z = np.transpose(Z)

#save to file
np.save('20131014_164320_3860258246', Z)
print('npy file saved')

#draw observation in centroid data
fig, ax = plt.subplots(1, 1)
mesh = ax.pcolormesh(Z)
fig.colorbar(mesh)
plt.xlabel( "time step" )
plt.ylabel( "y on slit" )
plt.savefig('smth.png')
plt.show()

