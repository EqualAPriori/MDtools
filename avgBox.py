import numpy as np
import mdtraj
from pymbar import timeseries

import argparse as ap
parser = ap.ArgumentParser(description = 'Calculate box dimension statistics')
parser.add_argument( 'traj', type=str, default='output.dcd', help = 'trajectory file')
parser.add_argument( 'top', type=str, default='chk_00.pdb', help = 'topology file')
args = parser.parse_args()

traj=args.traj
top=args.top
t = mdtraj.load(traj,top=top)
d = t.unitcell_lengths
#d = np.loadtxt('boxdimensions.dat')
xs = d[:,0]
ys = d[:,1]
zs = d[:,2]


n_fields = 6
fulldata = np.vstack( [xs,ys,zs, xs*ys, xs*zs, ys*zs, xs*ys*zs] )
print(xs.shape)
print(fulldata.shape)
summary = np.zeros( (fulldata.shape[0],n_fields) )


for ind in range(fulldata.shape[0]):
    row = fulldata[ind,:]
    t0,g,Neff = timeseries.detectEquilibration( row )
    data_equil = row[t0:]
    indices = timeseries.subsampleCorrelatedData(data_equil, g=g)
    sub_data = data_equil[indices]
    print('Detected correlation statistics: t0 {}, efficiency {}, Neff_max {}'.format(t0,g,Neff))

    avg = sub_data.mean()
    std = sub_data.std()
    err = sub_data.std()/np.sqrt( len(indices) )

    summary[ind,:] = [avg,std,err,t0,g,Neff]

print('rows: x,y,z, x*y,x*z,y*z, x*y*z\navg\tstd\terr \tt0 \tg_eq \tN_eff')
print(summary)
np.savetxt('box_stats.txt',summary,header='rows: x,y,z, x*y,x*z,y*z, x*y*z\navg\tstd\terr \tt0 \tg_eq \tN_eff')
np.savetxt('box_lengths.txt',d)
t[-1].save('lastframe.pdb')


