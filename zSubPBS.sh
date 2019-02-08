#!/bin/bash
######################################
# Replace everything inside <...> with
# suitable settings for your jobs
######################################
#PBS -q gpuq
# NOTE: reserve six cores as a way of blocking half of the node from other jobs.
#PBS -l nodes=1:ppn=6:pascal
#PBS -l walltime=75:00:00
#PBS -V
#PBS -j oe
#PBS -N N18
#PBS -M kevinshen@ucsb.edu
#PBS -m abe
######################################
inputfile=params.in
outputfile=z.log
polyftsdir=~/code/PolyFTS/bin/Release
######################################

cd $PBS_O_WORKDIR
outdir=${PBS_O_WORKDIR}
rundir=${outdir}
username=`whoami`

############# TO USE LOCAL SCRATCH FOR INTERMEDIATE IO, UNCOMMENT THE FOLLOWING
#if [ ! -d /scratch_local/${username} ]; then
#  rundir=/scratch_local/${username}/${PBS_JOBID}
#  mkdir -p $rundir
#  cp ${PSB_O_WORKDIR}/* $rundir
#  cd $rundir
#fi
#####################################################

echo CUDA_VISIBLE:  $CUDA_VISIBLE_DEVICES
echo PBS_GPUFILE:   `cat $PBS_GPUFILE`
echo " "

# Fetch the device ID for the GPU that has been assigned to the job
GPUDEV=`cat $PBS_GPUFILE | awk '{print $1}'` #Takes $PBS_GPUFILE, cats it into a file, then (|) pipes it to an awk script '{...}' which prints the $1 variable of the line
if [ -z $GPUDEV ]; then
  echo "ERROR finding $PBS_GPUFILE; using default GPU deviceid=0"
  GPUDEV=0
fi

echo "Assigned GPU device: $GPUDEV"
echo " "
echo "=== === === Begin Running === === ==="
echo " "

# Use default openmm test
python -m simtk.testInstallation
python ~/lib/MDtools/sim.py $inputfile --deviceid=$GPUDEV #> $outdir/${outfile}

# Now test my script. SLURM already masks the available GPUs,
#   so if running on single GPU, openmm's device index should be 0
#   (i.e. the index into the $CUDA_VISIBLE_DEVICES environt var.)
# see https://github.com/pandegroup/openmm/issues/2190
#srun --gres=gpu:1 python simulateSetup.py --deviceid=$GPUDEV
#srun --gres=gpu:1 python simulate_cont.py --deviceid=0

# Prepare the run by substituting the CUDA select device line
# Check whether the line exists first
#grep "CUDA_[Ss]elect[Dd]evice" ${inputfile} > /dev/null
#if [ $? -ne 0 ]; then
#  echo "CUDA_SelectDevice line not found in $inputfile"
#  exit 1
#fi
#sed -i "s/\(CUDA_[Ss]elect[Dd]evice\).*/\1 = ${GPUDEV}/g" ${inputfile}
#touch z.device
#devicefile = z.device
#sed -i "s/\(CUDA_[Ss]elect[Dd]evice\).*/\1 = ${GPUDEV}/g" ${devicefile}

# Run the job
#${polyftsdir}/PolyFTSGPU.x ${inputfile} > ${outdir}/${outputfile}

# Copy back results
if [ "$rundir" != "$outdir" ]; then
  mv * ${outdir}
fi

# Force good exit code here - e.g., for job dependency
exit 0

