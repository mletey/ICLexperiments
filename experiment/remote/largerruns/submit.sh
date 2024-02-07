#!/bin/bash
# taus40d200_arrays.sbatch
# 
#SBATCH --job-name=taus40d200
#SBATCH -c 10
#SBATCH -t 2-00:00:00
#SBATCH -p seas_gpu,kempner,pehlevan_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/largerruns/outputdump/run_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/largerruns/outputdump/run_%A_%a.err
#SBATCH --array=1-500
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    tauind=$(( ($1 - 1) / 10 ))
    avgind=$(( ($1 - 1) % 10 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="resultsdump"
newdir="$parentdir/job_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
pkldir="$parentdir/job_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/pickles"
errdir="$parentdir/job_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/errors"
mkdir "$newdir"
mkdir "$pkldir"
mkdir "$errdir"
python tau.py $newdir $tauind $avgind