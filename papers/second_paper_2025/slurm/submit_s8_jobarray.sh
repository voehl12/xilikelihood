#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=s8_copula
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --output=logs/s8_copula_%A_%a.out
#SBATCH --error=logs/s8_copula_%A_%a.err


# Default datapoint configurations
DATAPOINTS="3 6 15 30 54"

module load stack/2024-06 gcc/12.2.0 swig/4.1.1-ipvpwcc python/3.11.6 cuda/12.1.1 nccl/2.18.3-1 openblas/0.3.24 python_cuda/3.11.6 cmake/3.27.7 cudnn/9.2.0
export JAX_PLATFORMS=cpu
source /cluster/home/veoehl/xilikelihood/2ptlikelihood/bin/activate

# Run the analysis for the specific job array task
python /cluster/home/veoehl/xilikelihood/papers/second_paper_2025/analysis/s8_copula_comparison.py \
    --job-index ${SLURM_ARRAY_TASK_ID} \
    --n-datapoints ${DATAPOINTS} \
    --correlation-type all \
    --s8-grid medium \
    --student-t-dof 5.0 \
    --save-individual 

