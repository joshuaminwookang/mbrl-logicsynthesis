#!/bin/bash
# Job name:
#SBATCH --job-name=results
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio	
#
# Quality of Service:
#SBATCH --qos=savio_normal
#
# Wall clock limit:
#SBATCH --time=01:00:00
#
## Command(s) to run:
python /global/home/users/minwoo_kang/mlrl_synthesis/scripts/results.py --i /global/scratch/users/minwoo_kang/run_vtr_testset_rand
