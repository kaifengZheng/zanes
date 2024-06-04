#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=167:59:59
#SBATCH --partition=extended-40core
#SBATCH --output=array_test.%N_%a.log
#SBATCH --job-name=400
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module load slurm
module load  python/3.9.7
module load 10.0.0
git clone https://github.com/kaifengZheng/zanes.git

#module load intel/oneAPI/2021.4

#module load  mpi/2021.4.0

##module load gnu-parallel


for dir in input/*
   do
	   files=($dir)
	   echo "${files[0]}"
	   python fitting.py --input "${files[0]}" -o output/ -dt ProQEXAFS -bs -1 -os _rbkg1_ss &

   done
wait



#########write xyz######
#python trainingset_gen_mpi.py
