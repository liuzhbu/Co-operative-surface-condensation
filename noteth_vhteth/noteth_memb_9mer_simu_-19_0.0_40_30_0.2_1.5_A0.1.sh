#!/bin/bash -l
#$ -l h_rt=72:00:00
#$ -N t15.02.00.0_A1     #Give job a name, hteth_Jmem_Jb_lipAfrac
#$ -pe omp 4  #mpi_4_tasks_per_node 4
#$ -m ea

module load python3

mu=-19
Jb=0.0
L=40
D=30
Jmemb=0.2
Jteth=1.5
lipAfrac=0.1

#numbers on file name correspond to quantities below

python 9mer_memb_simu_noteth_pickup.py "$mu" "$Jb" "$L" "$D" "$Jmemb" "$Jteth" "$lipAfrac"
