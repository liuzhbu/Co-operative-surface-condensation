module load python3

mu=-19
Jb=0.1
L=40
D=30
Jmemb=0.1
Jteth=1
obstacle_frac=0.1
tether_frac=0.1

#numbers on file name correspond to quantities below

python 9mer_memb_simu_floating_obstacle.py "$mu" "$Jb" "$L" "$D" "$Jmemb" "$Jteth" "$obstacle_frac" "$tether_frac"
