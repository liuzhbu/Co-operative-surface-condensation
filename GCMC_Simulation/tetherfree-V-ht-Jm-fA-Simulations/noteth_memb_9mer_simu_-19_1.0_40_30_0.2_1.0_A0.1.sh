module load python3

mu=-19
Jb=1.0
L=40
D=30
Jmemb=0.2
Jteth=1.0
lipAfrac=0.1

#numbers on file name correspond to quantities below

python 9mer_memb_simu_noteth_pickup.py "$mu" "$Jb" "$L" "$D" "$Jmemb" "$Jteth" "$lipAfrac"
