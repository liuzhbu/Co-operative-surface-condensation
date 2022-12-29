module load python3

mu=-19
Jb=0.1
L=40
D=30
Jmemb=0.1
Jteth=1
lipAfrac=0.1
tethrlength=5

#numbers on file name correspond to quantities below

python 9mer_memb_simu_noteth_pickup_varytethlength.py "$mu" "$Jb" "$L" "$D" "$Jmemb" "$Jteth" "$lipAfrac" "$tethrlength"
