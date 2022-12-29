# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:42:03 2021

@author: liuzh
"""

''' This is the updated Pickup version of the simulation code, made to work at initial or pickup time of the simu'''

from IDP9mer_bulk_simu_functions import configuration_pool
from IDP9mer_bulk_simu_functions import adjacent6_mem

"functions for reading the initial states of the IDPs (5mer here)"
from IDP9mer_bulk_simu_functions import initialstate9mer
from IDP9mer_bulk_simu_functions import read_polympool_history_9mer
from IDP9mer_bulk_simu_functions import read_state_history

"functions for reading the initial states/history of the memb"
from IDP9mer_bulk_simu_functions import readinconfigs

from IDP9mer_bulk_simu_functions import config_pool_9mer
from IDP9mer_bulk_simu_functions import get_self_Enn_pool_9mer

from IDP9mer_bulk_simu_functions import Plot_3D_lattice_memb

import numpy as np
from numpy.random import rand
from numpy.random import randint

bulk_9mer_pool = config_pool_9mer()
dist3_surup_9mer_pool = config_pool_9mer(3,'up')
dist3_surdown_9mer_pool = config_pool_9mer(3,'down')
dist2_surup_9mer_pool = config_pool_9mer(2,'up')
dist2_surdown_9mer_pool = config_pool_9mer(2,'down')
dist1_surup_9mer_pool = config_pool_9mer(1,'up')
dist1_surdown_9mer_pool = config_pool_9mer(1,'down')
dist0_surup_9mer_pool = config_pool_9mer(0,'up')
dist0_surdown_9mer_pool = config_pool_9mer(0,'down')

bulk_9mer_Enn_pool = get_self_Enn_pool_9mer(bulk_9mer_pool)
dist3_surup_9mer_Enn_pool = get_self_Enn_pool_9mer(dist3_surup_9mer_pool)
dist3_surdown_9mer_Enn_pool = get_self_Enn_pool_9mer(dist3_surdown_9mer_pool)
dist2_surup_9mer_Enn_pool = get_self_Enn_pool_9mer(dist2_surup_9mer_pool)
dist2_surdown_9mer_Enn_pool = get_self_Enn_pool_9mer(dist2_surdown_9mer_pool)
dist1_surup_9mer_Enn_pool = get_self_Enn_pool_9mer(dist1_surup_9mer_pool)
dist1_surdown_9mer_Enn_pool = get_self_Enn_pool_9mer(dist1_surdown_9mer_pool)
dist0_surup_9mer_Enn_pool = get_self_Enn_pool_9mer(dist0_surup_9mer_pool)
dist0_surdown_9mer_Enn_pool = get_self_Enn_pool_9mer(dist0_surdown_9mer_pool)

# Temporary adjustment
import sys
mu = float(sys.argv[1])
Jb = float(sys.argv[2])
L = int(sys.argv[3])
D = int(sys.argv[4])
Jmemb = float(sys.argv[5])
Jteth = float(sys.argv[6])
obstacle_frac = float(sys.argv[7])
tether_frac = float(sys.argv[8])

foldername = '9mer_memb_simu_mu={0:}_Jb={1:}_Jmem={2:}_Jteth={3:}_Fobst_frac={4:}_tethfrac={5:}_L={6:}_D={7:}'.format(mu,Jb,Jmemb,Jteth,obstacle_frac,tether_frac,L,D)
import os

APPEND=0
if not os.path.exists(foldername+'/9mer_memb_state_history_int100swp'):
    os.makedirs(foldername)
    APPEND=False
else:
    APPEND=True

pickuplength=0
if APPEND:    
    state_history_pickup = read_state_history(foldername+'/9mer_memb_state_history_int100swp')    
    state_pickup = np.asarray(state_history_pickup[-1])
    
    polympool_history_pickup = read_polympool_history_9mer(foldername+'/9mer_memb_polymerpool_history_int100swp')   
    polympool_pickup = polympool_history_pickup[-1]
    
    board_history_pickup = readinconfigs(foldername+'/9mer_memb_board_history_int100swp',L)    
    board_pickup = board_history_pickup[-1]
    
    tether_history_pickup = readinconfigs(foldername+'/9mer_memb_tether_history_int100swp',L)    
    tether_pickup = tether_history_pickup[-1]

    pickuplength = len(state_history_pickup)
    
    if len(polympool_history_pickup)!=pickuplength or len(board_history_pickup)!=pickuplength or len(tether_history_pickup)!=pickuplength:
        print('VERY WRONG, STOP, unlucky that previous simulation stopped while writing int100swp files')
        with open('VERY_WRONG_STOP.txt','w') as wr:
            wr.writelines('VERY WRONG, STOP, unlucky that previous simulation stopped while writing int100swp files')
        
        # try to open a non-existent file to force the program to stop
        with open('time_to_stop.txt','r') as ww:
            alllines = ww.readlines()
        

    import gc
    del state_history_pickup
    del polympool_history_pickup
    del board_history_pickup
    del tether_history_pickup
    gc.collect()

class GCMC_9mer_memb():
    ''' Simulating the bulk phase polymers '''
    def __init__(self, L, obstacle_frac, Jmemb):
        
        # initialize the polymers 
        if APPEND:
            self.state = state_pickup          # stores lattice occupancy
            self.polymer_pool = polympool_pickup
            self.board  = board_pickup
            self.tether = tether_pickup
            
        else:
            self.init_polymer = initialstate9mer('9mer_state_init_160chain_30_40', '9mer_polymerpool_init_160chain_30_40')        
            self.state = self.init_polymer[0]           # stores lattice occupancy
            self.polymer_pool = self.init_polymer[1]    # stores 5mers (5 3d coordinates as one 5 mer)
            
            #self.state_history = [self.state]
            #self.polymer_pool_history = [self.polymer_pool]
            
            # initialize the membrane and tether
            if obstacle_frac==0:
                self.board  = readinconfigs('memb_obstaclfrac=0_tetherfrac=0.1_L=40_init',L)[0] 
                self.tether = readinconfigs('memb_obstaclfrac=0_tetherfrac=0.1_L=40_init',L)[1]
                
            elif obstacle_frac==0.1:
                self.board  = readinconfigs('memb_obstaclfrac=0.1_tetherfrac=0.1_L=40_init',L)[0]
                self.tether = readinconfigs('memb_obstaclfrac=0.1_tetherfrac=0.1_L=40_init',L)[1]
                
            elif obstacle_frac==0.2:
                self.board  = readinconfigs('memb_obstaclfrac=0.2_tetherfrac=0.1_L=40_init',L)[0]
                self.tether = readinconfigs('memb_obstaclfrac=0.2_tetherfrac=0.1_L=40_init',L)[1]
                
            elif obstacle_frac==0.3:
                self.board  = readinconfigs('memb_obstaclfrac=0.3_tetherfrac=0.1_L=40_init',L)[0]
                self.tether = readinconfigs('memb_obstaclfrac=0.3_tetherfrac=0.1_L=40_init',L)[1]
                
            elif obstacle_frac==0.4:
                self.board  = readinconfigs('memb_obstaclfrac=0.4_tetherfrac=0.1_L=40_init',L)[0]
                self.tether = readinconfigs('memb_obstaclfrac=0.4_tetherfrac=0.1_L=40_init',L)[1]
                
            elif obstacle_frac==0.5:
                self.board  = readinconfigs('memb_obstaclfrac=0.5_tetherfrac=0.1_L=40_init',L)[0]
                self.tether = readinconfigs('memb_obstaclfrac=0.5_tetherfrac=0.1_L=40_init',L)[1]
            
        #self.board_history = [self.board] # stores all the board configuration during the simulation
        #self.tether_history = [self.tether] # stores all tether positions (1 tether, 0 no tether)
        
        self.etable = []
        for t in np.arange(-16,17,1):
            self.etable.append(np.exp(Jmemb*t))
        print(len(self.etable))
        print('self.etable done')

        self.tot_mc_move = 0
        self.tot_add_move = 0
        self.tot_delet_move = 0
        self.tot_snake_move = 0
        self.tot_cluster_move = 0
        self.tot_ising_swap = 0
        self.tot_tether_move = 0
        
        self.accepted_addmove = 0
        self.accepted_deletmove = 0
        self.accepted_snakemove = 0
        self.accepted_clustermove = 0
        self.accepted_ising_swap = 0
        self.accepted_tether_move = 0
        
        self.tethernum0=np.sum(self.tether)
        
        
    # monte carlo moves sweep through all (memb-spin, tether), IDP, IDP addition/deletion 
    def mc_sweep_9mer_memb_Float(self, mu, Jb, Jmemb, Jteth, L, D):
        ''' This is to execute the monte carlo moves using Metropolis algorithm such that detailed
            balance condition is satisified'''
        ''' randomize move (snake move and cluster move) addition and deletion, 
            give 20% moves to addition, 20% to deletion, 60% to move'''
        ''' number of move step per sweep is the number of polymers in the system before each sweep'''
        
        'randomized the sequence of moves is safer !!!!!'
        V=L*L*D
        
        state = self.state.copy()
        polymers_pool = self.polymer_pool.copy()
        memb = self.board.copy()
        tethers = self.tether.copy()
        
        #tethernum = np.sum(tethers)
        polymnum=len(polymers_pool)
        mcstep = max(int((polymnum+self.tethernum0)*5/3), 100) # This is now the MC step without Ising swaps
        polym_ratio = polymnum/(polymnum+self.tethernum0)
        
        for i in range(mcstep):
            self.tot_mc_move+=1
            
            if rand()<0.6: # do movement move               # check 一些赋值会不会产生问题          
                
                if rand()<=polym_ratio:  # do polymer move
                    j = randint(0,len(polymers_pool))
                    
                    if rand() < 2.2/len(polymers_pool):      # do cluster move
                        self.tot_cluster_move+=1

                        acluster, to_check = [], []
                        for eap in polymers_pool[j]:
                            acluster.append(eap)
                            to_check.append(eap)
                        
                        while len(to_check)!=0:
                            for each in to_check:
                                for eac in adjacent6_mem(each,L,D):
                                    if eac[0] in [-1,D]: pass
                                    elif state[eac[0],eac[1],eac[2]]==1 and eac not in acluster:
                                        acluster.append(eac)
                                        to_check.append(eac)
                                to_check.remove(each)
                        if len(to_check)!=0:
                            print('cluster move error')
                                
                        dirr=configuration_pool[randint(0,6)]
                        oktogo=True
                        newcluster=[]
                        for ep in acluster:
                            if (ep[0]+dirr[0]) in [-1,D]:
                                oktogo=False
                                break
                            
                            else:
                                togo = list((np.asarray(ep)+dirr)%[D+200,L,L])
                                newcluster.append(togo)
                                if state[togo[0],togo[1],togo[2]]==1 and togo not in acluster:
                                    oktogo=False
                                    break
                            
                        if oktogo: 
                            for eachh in newcluster:
                                for eacc in adjacent6_mem(eachh,L,D):
                                    if eacc[0] in [-1,D]: pass
                                    elif state[eacc[0],eacc[1],eacc[2]]==1 and eacc not in acluster:
                                        # and not in newcluster, but if it's not in acluster, it can't be in newcluster
                                        oktogo=False
                                        break
                                if not oktogo:
                                    break
                                    
                        if oktogo:   
                            if len(acluster)!=len(newcluster):
                                print('cluster identification error')
                                
                            tethc_old, tethc_new  = 0,0
                    
                            for c in range(len(acluster)):
                                # first time showing tether length is 2
                                # nn interaction between polymer and tether
                                'tether length relevant'
                                if acluster[c][0] in [0,1,2,3,4]:    
                                    if tethers[acluster[c][1], acluster[c][2]]==1: tethc_old+=1
                                    
                                if newcluster[c][0] in [0,1,2,3,4]:
                                    if tethers[newcluster[c][1], newcluster[c][2]]==1: tethc_new+=1

                            cost = -Jteth*(tethc_new - tethc_old)
                            if cost<=0 or rand() < np.exp(-cost): 
                                self.accepted_clustermove += 1   

                                for em in acluster:
                                    state[em[0],em[1],em[2]]=0
                                for en in newcluster:
                                    state[en[0],en[1],en[2]]=1                        
                            
                                add_to_pool, dele_from_pool = [], []
                            
                                for eh in polymers_pool:
                                    moved=False
                                    if eh[0] in acluster:
                                        moved=True    
                                    if moved:
                                        new9mer=[]
                                        for pt in eh:
                                            npt0 = (pt[0]+dirr[0])
                                            npt1 = (pt[1]+dirr[1])%L
                                            npt2 = (pt[2]+dirr[2])%L
                                            new9mer.append([npt0,npt1,npt2])
                                        dele_from_pool.append(eh)
                                        add_to_pool.append(new9mer)

                                for q in range(len(add_to_pool)):
                                    polymers_pool.append(add_to_pool[q])
                                    polymers_pool.remove(dele_from_pool[q])


                    else:                                   # do snake move
                        self.tot_snake_move+=1
                        
                        head = [0,8][randint(0,2)]
                        dir = configuration_pool[randint(0,6)]
                        
                        if (polymers_pool[j][head][0]+dir[0]) in [-1,D] : pass
                        else:
                            headn = list((np.asarray(polymers_pool[j][head])+dir)%[D+200,L,L])
                            
                            if state[headn[0],headn[1],headn[2]]==1:
                                pass

                            else:
                                
                                helper=[0,8]
                                helper.remove(head)
                                tail = helper[0]

                                tailold = polymers_pool[j][tail].copy()
                                
                                sumold, sumnew = -1, -1
                                for dr in configuration_pool:
                                    
                                    if (tailold[0]+dr[0]) in [-1,D]: pass
                                    else: sumold += state[(tailold[0]+dr[0]), (tailold[1]+dr[1])%L, (tailold[2]+dr[2])%L]

                                    nbrn = [(headn[0]+dr[0]), (headn[1]+dr[1])%L, (headn[2]+dr[2])%L]
                                    if nbrn == tailold or nbrn[0] in [-1,D]: pass    #                                 if nbrn in siteso: pass
                                    else: sumnew +=state[nbrn[0], nbrn[1], nbrn[2]]

                                tethold, tethnew = 0,0
            
                                'tether length relevant'
                                if tailold[0] in [0,1,2,3,4]:
                                    if tethers[tailold[1], tailold[2]]==1: tethold=1
                                    
                                if headn[0] in [0,1,2,3,4]:
                                    if tethers[headn[1], headn[2]]==1: tethnew=1

                                cost = -Jb*(sumnew - sumold) - Jteth*(tethnew - tethold)        
                                if cost<=0 or rand() < np.exp(-cost):
                                    self.accepted_snakemove += 1
                                    state[headn[0], headn[1], headn[2]] = 1
                                    state[tailold[0], tailold[1], tailold[2]] = 0

                                    sitesn = [headn]
                                    sitesn.append(polymers_pool[j][head])
                                    if head==8:
                                        sitesn.append(polymers_pool[j][7])
                                        sitesn.append(polymers_pool[j][6])
                                        sitesn.append(polymers_pool[j][5])
                                        sitesn.append(polymers_pool[j][4])
                                        sitesn.append(polymers_pool[j][3])
                                        sitesn.append(polymers_pool[j][2])
                                        sitesn.append(polymers_pool[j][1])
                                    elif head==0:
                                        sitesn.append(polymers_pool[j][1])
                                        sitesn.append(polymers_pool[j][2])
                                        sitesn.append(polymers_pool[j][3])
                                        sitesn.append(polymers_pool[j][4])
                                        sitesn.append(polymers_pool[j][5])
                                        sitesn.append(polymers_pool[j][6])
                                        sitesn.append(polymers_pool[j][7])
                                    polymers_pool[j] = sitesn

                else: # do tether move        
                    self.tot_tether_move+=1
                    st=0
                    while st!=1:
                        tio, tjo = randint(0,L), randint(0,L)
                        st = tethers[tio, tjo]
                    dirt = [[1,0],[-1,0],[0,1],[0,-1]][randint(0,4)]
                    tin, tjn = (tio+dirt[0])%L, (tjo+dirt[1])%L
                    if memb[tin,tjn]!=1 or tethers[tin, tjn]==1 : pass
                    else:
                        'tether length relevant'

                        teth_new, teth_old =0,0
                        for z in range(5):
                            teth_new += state[z,tin,tjn]
                            teth_old += state[z,tio,tjo]
                        cost = -Jteth*(teth_new - teth_old) #len1 tether
                        
                        if cost<=0 or rand() < np.exp(-cost):
                            self.accepted_tether_move+=1
                            tethers[tio,tjo]=0
                            tethers[tin,tjn]=1
                                    
            else:           # do addition/deletion move
                if rand()<0.5:     # do addition
                    self.tot_add_move+=1
                    center = [randint(0,D), randint(0,L), randint(0,L)]

                    if center[0]==0:
                        xuhao = randint(0,55184)
                        tail = dist0_surdown_9mer_pool[xuhao]
                        summ = dist0_surdown_9mer_Enn_pool[xuhao]
                        polena = 55184
                        
                    elif center[0]==1:
                        xuhao = randint(0,138626)
                        tail = dist1_surdown_9mer_pool[xuhao]
                        summ = dist1_surdown_9mer_Enn_pool[xuhao]
                        polena = 138626
                        
                    elif center[0]==2:
                        xuhao = randint(0,184332)
                        tail = dist2_surdown_9mer_pool[xuhao]
                        summ = dist2_surdown_9mer_Enn_pool[xuhao]
                        polena = 184332
                    
                    elif center[0]==3:
                        xuhao = randint(0,193398)
                        tail = dist3_surdown_9mer_pool[xuhao]
                        summ = dist3_surdown_9mer_Enn_pool[xuhao]
                        polena = 193398
                        
                    #'middle'
                    
                    elif center[0]==(D-4):
                        xuhao = randint(0,193398)
                        tail = dist3_surup_9mer_pool[xuhao]
                        summ = dist3_surup_9mer_Enn_pool[xuhao]
                        polena = 193398
                        
                    elif center[0]==(D-3):
                        xuhao = randint(0,184332)
                        tail = dist2_surup_9mer_pool[xuhao]
                        summ = dist2_surup_9mer_Enn_pool[xuhao]
                        polena = 184332
                    
                    elif center[0]==(D-2):
                        xuhao = randint(0,138626)
                        tail = dist1_surup_9mer_pool[xuhao]
                        summ = dist1_surup_9mer_Enn_pool[xuhao]
                        polena = 138626
                        
                    elif center[0]==(D-1):
                        xuhao = randint(0,55184)
                        tail = dist0_surup_9mer_pool[xuhao]
                        summ = dist0_surup_9mer_Enn_pool[xuhao]
                        polena = 55184
                        
                    else:
                        xuhao = randint(0,193983)
                        tail = bulk_9mer_pool[xuhao]
                        summ = bulk_9mer_Enn_pool[xuhao]
                        polena = 193983
                
                    'choosing configurations from approapriate pools already gurantees no penetration'   
                    tail1 = list((np.array(center)+tail[0])%[D+200,L,L])
                    tail2 = list((np.array(center)+tail[1])%[D+200,L,L])
                    tail3 = list((np.array(center)+tail[2])%[D+200,L,L])
                    tail4 = list((np.array(center)+tail[3])%[D+200,L,L])
                    tail5 = list((np.array(center)+tail[4])%[D+200,L,L])
                    tail6 = list((np.array(center)+tail[5])%[D+200,L,L])
                    tail7 = list((np.array(center)+tail[6])%[D+200,L,L])
                    tail8 = list((np.array(center)+tail[7])%[D+200,L,L])
                    
                    Add = True
                    for st in [center, tail1, tail2, tail3, tail4, tail5, tail6, tail7, tail8]:
                        if state[st[0],st[1],st[2]]==1:
                            Add=False
                            break
                            
                    if Add:
                        for each in [tail1, tail2, tail3, tail4, center, tail5, tail6, tail7, tail8]:
                            'tether length relevant'
                            if each[0] in [0,1,2,3,4] and tethers[each[1],each[2]]==1:
                                Add=False
                                break
                            else:
                                for dr in configuration_pool:
                                    Zz = each[0]+dr[0]
                                    if Zz in [-1,D]: pass
                                    else: 
                                        # addin polymer cannot form any strong contact with polymer or tether
                                        if state[Zz, (each[1]+dr[1])%L, (each[2]+dr[2])%L]==1:
                                            Add=False
                                            break
                                    
                            if Add==False: break
                     
                    if Add:
                            
                        cost = - mu - Jb*summ
                        if rand() < np.exp(-cost)*(polena*V/(len(polymers_pool)+1)):
                            self.accepted_addmove+=1
                            for each in [tail1, tail2, tail3, tail4, center, tail5, tail6, tail7, tail8]:
                                state[each[0], each[1], each[2]]=1
                            polymers_pool.append([tail1, tail2, tail3, tail4, center, tail5, tail6, tail7, tail8])

                else:              # do deletion
                    if len(polymers_pool)==0:
                        pass
                    else:
                        self.tot_delet_move+=1
                        j=randint(0,len(polymers_pool))

                        Remove = True
                        for each in polymers_pool[j]:
                            'tether length relevant'
                            if each[0] in [0,1,2,3,4] and tethers[each[1],each[2]]==1:
                                Remove = False
                                break
                            else:
                                for dr in configuration_pool:
                                    Zz = each[0]+dr[0]
                                    if Zz in [-1,D]: pass
                                    else:
                                        if state[Zz, (each[1]+dr[1])%L, (each[2]+dr[2])%L]==1 and [Zz, (each[1]+dr[1])%L, (each[2]+dr[2])%L] not in polymers_pool[j]:
                                            Remove=False
                                            break
                            if not Remove: break 
                        
                        if Remove:
                            summ=-8
                            for each in polymers_pool[j]:
                                for dr in configuration_pool:
                                    Zz = each[0]+dr[0]
                                    if Zz in [-1,D]: pass
                                    else:
                                        if [Zz, (each[1]+dr[1])%L, (each[2]+dr[2])%L] in polymers_pool[j]:
                                            summ += 0.5
                                
                            cost = mu + Jb*summ
                            if polymers_pool[j][4][0] in [0,D-1]: polen = 55184
                            elif polymers_pool[j][4][0] in [1,D-2]: polen = 138626
                            elif polymers_pool[j][4][0] in [2,D-3]: polen = 184332
                            elif polymers_pool[j][4][0] in [3,D-4]: polen = 193398
                            else: polen = 193983

                            if rand() < np.exp(-cost)*(len(polymers_pool)/polen/V):
                                self.accepted_deletmove+=1
                                for each in polymers_pool[j]:
                                    state[each[0], each[1], each[2]]=0
                                polymers_pool.remove(polymers_pool[j])          
                    
        # Now sweep through every spin once    
        memb_sites = np.arange(L**2)
        random_memb_sites = np.random.permutation(memb_sites)
        for sitee in random_memb_sites:
            i= int(sitee/L)
            j= sitee%L
            if tethers[i,j]==1: pass
            else:
                self.tot_mc_move+=1
                self.tot_ising_swap+=1
                s1 = memb[i,j]

                s2=s1
                a,b=0,0
                while s2==s1 or tethers[a,b]==1:
                    a = np.random.randint(0, L)
                    b = np.random.randint(0, L)
                    s2 = memb[a,b]
                
                if b==j and a in [(i+1)%L, (i-1)%L]:
                    sum1 = memb[(i+1)%L,j] + memb[(i-1)%L,j] + memb[i,(j+1)%L] + memb[i,(j-1)%L]-s2
                    sum2 = memb[(a+1)%L,b] + memb[(a-1)%L,b] + memb[a,(b+1)%L] + memb[a,(b-1)%L]-s1
                elif a==i and b in [(j+1)%L, (j-1)%L]:
                    sum1 = memb[(i+1)%L,j] + memb[(i-1)%L,j] + memb[i,(j+1)%L] + memb[i,(j-1)%L]-s2
                    sum2 = memb[(a+1)%L,b] + memb[(a-1)%L,b] + memb[a,(b+1)%L] + memb[a,(b-1)%L]-s1
                else:
                    sum1 = memb[(i+1)%L,j] + memb[(i-1)%L,j] + memb[i,(j+1)%L] + memb[i,(j-1)%L]
                    sum2 = memb[(a+1)%L,b] + memb[(a-1)%L,b] + memb[a,(b+1)%L] + memb[a,(b-1)%L]
                
                cost = (s1*sum2+s2*sum1)-(s1*sum1+s2*sum2) # in np.arange(-16,18,2) ,17 possible value
                xu = int(cost+16)
                ecost = self.etable[xu]
                
                if cost >= 0:
                    self.accepted_ising_swap+=1
                    memb[i,j]=s2
                    memb[a,b]=s1
#                     elif rand() < np.exp(-cost):
                elif rand() < ecost:
                    self.accepted_ising_swap+=1
                    memb[i,j]=s2
                    memb[a,b]=s1
                    
        return [state, polymers_pool, memb, tethers]
    
memb_9mer = GCMC_9mer_memb(L, obstacle_frac, Jmemb)

# Create same Files recording information every 100 sweeps
if APPEND:
    pass

else:
    with open(foldername+'/9mer_memb_polymerpool_history_int100swp','w') as f:
        f.writelines('$')
        f.writelines(str(memb_9mer.polymer_pool))
        f.writelines('$')
          
    with open(foldername+'/9mer_memb_state_history_int100swp','w') as k:
        k.writelines('$')
        listit=[]
        for layer in memb_9mer.state:
            ceng = []
            for row in layer:
                hang=[]
                for each in row:
                    hang.append(each)
                ceng.append(hang)
            listit.append(ceng)
        k.writelines(str(listit))
        k.writelines('$')
          
    with open(foldername+'/9mer_memb_board_history_int100swp','w') as f:
        f.writelines('$')
        Nums=[]
        for lin in memb_9mer.board:
            for num in lin:
                Nums.append(num)
        f.writelines(str(Nums))
        f.writelines('$')
      
    with open(foldername+'/9mer_memb_tether_history_int100swp','w') as f:
        f.writelines('$')
        Nums=[]
        for lin in memb_9mer.tether:
            for num in lin:
                Nums.append(num)
        f.writelines(str(Nums))
        f.writelines('$')
          
    with open(foldername+'/9mer_memb_polypool_length_int100swp','w') as p:
        p.writelines('polymer pool length for each sweep \n')
        p.writelines(str(len(memb_9mer.polymer_pool))+'\n')
        
    with open(foldername+'/mc_status.txt','w') as p:
        p.writelines('mc status \n')

#for itr in range(1000000):
for itr in np.arange(pickuplength*100+1,1000001,1):
    if itr==20000:
        with open(foldername+'/paramcheck_mu={0:}_Jb={1:}_Jmem={2:}_Jteth={3:}_obst_frac={4:}_tethfrac={5:}_L={6:}_D={7:}.txt'.format(mu,Jb,Jmemb,Jteth,obstacle_frac,tether_frac,L,D),'w') as q:
            q.writelines('paramcheck')
    
    result = memb_9mer.mc_sweep_9mer_memb_Float(mu, Jb, Jmemb, Jteth, L, D)
    
    memb_9mer.state = result[0]
    memb_9mer.polymer_pool = result[1]
    memb_9mer.board = result[2]
    memb_9mer.tether = result[3]
        
   
    if itr%100==0:
        with open(foldername+'/9mer_memb_polymerpool_history_int100swp','a') as f:
            f.writelines(str(memb_9mer.polymer_pool))
            f.writelines('$')
        
        with open(foldername+'/9mer_memb_state_history_int100swp','a') as k:
            listit=[]
            for layer in memb_9mer.state:
                ceng = []
                for row in layer:
                    hang=[]
                    for each in row:
                        hang.append(each)
                    ceng.append(hang)
                listit.append(ceng)
            k.writelines(str(listit))
            k.writelines('$')
        
        with open(foldername+'/9mer_memb_board_history_int100swp','a') as f:
            Nums=[]
            for lin in memb_9mer.board:
                for num in lin:
                    Nums.append(num)
            f.writelines(str(Nums))
            f.writelines('$')
    
        with open(foldername+'/9mer_memb_tether_history_int100swp','a') as f:
            Nums=[]
            for lin in memb_9mer.tether:
                for num in lin:
                    Nums.append(num)
            f.writelines(str(Nums))
            f.writelines('$')
        
        with open(foldername+'/9mer_memb_polypool_length_int100swp','a') as p:
            p.writelines(str(len(memb_9mer.polymer_pool))+'\n')
            
    if itr%10000==0:
        figtitle = '9mer_memb_mu={0:}_Jb={1:}_Jmem={2:}_Jteth={3:}\nFloating_obstacle={4:}_teth={5:}'.format(mu,Jb,Jmemb,Jteth,obstacle_frac,tether_frac)
        figname = foldername+'/9mer_memb_mu=-{0:}_Jb={1:}_Jmem={2:}_Jteth={3:}_Floating_obstacle={4:}_teth={5:}_L={6:}_D={7:}_swp={8:}.png'.format(mu,Jb,Jmemb,Jteth,obstacle_frac,tether_frac,L,D,itr)

        Plot_3D_lattice_memb(memb_9mer.state, memb_9mer.polymer_pool,memb_9mer.board,memb_9mer.tether, D,L ,9,figtitle,figname)
        
        with open(foldername+'/mc_status.txt','a') as q:
            q.writelines('current sweep number = {0:}'.format(itr))
            q.writelines('\n total MC move = {0:}'.format(memb_9mer.tot_mc_move))
            q.writelines('\n total add move = {0:}, '.format(memb_9mer.tot_add_move))
            q.writelines('accepted add move = {0:}, '.format(memb_9mer.accepted_addmove))
            q.writelines('add move accept ratio = {0:}'.format(memb_9mer.accepted_addmove/(memb_9mer.tot_add_move+1e-6)))
            
            q.writelines('\n total delete move = {0:}, '.format(memb_9mer.tot_delet_move))
            q.writelines('accepted delete move = {0:}, '.format(memb_9mer.accepted_deletmove))
            q.writelines('delete move accept ratio = {0:}'.format(memb_9mer.accepted_deletmove/(memb_9mer.tot_delet_move+1e-6)))
            
            q.writelines('\n total snake move = {0:}, '.format(memb_9mer.tot_snake_move))
            q.writelines('accepted snake move = {0:}, '.format(memb_9mer.accepted_snakemove))
            q.writelines('snake move accept ratio = {0:}'.format(memb_9mer.accepted_snakemove/(memb_9mer.tot_snake_move+1e-6)))
            
            q.writelines('\n total cluster move = {0:}, '.format(memb_9mer.tot_cluster_move))
            q.writelines('accepted cluster move = {0:}, '.format(memb_9mer.accepted_clustermove))
            q.writelines('cluster move accept ratio = {0:}'.format(memb_9mer.accepted_clustermove/(memb_9mer.tot_cluster_move+1e-6)))
            
            q.writelines('\n total ising swap move = {0:}, '.format(memb_9mer.tot_ising_swap))
            q.writelines('accepted ising swap move = {0:}, '.format(memb_9mer.accepted_ising_swap))
            q.writelines('ising swap move accept ratio = {0:}'.format(memb_9mer.accepted_ising_swap/(memb_9mer.tot_ising_swap+1e-6)))
            
            q.writelines('\n total tether move = {0:}, '.format(memb_9mer.tot_tether_move))
            q.writelines('accepted tether move = {0:}, '.format(memb_9mer.accepted_tether_move))
            q.writelines('tether move accept ratio = {0:}'.format(memb_9mer.accepted_tether_move/(memb_9mer.tot_tether_move+1e-6)))
    