# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:14:21 2021

@author: liuzh
"""

def adjacent6(ss, L=20, D=15):
    return [ [(ss[0]-1)%D,ss[1],ss[2]], [(ss[0]+1)%D,ss[1],ss[2]], [ss[0],(ss[1]+1)%L,ss[2]], [ss[0],(ss[1]-1)%L,ss[2]] ,[ss[0],ss[1],(ss[2]-1)%L],[ss[0],ss[1],(ss[2]+1)%L]]

def adjacent6_mem(ss,L=20, D=15):
    return [ [(ss[0]-1),ss[1],ss[2]], [(ss[0]+1),ss[1],ss[2]], [ss[0],(ss[1]+1)%L,ss[2]], [ss[0],(ss[1]-1)%L,ss[2]] ,[ss[0],ss[1],(ss[2]-1)%L],[ss[0],ss[1],(ss[2]+1)%L]]

import numpy as np
import matplotlib.pyplot as plt

configuration_pool = [np.array([-1,0,0]), np.array([1,0,0]), np.array([0,-1,0]), np.array([0,1,0]), np.array([0,0,-1]), np.array([0,0,1])]

def config_pool_9mer(dist_tosurface='bulk', surface_posit=None):
    'generate the configurational pool for 9mer with center' 
    'I. in the bulk; II. 3_site to surface; III. 2_site to surface; IV. 1_site to surface; V. touching surface'
    
    Dirs=[]
    
    'immediate neighbor'
    for i in range(6):
        for j in range(i+1,6):
            dir1, dir2 = configuration_pool[i], configuration_pool[j]
            '2nd neighbor'
            for m in range(6):
                for n in range(6):
                    dir12, dir22 = (dir1 + configuration_pool[m]), (dir2 + configuration_pool[n])
                    if list(dir12)==[0,0,0] or list(dir22)==[0,0,0] or list(dir12)==list(dir22): pass
                    else:
                        '3rd neighbor'
                        for a in range(6):
                            for b in range(6):
                                dir13, dir23 = (dir12 + configuration_pool[a]), (dir22 + configuration_pool[b])
                                if list(dir13)==list(dir23): pass
                                else:
                                    '4th neighbor'
                                    for c in range(6):
                                        for d in range(6):
                                            dir14, dir24 = (dir13 + configuration_pool[c]), (dir23 + configuration_pool[d])
                                            if list(dir14)==[0,0,0] or list(dir24)==[0,0,0] or list(dir14)==list(dir24): pass
                                            else:
                                                'final check of validity of 9mer config'
                                                temp = [dir14,dir13,dir12,dir1, dir2,dir22,dir23,dir24]
                                                valid=True
                                                
                                                if dist_tosurface=='bulk': pass
                                                else:
                                                    if surface_posit=='up':
                                                        for each in temp:
                                                            if each[0]>dist_tosurface:
                                                                valid=False
                                                                break
                                                                
                                                    elif surface_posit=='down':
                                                        for each in temp:
                                                            if each[0]<(-dist_tosurface):
                                                                valid=False
                                                                break
                                                    
                                                
                                                if valid:
                                                    for e in range(8):
                                                        for f in range(e+1,8):
                                                            if list(temp[e])==list(temp[f]):
                                                                valid=False
                                                                break
                                                        if not valid: break

                                                if valid:
                                                    Dirs.append(temp)

                            
    print(['dist to surface = {0:}, {1:}'.format(dist_tosurface, surface_posit), 'pool length ={0:}'.format(len(Dirs))])
        
    return Dirs

def get_self_Enn_pool_9mer(configpool):
    'get self Enn for 9-mers'
    selfEnn_pool=[]
    for each in configpool:
        posits=[np.array([0,0,0])]
        for tailsite in each:
            posits.append(np.array([0,0,0])+tailsite)
            
        posits_list=[]
        for arrayy in posits: posits_list.append(list(arrayy))
        
        if len(posits)!=9:
            print('wrong')
            return
        
        NBRS = -8  # -4 is specific for 9-mers
        for eac in posits:
            for mv in configuration_pool:
                nbrr = eac+mv
                if list(nbrr) in posits_list:
                    NBRS += 0.5
        selfEnn_pool.append(NBRS)
    if len(selfEnn_pool)!=len(configpool):
        print('wrong')
        return
    print([max(selfEnn_pool),min(selfEnn_pool)])
    
    return selfEnn_pool

def read_polympool_history_9mer(polympool_histfile):
    with open(polympool_histfile,'r') as g:
        alllines=g.readlines()[0]
    content = alllines.split('$')[1:-1]
    readout=[]        # store all sweep
    for swp in content:
        temp = swp[3:-3].split(']], [[')
        swep=[]       # store 1 sweep
        for each in temp: # each is a single 3mer
            tempp=[] 
            heep = each.split('], [')
            for pnt in heep:
                pntp=[]
                for nm in pnt.split(','):
                    pntp.append(int(nm))
                tempp.append(pntp)
            swep.append(tempp)
        readout.append(swep)
        
    return readout

def read_state_history(state_histfile):
    with open(state_histfile,'r') as g:
        alllines=g.readlines()[0]
        
    content = alllines.split('$')[1:-1]
    readout_state=[]
    for swp in content:
        cube=[]
        temp = swp[3:-3].split(']], [[')

        for each in temp:
            plane=[]
            eac = each.split('], [')
            for ea in eac:
                hang=[]
                for nm in ea.split(','):
                    hang.append(int(nm))
                plane.append(hang)
            cube.append(plane)
        readout_state.append(cube)
    return readout_state

def initialstate9mer(init_statefile, init_polympoolfile):
    initstate = np.asarray(read_state_history(init_statefile)[0])
    initpolympool = read_polympool_history_9mer(init_polympoolfile)[0]
    return [initstate, initpolympool]

        
def readinconfigs(configfile, L):
    sizeH=L**2
    result_configs=[]
    
    with open(configfile,'r') as f:
        configlines=f.readlines()[0]
    
    resultnum=[] 
    for each in configlines.split('$')[1:-1]:
        for intt in each[1:-1].split(','):
            resultnum.append(int(intt))
        
    lenconfig=int(len(resultnum)/sizeH)
    for i in range(lenconfig):
        tempconfig = np.asarray(resultnum[i*sizeH:(i+1)*sizeH]).reshape(L,L)
        result_configs.append(tempconfig)
        
    return result_configs

def Plot_3D_lattice(lattice_array, polymerpool, polymlength=9, D=15, L=20, latfigtitle=None, latfigname=None):
    latt_fig = plt.figure(figsize=(6,6),dpi=300)
    lf = latt_fig.add_subplot((111),projection='3d')

    lf.set_xticks(np.arange(0,L))
    lf.set_yticks(np.arange(0,L))   
    lf.set_zticks(np.arange(0,D))

    for i in range(D):
        for j in range(L):
            for k in range(L):
                if lattice_array[i,j,k]==1:
                    lf.scatter(j,k,i, color='red', marker='o', s=1)
                
    if polymlength==9:
        for each in polymerpool:
            boundary_check = True
            for k in range(9):
                for h in range(k+1,9):
                    for i in range(3):
                        if abs(each[k][i]-each[h][i])>8:
                            boundary_check = False
#                             print(each)
                            break
                    if not boundary_check: break
                if not boundary_check: break
            if boundary_check:
                lf.plot([each[0][1], each[1][1]], [each[0][2], each[1][2]], [each[0][0], each[1][0]], 'r-')
                lf.plot([each[1][1], each[2][1]], [each[1][2], each[2][2]], [each[1][0], each[2][0]], 'r-')
                lf.plot([each[2][1], each[3][1]], [each[2][2], each[3][2]], [each[2][0], each[3][0]], 'r-')
                lf.plot([each[3][1], each[4][1]], [each[3][2], each[4][2]], [each[3][0], each[4][0]], 'r-')
                lf.plot([each[4][1], each[5][1]], [each[4][2], each[5][2]], [each[4][0], each[5][0]], 'r-')
                lf.plot([each[5][1], each[6][1]], [each[5][2], each[6][2]], [each[5][0], each[6][0]], 'r-')
                lf.plot([each[6][1], each[7][1]], [each[6][2], each[7][2]], [each[6][0], each[7][0]], 'r-')
                lf.plot([each[7][1], each[8][1]], [each[7][2], each[8][2]], [each[7][0], each[8][0]], 'r-')
    
#     for asdf in [[1, 0, 35], [1, 0, 36], [2, 0, 36], [2, 39, 36], [1, 39, 36]]:
#         lf.scatter(asdf[1], asdf[2], asdf[0], color='blue', marker='o', s=1)
    print([len(polymerpool),np.sum(lattice_array)])
    if latfigtitle:
        lf.set_title(latfigtitle)
    if latfigname:
        latt_fig.savefig(latfigname)
        
        
def Plot_3D_lattice_memb(lattice_array, polymerpool, memarr,tetharr, D,L, polymlength=5, latfigtitle=None, latfigname=None):
    latt_fig = plt.figure(figsize=(6,6),dpi=300)
    lf = latt_fig.add_subplot((111),projection='3d')
        
    lf.set_xticks(np.arange(0,L))
    lf.set_yticks(np.arange(0,L))   
    lf.set_zticks(np.arange(0,D))
    lf.set_xlim([-0,L])
    lf.set_ylim([-0,L])
    lf.set_zlim([-0.5,D-0.5])
    # plot membs
    for m in range(L):
        for n in range(L):
            x=np.asarray([m, m+1])
            y=np.asarray([n, n+1])
            X,Y = np.meshgrid(x,y)
            Z = np.zeros(4)-1.1
            Z=np.reshape(Z,(2,2))
            if memarr[m,n]==1:
                lf.plot_surface(X,Y,Z, color='white',zorder=0,alpha=0.2)
            elif memarr[m,n]==-1:
                lf.plot_surface(X,Y,Z, color='black',zorder=0,alpha=0.2)
            elif memarr[m,n]==0:
                lf.plot_surface(X,Y,Z, color='lime',zorder=0,alpha=0.2)
    
    
    for i in range(D):
        for j in range(L):
            for k in range(L):
                if lattice_array[i,j,k]==1:
                    lf.scatter(j,k,i, color='red', marker='o', s=2, zorder=5)
                
    if polymlength==9:
        for each in polymerpool:
            boundary_check = True
            for k in range(9):
                for h in range(k+1,9):
                    for i in range(3):
                        if abs(each[k][i]-each[h][i])>8:
                            boundary_check = False
#                             print(each)
                            break
                    if not boundary_check: break
                if not boundary_check: break
            if boundary_check:
#                 for pt in each:
#                     lf.scatter(pt[1], pt[2], pt[0], color='red', marker='o', s=2, zorder=5)
                lf.plot([each[0][1], each[1][1]], [each[0][2], each[1][2]], [each[0][0], each[1][0]], 'r-')
                lf.plot([each[1][1], each[2][1]], [each[1][2], each[2][2]], [each[1][0], each[2][0]], 'r-')
                lf.plot([each[2][1], each[3][1]], [each[2][2], each[3][2]], [each[2][0], each[3][0]], 'r-')
                lf.plot([each[3][1], each[4][1]], [each[3][2], each[4][2]], [each[3][0], each[4][0]], 'r-')
                lf.plot([each[4][1], each[5][1]], [each[4][2], each[5][2]], [each[4][0], each[5][0]], 'r-')
                lf.plot([each[5][1], each[6][1]], [each[5][2], each[6][2]], [each[5][0], each[6][0]], 'r-')
                lf.plot([each[6][1], each[7][1]], [each[6][2], each[7][2]], [each[6][0], each[7][0]], 'r-')
                lf.plot([each[7][1], each[8][1]], [each[7][2], each[8][2]], [each[7][0], each[8][0]], 'r-')
#     plot tethers   
    tshift=0.
    for m in range(L):
        for n in range(L):
            if tetharr[m,n]==1:
                if memarr[m,n]!=1:
                    print('wrong')
                    return
                lf.plot([m-tshift,m-tshift], [n-tshift,n-tshift], [-1.1,4],'y-',zorder=5,alpha=0.25)
    
    if latfigtitle:
        lf.set_title(latfigtitle)
    if latfigname:
        latt_fig.savefig(latfigname)
    plt.close(latt_fig)
