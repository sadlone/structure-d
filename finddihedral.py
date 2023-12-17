import numpy as np
import itertools as it
import os
import ttt
import copy
from scipy.optimize import minimize
import cconf_gen
corad =[0.32,0.93,1.23,0.90,0.82,0.77,0.75,0.73,0.72,0.71,1.54,1.36,1.18,1.11,1.06,1.02,
      0.99,0.98,2.03,1.74,1.44,1.32,1.22,1.18,1.17,1.17,1.16,1.15,1.17,1.25,1.26,1.22,1.20,1.44,1.14,
      1.12,2.16,1.91,1.62,1.45,1.34,1.30,1.27,1.25,1.25,1.28,1.34,1.48,1.44,1.41,1.40,1.42,1.33,1.31,
      2.35,1.98,1.69,1.65,1.65,1.64,1.63,1.62,1.85,1.61,1.59,1.59,1.58,1.57,1.56,1.74,1.56,1.44,1.34,
      1.30,1.28,1.26,1.27,1.30,1.34,1.49,1.48,1.47,1.46,1.46,1.45]
elment=["H","He", "Li","Be","B","C","N","O","F","Ne",
"Na","Mg","Al","Si","P","S","Cl","Ar",
"K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
"Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe", 
"Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu", 
"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At"]

pi=np.pi

#def verify_hybrid
#for i in range(natom):

def mkrad(vector1,vector2):
    len1=np.linalg.norm(vector1.astype(float))
    len2=np.linalg.norm(vector2.astype(float))
    rad=np.arccos(np.dot(vector1.astype(float),vector2.astype(float))/len1/len2)
    return rad

def verify_hybrid(rad,n_coord):
    if n_coord==1:
        return 0
    elif n_coord==2:
        if abs(rad-109/180*pi)<abs(rad-2/3*pi):
            return 3
        elif abs(rad-2/3*pi)<abs(rad-pi):
            return 2
        else:
            return 1
    elif n_coord==3:
        if abs(rad-109/180*pi)<abs(rad-2/3*pi):
            return 3
        else:
            return 2
    else :
        return 3
#print(linkmatrix[25,24])
# def finddihedral(atom,recordlist,target):
#     judge=0
#     if linkmatrix[atom,target]==1.0:
#         judge=1
#     else:
#         for i in range(natom):
#             if linkmatrix[atom,i]==1 and i not in recordlist:
#                 recordlist.append(i)
#                 judge=finddihedral(i,recordlist,target)
#                 if judge==1:
#                     break
#     return judge
# atom=21
# target=22
# recordlist=[atom]
# for i in range(natom):
#     if i != target and linkmatrix[atom,i]==1 and i not in recordlist:
#         recordlist.append(i)
#         judge=finddihedral(i,recordlist,target)
# print(judge)
# pass



#tmplist=list(linkmatrix[28,:])
#a=[[m,n] for m in range(natom) for n in range(natom) if m<n and tmplist[m]==1 and tmplist[n]==1]
#n_coord=tmplist.count(1)

def isosurface_bond(n,center,linkmatrix,natom):
    atom_list=[]
    atom_list.append(center)
    fromlist=[0]
    i=0
    exist_list=[center]
    while i<n:
        tmplist=[]
        tmplist=[m for l in range(len(atom_list)) for m in range(natom) if linkmatrix[atom_list[l],m]==1 and m!=fromlist[l] and m not in exist_list]
        tmpfromlist=[atom_list[l] for l in range(len(atom_list)) for m in range(natom) if linkmatrix[atom_list[l],m]==1 and m!=fromlist[l] and m not in exist_list]
        fromlist=tmpfromlist
        atom_list=[]
        for m in tmplist :
            if m not in atom_list:
                atom_list.append(m)
        exist_list=atom_list+exist_list
        i=i+1
    return atom_list

def findbestdihedral(n,fragment_list,orthmatrix_list,rot_list,coord_array,target_list,judgedist,judgesign,ctrlatom1,ctrlatom2,natom,atomnum,ratio,fix_n):
    n=n-1
    m=copy.deepcopy(n)
    tmpsign=copy.deepcopy(judgesign)
    tmpcoord=copy.deepcopy(coord_array)
    tmplink1=fragment_list[m][0]
    tmplink2=fragment_list[m][1]
    orthmatrix=ttt.mkaxis(tmpcoord[target_list[m][0]]-tmpcoord[target_list[m][1]])
    j=0
    mindist=judgedist
    minsign=judgesign
    while j<=9:
        arg=rot_list[j]
        urotx=ttt.mkrotx(orthmatrix[:,0],orthmatrix[:,1],orthmatrix[:,2],arg)
        # print("{}".format(np.linalg.norm(tmpcoord[21,:]-tmpcoord[19,:])))
        for i in range(len(tmplink1)):
            if tmplink1[i]!=natom+1:
                tmpcoord[tmplink1[i]]=np.dot(urotx,(coord_array[tmplink1[i],:]-coord_array[target_list[m][0],:]))+coord_array[target_list[m][0],:]
        # print("{}".format(np.linalg.norm(tmpcoord[21,:]-tmpcoord[19,:])))
        tmpsign[m]=j
        # if tmpsign==[1,0,3,8]:
        #     print("{}".format(np.linalg.norm(tmpcoord[3,:]-tmpcoord[36,:])))
        #     for i in range(tmpcoord.shape[0]):
        #         print("{}\t{}\t{}".format(tmpcoord[i,0],tmpcoord[i,1],tmpcoord[i,2]))
        #     a=1
        if m==0:
            for k in range(fix_n):
                tmplink1_k=fragment_list[k][0]
                tmplink2_k=fragment_list[k][1]
                frag1=[tmpcoord[i] for i in tmplink1_k if i!= natom+1]
                frag2=[tmpcoord[i] for i in tmplink2_k if i!= natom+1]
                residue=[tmpcoord[i]  for i in range(tmpcoord.shape[0]) if i not in tmplink1_k and i not in tmplink2_k]
                dist2=ttt.min_dist_3arg(frag1,frag2,residue)
                tmpdist_k=np.abs(np.linalg.norm(tmpcoord[ctrlatom1-1,:]-tmpcoord[ctrlatom2-1,:])-ratio*(corad[atomnum[ctrlatom1-1]]+corad[atomnum[ctrlatom2-1]]))-100*dist2
                if k==0:
                    tmpdist=np.abs(np.linalg.norm(tmpcoord[ctrlatom1-1,:]-tmpcoord[ctrlatom2-1,:])-ratio*(corad[atomnum[ctrlatom1-1]]+corad[atomnum[ctrlatom2-1]]))-100*dist2
                if tmpdist_k>tmpdist:
                    tmpdist=copy.deepcopy(tmpdist_k)
        else:
            tmpdist,tmpsign=findbestdihedral(m,fragment_list,orthmatrix_list,rot_list,tmpcoord,target_list,judgedist,tmpsign,ctrlatom1,ctrlatom2,natom,atomnum,ratio,fix_n)
            if tmpdist>tmpdist:
                tmpdist=tmpdist
        if tmpdist<mindist:
            mindist=copy.deepcopy(tmpdist)
            minsign=copy.deepcopy(tmpsign)
        j=j+1
    return mindist,minsign



def rot_sp3atom(file,ctrlatom1,ctrlatom2,nowpath,ratio=1.3,div=10):
    coord_array,elment_list,atomnum,chg_multi,natom=ttt.getcoord(file)
    linkmatrix=ttt.mklinkmatrix(coord_array,atomnum)
    hybrid_list=[]
    n_coord_list=[]
    for el in range(natom):
        tmplist=list(linkmatrix[el,:])
        a=[[m,n] for m in range(natom) for n in range(natom) if m<n and tmplist[m]==1 and tmplist[n]==1]
        n_coord=tmplist.count(1)
        n_coord_list.append(n_coord)
        rad=0
        if len(a)>=1:
            for i in a:
                vector1=coord_array[el,0:3]-coord_array[i[0],0:3]
                vector2=coord_array[el,0:3]-coord_array[i[1],0:3]
                rad=mkrad(vector1,vector2)/len(a)+rad
        hybrid_list.append(verify_hybrid(rad,n_coord))
    #a=[hybrid_list[x] for x in range(natom) if linkmatrix[79,x]==1][0] 
    for i in range(len(hybrid_list)):
        if hybrid_list[i] ==0:
            hybrid_list[i]=[hybrid_list[x] for x in range(natom) if linkmatrix[i,x]==1][0]
    linklist_ctrlatom1=ttt.mklink(ctrlatom1-1,1,linkmatrix)
    if ctrlatom2-1 not in linklist_ctrlatom1:
        seq_fragment1=np.array([[i,elment[atomnum[i]],coord_array[i,0],coord_array[i,1],coord_array[i,2]] for i in range(natom) if i in linklist_ctrlatom1])
        seq_fragment2=np.array([[i,elment[atomnum[i]],coord_array[i,0],coord_array[i,1],coord_array[i,2]] for i in range(natom) if i not in linklist_ctrlatom1])
        with open("fragment1.txt","w") as f1:
            f1.write("{}\n".format(seq_fragment1.shape[0]))
            for i in range(seq_fragment1.shape[0]):
                 f1.write("{}\t{}\t{}\t{}\n".format(seq_fragment1[i][1],seq_fragment1[i][2],seq_fragment1[i][3],seq_fragment1[i][4]))
        with open("fragment2.txt","w") as f1:
            f1.write("{}\n".format(seq_fragment2.shape[0]))
            for i in range(seq_fragment2.shape[0]):
                 f1.write("{}\t{}\t{}\t{}\n".format(seq_fragment2[i][1],seq_fragment2[i][2],seq_fragment2[i][3],seq_fragment2[i][4]))
        with open('input.txt','r') as f1:
            lines=f1.readlines()
        with open('control.txt','w') as f1:
                f1.write("{} {}\n".format([i+1 for i in range(len(seq_fragment1)) if int(seq_fragment1[i][0])==ctrlatom1-1][0],[i+1 for i in range(len(seq_fragment2)) if int(seq_fragment2[i][0])==ctrlatom2-1][0]))
                for i in range(len(lines)):
                    f1.write(lines[i])
        os.system("./scanRot")
        coord_array,_,_,_,_=ttt.getcoord("output.gjf")
        fragment2_coord=coord_array[0:seq_fragment2.shape[0],0:3]
        fragment1_coord=coord_array[seq_fragment2.shape[0]:,0:3]
        new_seq_fragment1=np.hstack([seq_fragment1[:,0].astype(int).reshape(seq_fragment1.shape[0],1),fragment1_coord])
        new_seq_fragment2=np.hstack([seq_fragment2[:,0].astype(int).reshape(seq_fragment2.shape[0],1),fragment2_coord])
        new_total_coord=np.vstack([new_seq_fragment1,new_seq_fragment2])
        new_total_coord=new_total_coord[np.argsort(new_total_coord[:,0])]
        ttt.mkeasyfile(os.path.join(nowpath,"output.gjf"),new_total_coord[:,1:4],atomnum,chg_multi)
    else:    
        # linklist_ctrlatom1_sp3_notincycle=[i for i in range(natom) if (hybrid_list[i]==3 or hybrid_list[i]==2) and n_coord_list[i]!=1 and i in linklist_ctrlatom1 and ttt.ifnotincycle(i,natom,linkmatrix)]
        # target_list=[]
        # for i in range(len(linklist_ctrlatom1_sp3_notincycle)):
        #     a=linklist_ctrlatom1_sp3_notincycle[i]
        #     list_near=[i for i in range(natom) if linkmatrix[a,i]==1 and np.sum(linkmatrix[i,:])>1]
        #     for j in range(len(list_near)):
        #         b=list_near[j]
        #         if ttt.judge_effective_rot(a,b,coord_array,linkmatrix,natom,ctrlatom1,ctrlatom2):
        #             if a<b:
        #                 target_list.append([a,b])
        #             else:
        #                 target_list.append([b,a])
        # tmp=[]
        # for i in target_list:
        #     if i not in tmp:
        #         if ctrlatom1-1 not in i and ctrlatom2-1 not in i:
        #             tmp.append(i)
        # target_list=tmp
        # n=len(target_list)
        # if n>4:
        #     n_list=[round(i*n/4)for i in range(4)]
        #     target_list=[target_list[n_list[i]]for i in range(4)]
        #     n=4
        # if n!=0:
        #     linkmatrix_list=[]
        #     fragment_list=[]
        #     orthmatrix_list=[]
        #     for i in range(n):
        #         tmplinkmatrix=copy.deepcopy(linkmatrix)
        #         tmplinkmatrix[target_list[i][0],target_list[i][1]]=0
        #         tmplinkmatrix[target_list[i][1],target_list[i][0]]=0
        #         linkmatrix_list.append(tmplinkmatrix)
        #         tmplink1=ttt.mklink(target_list[i][0],1,linkmatrix_list[i])
        #         tmplink2=ttt.mklink(target_list[i][1],1,linkmatrix_list[i])
        #         fragment_list.append([tmplink1,tmplink2])
        #         tmporthmatrix=ttt.mkaxis(coord_array[target_list[i][0]]-coord_array[target_list[i][1]])
        #         orthmatrix_list.append(tmporthmatrix)
        #     rot_list=[(2*pi*i+1)/div for i in range(div)]
        #     sign=[]
        #     for i in range(n):
        #         sign.append(0)
        #     dist=np.abs(np.linalg.norm(coord_array[ctrlatom1-1,:]-coord_array[ctrlatom2-1,:])-1.3*(corad[atomnum[ctrlatom1-1]]+corad[atomnum[ctrlatom2-1]]))
        #     judgedist,judgesign=findbestdihedral(n,fragment_list,orthmatrix_list,rot_list,coord_array,target_list,dist,sign,ctrlatom1,ctrlatom2,natom,atomnum,ratio,copy.deepcopy(n))

        #     target_coord=copy.deepcopy(coord_array)
        #     for m in range(n-1,-1,-1):
        #         orthmatrix=ttt.mkaxis(target_coord[target_list[m][0]]-target_coord[target_list[m][1]])
        #         tmplink1=fragment_list[m][0]
        #         arg=rot_list[judgesign[m]]
        #         urotx=ttt.mkrotx(orthmatrix[:,0],orthmatrix[:,1],orthmatrix[:,2],arg)
        #         for i in range(len(tmplink1)):
        #             if tmplink1[i]!=natom+1:
        #                 target_coord[tmplink1[i]]=np.dot(urotx,(target_coord[tmplink1[i],:]-target_coord[target_list[m][0],:]))+target_coord[target_list[m][0],:]
        linkmatrix=ttt.mklinkmatrix(coord_array,atomnum)
        linkmatrix=np.array(linkmatrix,np.int32)
        linkmatrix[ctrlatom1-1,ctrlatom2-1]=1
        linkmatrix[ctrlatom2-1,ctrlatom1-1]=1
        d0=np.zeros([natom,natom])
        for i in range(natom):
            for j in range(natom):
                if linkmatrix[i,j]==1:
                    d0[i,j]=ttt.corad[atomnum[i]]+ttt.corad[atomnum[j]]
        d0[ctrlatom1-1,ctrlatom2-1]=(ttt.corad[atomnum[ctrlatom1-1]]+ttt.corad[atomnum[ctrlatom2-1]])*1.3
        d0[ctrlatom2-1,ctrlatom1-1]=(ttt.corad[atomnum[ctrlatom1-1]]+ttt.corad[atomnum[ctrlatom2-1]])*1.3
        opt_coord=opt_structure(file,ctrlatom1,ctrlatom2,linkmatrix,d0)
        ttt.mkeasyfile(os.path.join(nowpath,"output.gjf"),opt_coord,atomnum,chg_multi)
        # else:
        #     os.system("cp {} {}".format(file,os.path.join(nowpath,"output.gjf")))

        # for i in range(len(tmplink1)):
        #     print("{} {} {}".format(target_coord[i][0],target_coord[i][1],target_coord[i][2]))



# file='/home/crz/disk_crz/auto_tssearch/test2/testoverall5-3/input2-2_44-1.gjf'
# nowpath='/home/crz/disk_crz/auto_tssearch/test2/testoverall5-3/'
# ctrlatom1=4
# ctrlatom2=37
# rot_sp3atom(file,ctrlatom1,ctrlatom2,nowpath)
def opt_structure(file,linkmatrix,d0,k=1,c=0.01,tol=0.0001,exponent=8,fix_natom=None):
    coord_array,elment_list,atomnum,chg_multi,natom=ttt.getcoord(file)
    natom=coord_array.shape[0]
    coords=coord_array.reshape(3*natom)

    # def get_optmize_structure(coords, bonds, k, c, d0, tol, exponent=8, fixed_bonds=None, fixed_idxs=None):
    result=minimize(cconf_gen.v,
                    coords,
                    args=(linkmatrix,k,d0,c,exponent,fix_natom),
                    tol=tol,
                    jac=cconf_gen.dvdr,
                    options={"maxiter":400}    
                    )
    opt_coord=result.x.reshape(natom, 3)
    return opt_coord,result.fun

# opt_structure(file,k=1,c=0.01,tol=0.00001,exponent=8,ctrlatom1=4,ctrlatom2=37)


# bond_list3_3=[[m,n] for m in range(natom) for n in range(natom) if m<n and linkmatrix[m,n]==1 and hybrid_list[m]==3 and hybrid_list[n]==3]
# bond_list3_2=[[m,n] for m in range(natom) for n in range(natom) if m<n and linkmatrix[m,n]==1 and hybrid_list[m]==3 and hybrid_list[n]==2]+ \
# [[m,n] for m in range(natom) for n in range(natom) if m<n and linkmatrix[m,n]==1 and hybrid_list[m]==2 and hybrid_list[n]==3]    
#hybrid_list
#tmplist=list(linkmatrix[79,:])
#a=[[m,n] for m in range(natom) for n in range(natom) if m<n and tmplist[m]==1 and tmplist[n]==1]
#n_coord=tmplist.count(1)
#rad=0
#if len(a)>=1:
#    for i in a:
#        vector1=coord_array[0,0:3]-coord_array[i[0],0:3]
#        vector2=coord_array[0,0:3]-coord_array[i[1],0:3]
#        rad=mkrad(vector1,vector2)/len(a)+rad
#hybrid=verify_hybrid(rad,n_coord)




