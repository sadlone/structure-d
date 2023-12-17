import os
import numpy as np
import ttt

def mkrot_dih(newreatant,new_coord_1,rot_list,degree,sign):

    coord_array1=np.array(newreatant)
    coord_array2=np.array(new_coord_1)
    vector_axis=coord_array2[rot_list[1],:]-coord_array1[rot_list[2],:]
    vector_arm1=coord_array1[rot_list[3],:]-coord_array1[rot_list[2],:]
    orth_vector_arm1=vector_arm1-np.dot(ttt.normalization(vector_axis),vector_arm1)*ttt.normalization(vector_axis)
    a=coord_array2[rot_list[0],:]-coord_array1[rot_list[2],:]                  
    orth_vector_arm2 = np.cross(a,vector_axis)*sign
    orthmatrix=ttt.mkaxis(vector_axis)
    targetUrotx=ttt.mkrotx(orthmatrix[:,0],orthmatrix[:,1],orthmatrix[:,2],degree+np.pi/2)
    target_vector=np.dot(targetUrotx,ttt.normalization(orth_vector_arm2)) # vector 为 arm2 沿着axis逆时针转 degree 
    theta=np.arccos(np.dot(ttt.normalization(orth_vector_arm1),ttt.normalization(target_vector)))
    Urotx=ttt.mkrotx(orthmatrix[:,0],orthmatrix[:,1],orthmatrix[:,2],theta)
    judge=np.dot(Urotx,ttt.normalization(orth_vector_arm1))
    if not np.allclose(judge,ttt.normalization(target_vector)):
        theta=2*np.pi-theta
    Urotx=ttt.mkrotx(orthmatrix[:,0],orthmatrix[:,1],orthmatrix[:,2],theta)
    # print(np.linalg.norm(coord_array1[4,:]-coord_array1[5,:]))
    tmp=[np.dot(Urotx,(coord_array1[i]-coord_array1[rot_list[2]]))+coord_array1[rot_list[2]] for i in range(len(coord_array1))]
    # print(np.linalg.norm(tmp[4]-tmp[5]))
    return tmp


def run(cat_path,group_path,keyword_path,pattern,top,relaxation,div,arg,div_arg,degree):

    nowpath=os.curdir
    cat_list=os.listdir(cat_path)
    group_list=os.listdir(group_path)
    keyword=os.listdir(keyword_path)
    cat_list=[cat for cat in cat_list if ".gjf" in cat]
    group_list=[group for group in group_list if ".gjf" in group]

    cat = cat_list[0]
    group = group_list[0]
    total_atomlist = [[[3,1],[4,2]],[[0,2],[1,3]]] # [[[t1,v1],[t2,v2],...片段1所有标记位点],[[t1,v1],[t2,v2],...片段2所有标记位点]]
    # atomlist1=[3,1]     #在平移和转动中使用 t原子 和 v原子 的序号
    atomlist1 = total_atomlist[0][0]
    # atomlist2=[0,2]     #在平移和转动中使用 t原子 和 v原子 的序号
    atomlist2 = total_atomlist[1][0]
    total_dih = [[[0,1],[4,6]],[[0,1],[3,5]]] # [[[d1,t1],[d2,t2],...片段1所有二面角标记位点],[[v1,d1],[v2,d2],...片段2所有二面角标记位点]]
    # rot_dih=[0,1,0,1]   #在旋转二面角中使用 确定二面角所需要的原子序号 arm_atom1——axis_atom1——axis_atom2——arm_atom2
    rot_dih = [total_dih[0][0][0],total_dih[0][0][1],total_dih[1][0][0],total_dih[1][0][1]]

    total_delete_list_cat = [[],[]] #每对标记位点对应的需要额外删除的原子
    delete_list_cat = total_delete_list_cat[0]
    total_delete_list_group = [[],[]]
    delete_list_group = total_delete_list_group[0]

    abs_cat=os.path.join(cat_path,cat)
    coord_array_1,elment_list_1,atomnum_1,chg_multi_1,natom_1=ttt.getcoord(abs_cat)
    coord_list_cat=[[atomnum_1[i]+1,coord_array_1[i,0],coord_array_1[i,1],coord_array_1[i,2]] for i in range(natom_1)]
    filename=""
    filename=filename+"_"+cat.split('.')[0]
    abs_group=os.path.join(group_path,group)
    filename=filename+"_"+group.split('.')[0]
    coord_array_2,elment_list_2,atomnum_2,chg_multi_2,natom_2=ttt.getcoord(abs_group)
    coord_list_group=[[atomnum_2[i]+1,coord_array_2[i,0],coord_array_2[i,1],coord_array_2[i,2]] for i in range(natom_2)]

    opt_coord,atomnum,natom_new,new_coord1_natom=ttt.rotation_vv(coord_list_cat,coord_list_group,atomlist1,atomlist2,delete_list_cat,delete_list_group,pattern,top,relaxation,div,arg,div_arg)
    
    total_delete_afterrot_cat = [atomlist1[1]] + delete_list_cat
    total_delete_afterrot_group = [atomlist2[0] + new_coord1_natom] + [n + new_coord1_natom for n in total_delete_list_group]
    total_delete_afterrot = total_delete_afterrot_cat + total_delete_afterrot_group
    
    # for i in range(len(total_atomlist[1])):
    #     total_atomlist[1][i][0] = total_atomlist[1][i][0] +new_coord1_natom
    #     total_atomlist[1][i][1] = total_atomlist[1][i][1] +new_coord1_natom

    # for i in range(len(total_dih[1])):
    #     total_dih[1][i][0] = total_dih[1][i][0] +new_coord1_natom
    #     total_dih[1][i][1] = total_dih[1][i][1] +new_coord1_natom 


        
    #处理删除特定原子后的原子排序的改变
    for el in total_delete_afterrot:
        for i in range(len(total_atomlist)):
            for j in range(len(total_atomlist[i])):
                for k in range(len(total_atomlist[i][j])):
                    if i == 1:
                        total_atomlist[i][j][k] = total_atomlist[i][j][k] + new_coord1_natom
                    if total_atomlist[i][j][k] > el:
                        total_atomlist[i][j][k] = total_atomlist[i][j][k] - 1

        for i in range(len(total_dih)):
            for j in range(len(total_dih[i])):
                for k in range(len(total_dih[i][j])):
                    if i == 1:
                        total_dih[i][j][k] = total_dih[i][j][k] + new_coord1_natom                    
                    if total_dih[i][j][k] > el:
                        total_dih[i][j][k] = total_dih[i][j][k] - 1

        for i in range(len(total_delete_list_cat)):
            for j in range(len(total_delete_list_cat[i])):
                if total_delete_list_cat[i][j] > el:
                    total_delete_list_cat[i][j] = total_delete_list_cat[i][j] - 1
        
        for i in range(len(total_delete_list_group)):
            for j in range(len(total_delete_list_group[i])):
                total_delete_list_group[i][j] = total_delete_list_group[i][j] + new_coord1_natom
                if total_delete_list_group[i][j] > el:
                    total_delete_list_group[i][j] = total_delete_list_group[i][j] - 1        

    new_coord_1=[[opt_coord[i,0],opt_coord[i,1],opt_coord[i,2]] for i in range(new_coord1_natom)]
    # new_atomnum_1 = [atomnum[i]+1 for i in range(new_coord1_natom)]
    atom_reactant = np.linspace(new_coord1_natom,(natom_new-1),(natom_new-new_coord1_natom)).astype(int)
    newreatant = [[opt_coord[i,0],opt_coord[i,1],opt_coord[i,2]] for i in atom_reactant]
    # atomnum_subs = [atomnum[i]+1 for i in atom_reactant]
    newreatant=mkrot_dih(newreatant,new_coord_1,rot_dih,degree,sign=1)
    opt_coord=np.array(new_coord_1+newreatant)
    # new_atomnum=new_atomnum_1+atomnum_subs
    # atomnum=[i+1 for i in new_atomnum]
    ttt.mkeasyfile("target.gjf",opt_coord,atomnum,"0 1\n")
    natom_new=opt_coord.shape[0]

cat_path="/data/changegroup/structure-derivative/cat"
group_path="/data/changegroup/structure-derivative/group_a"
keyword_path="/data/changegroup/structure-derivative/dft"

pattern = 1         #模式0是t t 原子重合；模式1是t v原子重合
top = False         #top指最后一次替换group后将group的坐标位置提前
relaxation =False    #是否弛豫group部分
                #转动部分
div = 1         #将2pi 划分为div份，并旋转 
arg = 0         #沿旋转轴在[-arg,arg]范围内摆动
div_arg = 1     #将摆动划分的份数 
                #1 0 1 就是不转动且不摆动

#转动二面角
degree=2*np.pi/3      #该角度为arm_atom1——axis_atom1所代表的边，逆时针旋转 degree 到达 axis_atom2——arm_atom2所代表的边

run(cat_path,group_path,keyword_path,pattern,top,relaxation,div,arg,div_arg,degree)