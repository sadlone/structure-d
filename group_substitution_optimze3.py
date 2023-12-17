
import numpy as np
import copy
import ttt
import pandas as pd
import os
def getcoord_struct_csv(path,struct_csv):
    abs_struct_csv=os.path.join(path,struct_csv)
    csv_data=pd.read_csv(abs_struct_csv,encoding="gb18030")
    try:
        target_list=csv_data['标记'].fillna("").tolist()
    except Exception as e:
        target_list=csv_data['target'].fillna("").tolist()
    try:
        group_list = csv_data['group'].fillna("").tolist()
    except Exception as e:
        group_list = []
    try:
        joint_list = csv_data['joint'].fillna("").tolist()
    except Exception as e:
        joint_list = []
    try:
        dihedral_angle = csv_data['dihedralangle'].fillna("").tolist()
        dihedral_angle = [dih for dih in dihedral_angle if dih != ""]
        dih_dict = {}
        for dih in dihedral_angle:
            dih_dict[dih.split("_")[0]]=int(dih.split("_")[1])
    except Exception as e:
        dih_dict = {}    
    atomnum_list=np.array(csv_data['atomic number'].tolist()).reshape(csv_data['atomic number'].shape[0],1)
    coord_array=np.array(csv_data[['x','y','z']])
    try:
        chrg=int(csv_data['charge'][0])
        multi=int(csv_data['multi'][0])
    except Exception :
        chrg=""
        multi=""
    coord_list=np.hstack((atomnum_list,coord_array)).tolist()
    return coord_list,target_list,chrg,multi,group_list,joint_list,dih_dict

def adjConcat(a, b):
    '''
    将a,b两个矩阵沿对角线方向斜着合并，空余处补零[a,0.0,b]
    得到a和b的维度，先将a和b*a的零矩阵按行（竖着）合并得到c，再将a*b的零矩阵和b按行合并得到d
    将c和d横向合并
    '''
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    right = np.row_stack((np.zeros((lena, lenb)), b))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    result = np.hstack((left, right))  # 将左右矩阵水平拼接
    return result

def run(cat_paths,group_paths,keyword_path,outputpath,pattern,top,relaxation,div,arg,div_arg):
    try:
        os.mkdir(outputpath)
    except Exception as e:
        pass
    nowpath=os.getcwd()
    for cat_path in cat_paths:
        cat_csvs=os.listdir(cat_path)
        total_group_list = [[]]
        tmp = []
        # 提取所有替换片段的可能性，同为一个group的（如group_a）总是同时替换
        for group_path in group_paths:
            for group_list in total_group_list:
                for group in os.listdir(group_path):
                    newgroup_list = group_list + [group]
                    tmp.append(newgroup_list)           
                # total_group_list.remove(group_list)
            total_group_list = tmp
            tmp = []
        keyword_csvs=os.listdir(keyword_path)
    
        for cat_csv in cat_csvs:
            if ".csv" in cat_csv:
                #output命名
                outputname = outputpath + "/{}".format(cat_csv.split('.')[0])
                #提取csv
                for keyword_csv in keyword_csvs:
                    if ".csv" in keyword_csv:
                        mem,procs,keyword,link1=ttt.getkeword_csv(keyword_path,keyword_csv)
                        coord_list_cat,target_list_cat,chrg_cat,multi_cat,group_list_cat,joint_list_cat,dihedral_angle=getcoord_struct_csv(cat_path,cat_csv)
                        total_joint_list = {joint_list_cat[i]:i for i in range(len(joint_list_cat)) if joint_list_cat[i] != ""}
                        chg_multi="{} {}\n".format(chrg_cat,multi_cat)               
                        natom_cat=len(coord_list_cat)
                        coord_array_cat=np.array([[coord_list_cat[i][1],coord_list_cat[i][2],coord_list_cat[i][3]] for i in range(len(coord_list_cat))])
                        atomnum=[int(i[0])-1 for i in coord_list_cat]
                        linkmatrix = ttt.mklinkmatrix(coord_array_cat,atomnum)
                        total_target_list_cat = [i for i in range(natom_cat) if "t" in str(target_list_cat[i]) ]
                        # total_dict_cat1 = [[total_target_list_cat[i],[j for j in range(natom_cat) if "v" in str(target_list_cat[j]) and str(target_list_cat[j]).split("v")[-1] == str(target_list_cat[total_target_list_cat[i]]).split("t")[-1]],group_list_cat[total_target_list_cat[i]],[l for l in range(natom_cat) if "d" in str(target_list_cat[l]) and str(target_list_cat[l]).split("d")[-1] == str(target_list_cat[total_target_list_cat[i]]).split("t")[-1]]] for i in range(len(total_target_list_cat)) ]                                
                        total_dict_cat = [[total_target_list_cat[i],j,group_list_cat[j],[l for l in range(natom_cat) if "d" in str(target_list_cat[l]) and str(target_list_cat[l]).split("d")[-1] == str(target_list_cat[j]).split("v")[-1]],] for i in range(len(total_target_list_cat)) for j in range(natom_cat) if "v" in str(target_list_cat[j]) and "t"+str(target_list_cat[j]).split("v")[-1] in target_list_cat[total_target_list_cat[i]]]                                
                        #对某一个csv进行group替换
                        for group_csvs_list in total_group_list:
                            outputname = outputpath + "/{}".format(cat_csv.split('.')[0])
                            coord_list_cat1 = copy.deepcopy(coord_list_cat)
                            atomnum1 = copy.deepcopy(atomnum)
                            total_dict_cat1 = copy.deepcopy(total_dict_cat)
                            for g,group_csv in enumerate(group_csvs_list):
                                if ".csv" in group_csv:
                                    # try: 
                                        #名字拓展
                                        outputname=outputname+"_{}".format(group_csv.split('.')[0])    
                                        #提取csv
                                        coord_list_group,target_list_group,chrg_group,multi_group,group_list_group,_,_=getcoord_struct_csv(group_paths[g],group_csv)                                    
                                        natom_group=len(coord_list_group)
                                        # target_list_cat1=[i for i in range(natom_cat) if "t" in target_list_cat[i] and group_list_cat[i] == group_paths[g].split("/")[-1]]
                                        target_list_group1=[i for i in range(natom_group) if "t" in target_list_group[i] ]
                                        delete_list_group = [i for i in range(natom_group) if "d" in target_list_group[i] ]                                                           
                                        atomnum_group=[int(i[0])-1 for i in coord_list_group]
                                        dict_cat1=[]                                  
                                        dict_cat1=[[total_dict_cat1[i][0],total_dict_cat1[i][1]] for i in range(len(total_dict_cat1)) if total_dict_cat1[i][2] in group_paths[g].split("/")[-1]]
                                        dict_group1=[]
                                        dict_group1=[target_list_group1[0],[j for j in range(natom_group) if "v" in target_list_group[j]][0]]
                                        if len(dict_cat1)>0:                                            
                                            for k in range(len(dict_cat1)):
                                                #更新连接原子对的序号（如果有增减原子）                                 
                                                dict_cat1=[[total_dict_cat1[i][0],total_dict_cat1[i][1]] for i in range(len(total_dict_cat1)) if total_dict_cat1[i][2] in group_paths[g].split("/")[-1]]
                                                #根据pattern确定移动形式
                                                if pattern ==0 :
                                                    atomlist1=[dict_cat1[k][0],dict_cat1[k][1]]
                                                    atomlist2=[dict_group1[0],dict_group1[1]]
                                                else :
                                                    atomlist1=[dict_cat1[k][1],dict_cat1[k][0]]
                                                    atomlist2=[dict_group1[0],dict_group1[1]]
                                                if natom_group>1:#group是否只有一个原子（如Br）
                                                    delete_list_cat = [j[3] for j in total_dict_cat1 if j[0] == dict_cat1[k][0]][0]
                                                    #替换片段脚本ttt.rotation_vv                                                        
                                                    opt_coord,atomnum1,natom_new,new_coord1_natom=ttt.rotation_vv(coord_list_cat1,coord_list_group,atomlist1,atomlist2,delete_list_cat,delete_list_group,pattern,top,relaxation,div,arg,div_arg)
                                                    natom_1=len(coord_list_cat)
                                                    coord_list_cat1=[[atomnum1[i]+1,opt_coord[i,0],opt_coord[i,1],opt_coord[i,2]] for i in range(natom_new)]
                                                    if pattern == 0:
                                                        total_delete_list = atomlist1 + delete_list_cat
                                                    else:
                                                        total_delete_list = [atomlist1[0]] + delete_list_cat
                                                    tmp_total_dict_cat1 = copy.deepcopy(total_dict_cat1)
                                                    tmp_total_joint_list = copy.deepcopy(total_joint_list)
                                                    for delete_el in total_delete_list: #处理删除特定原子后的原子排序的改变
                                                        for i in range(len(total_dict_cat1)):
                                                            if total_dict_cat1[i][0]>delete_el:
                                                                tmp_total_dict_cat1[i][0]=tmp_total_dict_cat1[i][0]-1
                                                            if total_dict_cat1[i][1]>delete_el:                                                                
                                                                tmp_total_dict_cat1[i][1]=tmp_total_dict_cat1[i][1]-1
                                                            for j in range(len(total_dict_cat1[i][3])):
                                                                if total_dict_cat1[i][3][j]>delete_el:                                                                
                                                                    tmp_total_dict_cat1[i][3][j]=tmp_total_dict_cat1[i][3][j]-1
                                                        for key_joint in total_joint_list:
                                                            if total_joint_list[key_joint] >delete_el:
                                                                tmp_total_joint_list[key_joint] = tmp_total_joint_list[key_joint] -1 
                                                    total_dict_cat1 = tmp_total_dict_cat1
                                                    total_joint_list = tmp_total_joint_list
                                                    pass

                                                                                                                                                                              
                                                    pass
                                                else:
                                                    for i in range(len(coord_list_cat1)):
                                                        if i==dict_cat1[k][0]:
                                                            coord_list_cat1[i]=[atomnum_group[target_list_group1[0]]+1,coord_list_cat1[i][1],coord_list_cat1[i][2],coord_list_cat1[i][3]]
                                                            break
                                                    atomnum1=[int(coord_list_cat1[i][0]-1) for i in range(len(coord_list_cat))]
                                                    opt_coord=np.array([[coord_list_cat1[i][1],coord_list_cat1[i][2],coord_list_cat1[i][3]] for i in range(len(coord_list_cat1)) ])
                                            atomnum1=[i+1 for i in atomnum1]
    
                            # columns = ['atomic number', 'x', 'y', 'z']
                            # coor_df = pd.DataFrame(data=coord_list_cat, columns=columns)
                            # coor_df['center number'] = list(range(1, len(coord_list_cat)+1))
                            # coor_df['charge'] = np.nan
                            # coor_df['multi'] = np.nan
                            # coor_df["target"]=np.nan
                            # coor_df["joint"]=np.nan
                            # coor_df["dihedralangle"]=np.nan
                            # # print(coor_df)
                            # coor_df.loc[0, 'charge'] = chrg_cat
                            # coor_df.loc[0, 'multi'] = multi_cat
                            # if len(dihedral_angle) != 0:
                            #     coor_df['dihedralangle'] = dihedral_angle
                            # for key_joint in total_joint_list:
                            #     coor_df.loc[total_joint_list[key_joint], "joint"] = key_joint
                            # coor_df = coor_df[["target","joint","dihedralangle",'center number', 'atomic number', 'x', 'y', 'z', 'charge', 'multi']]
                            # coor_df.to_csv(outputname + '.csv')
                        
                            if top:#实行top的功能
                                opt_coord = np.vstack((opt_coord[new_coord1_natom:len(opt_coord),:],opt_coord[0:new_coord1_natom,:]))
                                atomnum = np.vstack((atomnum[new_coord1_natom:len(atomnum),:],opt_coord[0:new_coord1_natom,:]))
                            ttt.mkoutput(opt_coord,atomnum1,mem,procs,keyword,chrg_cat,multi_cat,link1,outputname)

cat_path=["/data/changegroup/structure-derivative/cat"]
group_paths=["/data/changegroup/structure-derivative/group_1","/data/changegroup/structure-derivative/group_2"]
keyword_path="/data/changegroup/structure-derivative/dft"
outoutpath = "/data/changegroup/structure-derivative/output"
pattern = 1         #模式0是t t 原子重合；模式1是t v原子重合
top = False         #top指最后一次替换group后将group的坐标位置提前
relaxation =True    #是否弛豫group部分
                #转动部分
div = 1         #将2pi 划分为div份，并旋转 
arg = 0         #沿旋转轴在[-arg,arg]范围内摆动
div_arg = 1     #将摆动划分的份数 
                #三者1 0 1 就是不转动且不摆动

run(cat_path,group_paths,keyword_path,outoutpath,pattern,top,relaxation,div,arg,div_arg)