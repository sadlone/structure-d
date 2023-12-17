import numpy as np
import itertools as it
import os
import copy 
import pandas as pd
import finddihedral
np.set_printoptions(suppress=True)
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
metal=["Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
"Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Cs","Ba","La","Ce",
"Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu", 
"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"]
pi=np.pi
def getcoord(file1):
    blanklist=[]
    atomnum=[]
    coord_list=[]
    elment_list=[]
    with open(file1,'r') as file:
        f2_lines=file.readlines()
        nblank=0
        for f_line in f2_lines:
            if f_line in ['\n',' \n','\r\n',' \r\n']:
                blanklist.append(nblank)
            nblank=nblank+1
        coord_in_f=f2_lines[blanklist[1]+2:blanklist[2]]
        chg_multi=f2_lines[blanklist[1]+1]
        for r in coord_in_f:
            atomnum.append(r.split()[0])
            coord_list.append([r.split()[1],r.split()[2],r.split()[3]])
    coord_array=np.array(coord_list)
    coord_array=coord_array.astype(float)
    natom=blanklist[2]-blanklist[1]-2
    for i in atomnum:
        if not i.isdigit():
            elment_list.append(i)
            atomnum[atomnum.index(i)]=[elment.index(j) for j in elment if str(j)==str(i) ][0]
    if len(elment_list)==0:
        atomnum=[int(i)-1 for i in atomnum]
    return coord_array,elment_list,atomnum,chg_multi,natom

def mkeasyfile(file,coord,atomnum,chrg_spin):
    with open(file,'w') as f:
        f.write("%chk={}.chk\n".format(file.split("/")[-1].split(".")[0]))
        f.write("%nprocs=1\n")
        f.write("%mem=2gb\n")
        f.write("#opt=nomicro ugbs external='./xtb.sh'\n")
        f.write("\n")
        f.write("aaa\n")
        f.write("\n")
        f.write("{}".format(chrg_spin))
        for i in range(len(atomnum)):
            f.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\t\n".format(atomnum[i]+1,coord[i,0],coord[i,1],coord[i,2]))
        f.write("\n")
def find_same_H(linkmatrix,elimination):
    H_list=[]
    for i in range(len(elimination)):
        if elimination[i][0] not in H_list:
            H_list.append(elimination[i][0])
    tmplinkmatrix=np.matmul(linkmatrix,linkmatrix)
    sameH_list=[]
    for i in H_list:
        judge=0
        tmplist=[atom+1 for atom in range(tmplinkmatrix.shape[0]) if tmplinkmatrix[atom,i-1]==1]
        for j in tmplist:
            if j in sameH_list:
                judge=1
                break
        if judge==0:
            sameH_list.append(i)
    elimination=[i for i in elimination if i[0] in sameH_list]
    return np.array(elimination)
def mkopt(file,coord,atomnum,chrg_spin):
    os.system("rm -f opt*")
    with open(file,'w') as f:
        f.write("%chk={}.chk\n".format(file.split('.')[0]))
        f.write("%mem=2gb\n")
        f.write("# opt=nomicro external='./xtb.sh'\n")
        f.write("\n")
        f.write("aaa\n")
        f.write("\n")
        f.write(chrg_spin)
        for i in range(len(atomnum)):
            f.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\t\n".format(atomnum[i]+1,coord[i,0],coord[i,1],coord[i,2]))
        f.write("\n")

def mkcfopt(file,coord,atomnum,chrg_spin,ctrlatom1,ctrlatom2):
    os.system("rm -f cfopt*")
    with open(file,'w') as f:
        f.write("%chk={}.chk\n".format(file.split('.')[0]))
        f.write("%mem=2gb\n")
        f.write("# opt=(nomicro,modredundant) external='./xtb.sh'\n")
        f.write("\n")
        f.write("aaa\n")
        f.write("\n")
        f.write(chrg_spin)
        for i in range(len(atomnum)):
            f.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\t\n".format(atomnum[i]+1,coord[i,0],coord[i,1],coord[i,2]))
        f.write("\n") 
        f.write("B {} {} F\n".format(ctrlatom1,ctrlatom2)) 
        f.write("\n")        

def mkcfoptgdiis(file,ctrlatom1,ctrlatom2):
    os.system("rm -f {}.log".format(file.split('.')[0]))
    file2=file.split('.')[0]+'.gjf'
    with open(file2,'w') as f:
        f.write("%chk={}.chk\n".format(file.split('.')[0]))
        f.write("%mem=2gb\n")
        f.write("# opt=(modredundant,nomicro,gdiis,maxstep=5,notrust) UGBS external='./xtb.sh' geom=allcheck\n")
        f.write("\n")
        f.write("B {} {} F\n".format(ctrlatom1,ctrlatom2)) 
        f.write("\n")

def mkoptgdiis(file):
    os.system("rm -f {}.log".format(file.split('.')[0]))
    file2=file.split('.')[0]+'.gjf'
    with open(file2,'w') as f:
        f.write("%chk={}.chk\n".format(file.split('.')[0]))
        f.write("%mem=2gb\n")
        f.write("# opt=(nomicro,gdiis,maxstep=5,notrust) UGBS external='./xtb.sh' geom=allcheck\n")
        f.write("\n")
        f.write("\n") 
        f.write("\n")

def gdiisloop(log,ctrlatom1,ctrlatom2) :
    judgedelta=0
    interia=0
    with open(log,'r') as f:
        lines=f.readlines()
    judgegdiis=[line for line in lines if "Number of steps exceeded" in line]
    judgebond1=[line for line in lines if "FormBX had a problem" in line]
    judgebond2=[line for line in lines if "Linear angle in Tors" in line]
    judgebond3=[line for line in lines if "Linear angle in Bend" in line]
    judgebond4=[line for line in lines if "Error imposing constraints" in line]
    while len(judgegdiis)!=0 or len(judgebond1)!=0 or len(judgebond2)!=0 or len(judgebond3)!=0 or len(judgebond4)!=0 or judgedelta != 1:
        if interia>=10:
            break
        if "cf" not in log:
            mkoptgdiis(log)
            os.system("g09 {}.gjf >error.msg 2>&1".format(log.split('.')[0]))
        else:
            mkcfoptgdiis(log,ctrlatom1,ctrlatom2)
            os.system("g09 cfopt.gjf >error.msg 2>&1")
        with open(log,'r') as f:
            lines=f.readlines()        
        judgegdiis=[line for line in lines if "Number of steps exceeded" in line]
        judgebond1=[line for line in lines if "FormBX had a problem" in line]
        judgebond2=[line for line in lines if "Linear angle in Tors" in line]
        judgebond3=[line for line in lines if "Linear angle in Bend" in line]
        judgebond4=[line for line in lines if "Error imposing constraints" in line]
        try:
            energy1=float([line for line in lines if "Recovered energy=" in line][-1].split()[2])
            energy2=float([line for line in lines if "Recovered energy=" in line][-2].split()[2])
            if np.abs(energy1-energy2) <0.0001:
                judgedelta=1
        except Exception as e:
            nowpath=os.getcwd()
            print("{} in {}".format(e,nowpath))
        interia=interia+1


def change_coord_gjf(file,target_coord,atomnum):
    blanklist=[]
    with open(file,'r') as f1:
        lines=f1.readlines()
    nblank=0
    for f_line in lines:
        if f_line in ['\n',' \n','\r\n',' \r\n']:
            blanklist.append(nblank)
        nblank=nblank+1
    with open(file,'w') as f1:
        for i in range(blanklist[1]+2):
            f1.write(lines[i])
        for i in range(len(atomnum)):
            f1.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\t\n".format(atomnum[i]+1,target_coord[i,0],target_coord[i,1],target_coord[i,2]))
        f1.write("\n")
        

def ismetal(elment_list):
    metal_list={}
    for i in range(len(elment_list)):
        if elment_list[i] in metal:
            metal_list[i]=elment_list[i]
    return metal_list
    

def getatomnum(elment_list):
    atomnum=[]
    for i in elment_list:
        if not i.isdigit():
            atomnum.append([str(elment.index(j)) for j in elment if str(j)==str(i) ][0])
    return atomnum
def getelment(atomnum):
    elment_list=[]
    for i in atomnum:
            elment_list.append([elment[j] for j in range(len(elment)) if j+1==i ][0])
    return elment_list

def mklinkmatrix(coord_array,atomnum):
    natom=coord_array.shape[0]
    linkmatrix=np.zeros([natom,natom])
    for i in range(natom):
        for j in range(i):
            dist=np.linalg.norm(coord_array[i,0:3].astype(float)-coord_array[j,0:3].astype(float))
            coradius=(corad[int(atomnum[i])]+corad[int(atomnum[j])])*1.2
            if dist <= coradius :
                linkmatrix[j,i]=1
                linkmatrix[i,j]=linkmatrix[j,i]
            else:
                linkmatrix[j,i]=0
                linkmatrix[i,j]=linkmatrix[j,i]
        linkmatrix[i,i]=0
    return linkmatrix

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
# def finddihedral(atom,recordlist,target,linkmatrix,natom):
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

def mklink(seqatom,judge,tmplinkmatrix):
    natom=tmplinkmatrix.shape[0]
    tmplink=np.zeros(natom)
    tmplink=tmplink+natom+1
    tmplink[0]=seqatom
    seqoflist=0
    i=0
    if judge==1:
        while tmplink[i]!=natom+1:
            for j in range(natom):
                    k=0
                    if tmplinkmatrix[j,int(tmplink[i])]==1:
                        for counter in range(seqoflist+1):
                            if j==int(tmplink[counter]):
                                k=1
                                break
                        if k!=1:
                            seqoflist=seqoflist+1
                            tmplink[seqoflist]=j
                    if k==1:
                        continue
            if i==natom-1:
                break                    
            i=i+1 
                       
    return list(map(int,tmplink))

def nelment(tmplink):
    elment_list=[]
    dict_nelment={}
    for i in tmplink:
        if i not in elment_list:
            elment_list.append(i)
    for i in elment_list:
        n=0
        for j in tmplink:
            if i==j:
                n=n+1
        dict_nelment[i]=n
    return dict_nelment,elment_list

def ifnotincycle(seqatom,natom,linkmatrix,metalcenter=None):
    judge=True
    nearlist=[i for i in range(natom) if linkmatrix[i,seqatom]==1]
    for i in nearlist:
        tmplinkmatrix=copy.deepcopy(linkmatrix)
        if metalcenter!=None:
            tmplinkmatrix[:,metalcenter]=0
            tmplinkmatrix[metalcenter,:]=0
        tmplinkmatrix[seqatom,int(i)]=0
        tmplinkmatrix[int(i),seqatom]=0
        link_ifcycle=mklink(i,1,tmplinkmatrix)
        for j in nearlist:
            if j in link_ifcycle and j!=i:
                judge=False
                return judge
    return judge

def normalization(vector):#归一化
    modulus=np.linalg.norm(vector)
    return vector/modulus
def sort_order(coord,tmpnatom,tmpseqatom):
    distlist=[np.linalg.norm(coord[i]-coord[tmpseqatom]) for i in range(tmpnatom)]
    seqlist=[i for i in range(tmpnatom)]
    for i in range(tmpnatom):
     for j in range(tmpnatom-i-1):
        if distlist[j]>distlist[j+1]:
           tmp=distlist[j]
           tmpseq=seqlist[j]
           distlist[j]=distlist[j+1]
           seqlist[j]=seqlist[j+1]
           distlist[j+1]=tmp
           seqlist[j+1]=tmpseq
    return seqlist
def mkrotx(ortha,orthb,orthc,theta):
    Urot=np.zeros([3,3])
    orth_matrix=np.vstack([ortha,orthb,orthc]).T
    orth_matrix_T=np.vstack([ortha,orthb,orthc])
    
    Urot[1,1]=np.cos(theta)
    Urot[1,2]=(-1)*np.sin(theta)
    Urot[2,1]=np.sin(theta)
    Urot[2,2]=np.cos(theta)
    for i in range(3):
       Urot[0,i]=0.0
       Urot[i,0]=0.0
    Urot[0,0]=1.0
    rotx=np.matmul(np.matmul(orth_matrix,Urot),orth_matrix_T)
    return rotx

def mkaxis(vector):
    vector=normalization(vector)
    vector2=copy.deepcopy(vector)
    vector2[0]=vector[0]+0.1
    orth_vector2=vector2-np.dot(vector,vector2)*vector
    orth_vector2=normalization(orth_vector2)
    orth_vector3=np.cross(vector,orth_vector2)
    orth_matrix=np.vstack([vector,orth_vector2,orth_vector3]).T
    return orth_matrix

def min_dist(frag1,frag2,atomlist1,atomlist2):
    min_dist=np.inf
    for c in range(len(frag1)):
        if c not in atomlist2:
            for r in range(len(frag2)):
                if r in atomlist1:
                    continue
                tmpdist=np.linalg.norm(frag2[r]-frag1[c])
                if tmpdist<=min_dist:
                    min_dist=tmpdist
    return min_dist

def min_dist_3arg(frag1,frag2,frag3):
    min_dist=1000
    for c in range(len(frag1)):
        for r in range(len(frag2)):
            tmpdist=np.linalg.norm(frag2[r]-frag1[c])
            if tmpdist<=min_dist:
                min_dist=tmpdist
        for k in range(len(frag3)):
            tmpdist=np.linalg.norm(frag3[k]-frag1[c])
            if tmpdist<=min_dist:
                min_dist=tmpdist
    for c in range(len(frag3)):
        for r in range(len(frag2)):
            tmpdist=np.linalg.norm(frag2[r]-frag3[c])
            if tmpdist<=min_dist:
                min_dist=tmpdist
    return min_dist

def getkeword_csv(path,csv):
    csv=os.path.join(path,csv)
    csv_data=pd.read_csv(csv,encoding='gb18030')
    csv_data.fillna(100)
    mem=csv_data['%mem'][0]
    procs=csv_data['%nprocs'][0]
    keyword=csv_data['keyword'][0]
    try:
        link1=csv_data['link1'].fillna("")[0]
    except:
        link1=''
    return mem,procs,keyword,link1
    

def mkoutput(coord,atomnum,mem,procs,keyword,charge,spin,link1,outputname,fix_list=[]):
    with open("{}.gjf".format(outputname),'w') as f:
        f.write("%chk={}.chk\n".format(outputname.split("/")[-1]))
        f.write("%nprocs={}\n".format(int(procs)))
        f.write("%mem={}\n".format(mem))
        f.write("#{}\n".format(keyword))
        f.write("\n")
        f.write("{}\n".format(outputname.split("/")[-1]))
        f.write("\n")
        f.write("{} {}\n".format(charge,spin))
        for i in range(len(atomnum)):
            f.write("{}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(atomnum[i],coord[i][0],coord[i][1],coord[i][2]))
        f.write("\n")
        for i in range(len(fix_list)):
            f.write("B {} {} F\n".format(fix_list[i][0],fix_list[i][1])) 
        f.write("\n")  
        if len(link1)!=0:
            f.write("--Link1--\n")
            f.write("%chk={}.chk\n".format(outputname))
            f.write("%nprocs={}\n".format(procs))
            f.write("%mem={}\n".format(mem))
            f.write("# {} {} {}\n".format(keyword,link1))
            f.write("\n")
            f.write("\n")
            
def judge_effective_rot(a,b,coord_array,linkmatrix,natom,ctrlatom1,ctrlatom2):
    tmplinkmatrix=copy.deepcopy(linkmatrix)
    tmplinkmatrix[a,b]=0
    tmplinkmatrix[b,a]=0
    tmplink1=mklink(a,1,tmplinkmatrix)
    tmplink2=mklink(b,1,tmplinkmatrix)
    if (ctrlatom1-1 in tmplink1 or ctrlatom2-1 in tmplink1) and (ctrlatom1-1 in tmplink2 or ctrlatom2-1 in tmplink2):
        return True
    else:
        return False

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

def rotation_vv(coord_list_1,coord_list_subs,atomlist_1,atomlist_subs,delete_list_cat,delete_list_group,pattern,top,relaxation,div,arg,div_arg):
    ctrlatom1=atomlist_1[0]
    ctrlatom2=atomlist_1[1]
    ctrlatom1_subs=atomlist_subs[0]
    ctrlatom2_subs=atomlist_subs[1]

    # div=30
    # arg=pi/6
    # div_arg=10
    pi=np.pi

    natom_1=len(coord_list_1)
    natom_subs=len(coord_list_subs)
    coord_array_1=np.array([coord_list_1[i][1:4] for i in range(natom_1)])
    atomnum_1=[int(coord_list_1[i][0]) for i in range(natom_1)]
    atomnum_subs=[int(coord_list_subs[i][0]) for i in range(natom_subs)]
    coord_array_subs=np.array([coord_list_subs[i][1:4] for i in range(natom_subs)])
    dirvector=coord_array_1[ctrlatom1,:]-coord_array_subs[ctrlatom1_subs,:]
    tmp=[coord_array_subs[i,:]+dirvector for i in range(coord_array_subs.shape[0])]
    coord_array_subs=tmp
    # seqlist1=sort_order(coord_array_1,natom_1,ctrlatom1)
    # seqlist2=sort_order(coord_array_subs,natom_subs,ctrlatom1_subs)
    if natom_1>=2 and natom_subs>=2:
        vector=coord_array_1[ctrlatom2]-coord_array_1[ctrlatom1]
        vector=normalization(vector)
        vector_subs=coord_array_subs[ctrlatom2_subs]-coord_array_subs[ctrlatom1_subs]
        vector_subs=normalization(vector_subs)
        orth_vector_subs=vector_subs-np.dot(vector,vector_subs)*vector
        orth_vector_subs=normalization(orth_vector_subs)
        normalvector_subs=np.cross(vector,orth_vector_subs)
        normalvector_subs=normalization(normalvector_subs)
        # normalvector_subs=[0,1,0]
        # vector=[1,0,0]
        # orth_vector_subs=[0,0,1]
        # theta=(np.pi)

        theta=np.arccos(np.dot(vector,vector_subs))
        Urotx=mkrotx(normalvector_subs,vector,orth_vector_subs,theta)
        judge=np.dot(Urotx,vector_subs)
        if not np.allclose(judge,vector):
            theta=2*pi-theta
        Urotx=mkrotx(normalvector_subs,vector,orth_vector_subs,theta)
        tmp=[np.dot(Urotx,(coord_array_subs[i]-coord_array_subs[ctrlatom1_subs]))+coord_array_subs[ctrlatom1_subs] for i in range(len(coord_array_subs))]
        coord_array_subs=tmp
        rot_list=[(2*pi*i)/div for i in range(div)]
        rot_arg_list=np.linspace((-1)*arg,arg,div_arg)
        judge_index=[0,0,0]
        target_index=[0,0,0]    
        target_dist=0
        #以上是平移和转动，解释可参考rotation_vv2

        #实行转动和摆动
        for  y in range(len(rot_arg_list)):
            tmpUroty=mkrotx(normalvector_subs,vector,orth_vector_subs,rot_arg_list[y])
            for  z in range(len(rot_arg_list)):
                tmpUrotz=mkrotx(orth_vector_subs,normalvector_subs,vector,rot_arg_list[z])
                for rot_i in range(len(rot_list)):
                    tmp_atomlist_1 = copy.deepcopy(atomlist_1)
                    tmp_atomlist_subs = copy.deepcopy(atomlist_subs)
                    tmpUrotx=mkrotx(vector,normalvector_subs,orth_vector_subs,rot_list[rot_i])
                    judge_coord_array=[np.dot(tmpUroty,np.dot(tmpUrotz,np.dot(tmpUrotx,(coord_array_subs[i]-coord_array_subs[ctrlatom1_subs]))))+coord_array_subs[ctrlatom1_subs] for i in range(len(coord_array_subs))]
                    if pattern == 1:
                        new_coord_1=[coord_array_1[i] for i in range(natom_1) if i != atomlist_1[0] and i not in delete_list_cat] #去掉片段1中的向量1原子
                        new_atomnum_1=[atomnum_1[i] for i in range(natom_1) if i != atomlist_1[0] and i not in delete_list_cat]
                        newreatant=[judge_coord_array[i]for i in range(natom_subs) if i !=atomlist_subs[1] and i not in delete_list_group] #去掉片段2中的向量2原子
                        atomnum_subs_1=[atomnum_subs[i] for i in range(natom_subs) if i!=atomlist_subs[1] and i not in delete_list_group]
                    else:
                        new_coord_1=[coord_array_1[i] for i in range(natom_1) if i not in atomlist_1 and i not in delete_list_cat] #去掉片段1中的向量1原子
                        new_atomnum_1=[atomnum_1[i] for i in range(natom_1) if i not in atomlist_1 and i not in delete_list_cat]  
                        newreatant=[judge_coord_array[i]for i in range(natom_subs) ]
                        atomnum_subs_1=[atomnum_subs[i] for i in range(natom_subs) ]
                    #在删减原子后，调整原子序号
                    if pattern == 1:
                        if atomlist_1[0]<atomlist_1[1]:
                            tmp_atomlist_1[1]=tmp_atomlist_1[1]-1
                        if atomlist_subs[0] > atomlist_subs[1]:
                            tmp_atomlist_subs[0] = tmp_atomlist_subs[0] - 1
                        for delete_el in delete_list_cat:
                            if tmp_atomlist_1[1]>delete_el:
                                tmp_atomlist_1[1]=tmp_atomlist_1[1]-1
                        for delete_el in delete_list_group:
                            if atomlist_subs[0] > delete_el:
                                tmp_atomlist_subs[0] = tmp_atomlist_subs[0] - 1
                    else:
                        tmp_atomlist_1 = []                            

                      
                    opt_coord=np.array(new_coord_1+newreatant)
                    atomnum=new_atomnum_1+atomnum_subs_1
                    atomnum=[i-1 for i in atomnum]
                    #实行或不实行结构弛豫
                    if relaxation:
                        natom_new=opt_coord.shape[0]
                        d0=np.zeros([natom_new,natom_new])
                        linkmatrix1=mklinkmatrix(opt_coord[0:len(new_coord_1),:],atomnum[0:len(new_coord_1)]) #构建片段1的连接矩阵
                        linkmatrix2=mklinkmatrix(opt_coord[len(new_coord_1):natom_new,:],atomnum[len(new_coord_1):natom_new]) #构建第一个片
                        linkmatrix=adjConcat(linkmatrix1,linkmatrix2)
                        if pattern == 1:
                            linkmatrix[tmp_atomlist_1[1],len(new_atomnum_1)+tmp_atomlist_subs[0]]=1
                            linkmatrix[len(new_atomnum_1)+tmp_atomlist_subs[0],tmp_atomlist_1[1]]=1
                        linkmatrix=np.array(linkmatrix,np.int32)
                        fix_natom=np.linspace(len(new_coord_1),len(new_coord_1)+len(newreatant)-1,len(newreatant)) # 弛豫中，非冻结的原子，此处为group中所有原子
                        fix_natom=fix_natom.astype(np.int32)
                        for i in range(natom_new):
                            for j in range(natom_new):
                                if linkmatrix[i,j]==1:
                                    d0[i,j]=corad[atomnum[i]]+corad[atomnum[j]]
                        mkeasyfile("input.gjf",opt_coord,atomnum,"0 1\n")
                        opt_coord,energy=finddihedral.opt_structure("input.gjf",linkmatrix,d0,fix_natom=fix_natom) 
                        # a=linkmatrix[42,44]  

                        judge_dist=energy
                    else:
                        judge_dist = (-1)*min_dist(new_coord_1,newreatant,tmp_atomlist_subs,tmp_atomlist_1)
                    judge_index[0]=y
                    judge_index[1]=z
                    judge_index[2]=rot_i
                    if rot_i==0:
                        target_dist=copy.deepcopy(judge_dist)
                        if relaxation:
                            target_linkmatrix=copy.deepcopy(linkmatrix)
                            target_d0=copy.deepcopy(d0)
                    elif judge_dist<=target_dist:
                        target_dist=copy.deepcopy(judge_dist)
                        target_index=copy.deepcopy(judge_index)
                        if relaxation:
                            target_linkmatrix=copy.deepcopy(linkmatrix)
                            target_d0=copy.deepcopy(d0)                        
        Uroty=mkrotx(normalvector_subs,(-1)*vector,orth_vector_subs,rot_arg_list[target_index[0]])
        Urotz=mkrotx(orth_vector_subs,normalvector_subs,(-1)*vector,rot_arg_list[target_index[1]])  
        Urotx=mkrotx(vector,normalvector_subs,orth_vector_subs,rot_list[target_index[2]])
        newreatant=[np.dot(Uroty,np.dot(Urotz,np.dot(Urotx,(coord_array_subs[i]-coord_array_subs[ctrlatom1_subs]))))+coord_array_subs[ctrlatom1_subs] for i in range(len(coord_array_subs))]
        # for i in range(natom_subs):
        #     print("{}\t{}\t{}\t{}\t".format(atomnum_subs[i],newreatant[i][0],newreatant[i][1],newreatant[i][2]))

        if pattern == 1:
            new_coord_1=[coord_array_1[i] for i in range(natom_1) if i != atomlist_1[0] and i not in delete_list_cat] #去掉片段1中的向量1原子
            new_atomnum_1=[atomnum_1[i] for i in range(natom_1) if i != atomlist_1[0] and i not in delete_list_cat]
            newreatant=[newreatant[i]for i in range(natom_subs) if i !=atomlist_subs[1] and i not in delete_list_group] #去掉片段2中的向量2原子
            atomnum_subs=[atomnum_subs[i] for i in range(natom_subs) if i!=atomlist_subs[1] and i not in delete_list_group]
        else:
            new_coord_1=[coord_array_1[i] for i in range(natom_1) if i not in atomlist_1 and i not in delete_list_cat] #去掉片段1中的向量1原子
            new_atomnum_1=[atomnum_1[i] for i in range(natom_1) if i not in atomlist_1 and i not in delete_list_cat]            
            newreatant=[newreatant[i]for i in range(natom_subs) ] #去掉片段2中的向量2原子
            atomnum_subs=[atomnum_subs[i] for i in range(natom_subs)]
        
        opt_coord=np.array(new_coord_1+newreatant)
        atomnum=new_atomnum_1+atomnum_subs
        atomnum=[i-1 for i in atomnum]
        if relaxation:
            mkeasyfile("target.gjf",opt_coord,atomnum,"0 1\n")
            # a = target_linkmatrix[42,44]
            opt_coord,energy=finddihedral.opt_structure("target.gjf",target_linkmatrix,target_d0,fix_natom=fix_natom)          
        natom_new=opt_coord.shape[0]
        # if top:
        #     top_opt_coord = np.vstack((opt_coord[len(new_coord_1):len(opt_coord),:],opt_coord[0:len(new_coord_1),:]))
        #     top_atomnum = np.vstack((atomnum[len(new_atomnum_1):len(atomnum),:],opt_coord[0:len(new_atomnum_1),:]))
        mkeasyfile("target.gjf",opt_coord,atomnum,"0 1\n")
        return opt_coord,atomnum,natom_new,len(new_coord_1)

def rotation_vv2(coord_list_1,coord_list_subs,atomlist_1,atomlist_subs,div=30,arg=np.pi/6,div_arg=10): # 将片段1中atomlist1确定的向量通过平移和转动使之与片段2中atomlist2确定的向量完全重合，并以此带动片段的
                                                                                                       #整体平移与旋转
    ctrlatom1=atomlist_1[0]
    ctrlatom2=atomlist_1[1]
    ctrlatom1_subs=atomlist_subs[0]
    ctrlatom2_subs=atomlist_subs[1]
    # div=30
    # arg=pi/6
    # div_arg=10
    pi=np.pi
    natom_1=len(coord_list_1)
    natom_subs=len(coord_list_subs)
    coord_array_1=np.array([coord_list_1[i][1:4] for i in range(natom_1)])
    atomnum_1=[int(coord_list_1[i][0]) for i in range(natom_1)]
    atomnum_subs=[int(coord_list_subs[i][0]) for i in range(natom_subs)]
    coord_array_subs=np.array([coord_list_subs[i][1:4] for i in range(natom_subs)])

    dirvector=coord_array_1[ctrlatom1,:]-coord_array_subs[ctrlatom1_subs,:] #确定平移始末的方向向量
    tmp=[coord_array_subs[i,:]+dirvector for i in range(coord_array_subs.shape[0])] #将片段2根据方向向量平移
    coord_array_subs=tmp #平移后坐标
    # seqlist1=sort_order(coord_array_1,natom_1,ctrlatom1)
    # seqlist2=sort_order(coord_array_subs,natom_subs,ctrlatom1_subs)
    if natom_1>=2 and natom_subs>=2:
        vector=coord_array_1[ctrlatom2]-coord_array_1[ctrlatom1]
        vector=normalization(vector)
        vector_subs=coord_array_subs[ctrlatom2_subs]-coord_array_subs[ctrlatom1_subs]
        vector_subs=normalization(vector_subs)
        vector_subs_tmp=copy.deepcopy(vector_subs)
        if np.allclose(vector,vector_subs):
            vector_subs_tmp=vector_subs+0.1
        orth_vector_subs=vector_subs_tmp-np.dot(vector,vector_subs_tmp)*vector #由vector_subs衍生出垂直于vector的向量
        orth_vector_subs=normalization(orth_vector_subs)
        normalvector_subs=np.cross(vector,orth_vector_subs) #构建vector,orth_vector_subs所在平面的法向量
        normalvector_subs=normalization(normalvector_subs) #至此构建出新的三个正交归一向量
        # normalvector_subs=[0,1,0]
        # vector=[1,0,0]
        # orth_vector_subs=[0,0,1]
        # theta=(np.pi)

        theta=np.arccos(np.dot(vector,vector_subs)) #构建atomlist1确定的向量与atomlist2确定的向量之间的夹角theta
        Urotx=mkrotx(normalvector_subs,vector,orth_vector_subs,theta) #构建绕法向量逆时针转动theata角度的矩阵
        judge=np.dot(Urotx,vector_subs)
        if not np.allclose(judge,vector):# 以下if判断用于判断该夹角是否是逆时针的夹角，因为点积运算得到的角度无法判断方向，但转动矩阵固定绕逆时针方向
            theta=2*np.pi-theta #若不是逆时针方向，即judge向量与vector向量不相等，意味着之前内积得到的夹角是由vector顺时针转到target，因此把转动矩阵的旋转角度变为2pi-theta（逆时针）
        Urotx=mkrotx(normalvector_subs,vector,orth_vector_subs,theta) #生成新旋转矩阵
        tmp=[np.dot(Urotx,(coord_array_subs[i]-coord_array_subs[ctrlatom1_subs]))+coord_array_subs[ctrlatom1_subs] for i in range(len(coord_array_subs))]
        # 生成经过转动theta角度后片段subs的新坐标
        coord_array_subs=tmp
        #     print("{}\t{}\t{}\t{}\t".format(atomnum_subs[i],newreatant[i][0],newreatant[i][1],newreatant[i][2]))
        new_coord_1=[coord_array_1[i] for i in range(natom_1) if i != atomlist_1[0]] #去掉片段1中的向量1原子
        new_atomnum_1=[atomnum_1[i] for i in range(natom_1) if i != atomlist_1[0]]
        newreatant=[coord_array_subs[i]for i in range(natom_subs) if i !=atomlist_subs[1]] #去掉片段2中的向量2原子
        atomnum_subs=[atomnum_subs[i] for i in range(natom_subs) if i!=atomlist_subs[1]]
        return new_coord_1,new_atomnum_1,newreatant,atomnum_subs
    else:
        print('too less number of atoms')
        
def getcoord_log(log):
    with open(log,'r') as f:
        f1_lines=f.readlines()
    natom=[f1_lines[i] for i in range(len(f1_lines)) if 'NAtoms' in f1_lines[i] ][0].split()[1]
    chrg_multi=[f1_lines[i] for i in range(len(f1_lines)) if 'Charge =' in f1_lines[i] and 'Multiplicity' in f1_lines[i]][0].split()
    chrg_multi="{} {}\n".format(chrg_multi[2],chrg_multi[5])
    standorient_list=[i for i in range(len(f1_lines)) if 'Standard orientation:' in f1_lines[i] ]
    try:
        coord=f1_lines[standorient_list[-1]+5:standorient_list[-1]+5+int(natom)]
        coord_list=[[int(coord[i].split()[1]),float(coord[i].split()[3]),float(coord[i].split()[4]),float(coord[i].split()[5])] for i in range(len(coord))]
    except Exception as e:
        coord_list=[]
        print("{} in getcoord_log".format(e))
    return coord_list,chrg_multi
        


#tmplist=list(linkmatrix[28,:])
#a=[[m,n] for m in range(natom) for n in range(natom) if m<n and tmplist[m]==1 and tmplist[n]==1]
#n_coord=tmplist.count(1)

# changelment=4
# path='D:\ml\\tmpfile'
# str_fault_dir='D:\ml\\tmpfault'
# files=os.listdir(path) 
### change the link atom
# for file1 in files:
#     absfile1=os.path.join(path,file1)
#     coord_array,elment_list,atomnum,chrg_multi=getcoord(absfile1)
#     linkmatrix=mklinkmatrix(coord_array,atomnum)
#     hybrid_list=[]
#     for el in range(natom):
#         tmplist=list(linkmatrix[el,:])
#         a=[[m,n] for m in range(natom) for n in range(natom) if m<n and tmplist[m]==1 and tmplist[n]==1]
#         n_coord=tmplist.count(1)
#         rad=0
#         if len(a)>=1:
#             for i in a:
#                 vector1=coord_array[el,0:3]-coord_array[i[0],0:3]
#                 vector2=coord_array[el,0:3]-coord_array[i[1],0:3]
#                 rad=mkrad(vector1,vector2)/len(a)+rad
#         hybrid_list.append(verify_hybrid(rad,n_coord))
#     sp3C_list=[i for i in range(natom) if elment[int(atomnum[i])]=='C' and hybrid_list[i]==3]
#     target_list=[]
#     for i in sp3C_list:
#         judge=ifnotincycle(int(i),0)
#         if judge==1:
#             target_list.append(i)
#     if len(target_list)==1:
#         newatomnum=copy.deepcopy(atomnum)
#         newatomnum[target_list[0]]=changelment
#         nearH=[i for i in range(natom) if linkmatrix[i,target_list[0]]==1 if atomnum[i]=="0"]
#         A=file1.split('B')[0]
#         CE=file1.split('C')[1]
#         newfile1=A+'B4C'+CE
#         absnewfile1=os.path.join(str_fault_dir,newfile1)
#         with open(absnewfile1,'w') as f1:
#             f1.write("%chk={}.chk\n".format(newfile1.split('.')[0]))
#             f1.write('%nprocshared=1\n%mem=2Gb\n# opt=(nomicro) UGBS external=\'./xtb.sh\'\n')
#             f1.write('\n')
#             f1.write('title\n')
#             f1.write('\n')
#             f1.write("{}".format(chrg_multi))
#             for i in range(natom):
#                 if i !=nearH[0]:
#                     f1.write("{}\t{}\t{}\t{}\n".format(elment[int(newatomnum[i])],coord_array[i,0],coord_array[i,1],coord_array[i,2]))
#             f1.write('\n')
#             f1.write('\n')
#     else:
#         print("{} was wrong".format(file1))           
### change the link atom

### change the filename
# files=os.listdir(path) 
# for file in files:
#     oldname=os.path.join(path,file)
#     B=file.split('B')[1]
#     newname=os.path.join(path,'A203B'+B)
#     os.rename(oldname,newname)
### change the filename

# a=[hybrid_list[x] for x in range(natom) if linkmatrix[79,x]==1][0] 
# for i in range(len(hybrid_list)):
#     if hybrid_list[i] ==0:
#         hybrid_list[i]=[hybrid_list[x] for x in range(natom) if linkmatrix[i,x]==1][0]
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




