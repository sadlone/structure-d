import re
import os
import pandas as pd
import numpy as np

elementTable = {'1': 'H', 'H': '1', '2': 'He', 'He': '2', '3': 'Li', 'Li': '3', '4': 'Be', 'Be': '4', '5': 'B', 'B': '5', '6': 'C', 'C': '6', '7': 'N', 'N': '7', '8': 'O', 'O': '8', '9': 'F', 'F': '9', '10': 'Ne', 'Ne': '10', '11': 'Na', 'Na': '11', '12': 'Mg', 'Mg': '12', '13': 'Al', 'Al': '13', '14': 'Si', 'Si': '14', '15': 'P', 'P': '15', '16': 'S', 'S': '16', '17': 'Cl', 'Cl': '17', '18': 'Ar', 'Ar': '18', '19': 'K', 'K': '19', '20': 'Ca', 'Ca': '20', '21': 'Sc', 'Sc': '21', '22': 'Ti', 'Ti': '22', '23': 'V', 'V': '23', '24': 'Cr', 'Cr': '24', '25': 'Mn', 'Mn': '25', '26': 'Fe', 'Fe': '26', '27': 'Co', 'Co': '27', '28': 'Ni', 'Ni': '28', '29': 'Cu', 'Cu': '29', '30': 'Zn', 'Zn': '30', '31': 'Ga', 'Ga': '31', '32': 'Ge', 'Ge': '32', '33': 'As', 'As': '33', '34': 'Se', 'Se': '34', '35': 'Br', 'Br': '35', '36': 'Kr', 'Kr': '36', '37': 'Rb', 'Rb': '37', '38': 'Sr', 'Sr': '38', '39': 'Y', 'Y': '39', '40': 'Zr', 'Zr': '40', '41': 'Nb', 'Nb': '41', '42': 'Mo', 'Mo': '42', '43': 'Tc', 'Tc': '43', '44': 'Ru', 'Ru': '44', '45': 'Rh', 'Rh': '45', '46': 'Pd', 'Pd': '46', '47': 'Ag', 'Ag': '47', '48': 'Cd', 'Cd': '48', '49': 'In', 'In': '49', '50': 'Sn', 'Sn': '50', '51': 'Sb', 'Sb': '51', '52': 'Te', 'Te': '52', '53': 'I', 'I': '53', '54': 'Xe', 'Xe': '54', '55': 'Cs', 'Cs': '55', '56': 'Ba', 'Ba': '56', '57': 'La', 'La': '57', '58': 'Ce', 'Ce': '58', '59': 'Pr', 'Pr': '59', '60': 'Nd', 'Nd': '60', '61': 'Pm', 'Pm': '61', '62': 'Sm', 'Sm': '62', '63': 'Eu', 'Eu': '63', '64': 'Gd', 'Gd': '64', '65': 'Tb', 'Tb': '65', '66': 'Dy', 'Dy': '66', '67': 'Ho', 'Ho': '67', '68': 'Er', 'Er': '68', '69': 'Tm', 'Tm': '69', '70': 'Yb', 'Yb': '70', '71': 'Lu', 'Lu': '71', '72': 'Hf', 'Hf': '72', '73': 'Ta', 'Ta': '73', '74': 'W', 'W': '74', '75': 'Re', 'Re': '75', '76': 'Os', 'Os': '76', '77': 'Ir', 'Ir': '77', '78': 'Pt', 'Pt': '78', '79': 'Au', 'Au': '79', '80': 'Hg', 'Hg': '80', '81': 'Tl', 'Tl': '81', '82': 'Pb', 'Pb': '82', '83': 'Bi', 'Bi': '83', '84': 'Po', 'Po': '84', '85': 'At', 'At': '85', '86': 'Rn', 'Rn': '86', '87': 'Fr', 'Fr': '87', '88': 'Ra', 'Ra': '88', '89': 'Ac', 'Ac': '89', '90': 'Th', 'Th': '90', '91': 'Pa', 'Pa': '91', '92': 'U', 'U': '92', '93': 'Np', 'Np': '93', '94': 'Pu', 'Pu': '94', '95': 'Am', 'Am': '95', '96': 'Cm', 'Cm': '96', '97': 'Bk', 'Bk': '97', '98': 'Cf', 'Cf': '98', '99': 'Es', 'Es': '99', '100': 'Fm', 'Fm': '100', '101': 'Md', 'Md': '101', '102': 'No', 'No': '102', '103': 'Lr', 'Lr': '103', '104': 'Rf', 'Rf': '104', '105': 'Db', 'Db': '105', '106': 'Sg', 'Sg': '106', '107': 'Bh', 'Bh': '107', '108': 'Hs', 'Hs': '108', '109': 'Mt', 'Mt': '109', '110': 'Ds', 'Ds': '110', '111': 'Rg', 'Rg': '111', '112': 'Cn', 'Cn': '112', '113': 'Nh', 'Nh': '113', '114': 'Fl', 'Fl': '114', '115': 'Mc', 'Mc': '115', '116': 'Lv', 'Lv': '116', '117': 'Ts', 'Ts': '117', '118': 'Og', 'Og': '118'}


def get_charge_multi(f):
    if os.path.splitext(f)[1] == '.gjf':
        with open(f, 'r') as f_gjf:
            lines = f_gjf.readlines()

        coors = []
        pa = ' *(\w+)  *(-?\d+\.\d+)  *(-?\d+\.\d+)  *(-?\d+\.\d+) *'
        for i in range(len(lines)):
            m = re.search(pa, lines[i])
            if m:
                charge_multi = lines[i - 1]
                return tuple(charge_multi.split())
    if os.path.splitext(f)[1] == '.log':
        with open(f, 'r') as f_log:
            lines =  f_log.readlines()
        pa = ' Charge *=  *(-?\d+) *Multiplicity *= *(\d+)'
        for line in lines:
            m = re.search(pa, line)
            if m:
                charge = m.group(1)
                multi = m.group(2)
                return (charge, multi)            
        
    
    
def get_gjfcoor(f):
    if os.path.splitext(f)[1] == '.gjf':
        with open(f, 'r') as f_gjf:
            lines =  f_gjf.readlines()
            
        coors = []
        pa = ' *(\w+)  *(-?\d+\.\d+)  *(-?\d+\.\d+)  *(-?\d+\.\d+) *'
        for l in lines:
            m = re.search(pa, l)
            if m:
                coors.append(l.split())
        for i in range(len(coors)):
            e = coors[i][0]
            if e.isdigit(): continue
            else:
                coors[i][0] = elementTable[e]
        return coors
    
def get_logcoor(log_path):
    
    log = []
    with open(log_path, 'r', encoding='utf-8') as f:
        log = f.readlines()
    logStr = ''.join(log)
    Natoms = re.search('NAtoms= *(\d+)', logStr).group(1)
    coordinate_index = 0
    flag='Standard orientation:'
    flag1 = 'Input orientation:'
    for index in range(len(log)):
        if flag in log[index]:
            coordinate_index = index
    if coordinate_index == 0:
            for index in range(len(log)):
                if flag1 in log[index]:
                    coordinate_index = index
                
    coor_list = []

    for index in range(coordinate_index + 5, coordinate_index + 5 + int(Natoms)):
        line = log[index]
        pattern = ' *\d+  *(\d+)  *\d+ *(-?\d+\.\d+) *(-?\d+\.\d+) *(-?\d+\.\d+)'
        s_result = re.search(pattern, line)
        coor_list.append([s_result.group(1), s_result.group(2), s_result.group(3), s_result.group(4)])

    return coor_list

if __name__ == '__main__':
    path = './'
    fileList = []
    for file in os.listdir(path):
        fileType = os.path.splitext(file)[1]
        if fileType == '.gjf' or fileType == '.log':
            fileList.append(file)
    for file in fileList:
        try:
            fileName = os.path.splitext(file)[0]
            fileType = os.path.splitext(file)[1]
            coors = []
            if fileType == '.gjf':
                coors = get_gjfcoor(file)
            elif fileType == '.log':
                coors = get_logcoor(file)
            chargeMulti = get_charge_multi(file)
            indexRange = list(range(1, len(coors)+1))
            columns = ['atomic number', 'x', 'y', 'z']
            coor_df = pd.DataFrame(data=coors, columns=columns)
            coor_df['center number'] = indexRange
            coor_df['charge'] = np.nan
            coor_df['multi'] = np.nan
            coor_df["target"]=np.nan
            # print(coor_df)
            coor_df.loc[0, 'charge'] = chargeMulti[0]
            coor_df.loc[0, 'multi'] = chargeMulti[1]
            coor_df = coor_df[["target",'center number', 'atomic number', 'x', 'y', 'z', 'charge', 'multi']]
            coor_df.to_csv(fileName + '.csv')
        except Exception as e:
            print(e)