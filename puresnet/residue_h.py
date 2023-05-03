from .atom import atom
import os
from .featurizer import Featurizer
from openbabel import pybel
pybel.ob.obErrorLog.StopLogging()
ch_path=os.getcwd()
if not os.path.exists(os.path.join(ch_path,'residues')):
    os.makedirs(os.path.join(ch_path,'residues'),exist_ok=True)
feature=Featurizer()
def is_standard(obj):
    names=[x.name[0] for x in vars(obj).values() if type(x)==type(atom())]
    if all([True if x in names else False for x in ['N','C','O']]):
        return True
    else:
        return False
def res_iter(obj):
    return (x for x in vars(obj).values() if type(x)==type(atom()))
def get_url(residue):
    return 'https://files.rcsb.org/ligands/download/'+residue+'_model.sdf'
def get_residue(residue_name):
    residue_name=residue_name.upper().strip()
    url=' https://files.rcsb.org/ligands/download/'+residue_name+'.cif'
    url2=get_url(residue_name)
    
    if residue_name+'.cif' not in os.listdir(ch_path+'/residues'):
        cmd1='wget '+url+' -O '+ch_path+'/residues/'+residue_name+'.cif'+' >/dev/null 2>&1'
        os.system(cmd1)
    if residue_name+'.sdf' not in os.listdir(ch_path+'/residues'):
        cmd2='wget '+url2+' -O '+ch_path+'/residues/'+residue_name+'.sdf'+' >/dev/null 2>&1'
        os.system(cmd2)
    tabular_format=False
    loop_count=0
    temp=None
    mol=next(pybel.readfile('sdf',ch_path+'/residues/'+residue_name+'.sdf'))
    feat=feature.get_features(mol)
    f=open(ch_path+'/residues/'+residue_name+'.cif','r')
    for i,lines in enumerate(f):
        if i==0:
            name=lines.strip().split('_')[1]
            temp = type('_'+name,(object,),{
                "res_iter": res_iter
            })
        if lines.strip():
            if lines.strip()[0]== '#':
                tabular_format=False
                continue
        if lines.strip()=='loop_':
            tabular_format=True
            res_details=False
            res_bond=False
            continue
        if lines.strip().split('.')[0]=='_chem_comp_atom':
            res_details=True
            continue
        if lines.strip().split('.')[0]=='_chem_comp_bond':
            res_bond=True
            continue
        if tabular_format:
            if lines[:len(name)]==name:
                if res_details:
                    atom_name=lines[len(name):].split()[0]
                    atom_type=lines[len(name):].split()[2]
                    atom_num=int(lines[len(name):].split()[-1])
                    leaving=lines[len(name):].split()[6]
                    config=lines[len(name):].split()[7]
                    if '"' in atom_name:
                        atom_name=atom_name[1:-1]
                    if atom_type=='H' or atom_name=='OXT' or atom_name=='O1'or atom_name=='O3':
                        continue
                    try:
                        feat[atom_num-1]
                    except Exception:
                        continue
                    setattr(temp,atom_name,atom())
                    vars(temp)[atom_name].name=(atom_name,atom_type)
                    vars(temp)[atom_name].stereo_config=config
                    vars(temp)[atom_name].leaving_atom_flag=leaving
                    vars(temp)[atom_name].properties=feat[atom_num-1]
        if loop_count==3:
            f.close()
            break
    return temp