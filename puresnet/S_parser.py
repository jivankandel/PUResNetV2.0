from .residue_h import get_residue,res_iter
from .error import AAError,ModelError,MissingAtom
from urllib import request
class molecule:
    def __init__(self) -> None:
        self.chains={}
class chain:
    def __init__(self,name) -> None:
        self.index={}
        self.residues={}
        self.name=name
class pdb_structure:
    def __init__(self) -> None:
        self.HEADER=None
        self.TITLE=None
        self.NUMTER=0
        self.NUMMDL=None
        self.REMARK=None
        self.MASTER=None
        self.DATA=[]
        self.MODELS_INDEXES=[]
        self.CHAINS=[]
    def check_model(self):
        if len(self.MODELS_INDEXES)==0:
            return 1
        else:
            diff=set(y-x for [x,y] in self.MODELS_INDEXES)
            if len(diff)==1:
                s,e=self.MODELS_INDEXES[0]
                self.DATA=self.DATA[s+1:e]
                return 2
            else:
                raise ModelError
def file_reader(pdb_path,rscb=False,uniprot=False):
    allowed=['HEADER','TITLE','NUMMDL','REMARK','HELIX','SHEET','TURN','CISPEP','SSBOND','SITE','CONECT','MASTER','ATOM','MODEL','TER','ENDMDL','SEQRES','MODRES','HETATM','LINK']
    if  rscb:
        url=f'https://files.rcsb.org/download/{pdb_path.upper()}.pdb'
        with request.urlopen(url) as f:
            for lines in f.readlines():
                lines=lines.decode('utf-8').strip()
                if any(lines.startswith(s,0,len(s)) for s in allowed):
                    yield lines
    elif uniprot:
        url=f'https://alphafold.ebi.ac.uk/files/AF-{pdb_path.upper()}-F1-model_v4.pdb'
        with request.urlopen(url) as f:
            for lines in f.readlines():
                lines=lines.decode('utf-8').strip()
                if any(lines.startswith(s,0,len(s)) for s in allowed):
                    yield lines
    else:
        file_path=pdb_path
        f=open(file_path,'r')
        for lines in f:
            if any(lines.startswith(s,0,len(s)) for s in allowed):
                yield lines
        f.close()
        
def check_residue(x):
    residues=['A', 'G', 'C', 'U', 'I','DA', 'DG', 'DC', 'DT', 'DI','N']
    if x in residues:
        return True
    return False
def parse_atom(line):
    '''Parser to parse atom information
        line: line containing atom information
        Returns: atom_serial_number,atom_number
                atom_alt_loc,residue_name,chain,residue_no,
                residue_insertion_code,atom_x,atom_y,atom_z
                atom_occupancy,atom_type'''
    atom_serial_number=line[6:11].strip()
    atom_name=line[12:16].strip()
    atom_alt_loc=line[16].strip()
    residue_name=line[17:20].strip()
    chain=line[21].strip()
    residue_no=line[22:26].strip()
    residue_insertion_code=line[26].strip()
    atom_x=line[30:38].strip()
    atom_y=line[38:46].strip()
    atom_z=line[46:54].strip()
    atom_occupancy=line[54:60].strip()
    atom_type=line[76:78].strip()
    return atom_serial_number,atom_name,atom_alt_loc,residue_name,chain,residue_no,residue_insertion_code,atom_x,atom_y,atom_z,atom_occupancy,atom_type
def parser(path,rscb=False,uniprot=False):
    temp=pdb_structure()
    i=0
    for lines in file_reader(path,rscb=rscb,uniprot=uniprot):
        try:
            if lines[:6].strip() in ['ATOM','MODEL','TER','ENDMDL','HETATM']:
                if lines[:6].strip()=='ATOM':
                    vars(temp)['CHAINS'].append(lines[21].strip())
                if lines[:6].strip()=='MODEL':
                    temp_index=[i]
                elif lines[:6].strip()=='ENDMDL':
                    temp_index.append(i)
                    vars(temp)['MODELS_INDEXES'].append(temp_index)
                else:
                    if lines[:6].strip()=='TER':
                        temp.NUMTER+=1
                    vars(temp)['DATA'].append(lines.strip())
                    i+=1
            else:
                vars(temp)[lines[:6].strip()]=lines
        except KeyError:
            pass
    return temp
def get_molecule(pdb_struc):
    mol=molecule()
    ter=pdb_struc.NUMTER
    prev_res_key=None
    if pdb_struc.NUMMDL:
        ter=ter/int(pdb_struc.NUMMDL[6:])
    elif ter!=len(set(pdb_struc.CHAINS)):
        ter=len(set(pdb_struc.CHAINS))
    ter_count=0
    for line in pdb_struc.DATA:
        if line[:6].strip()=='TER':
            ter_count+=1
        elif ter_count<ter:
            atom_serial_number,atom_name,atom_alt_loc,residue_name,chain_id,residue_no,residue_insertion_code,atom_x,atom_y,atom_z,atom_occupancy,atom_type=parse_atom(line)
            if check_residue(residue_name):
                continue
            if atom_name[0]=='D' and atom_type=='D':
                atom_type='H'
                atom_name=atom_name.replace('D','H',1)
            if chain_id not in mol.chains.keys():
                mol.chains[chain_id]=chain(chain_id)
            key=residue_no+residue_insertion_code
            if key not in mol.chains[chain_id].residues.keys():
                mol.chains[chain_id].residues[key]=get_residue(residue_name)
                if atom_name in vars(mol.chains[chain_id].residues[key]).keys():
                    if atom_alt_loc and mol.chains[chain_id].residues[key].__name__[1:]==residue_name:
                        if not prev_res_key:
                            prev_res_key=key
                        if vars(mol.chains[chain_id].residues[key])[atom_name].serial_num:
                            if float(atom_occupancy)>float(vars(mol.chains[chain_id].residues[key])[atom_name].occupancy):
                                vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                                vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                                vars(mol.chains[chain_id].residues[key])[atom_name].alt_serial_num=atom_serial_number
                            else:
                                vars(mol.chains[chain_id].residues[key])[atom_name].alt_serial_num=atom_serial_number
                        else:
                            vars(mol.chains[chain_id].residues[key])[atom_name].serial_num=atom_serial_number
                            vars(mol.chains[chain_id].residues[key])[atom_name].alt_loc=atom_alt_loc
                            vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                            vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                            vars(mol.chains[chain_id].residues[key])[atom_name].type=atom_type
                            mol.chains[chain_id].index[atom_serial_number]=(key,atom_name)
                    
                    else:
                        vars(mol.chains[chain_id].residues[key])[atom_name].serial_num=atom_serial_number
                        vars(mol.chains[chain_id].residues[key])[atom_name].alt_loc=atom_alt_loc
                        vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                        vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                        vars(mol.chains[chain_id].residues[key])[atom_name].type=atom_type
                        mol.chains[chain_id].index[atom_serial_number]=(key,atom_name)
                        prev_res_key=None
            else:
                if atom_name in vars(mol.chains[chain_id].residues[key]).keys():
                    if atom_alt_loc and prev_res_key==key and mol.chains[chain_id].residues[key].__name__[1:]==residue_name:
                        if vars(mol.chains[chain_id].residues[key])[atom_name].serial_num:
                            if float(atom_occupancy)>float(vars(mol.chains[chain_id].residues[key])[atom_name].occupancy):
                                vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                                vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                                vars(mol.chains[chain_id].residues[key])[atom_name].alt_serial_num=atom_serial_number
                            else:
                                vars(mol.chains[chain_id].residues[key])[atom_name].alt_serial_num=atom_serial_number
                        else:
                            vars(mol.chains[chain_id].residues[key])[atom_name].serial_num=atom_serial_number
                            vars(mol.chains[chain_id].residues[key])[atom_name].alt_loc=atom_alt_loc
                            vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                            vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                            vars(mol.chains[chain_id].residues[key])[atom_name].type=atom_type
                            mol.chains[chain_id].index[atom_serial_number]=(key,atom_name)
                        prev_res_key=key
                    elif atom_alt_loc and prev_res_key==key and mol.chains[chain_id].residues[key].__name__[1:]!=residue_name:
                        occ=next(res_iter(mol.chains[chain_id].residues[key])).occupancy
                        if float(occ)<float(atom_occupancy):
                            mol.chains[chain_id].residues[key]=get_residue(residue_name)
                            vars(mol.chains[chain_id].residues[key])[atom_name].serial_num=atom_serial_number
                            vars(mol.chains[chain_id].residues[key])[atom_name].alt_loc=atom_alt_loc
                            vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                            vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                            vars(mol.chains[chain_id].residues[key])[atom_name].type=atom_type
                            mol.chains[chain_id].index[atom_serial_number]=(key,atom_name)
                        else:
                            continue

                    else:
                        vars(mol.chains[chain_id].residues[key])[atom_name].serial_num=atom_serial_number
                        vars(mol.chains[chain_id].residues[key])[atom_name].alt_loc=atom_alt_loc
                        vars(mol.chains[chain_id].residues[key])[atom_name].coord=(float(atom_x),float(atom_y),float(atom_z))
                        vars(mol.chains[chain_id].residues[key])[atom_name].occupancy=atom_occupancy
                        vars(mol.chains[chain_id].residues[key])[atom_name].type=atom_type
                        mol.chains[chain_id].index[atom_serial_number]=(key,atom_name)
                        prev_res_key=None
        elif ter_count==ter:
            break
    return mol
def get_graph_data(pdb_filepath,rscb=False,uniprot=False):
    temp_serial=1
    i=0
    pdb=pdb_filepath
    pdb_chain=chain
    file_names=pdb+'.pdb'
    try:
        temp=parser(pdb_filepath,rscb=rscb,uniprot=uniprot)
        temp.check_model()
        mol=get_molecule(temp)
        vertices={}
        keys={}
        for pdb_chain in mol.chains.keys():
            for key in mol.chains[pdb_chain].residues.keys():
                for atm in res_iter(mol.chains[pdb_chain].residues[key]):
                    if atm.serial_num:
                        if pdb_chain in vertices:
                            vertices[pdb_chain].append([atm.serial_num,atm.name[0],str(atm),*list(atm.coord),mol.chains[pdb_chain].residues[key].__name__[1:],str(key),pdb_chain,*atm.properties])
                            keys[atm.serial_num]=i
                            i+=1
                        else:
                            vertices[pdb_chain]=[[atm.serial_num,atm.name[0],str(atm),*list(atm.coord),mol.chains[pdb_chain].residues[key].__name__[1:],str(key),pdb_chain,*atm.properties]]
                            keys[atm.serial_num]=i
                            i+=1
                    else:
                        atm.serial_num='*'+str(temp_serial)
                        if pdb_chain in vertices: 
                            vertices[pdb_chain].append([atm.serial_num,atm.name[0],str(atm),*list(atm.coord),mol.chains[pdb_chain].residues[key].__name__[1:],str(key),pdb_chain,*atm.properties])
                            temp_serial+=1
                            keys[atm.serial_num]=i
                            i+=1
                        else:
                            vertices[pdb_chain]=[[atm.serial_num,atm.name[0],str(atm),*list(atm.coord),mol.chains[pdb_chain].residues[key].__name__[1:],str(key),pdb_chain,*atm.properties]]
                            temp_serial+=1
                            keys[atm.serial_num]=i
                            i+=1
        return vertices
    except AAError as e:
        print('Amino Acid Error',file_names)
    except ModelError as e:
        print('Model Error',file_names)
    except MissingAtom as e:
        print('Missing Atom Error',e)

     
