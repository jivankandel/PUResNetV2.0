from .model import get_trained_model
from .sparse import *
import os
import MinkowskiEngine as ME
from sklearn.cluster import DBSCAN

def flttostring(num):
    str_num=f'{"{:.3f}".format(float(num))}'
    return str_num
def save_predictions(vert_dict,pdb_name,chain,model,device):
    coord,feat,information=get_coordinates_features(vert_dict,chain)
    coord=ME.utils.batched_coordinates([coord],device=device)
    inputs = ME.SparseTensor(feat.float(), coordinates=coord, device=device)
    out=model(inputs)
    outs=torch.sigmoid(out.F).detach().cpu()
    prob=outs
    thresh=0.34
    out=outs>=thresh
    predicted=information[out.view(-1)]
    prob=prob[out.view(-1)]
    clustering = DBSCAN(eps=6,algorithm='kd_tree',leaf_size=100,min_samples=5).fit(np.asanyarray(predicted[:,-3:],dtype=float))
    clus=np.unique(clustering.labels_)
    lbl=clustering.labels_
    name=os.path.basename(pdb_name).split('.')[0]
    if isinstance(chain,list):
        chain=''.join(chain)
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists(f'results/{name}'):
        os.mkdir(f'results/{name}')
    for a in clus:
        if a==-1:
            continue
        mask=lbl==a
        pred=predicted[mask,:]
        probs=prob[mask]
        sorted_indices=np.argsort(pred[:,0].astype(float))
        pred=pred[sorted_indices]
        probs=probs[sorted_indices]
        if np.sum(mask)==1:
            continue
        with open(f'results/{name}/{a}_{chain}_{name}.pdb','w') as f:
            for i,pos in enumerate(pred):
                occupancy=probs[i].item()
                temp_factor=0.00
                atm_serial_num,atm_name,atm_type,res_name,res_num,res_chain,x,y,z=pos
                if isinstance(x,type(None)) or isinstance(y,type(None)) or isinstance(z,type(None)):
                    continue
                # line=f'{record_type:<6}{atm_serial_num:>5} {atm_name:<4} {res_name:<3} {res_chain:<1}{res_num:>4}    {flttostring(x):>8}{flttostring(y):>8}{flttostring(z):>8}{" "*23}{atm_type:<2}{" "*2}'
                line=(f"ATOM  {int(atm_serial_num):5d} {atm_name:<4s} {res_name:3s} {res_chain:1s}{int(res_num):4d}    "
                 f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {atm_type:>2s}  ")
                f.write(line+'\n')
    with open(f'results/{pdb_name}/withoutclustering.pdb','w') as f:
        s_indices=np.argsort(predicted[:,0].astype(float))
        predicted=predicted[s_indices]
        prob=prob[s_indices]
        for i,pos in enumerate(predicted):
            occupancy=prob[i].item()
            temp_factor=0.00
            atm_serial_num,atm_name,atm_type,res_name,res_num,res_chain,x,y,z=pos
            if isinstance(x,type(None)) or isinstance(y,type(None)) or isinstance(z,type(None)):
                continue
            # line=f'{record_type:<6}{atm_serial_num:>5} {atm_name:<4} {res_name:<3} {res_chain:<1}{res_num:>4}    {flttostring(x):>8}{flttostring(y):>8}{flttostring(z):>8}{" "*23}{atm_type:<2}{" "*2}'
            line=(f"ATOM  {int(atm_serial_num):5d} {atm_name:<4s} {res_name:3s} {res_chain:1s}{int(res_num):4d}    "
                f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {atm_type:>2s}  ")
            f.write(line+'\n')
def make_prediction(pdb_path,rscb=False,uniprot=False,chain=[],mode='A',device='cpu'):
    '''pdb_file : pdb_file path
                : if rscb True then pdb name
                : if uniprot then uniprot id (Alpha predicted structures are taken)
        mode    : must be on of (A,B,C)
                : (A) whole structure is predicted as single entity
                : (B) Each individual chain as entity 
                : (C) The provided chain is predicited as single entity
        chain   : Should be string or list of chains (Case Sensative) works with only mode C
        device  : cpu or GPU number(can be any gpu like 0,1,2 etc)
        
        Saves the prediction made in result folder. 
    '''
    assert mode in ['A','B','C'], f'Invalid mode {mode}'
    assert isinstance(chain,str) or isinstance(chain,list), f'Chain should be either string or list passed {type(chain)}'
    if torch.cuda.is_available() and device!='cpu':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{device}"
        device='cuda'
    else:
        device='cpu'
    model=get_trained_model()
    model=model.eval()
    model=model.to(device)
    pdb_name=os.path.basename(pdb_path)
    v_dict,chains=get_data(pdb_file=pdb_path,rscb=rscb,uniprot=uniprot)
    if mode=='A':
        try:
            save_predictions(v_dict,pdb_name=pdb_name,chain=chains,model=model,device=device)
        except:
            pass
    elif mode=='B':
        for ch in chains:
            try:
                save_predictions(v_dict,pdb_name=pdb_name,chain=ch,model=model,device=device)
            except Exception:
                pass
    elif mode=='C':
        try:
            save_predictions(v_dict,pdb_name=pdb_name,chain=chain,model=model,device=device)
        except Exception:
            pass


