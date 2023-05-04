from .model import get_model
from .dataset import get_trainVal_loder
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import AdamW 
import MinkowskiEngine as ME
import tqdm
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.148021,weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
    def forward(self, inputs, targets,smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss =  self.alpha * (1-BCE_EXP)** self.gamma * BCE
                       
        return focal_loss

def validation(epoch_iterator_val,model,loss_function,device):
    model.eval()
    total_loss=0
    total_dice=0
    roc_auc=0
    f1_score=0
    with torch.no_grad():
        for step, batchs in enumerate(epoch_iterator_val):
            coords, feat, label = batchs
            inputs = ME.SparseTensor(feat, coordinates=coords, device=device)
            labels = label.view(-1,1).to(device)
            outputs = model(inputs)
            loss = loss_function(outputs.F, labels.float())
            y_pred=torch.sigmoid(outputs.F)
            total_dice+=metrics.average_precision_score(label.detach().cpu().numpy(),y_pred.detach().cpu().numpy())
            roc_auc+=metrics.roc_auc_score(label.detach().cpu().numpy(),y_pred.detach().cpu().numpy())
            f1_score+=metrics.f1_score(label.detach().cpu().numpy(),(y_pred.detach().cpu().numpy()>=0.5)*1)
            total_loss+=loss.item()
    return total_dice/len(epoch_iterator_val),roc_auc/len(epoch_iterator_val),f1_score/len(epoch_iterator_val),total_loss/len(epoch_iterator_val)

def train(path_to_save,device,dataset_path='sparse'):
    if torch.cuda.is_available() and device!='cpu':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=f"{device}"
        device='cuda'
    else:
        device='cpu'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    prev_loss=1000000
    count=0
    step=0
    t_loss=0
    t_acc=0
    criterion = FocalLoss()
    model=get_model()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.000789)
    train_dataloader,test_dataloder=get_trainVal_loder(dataset_path)
    with open(path_to_save+'/'+'training.txt',mode='a') as f:
        f.write('Epoch,Training_Loss,Validation_loss,Training_PRC,Validation_PRC,Validation_ROC,Validation_F1score\n')
    
    for i in range(10000):
        epoch_iterator = tqdm.tqdm(train_dataloader,unit='Batch',leave=True)
        epoch_iterator.set_description(f'Epoch {i} ')
        for batch in epoch_iterator:
            model.train()
            step+=1
            coords, feat, label = batch
            input = ME.SparseTensor(feat, coordinates=coords, device=device)
            label = label.view(-1,1).to(device)
            optimizer.zero_grad()
            output = model(input)           
            loss = criterion(output.F, label.float())
            ME.clear_global_coordinate_manager() 
            loss.backward()
            optimizer.step()
            model.eval()
            y_pred=torch.sigmoid(output.F)
            t_loss+=loss.item()
            t_acc+=metrics.average_precision_score(label.detach().cpu().numpy(),y_pred.detach().cpu().numpy())
            ME.clear_global_coordinate_manager()
        t_loss=t_loss/len(epoch_iterator)
        t_acc=t_acc/len(epoch_iterator)
        val_score,roc_auc,f1_score,val_loss=validation(test_dataloder,model,criterion,device)
        with open(path_to_save+'/'+'training.txt',mode='a') as f:
            f.write(f"{i},{t_loss},{val_loss},{t_acc},{val_score},{roc_auc},{f1_score}\n")
        t_loss=0
        t_acc=0
        if prev_loss<=round(val_loss,5):
            count+=1
        else:
            prev_loss=round(val_loss,5)
            count=0
            torch.save(model.state_dict(),f'{path_to_save}/best_model.pt')
        if count==12:
            return model
