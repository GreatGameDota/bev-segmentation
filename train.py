import os
import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
import gc
import yaml
import argparse
import random
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from effdet import get_efficientdet_config

from src.utils.datasets import ImageDataset
from src.utils.augmentations import train_transform, val_transform
from src.models.model import BiFPN_CrossTD
from src.utils.schedulers import *
from src.utils.train_utils import train_model, evaluate_model, id_collate
from src.utils.utils import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a bev segmentation model.')
    parser.add_argument("--model-cfg", dest="model_cfg", help="Config file for model parameters", default="model.yaml", type=str)
    parser.add_argument("--train-cfg", dest="train_cfg", help="Config file for training parameters", default="train_cfg.yaml", type=str)
    parser.add_argument("--drive", dest="drive", help="Save result files in Google Drive", action='store_true', default=False)
    args = parser.parse_args()
    
    with open(f"src/{args.model_cfg}") as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(f"src/{args.train_cfg}") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    seed_everything(train_cfg['seed'])

    def _init_fn(worker_id):
        np.random.seed(train_cfg['seed'])
        random.seed(train_cfg['seed'])

    validations = []

    if not os.path.isdir("src/runs"):
        os.mkdir("src/runs")
    exp_dir = f"src/runs/exp{len(os.listdir('src/runs/')) + 1}"
    os.mkdir(exp_dir)

    log_name = f"{exp_dir}/log.log"

    tars_,preds_,ids_=[],[],[]

    for fold in range(train_cfg['folds']):
        print(f'Train Fold {fold+1}')
        with open(log_name, 'a') as f:
            f.write(f'Train Fold {fold+1}\n\n')

        history = pd.DataFrame()
        history2 = pd.DataFrame()

        torch.cuda.empty_cache()
        gc.collect()

        best = -1e10
        best2 = 1e10
        n_epochs = train_cfg['epochs']
        early_epoch = 0
        early_stop = 0
        
        ids = os.listdir(train_cfg['label_path'])
        ids.sort()
        train_df = pd.DataFrame({"id": ids, "label": np.ones(len(ids))})

        train_df['fold'] = -1
        skf = KFold(n_splits=train_cfg['folds'], random_state=train_cfg['seed'], shuffle=True)
        for fld, (_,test_idx) in enumerate(skf.split(train_df['id'], train_df['label'])):
            train_df.iloc[test_idx, -1] = fld

        val_dataset = ImageDataset(train_cfg, train_df, train_cfg['image_path'], 
                                    seg_dir=train_cfg['label_path'], folds=[fold], mode='val', transform=val_transform)

        train_dataset = ImageDataset(train_cfg, train_df, train_cfg['image_path'], seg_dir=train_cfg['label_path'],
                                    folds=[i for i in np.arange(train_cfg['folds']) if i != fold], 
                                    transform=train_transform, mode='train')

        BATCH_SIZE = train_cfg['batch_size']

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=_init_fn, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=0, worker_init_fn=_init_fn, collate_fn=id_collate)

        scaler = amp.GradScaler()

        fpn_config = get_efficientdet_config(model_cfg['bifpn'])
        model = BiFPN_CrossTD(fpn_config, model_cfg['backbone'], hidden_dim=model_cfg['hidden_dim'], out_sz=train_cfg['output_sz'], num_classes=train_cfg['classes'])
        # model.freeze_backbone()
        model = model.cuda()

        # optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        # optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg['lr'], momentum=0.9, weight_decay=1e-4)

        updates_per_epoch = len(train_loader)
        num_updates = int(n_epochs * updates_per_epoch)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', factor=0.75, verbose=True, min_lr=1e-7)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=num_updates)
        scheduler = WarmupCosineSchedule(warmup=0.002, t_total=num_updates)
        
        for epoch in range(n_epochs-early_epoch):
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            # if epoch == 10:
            #   model.unfreeze_backbone()

            with open(log_name, 'a') as f:
                f.write(f'XXXXXXXXXXXXXX-- CYCLE INTER: {epoch+1} --XXXXXXXXXXXXXXXXXXX\n')
                f.write(f"curr lr: {optimizer.state_dict()['param_groups'][0]['lr']}\n")

            train_model(train_cfg, model, train_loader, epoch, optimizer, scaler=scaler, scheduler=scheduler, history=history)

            _, _, _, loss, kaggle = evaluate_model(train_cfg, model, val_loader, epoch, scheduler=None, history=history2)
            
            if loss < best2:
                best2 = loss
                early_stop = 0
                print(f'Saving best model... (loss)')
                torch.save({
                    'model_state': model.state_dict(),
                }, f'{exp_dir}/model-fld{fold+1}.pth')

                with open(log_name, 'a') as f:
                    f.write('Saving Best model...\n\n')
            else:
                early_stop += 1
                with open(log_name, 'a') as f:
                    f.write('\n')

            if early_stop == train_cfg['early_stop']:
                print("Stopping early")
                with open(log_name, 'a') as f:
                    f.write('Stopping early\n\n')
                break
        print()
        validations.append(best)
        model = BiFPN_CrossTD(fpn_config, model_cfg['backbone'], hidden_dim=model_cfg['hidden_dim'], out_sz=train_cfg['output_sz'], num_classes=train_cfg['classes'])
        model.load_state_dict(torch.load(f'{exp_dir}/model-fld{fold+1}.pth')['model_state'])
        model.cuda()
        ids, pred, tars, loss, kaggle = evaluate_model(train_cfg, model, val_loader, 0, scheduler=None, history=history2)
        for k,p in enumerate(pred):
            tars_.append(tars[k])
            preds_.append(p)
            ids_.append(ids[k])
        print()
        break
    
    if args.drive:
        os.system(f"zip -q -r runs.zip {exp_dir}")
        os.system(f"mv runs.zip ../drive/MyDrive/runs.zip")