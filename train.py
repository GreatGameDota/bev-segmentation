import os
import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
import gc
from torch.utils.data.dataloader import default_collate

def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids

if __name__ == 'main':
    validations = []

    log_name = f"drive/My Drive/logs/log-{len(os.listdir('drive/My Drive/logs/')) + 1}.log"

    tars_,preds_,ids_=[],[],[]

    for fold in range(config.folds):
        # fold = config.single_fold
        # fold = 4
        print(f'Train Fold {fold+1}')
        with open(log_name, 'a') as f:
            f.write(f'Train Fold {fold+1}\n\n')

        history = pd.DataFrame()
        history2 = pd.DataFrame()

        torch.cuda.empty_cache()
        gc.collect()

        best = -1e10
        best2 = 1e10
        n_epochs = config.epochs
        early_epoch = 0
        early_stop = 0

        val_dataset = ImageDataset(train_df, config.IMAGE_PATH, seg_dir=config.SEG_PATH, folds=[fold], mode='val', transform=val_transform)

        train_dataset = ImageDataset(train_df, config.IMAGE_PATH, seg_dir=config.SEG_PATH,
                                    folds=[i for i in np.arange(config.folds) if i != fold], 
                                    transform=train_transform, mode='train')

        BATCH_SIZE = config.batch_size

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=_init_fn, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=0, worker_init_fn=_init_fn, collate_fn=id_collate)

        scaler = amp.GradScaler()

        model = BiFPN_CrossTD(fpn_config, "tf_efficientnet_b0_ns", hidden_dim=512, out_sz=config.output_H, num_classes=config.n_classes)
        # model.freeze_backbone()
        model = model.cuda()

        # optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)

        if config.apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        updates_per_epoch = len(train_loader)
        num_updates = int(config.epochs * updates_per_epoch)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', factor=0.75, verbose=True, min_lr=1e-7)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=num_updates)
        scheduler = WarmupCosineSchedule(warmup=0.002, t_total=num_updates)
        
        for epoch in range(n_epochs-early_epoch):
            # break
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            # if epoch == 10:
            #   model.unfreeze_backbone()

            with open(log_name, 'a') as f:
                f.write(f'XXXXXXXXXXXXXX-- CYCLE INTER: {epoch+1} --XXXXXXXXXXXXXXXXXXX\n')
                lr_ = optimizer.state_dict()['param_groups'][0]['lr']
                f.write(f'curr lr: {lr_}\n')

            train_model(epoch, optimizer, scaler=scaler, scheduler=scheduler, history=history)

            _, _, _, loss, kaggle = evaluate_model(epoch, scheduler=None, history=history2)
            # break
            if loss < best2:
                best2 = loss
                early_stop = 0
                print(f'Saving best model... (loss)')
                torch.save({
                    'model_state': model.state_dict(),
                }, f'drive/My Drive/Models/model-fld{fold+1}.pth')
                with open(log_name, 'a') as f:
                    f.write('Saving Best model...\n\n')
            else:
                early_stop += 1
                with open(log_name, 'a') as f:
                    f.write('\n')
            if early_stop == config.early_stop:
                print("Stopping early")
                with open(log_name, 'a') as f:
                    f.write('Stopping early\n\n')
                break
        print()
        validations.append(best)
        model = BiFPN_CrossTD(fpn_config, "tf_efficientnet_b0_ns", hidden_dim=512, out_sz=config.output_H, num_classes=config.n_classes)
        model.load_state_dict(torch.load(f'drive/My Drive/Models/model-fld{fold+1}.pth')['model_state'])
        model.cuda()
        ids, pred, tars, loss, kaggle = evaluate_model(0, scheduler=None, history=history2)
        for k,p in enumerate(pred):
            tars_.append(tars[k])
            preds_.append(p)
            ids_.append(ids[k])
        print()
        break