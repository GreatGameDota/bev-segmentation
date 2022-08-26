

def train_model(epoch, optimizer, scaler=None, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    for i, (img_batch, y_batch) in enumerate(t):
        img_batch1 = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()
    
        if config.scale:
          with amp.autocast():
            output1 = model(img_batch1)
            loss = criterion1(output1, y_batch) / config.accumulation_steps
        else:
          output1 = model(img_batch1)
          loss = criterion1(output1, y_batch) / config.accumulation_steps

        total_loss += loss.data.cpu().numpy() * config.accumulation_steps
        t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

        if history is not None:
          history.loc[epoch + i / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
          history.loc[epoch + i / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        if config.scale:
          scaler.scale(loss).backward()
        elif config.apex:
          with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          loss.backward()
        
        if (i+1) % config.accumulation_steps == 0:
          if config.scale:
            scaler.step(optimizer)
            scaler.update()
          else:
            optimizer.step()
          optimizer.zero_grad()
        
        if scheduler is not None:
          lr = scheduler.get_lr((epoch * len(train_loader)) + (i + 1))
          optimizer.param_groups[0]['lr'] = config.lr * lr

def evaluate_model(epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred = []
    real = []
    P = []
    with torch.no_grad():
        for batch, ids in tqdm(val_loader):
            img_batch, y_batch = batch
            img_batch1 = img_batch.cuda().float()
            y_batch = y_batch.cuda().float()

            o1 = model(img_batch1)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            for i,batch in enumerate(o1):
              P.append(ids[i])
              # pred.append(torch.argmax(F.softmax(batch), 0).cpu().numpy())
              pred.append(F.sigmoid(batch).cpu().numpy())
            for tar in y_batch:
              real.append(tar.cpu().numpy())
    
    pred = np.array(pred)
    real = np.array(real)
    
    kaggle = dice_coef(pred, real)
    jaccard = iou_coef(pred, real)
    
    loss /= len(val_loader)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    if scheduler is not None:
      scheduler.step()

    print(f'Dev loss: %.4f, Kaggle: %.6f, Jaccard: %.6f'%(loss,kaggle,jaccard))
    
    with open(log_name, 'a') as f:
      f.write(f'val loss: {loss}\n')
      f.write(f'val Metric: {kaggle}\n')

    return P, pred, real, loss, kaggle