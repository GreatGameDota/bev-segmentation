from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, train_df, root_dir, seg_dir, folds=None, label_smoothing=0.01, transform=None, mode='train'):
        self.train_df = train_df[train_df.fold.isin(folds).reset_index(drop=True)].reset_index(drop=True)
        # self.train_df = train_df
        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.folds = folds
        self.mode = mode
        self.label_smoothing = label_smoothing

        # self.labels = self.train_df['outlier'].values.astype(np.float32)
        self.ids = self.train_df['id']

    def __len__(self):
        # return len(self.train_df)
        if self.mode == 'train':
          return len(self.ids) * 1
        else:
          return len(self.ids)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
          # idx = random.randint(0, len(self.ids)-1)
          idx %= len(self.ids)
          
        img_name = "0000" + self.ids[idx].split("_")[-1].split(".")[0]
        
        img = cv2.imread(f'{self.root_dir}{img_name}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.input_H, config.input_W))
        
        labels = np.load(f'{self.seg_dir}{self.ids[idx]}')
        labels = cv2.resize(labels.astype(np.float32), (config.output_H, config.output_W)).astype(np.int16)

        if self.transform is not None:
          res = self.transform(image=img, mask=labels)
          img = res['image']
          labels = res['mask']

        img = np.array(img).astype(np.float32)
        img = np.rollaxis(img, -1, 0)

        labels = np.array(labels).astype(np.float32)
        labels = np.rollaxis(labels, -1, 0)
        
        if self.mode != 'test' and self.mode != 'val':
          # labels = np.array(self.labels[idx]).astype(np.float32)
          # labels = np.clip(np.array(self.labels[idx]).astype(np.float32), self.label_smoothing, 1 - self.label_smoothing)[None]
          return [img, labels]
        elif self.mode == 'val':
          # labels = np.array(self.labels[idx]).astype(np.float32)
          return [img, labels, np.array(self.ids[idx])[None]]
        else:
          return [img]