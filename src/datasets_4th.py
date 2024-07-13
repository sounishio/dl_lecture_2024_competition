import os
import torch
from typing import Tuple
from termcolor import cprint
from PIL import Image
from torchvision import transforms

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", image_base_dir: str = "/content/Images", image_paths_file: str = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # 画像データの前処理の設定
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 画像パスの読み込み
        if image_paths_file:
            with open(image_paths_file, 'r') as f:
                self.image_paths = [os.path.join(image_base_dir, line.strip()) for line in f.readlines()]
        else:
            self.image_paths = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if self.image_paths:
            image_path = self.image_paths[i]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File {image_path} not found")
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)  # 前処理をここで適用
        else:
            image = torch.zeros(3, 224, 224)  # ダミーの画像データ

        if hasattr(self, "y"):
            return self.X[i], image, self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], image, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
