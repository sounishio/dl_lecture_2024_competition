import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        # EEGデータの処理用のブロック
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        # 画像データの処理用のブロック
        self.image_model = models.resnet18(weights="IMAGENET1K_V1")
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 512)
        
        # 結合部分
        self.head = nn.Sequential(
            nn.Linear(hid_dim + 512, num_classes),
        )

    def forward(self, X: torch.Tensor, image: torch.Tensor = None) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): EEGデータ
            image ( b, 3, H, W ): 画像データ（テスト時には None）
        Returns:
            X ( b, num_classes ): 出力クラス
        """
        # EEGデータの処理
        eeg_features = self.blocks(X)
        eeg_features = nn.AdaptiveAvgPool1d(1)(eeg_features)
        eeg_features = eeg_features.view(eeg_features.size(0), -1)

        if image is not None:
            # 画像データの処理
            image_features = self.image_model(image)
            # 結合
            combined_features = torch.cat((eeg_features, image_features), dim=1)
        else:
            # 画像データがない場合は、画像の特徴量をゼロテンソルとする
            image_features = torch.zeros(eeg_features.size(0), 512, device=eeg_features.device)
            combined_features = torch.cat((eeg_features, image_features), dim=1)

        return self.head(combined_features)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,   # baselineでは0.1だったのを0.3に変更
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
