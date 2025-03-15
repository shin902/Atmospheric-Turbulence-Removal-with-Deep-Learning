import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import csv


train_losses = []
valid_losses = []



# Double Convolution block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),  # padding='same' に変更
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),  # padding='same' に変更
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # 入力サイズを保存
        input_size = x.shape[2:]

        # Encoder
        enc1 = self.encoder1(x)
        p1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(p1)
        p2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(p2)
        p3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(p3)
        p4 = F.max_pool2d(enc4, 2)

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoder with size matching
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat([d4, F.interpolate(enc4, size=d4.shape[2:])], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, F.interpolate(enc3, size=d3.shape[2:])], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, F.interpolate(enc2, size=d2.shape[2:])], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, F.interpolate(enc1, size=d1.shape[2:])], dim=1)
        d1 = self.decoder1(d1)

        # 最終出力を入力サイズにリサイズ
        output = self.final_conv(d1)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        """
        print(f"Input size: {x.shape}")
        print(f"Encoder1 output: {enc1.shape}")
        print(f"Decoder1 output: {d1.shape}")
        """

        return torch.sigmoid(output)
# Dataset for noisy image pairs


class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        # 基本的な変換を定義（ToTensorとNormalize）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # RGB各チャンネルの平均
                                 std=[0.5, 0.5, 0.5])     # RGB各チャンネルの標準偏差
        ])
        self.custom_transform = transform  # 追加の変換が必要な場合用
        self.image_pairs = []

        # Find all image pairs
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory {root_dir} not found")

        # Initialize image pairs
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                noisy_images = list(folder.glob('*.jpg'))
                if len(noisy_images) >= 2:
                    self.image_pairs.append((noisy_images[0], noisy_images[1]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if idx >= len(self.image_pairs):
            raise IndexError("Index out of bounds")

        img1_path, img2_path = self.image_pairs[idx]

        # Load and convert images
        input_img = Image.open(str(img1_path)).convert('RGB')
        target_img = Image.open(str(img2_path)).convert('RGB')

        # Resize images
        size = (780, 972)
        input_img = input_img.resize(size, Image.BILINEAR)
        target_img = target_img.resize(size, Image.BILINEAR)

        # 基本的な変換を適用（ToTensorとNormalize）
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        # 追加の変換がある場合は適用
        if self.custom_transform:
            input_tensor = self.custom_transform(input_tensor)
            target_tensor = self.custom_transform(target_tensor)

        return input_tensor, target_tensor

# Training class
class Noise2Noise:
    def __init__(self, train_dir, valid_dir, model_dir, device):
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.transform = transforms.ToTensor()  # 必要なら他の変換も追加
        # Initialize model
        self.model = UNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Setup datasets with additional transforms if needed
        additional_transforms = None  # 必要に応じて追加の transforms を定義

        try:
            self.train_dataset = NoisyDataset(train_dir, additional_transforms)
            self.valid_dataset = NoisyDataset(valid_dir, additional_transforms)
        except FileNotFoundError as e:
            print(f"Error initializing datasets: {e}")
            raise

        # Setup dataloaders with smaller batch size
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # MacのMPSデバイスを使用する場合は0を推奨
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
    def train(self, epochs):
        best_valid_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, (input_img, target_img) in enumerate(self.train_loader):
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(input_img)
                loss = self.criterion(output, target_img)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(self.train_loader)}] '
                            f'Loss: {loss.item():.6f}')

            # Validation
            valid_loss = self.validate()
            print(f'Epoch {epoch+1} Average Train Loss: {train_loss/len(self.train_loader):.6f} '
                    f'Valid Loss: {valid_loss:.6f}')
            train_losses.append(train_loss / len(self.train_loader))
            valid_losses.append(valid_loss)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model('best_model.pth')
        self.save_losses_to_csv('losses-best.csv') # 追記


    def validate(self):
        self.model.eval()
        valid_loss = 0

        with torch.no_grad():
            for input_img, target_img in self.valid_loader:
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                output = self.model(input_img)
                loss = self.criterion(output, target_img)
                valid_loss += loss.item()

        return valid_loss / len(self.valid_loader)

    def save_model(self, filename):
        save_path = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def save_losses_to_csv(self, filename): # 追記
        filepath = self.model_dir / filename # 追記
        with open(filepath, 'w', newline='') as csvfile: # 追記
            csvwriter = csv.writer(csvfile) # 追記
            csvwriter.writerow(['Train Loss', 'Valid Loss']) # 追記
            for i in range(len(train_losses)): # 追記
                csvwriter.writerow([train_losses[i], valid_losses[i]]) # 追記
        print(f'Losses saved to {filename}') # 追記


    def load_model(self, filename):
        load_path = self.model_dir / filename
        checkpoint = torch.load(load_path, weights_only=True)  # weights_only=True を追加
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def denoise_image(self, input_path, output_path):
        self.model.eval()

        # 画像の読み込みと変換
        input_img = Image.open(input_path).convert('RGB')
        input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)

        # モデルの推論
        with torch.no_grad():
            output = self.model(input_tensor)

        # 出力テンソルを入力画像サイズにリサイズ
        output = F.interpolate(output, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # テンソルを画像に変換して保存
        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(output_path)






# Example usage
if __name__ == "__main__":
    # Setup device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # Initialize trainer
    trainer = Noise2Noise(
        train_dir="../../Resources/AI/train_data",
        valid_dir="../../Resources/AI/valid_data",
        model_dir="../../Resources/AI/model_dir",
        device=device
    )

    # Train model
    # trainer.train(epochs=1000)
    trainer.load_model('bright_model.pth')

    # Denoise a single image
    trainer.denoise_image("../../Resources/Images/19_57_44/001.jpg", "../../Resources/Input and Output/output/001-19_57_44.jpg")

"""
    img_path = Path("./img")
    img_path.mkdir(exist_ok=True)

    # グラフの描画
    plt.figure(figsize=(6, 4)) # 縦方向に少し大きく
    plt.plot(train_losses, marker='.', label='Train Loss') # train_losses をプロット、ラベルを追加
    plt.plot(valid_losses, marker='.', label='Valid Loss') # valid_losses をプロット、ラベルを追加
    plt.xlabel('Epoch') # X軸ラベルを追加
    plt.ylabel('Loss') # Y軸ラベルを追加
    plt.title('Training and Validation Loss') # タイトルを追加
    plt.legend() # 凡例を表示
    plt.grid(True) # グリッド線を追加
    plt.savefig("./img/best_loss_graph.png") # ファイル名を変更
"""
