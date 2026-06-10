import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def resolve_device(device=None):
    """device 未指定なら cuda → mps → cpu の順で自動選択する"""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Double Convolution block for UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
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

        return torch.tanh(output)


# Dataset for noisy image pairs
class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_pairs=None):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.custom_transform = transform
        self.size = (780, 972)
        self.image_pairs = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory {root_dir} not found")

        # フォルダ内の連続する2枚を全ペア化（N枚→N-1ペア）
        for folder in sorted(self.root_dir.iterdir()):
            if folder.is_dir():
                images = sorted(folder.glob('*.jpg'))
                for i in range(len(images) - 1):
                    self.image_pairs.append((images[i], images[i + 1]))

        if not self.image_pairs:
            raise FileNotFoundError(f"No consecutive image pairs found in {root_dir}")

        if max_pairs is not None:
            self.image_pairs = self.image_pairs[:max_pairs]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]

        input_img = Image.open(str(img1_path)).convert('RGB').resize(self.size, Image.BILINEAR)
        target_img = Image.open(str(img2_path)).convert('RGB').resize(self.size, Image.BILINEAR)

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        if self.custom_transform:
            input_tensor = self.custom_transform(input_tensor)
            target_tensor = self.custom_transform(target_tensor)

        return input_tensor, target_tensor


class Denoiser:
    """推論専用クラス。学習データなしでモデル読み込み・ノイズ除去ができる"""

    def __init__(self, device=None, model_dir=None):
        self.device = resolve_device(device)
        self.model_dir = Path(model_dir) if model_dir is not None else None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.model = UNet().to(self.device)

    def _resolve_model_path(self, filename):
        # パス区切りを含む・絶対パス・model_dir 未設定の場合はそのままのパスとして扱い、
        # 単なるファイル名なら model_dir からの相対とみなす（従来の load_model 互換）
        path = Path(filename)
        if len(path.parts) > 1 or self.model_dir is None:
            return path
        return self.model_dir / path

    def load_model(self, filename):
        load_path = self._resolve_model_path(filename)
        checkpoint = torch.load(load_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def denoise_image(self, input_path, output_path):
        self.model.eval()

        # 画像の読み込みと変換
        input_img = Image.open(input_path).convert('RGB')
        input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)

        # モデルの推論
        with torch.no_grad():
            output = self.model(input_tensor)

        # tanh出力 [-1,1] を [0,1] に逆正規化してから保存
        output = output.squeeze(0).cpu()
        output = output * 0.5 + 0.5
        output = output.clamp(0, 1)
        output_img = transforms.ToPILImage()(output)
        output_img.save(output_path)

    def _empty_cache(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


# Training class
class Noise2Noise(Denoiser):
    def __init__(self, train_dir=None, valid_dir=None, model_dir=None, device=None, max_pairs=None):
        super().__init__(device=device, model_dir=model_dir)

        if self.model_dir is not None:
            self.model_dir.mkdir(exist_ok=True)

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.max_pairs = max_pairs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.valid_losses = []

        # データセットは train() 呼び出し時まで遅延生成する
        # （推論だけの利用なら train_dir / valid_dir は不要）
        self.train_loader = None
        self.valid_loader = None

    def _setup_data(self):
        if self.train_loader is not None:
            return

        if self.train_dir is None or self.valid_dir is None:
            raise ValueError("train_dir and valid_dir are required for training")

        # Setup datasets with additional transforms if needed
        additional_transforms = None  # 必要に応じて追加の transforms を定義

        try:
            train_dataset = NoisyDataset(self.train_dir, additional_transforms, max_pairs=self.max_pairs)
            valid_dataset = NoisyDataset(self.valid_dir, additional_transforms, max_pairs=self.max_pairs)
        except FileNotFoundError as e:
            print(f"Error initializing datasets: {e}")
            raise

        # Setup dataloaders with smaller batch size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

    def train(self, epochs, run_name="fixed_model"):
        self._setup_data()

        best_valid_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, (input_img, target_img) in enumerate(self.train_loader):
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(input_img)
                loss = self.criterion(output, target_img)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} [{batch_idx}/{len(self.train_loader)}] '
                            f'Loss: {loss.item():.6f}')

            self._empty_cache()

            # Validation
            valid_loss = self.validate()
            print(f'Epoch {epoch+1} Average Train Loss: {train_loss/len(self.train_loader):.6f} '
                    f'Valid Loss: {valid_loss:.6f}')
            self.train_losses.append(train_loss / len(self.train_loader))
            self.valid_losses.append(valid_loss)

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(run_name + '.pth')
        self.save_losses_to_csv(run_name + '.csv')

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
        save_path = self._resolve_model_path(filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def save_losses_to_csv(self, filename):
        filepath = self._resolve_model_path(filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Train Loss', 'Valid Loss'])
            for i in range(len(self.train_losses)):
                csvwriter.writerow([self.train_losses[i], self.valid_losses[i]])
        print(f'Losses saved to {filename}')

    def load_model(self, filename):
        checkpoint = super().load_model(filename)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def _build_parser():
    parser = argparse.ArgumentParser(description="Noise2Noise の学習・推論 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="モデルを学習する")
    train_parser.add_argument("--train-dir", default="../../Resources/AI/train_data",
                              help="学習データのディレクトリ")
    train_parser.add_argument("--valid-dir", default="../../Resources/AI/valid_data",
                              help="検証データのディレクトリ")
    train_parser.add_argument("--model-dir", default="../../Resources/AI/model_dir",
                              help="モデル・損失CSVの保存先ディレクトリ")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--max-pairs", type=int, default=200,
                              help="使用する画像ペア数の上限（0 で無制限）")
    train_parser.add_argument("--run-name", default="fixed_model",
                              help="保存するモデル・CSVのファイル名（拡張子なし）")
    train_parser.add_argument("--device", default=None,
                              help="cuda / mps / cpu（省略時は自動選択）")
    train_parser.add_argument("--resume", default=None,
                              help="学習を再開するモデルファイル（model-dir 内のファイル名かフルパス）")

    denoise_parser = subparsers.add_parser("denoise", help="学習済みモデルでノイズ除去する")
    denoise_parser.add_argument("--model", required=True,
                                help="モデルファイル（.pth）のパス")
    denoise_parser.add_argument("--input", required=True,
                                help="入力画像ファイル、または jpg を含むディレクトリ")
    denoise_parser.add_argument("--output", required=True,
                                help="出力画像ファイル、または出力先ディレクトリ")
    denoise_parser.add_argument("--device", default=None,
                                help="cuda / mps / cpu（省略時は自動選択）")

    return parser


def _run_train(args):
    trainer = Noise2Noise(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        model_dir=args.model_dir,
        device=args.device,
        max_pairs=args.max_pairs if args.max_pairs > 0 else None,
    )
    if args.resume:
        trainer.load_model(args.resume)
    trainer.train(epochs=args.epochs, run_name=args.run_name)


def _run_denoise(args):
    denoiser = Denoiser(device=args.device)
    denoiser.load_model(args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        images = sorted(input_path.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No jpg files found in {input_path}")
        for i, img in enumerate(images, start=1):
            denoiser.denoise_image(str(img), str(output_path / img.name))
            print(f"[{i}/{len(images)}] {img.name}")
    else:
        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        denoiser.denoise_image(str(input_path), str(output_path))
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    cli_args = _build_parser().parse_args()
    if cli_args.command == "train":
        _run_train(cli_args)
    else:
        _run_denoise(cli_args)
