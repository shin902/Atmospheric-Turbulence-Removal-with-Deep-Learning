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
import cv2

import flet as ft



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

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss


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


def detect_ellipse(image):
    """
    画像から楕円を検出する関数（木星画像用に最適化）
    """
    # 画像の前処理
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # コントラストの強調
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # ノイズ除去
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # 2値化
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # モルフォロジー処理
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # デバッグ用コンテナ
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 最大の輪郭を見つける
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            try:
                # 楕円フィッティング
                ellipse = cv2.fitEllipse(largest_contour)
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]

                return center, axes, angle

            except Exception as e:
                print(f"楕円フィッティングエラー: {str(e)}")
        else:
            print(f"輪郭の点数が不足しています（{len(largest_contour)} < 5点）")

    # デバッグ画像を表示
    cv2.imshow('Debug Image', debug_image)
    return None, None, None

def ellipse_to_circle(image_path, out_path=None):
    """
    楕円を真円に変換する関数（行列計算を修正）
    """
    image = cv2.imread(image_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return

    # 前処理
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # 楕円検出
    center, axes, angle = detect_ellipse(image)

    if center is None:
        print("楕円検出に失敗しました")
        return image, image, (None, None, None)

    # 楕円の軸
    major_axis, minor_axis = axes
    target_radius = max(major_axis, minor_axis) / 2  # 半径に修正

    # スケール計算
    scale_x = target_radius / (major_axis / 2)
    scale_y = target_radius / (minor_axis / 2)

    # アフィン変換のための準備
    height, width = image.shape[:2]
    center_x, center_y = center

    # 変換行列の計算（3x3形式）
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 3x3行列として計算
    # 1. 中心を原点に移動
    T1 = np.array([[1, 0, -center_x],
                   [0, 1, -center_y],
                   [0, 0, 1]], dtype=np.float32)

    # 2. 回転
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]], dtype=np.float32)

    # 3. スケーリング
    S = np.array([[scale_x, 0, 0],
                  [0, scale_y, 0],
                  [0, 0, 1]], dtype=np.float32)

    # 4. 逆回転
    R_inv = np.array([[cos_theta, sin_theta, 0],
                      [-sin_theta, cos_theta, 0],
                      [0, 0, 1]], dtype=np.float32)

    # 5. 中心を元に戻す
    T2 = np.array([[1, 0, center_x],
                   [0, 1, center_y],
                   [0, 0, 1]], dtype=np.float32)

    # 行列の結合（右から順に適用）
    M = T2 @ R_inv @ S @ R @ T1

    # 2x3行列に変換（アフィン変換用）
    M_affine = M[:2, :]

    # 変換を適用
    transformed = cv2.warpAffine(image, M_affine, (width, height), flags=cv2.INTER_CUBIC)

    if out_path:
        cv2.imwrite(out_path, transformed)

    return transformed


def main(page: ft.Page):
    page.title = "Noise Reduction App"

    input_folder_path = ft.TextField(label="Input Folder")
    output_folder_path = ft.TextField(label="Output Folder")
    temp_folder_path = ft.TextField(label="Temp Folder")
    affine_checkbox = ft.Checkbox(label="アフィン変換を実行する")
    status_text = ft.Text("")

    def denoise_images(e):
        status_text.value = "処理開始..."
        page.update()

        input_dir = input_folder_path.value
        output_dir = output_folder_path.value
        temp_dir = temp_folder_path.value # Currently not used in denoise process

        if not input_dir or not output_dir:
            status_text.value = "入力フォルダーと出力フォルダーを指定してください。"
            page.update()
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        use_affine = affine_checkbox.value or False  # チェックボックスの状態を取得

        device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"  # Use MPS if available
        noise_reducer = Noise2Noise(
            train_dir=input_dir,  # train_dir and valid_dir are not actually used in denoise_image function
            valid_dir=input_dir,  # but are required for Noise2Noise class initialization
            model_dir="model",
            device=device
        )
        noise_reducer.load_model("noise2noise_model.pth")  # Assuming a pretrained model exists

        for img_file in input_path.glob("*.jpg"):  # Process only jpg files for simplicity
            output_file = output_path / f"denoised_{img_file.name}"
            temp_file = Path(temp_dir) / f"temp_{img_file.name}" if temp_dir else None # temp_dir があれば一時ファイルパスを設定

            try:
                # アフィン変換を行う場合
                if use_affine:
                    if not temp_dir:
                        status_text.value = "一時フォルダーを指定してください。"
                        page.update()
                        return

                    temp_dir_path = Path(temp_dir)
                    temp_dir_path.mkdir(parents=True, exist_ok=True) # Ensure temp directory exists

                    transformed_image = ellipse_to_circle(str(img_file), str(temp_file))
                    if transformed_image is None:
                        status_text.value = f"アフィン変換エラー: {img_file.name}"
                        page.update()
                        continue  # アフィン変換が失敗したら次のファイルへ

                    denoised_temp_path = temp_dir_path / f"denoised_temp_{img_file.name}"
                    noise_reducer.denoise_image(str(temp_file), str(denoised_temp_path))
                    # Denoise処理後の画像をoutput_fileに保存 (アフィン変換後の画像に対してノイズ除去)
                    cv2.imwrite(str(output_file), cv2.imread(str(denoised_temp_path)))


                # アフィン変換を行わない場合
                else:
                    noise_reducer.denoise_image(str(img_file), str(output_file))

            except Exception as err:
                status_text.value = f"ノイズ除去エラー: {img_file.name} - {err}"
                page.update()
                return  # Stop processing if error occurs

        status_text.value = "処理が完了しました。"
        page.update()

    denoise_button = ft.ElevatedButton("処理開始", on_click=denoise_images)

    page.add(
        ft.Column(
            [
                input_folder_path,
                output_folder_path,
                temp_folder_path,
                affine_checkbox,
                denoise_button,
                status_text,
            ]
        )
    )



if __name__ == "__main__":

    ft.app(target=main)
