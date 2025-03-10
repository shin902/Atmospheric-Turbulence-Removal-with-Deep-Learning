import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


class AffineCorrectionAgent(nn.Module):
    def __init__(self):
        super(AffineCorrectionAgent, self).__init__()
        # カラー画像用に入力サイズを128*128*3に変更
        self.fc1 = nn.Linear(128 * 128 * 3, 128)  # 128ユニットの隠れ層
        self.fc2 = nn.Linear(128, 6)  # 出力層

    def forward(self, state):
        # view()の代わりにreshape()を使用し、連続していないメモリレイアウトも処理できるようにする
        state = state.reshape(-1)  # 入力を平坦化
        hidden = F.relu(self.fc1(state))  # 隠れ層にReLU活性化関数を適用
        affine_params = torch.tanh(self.fc2(hidden))  # 出力層にtanh活性化関数を適用

        # パラメータを適切な範囲にスケーリング
        tx = affine_params[0] * 100  # x方向の平行移動
        ty = affine_params[1] * 100  # y方向の平行移動
        shear_x = affine_params[2] * 0.5  # x方向のせん断
        shear_y = affine_params[3] * 0.5  # y方向のせん断
        scale_x = affine_params[4] * 0.5 + 1  # x方向のスケール (1〜1.5)
        scale_y = affine_params[5] * 0.5 + 1  # y方向のスケール (1〜1.5)

        return torch.stack([tx, ty, shear_x, shear_y, scale_x, scale_y])


# 微分可能なアフィン変換関数
def differentiable_affine_transform(image_tensor, params):
    """
    微分可能な方法でアフィン変換を適用する（PyTorchのグリッドサンプリングを使用）

    引数:
        image_tensor: 形状 [C, H, W] の画像テンソル
        params: アフィンパラメータ [tx, ty, shear_x, shear_y, scale_x, scale_y]
    """
    # チャンネル数、高さ、幅を取得
    batch_size = 1
    if len(image_tensor.shape) == 3:
        c, h, w = image_tensor.shape
        image_tensor = image_tensor.unsqueeze(0)  # バッチ次元を追加: [1, C, H, W]
    else:
        batch_size, c, h, w = image_tensor.shape

    # パラメータを抽出
    tx, ty, shear_x, shear_y, scale_x, scale_y = params

    # アフィン変換行列を作成
    theta = torch.zeros(batch_size, 2, 3, device=image_tensor.device)
    theta[:, 0, 0] = scale_x
    theta[:, 0, 1] = shear_x
    theta[:, 0, 2] = tx / w * 2  # グリッド座標系に正規化 [-1, 1]
    theta[:, 1, 0] = shear_y
    theta[:, 1, 1] = scale_y
    theta[:, 1, 2] = ty / h * 2  # グリッド座標系に正規化 [-1, 1]

    # アフィン変換グリッドを作成
    grid = F.affine_grid(theta, [batch_size, c, h, w], align_corners=True)

    # グリッドサンプリングを使用して変換を適用
    output = F.grid_sample(image_tensor, grid, align_corners=True)

    if batch_size == 1:
        return output.squeeze(0)  # バッチ次元を削除
    return output


def calculate_similarity_loss(original_tensor, transformed_tensor):
    """
    元画像と変換後画像のテンソル間のMSE損失を計算
    """
    mse_loss = nn.MSELoss()
    return mse_loss(transformed_tensor, original_tensor)


def calculate_sharpness(image):
    """画像のシャープさを計算"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness


def load_and_compare_sharpness(image_path1, image_path2):
    """2つの画像を読み込み、シャープさを比較"""
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None:
        raise ValueError(f"エラー: 画像 '{image_path1}' の読み込みに失敗しました。パスを確認してください。")
    if image2 is None:
        raise ValueError(f"エラー: 画像 '{image_path2}' の読み込みに失敗しました。パスを確認してください。")

    sharpness1 = calculate_sharpness(image1)
    sharpness2 = calculate_sharpness(image2)

    if sharpness1 >= sharpness2:
        sharper_image = image1
        to_correct_image = image2
        sharper_image_path = image_path1
        to_correct_image_path = image_path2
        sharpness_sharper = sharpness1
        sharpness_to_correct = sharpness2
    else:
        sharper_image = image2
        to_correct_image = image1
        sharper_image_path = image_path2
        to_correct_image_path = image_path1
        sharpness_sharper = sharpness2
        sharpness_to_correct = sharpness1

    print(
        f"シャープな画像: {sharper_image_path} (シャープさ: {sharpness_sharper:.2f})")
    print(
        f"より低シャープな画像: {to_correct_image_path} (シャープさ: {sharpness_to_correct:.2f})")

    return sharper_image, to_correct_image, sharper_image_path, to_correct_image_path


def train_affine_correction_agent(sharper_image, image_to_correct):
    """アフィン補正エージェントの訓練"""
    # 画像をリサイズ
    img_size = 128
    sharper_resized = cv2.resize(sharper_image, (img_size, img_size))
    to_correct_resized = cv2.resize(image_to_correct, (img_size, img_size))

    # PyTorchテンソルに変換し、[0, 1]に正規化
    # OpenCVは画像を[B,G,R]の順で読み込むため、チャネルの順序を[C,H,W]に変更
    # contiguous()を呼び出して連続したメモリレイアウトを確保
    sharper_tensor = torch.tensor(sharper_resized, dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    to_correct_tensor = torch.tensor(to_correct_resized, dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0

    # モデルとオプティマイザを初期化
    agent = AffineCorrectionAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    # 最良の結果を追跡
    best_loss = float('inf')
    best_params = None

    # 学習ループ
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()

        # エージェントからパラメータを取得
        params = agent(to_correct_tensor)

        # 変換を適用（微分可能な関数を使用）
        transformed_tensor = differentiable_affine_transform(to_correct_tensor, params)

        # 損失を計算（MSEを最小化）
        loss = calculate_similarity_loss(sharper_tensor, transformed_tensor)

        # バックプロパゲーション
        loss.backward()

        # デバッグ用にグラディエント情報を表示
        for name, param in agent.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"レイヤー: {name}, グラディエントノルム: {grad_norm}")
                # グラディエントがゼロの場合、警告を表示
                if grad_norm == 0:
                    print(f"警告: {name} のグラディエントがゼロです")
            else:
                print(f"レイヤー: {name}, グラディエントなし")

        # 重みを更新
        optimizer.step()

        print(f"エポック {epoch + 1}/{epochs}, 損失: {loss.item():.6f}")

        # 最良のパラメータを保存
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().cpu().numpy()

    print("学習完了")

    # 最良の変換を元の画像に適用（出力用にOpenCVを使用）
    best_transformed = apply_affine_transform(image_to_correct, best_params)
    return best_transformed


def apply_affine_transform(image, params):
    """OpenCVを使用してアフィン変換を適用（最終出力用）"""
    tx, ty, shear_x, shear_y, scale_x, scale_y = params
    M = np.float32([[scale_x, shear_x, tx], [shear_y, scale_y, ty]])
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image


def main():
    image_path1 = '../../../Resources/Images/19_57_44/001.jpg'
    image_path2 = '../../../Resources/Images/19_57_44/012.jpg'
    save_path = '../../../Resources/transformed_image.jpg'
    sharper_image, to_correct_image, _, _ = load_and_compare_sharpness(image_path1, image_path2)
    transformed_image = train_affine_correction_agent(sharper_image, to_correct_image)

    cv2.imwrite(save_path, transformed_image)
    print(f"変換された画像を {save_path} として保存しました")


if __name__ == "__main__":
    main()
