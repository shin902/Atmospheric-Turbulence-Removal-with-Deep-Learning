import torch
import cv2
from tqdm import tqdm
from glob import glob
import os
import sys


module_dir = os.path.abspath("../")
sys.path.append(module_dir)

from Modules.noise2noise import Noise2Noise
from Modules.generate_movie import generate_movie



def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{ base_path }_{ str(n).zfill(digit) }.{ ext }', frame)
            n += 1
        else:
            return


if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    trainer = Noise2Noise(
        train_dir = "../../Resources/AI/train_data",
        valid_dir = "../../Resources/AI/valid_data",
        model_dir = "../../Resources/AI/model_dir",
        device=device
    )
    trainer.load_model('test_model.pth')


    # 各変数の初期設定
    img_folder = "../../Resources/Input and Output/affine3"
    out_folder = "../../Resources/Input and Output/denoise_2"
    movie_path = "../../Resources/Input and Output/affine_and_denoise_2.mp4"
    os.makedirs(out_folder, exist_ok=True)

    # 動画から連番画像を生成する
    # save_all_frames("input.mp4", img_folder)

    img_list = sorted(glob(img_folder+"/*.jpg"))
    frames = len(img_list)

    i = 1
    # ノイズ除去画像を生成
    for img_name in tqdm(img_list):
        trainer.denoise_image(img_name, os.path.join(out_folder, str(i)+".jpg"))
        i += 1




    # 動画を生成
    generate_movie(out_folder, movie_path)
