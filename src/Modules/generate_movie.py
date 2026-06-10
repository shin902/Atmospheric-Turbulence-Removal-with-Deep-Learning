from glob import glob
import os

import cv2
from tqdm import tqdm


def generate_movie(input_folder, mv_path):
    img_list = sorted(glob(input_folder+"/*.jpg"))
    if not img_list:
        raise FileNotFoundError(f"No jpg files found in {input_folder}")
    frames = len(img_list)

    mv_dir = os.path.dirname(mv_path)
    if mv_dir:
        os.makedirs(mv_dir, exist_ok=True)

    img = cv2.imread(img_list[0])
    h, w = img.shape[:2]
    # 作成する動画
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    #codec = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(mv_path, codec, 30000/1001, (w, h),1)

    bar = tqdm(total=frames, dynamic_ncols=True)
    for path in tqdm(img_list):
        # 画像を1枚ずつ読み込んで 動画へ出力する
        img = cv2.imread(path)
        writer.write(img)
        bar.update(1)

    bar.close()
    writer.release()
