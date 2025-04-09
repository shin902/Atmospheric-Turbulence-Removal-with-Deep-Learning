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
import os
import sys

import flet as ft

module_dir = os.path.abspath("../")
sys.path.append(module_dir)

from Modules.noise2noise import Noise2Noise
from Modules.ellipse import ellipse_to_circle



def main(page: ft.Page):
    page.title = "Noise Reduction App"

    input_folder_path = ft.TextField(label="Input Folder")
    output_folder_path = ft.TextField(label="Output Folder")
    temp_folder_path = ft.TextField(label="Temp Folder")
    models_folder_path = ft.TextField(label="Models Folder")
    affine_checkbox = ft.Checkbox(label="アフィン変換を実行する")
    status_text = ft.Text("")

    def denoise_images(e):
        status_text.value = "処理開始..."
        page.update()

        input_dir = input_folder_path.value
        output_dir = output_folder_path.value
        temp_dir = temp_folder_path.value
        models_dir = models_folder_path.value # モデルフォルダパスを取得

        if not input_dir or not output_dir or not models_dir:
            status_text.value = "入力フォルダー、出力フォルダー、モデルフォルダーを指定してください。"
            page.update()
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        models_path = Path(models_dir) # モデルフォルダパスをPathオブジェクトに変換
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        use_affine = affine_checkbox.value or False  # チェックボックスの状態を取得

        if (torch.backends.mps.is_available()):
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        noise_reducer = Noise2Noise(
            train_dir = "./train_data",  # train_dir and valid_dir are not actually used in denoise_image function
            valid_dir = "./valid_data",  # but are required for Noise2Noise class initialization
            model_dir = models_dir, # models_dir text field value を使用
            device=device
        )
        noise_reducer.load_model("bright_model.pth")  #TODO モデルファイル名もTextFieldから取得できるように変更するのも良いかもしれません

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
                models_folder_path,
                affine_checkbox,
                denoise_button,
                status_text,
            ]
        )
    )



if __name__ == "__main__":

    ft.app(target=main)
