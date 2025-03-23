import PySimpleGUI as sg
import os
from glob import glob
import subprocess

def denoise_images(img_folder, out_folder):
    command = [
        "python",
        "src/main/movie_denoise.py",
        "--img_folder", img_folder,
        "--out_folder", out_folder
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate()
    if errors:
        raise Exception(errors.decode())
    return output.decode()

def affine_transform_images(img_folder, out_folder):
    command = [
        "python",
        "src/main/movie_affine.py",
        "--input_dir", img_folder,
        "--output_dir", out_folder
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate()
    if errors:
        raise Exception(errors.decode())
    return output.decode()

def generate_movie(img_folder, movie_path):
    command = [
        "python",
        "src/Modules/generate_movie.py",
        "--img_folder", img_folder,
        "--movie_path", movie_path
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = process.communicate()
    if errors:
        raise Exception(errors.decode())
    return output.decode()


def main():
    layout = [
        [sg.Text("画像フォルダ"), sg.Input(key="-IMG_FOLDER-"), sg.FolderBrowse(key="-BROWSE_IMG_FOLDER-")],
        [sg.Text("出力フォルダ"), sg.Input(key="-OUTPUT_FOLDER-"), sg.FolderBrowse(key="-BROWSE_OUTPUT_FOLDER-")],
        [sg.Text("動画ファイル名"), sg.Input(key="-MOVIE_NAME-")],
        [sg.Button("Noise2Noise 実行", key="-DENOISE-"),
         sg.Button("アフィン変換 実行", key="-AFFINE-"),
         sg.Button("Noise2Noise + アフィン変換 実行", key="-DENOISE_AFFINE-")],
        [sg.Output(size=(80, 10), key="-OUTPUT-")]
    ]

    window = sg.Window("画像処理GUI", layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-DENOISE-":
            img_folder = values["-IMG_FOLDER-"]
            out_folder = values["-OUTPUT_FOLDER-"] or "denoised_images"  # デフォルト出力フォルダ
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder, exist_ok=True)
            try:
                window["-OUTPUT-"].update("")  # Outputクリア
                print("Noise2Noise 処理開始...")
                output = denoise_images(img_folder, out_folder)
                print("Noise2Noise 処理完了")
                movie_name = values["-MOVIE_NAME-"] or "denoised_movie.mp4"  # デフォルト動画ファイル名
                movie_path = os.path.join(out_folder, movie_name)
                print("動画生成開始...")
                movie_output = generate_movie(out_folder, movie_path)
                print(f"動画生成完了: {movie_path}")

            except Exception as e:
                print(f"エラー発生: {e}")


        if event == "-AFFINE-":
            img_folder = values["-IMG_FOLDER-"]
            out_folder = values["-OUTPUT_FOLDER-"] or "affine_images"  # デフォルト出力フォルダ
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder, exist_ok=True)
            try:
                window["-OUTPUT-"].update("")  # Outputクリア
                print("アフィン変換 処理開始...")
                output = affine_transform_images(img_folder, out_folder)
                print("アフィン変換 処理完了")
                movie_name = values["-MOVIE_NAME-"] or "affine_movie.mp4"  # デフォルト動画ファイル名
                movie_path = os.path.join(out_folder, movie_name)
                print("動画生成開始...")
                movie_output = generate_movie(out_folder, movie_path)
                print(f"動画生成完了: {movie_path}")
            except Exception as e:
                print(f"エラー発生: {e}")

        if event == "-DENOISE_AFFINE-":
            img_folder = values["-IMG_FOLDER-"]
            denoise_out_folder = "denoised_temp_images"  # Noise2Noise一時出力フォルダ
            affine_out_folder = values["-OUTPUT_FOLDER-"] or "denoised_affine_images"  # デフォルト出力フォルダ

            os.makedirs(denoise_out_folder, exist_ok=True)
            os.makedirs(affine_out_folder, exist_ok=True)
            try:
                window["-OUTPUT-"].update("")  # Outputクリア
                print("Noise2Noise 処理開始...")
                denoise_output = denoise_images(img_folder, denoise_out_folder)
                print("Noise2Noise 処理完了")
                print("アフィン変換 処理開始...")
                affine_output = affine_transform_images(denoise_out_folder, affine_out_folder)
                print("アフィン変換 処理完了")

                movie_name = values["-MOVIE_NAME-"] or "denoised_affine_movie.mp4"  # デフォルト動画ファイル名
                movie_path = os.path.join(affine_out_folder, movie_name)
                print("動画生成開始...")
                movie_output = generate_movie(affine_out_folder, movie_path)
                print(f"動画生成完了: {movie_path}")

                # 一時フォルダ削除
                import shutil
                shutil.rmtree(denoise_out_folder)

            except Exception as e:
                print(f"エラー発生: {e}")


    window.close()

if __name__ == "__main__":
    main()
