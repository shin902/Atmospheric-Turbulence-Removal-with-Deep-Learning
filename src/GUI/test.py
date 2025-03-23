import PySimpleGUI as sg

layout = [
    [sg.Text("画像フォルダ"), sg.Input(key="-IMG_FOLDER-"), sg.FolderBrowse(key="-BROWSE_IMG_FOLDER-")],
    [sg.Text("出力フォルダ"), sg.Input(key="-OUTPUT_FOLDER-"), sg.FolderBrowse(key="-BROWSE_OUTPUT_FOLDER-")],
    [sg.Text("動画ファイル名"), sg.Input(key="-MOVIE_NAME-")],
    [sg.Button("Noise2Noise 実行", key="-DENOISE-"),
     sg.Button("アフィン変換 実行", key="-AFFINE-"),
     sg.Button("Noise2Noise + アフィン変換 実行", key="-DENOISE_AFFINE-")],
    [sg.Output(size=(80, 10), key="-OUTPUT-")]
]

window = sg.Window("画像処理GUI テスト", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()
