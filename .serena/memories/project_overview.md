# プロジェクト概要

## プロジェクトの目的
機械学習を利用した大気のゆらぎの補正技術の開発。具体的には、木星の画像（動画）を、CNN、U-net、Noise2Noise、affine変換を使用して大気の揺らぎを補正するPythonアプリケーション。

## 技術スタック
- **メイン言語**: Python 3.9+
- **機械学習**: PyTorch, torchvision, torchaudio
- **画像処理**: OpenCV
- **GUI**: Flet
- **パッケージ管理**: uv
- **その他**: numpy, ipywidgets

## プロジェクト構造
```
src/
├── Modules/          # 自作モジュール（importで読み込むやつ）
├── main/             # メインアプリケーション
├── GUI/              # GUI関連
└── demo/             # デモコード
```

## 開発環境
- パッケージ管理: uv
- 仮想環境: uvで管理
- 設定ファイル: pyproject.toml