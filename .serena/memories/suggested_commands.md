# 推奨コマンド

## 基本的な開発コマンド

### パッケージ管理 (uv)
- `uv sync` - 依存関係を同期し、ロックファイルを生成（初回セットアップ、依存の更新時）
- `uv run python app.py` - 仮想環境内でPythonファイルを実行
- `uv add パッケージ名` - 新しいパッケージを追加

### 実行コマンド
- `uv run python src/main/movie_denoise.py` - 動画ノイズ除去
- `uv run python src/main/movie_affine.py` - アフィン変換適用
- `uv run python src/GUI/GUI.py` - GUI版アプリケーション

### システムコマンド (Darwin)
- `ls` - ファイル一覧表示
- `cd` - ディレクトリ移動
- `find` - ファイル検索
- `grep` - テキスト検索
- `git` - バージョン管理