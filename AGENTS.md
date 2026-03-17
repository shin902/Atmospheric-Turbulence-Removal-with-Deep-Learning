# Repository Guidelines

## プロジェクト構成とモジュール配置
- `src/Modules/` はUNetによるノイズ低減や楕円補正などの再利用モジュール群と、`noise_evaluation/` 以下の評価スクリプトを格納します。新しいアルゴリズムもこの中で独立させてください。
- `src/main/` にはパイプライン実装（`movie_denoise.py`, `movie_affine.py`）があり、画像ディレクトリを入力として連番出力や動画生成を行います。
- `src/GUI/` はFlet製GUI。学習済みモデルは `models/`、教師データは `train_data` / `valid_data`、生成物は `output` や `temp` に置く運用です。
- `src/demo/` は軽量なTorchデモや検証スクリプト置き場。試験的な処理はここで共有してから本体へ移設します。
- 参考資料は `Resources/`、ライセンスは `licenses/`。大容量データはリポジトリ外に保管し、README等で場所を案内します。

## ビルド・テスト・開発コマンド
- `uv sync` — Python 3.9環境を同期し、`pyproject.toml` に基づく依存を確定します。
- `uv run python src/main/movie_denoise.py --input ./data/jupiter --output ./artifacts/denoised` — UNetパイプラインでフォルダ内画像を一括ノイズ除去。
- `uv run python src/main/movie_affine.py --input ./data/jupiter --output ./artifacts/aligned` — 動画化前にアフィン補正を適用。
- `uv run python src/GUI/GUI.py` — デスクトップGUIを起動。`src/GUI/models/` に推論用重みが必要です。
- `uv run python -m src.Modules.noise2noise` — モジュール単体のデモを実行し、リファクタ後の動作確認に利用します。

## コーディング規約と命名
- 4スペースインデント、Python 3.9準拠でPEP 8を概ね維持。パス操作は `pathlib.Path` を優先します。
- 変数・関数は `snake_case`、クラスは `PascalCase`、設定値はモジュール冒頭の ALL_CAPS 定数にまとめます。
- 新しいエントリーポイントは既存ファイル名に倣い、モジュールレベルのdocstringで役割を明記します。
- 大規模変更前には `uv run python -m black src` などフォーマッタ／リンタを実行してください。

## テスト指針
- 現状公式テストスイートは未整備。テストを追加する場合は `tests/` 以下に配置し、`uv run pytest` で実行します。
- 回帰確認には `src/Modules/noise_evaluation/` 内の指標スクリプト（例：`snr.py`, `psmr.py`）を利用し、`uv run python ... --before --after` 形式で差分を計測します。
- PRでは期待されるSNRやPSMRの変化量、再現手順を記載し、レビュー時の検証コストを下げてください。

## コミットとプルリクエスト運用
- Git履歴に倣い、1コミット1論点で簡潔な現在形サマリ（日本語可）を記述し、必要に応じて変更領域の接頭辞（`GUI`, `Modules` 等）を付けます。
- 生成メディアや学習済みモデルはコミット対象外。コード・設定・ドキュメントの更新をまとめて送ってください。
- PRには概要、実行コマンド、使用データパス、ビフォーアフターの可視化やメトリクスを添付し、関連Issueのリンクと想定レビュワーを明記します。
- レビュー依頼前に主要パイプラインと `uv sync` の成功を確認し、CUDA/CPUなど動作条件の差異があれば注意書きを入れます。

## モデルとデータの扱い
- `.pt` など大容量モデルや生の観測データはコミットせず、`src/GUI/models/` や外部ストレージに配置しパスをドキュメント化します。
- 観測メタデータには機密が含まれる可能性があるため、共有前にファイル名・EXIF情報を必ずマスクしてください。
- 新しい前処理やデータ前提が増えた場合は README または該当モジュールのdocstringに追記し、次の実装者が迷わないようにします。
