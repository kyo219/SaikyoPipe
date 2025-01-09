# SaikyoPipe

# めっちゃ書いてる途中

LightGBMのハイパーパラメータ最適化を簡単に行うためのライブラリです。

## 特徴

- LightGBMのハイパーパラメータを自動最適化
- 二値分類、回帰問題に対応
- 不均衡データに対する最適化機能
- カテゴリカル変数の自動処理
- 時系列データ、層化サンプリングによるデータ分割
- 豊富な可視化機能（SHAP値、混同行列、予測vs実測プロット）

```
lightgbm-optuna-pipeline/
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
└── src/
    └── lgbopt/
        ├── __init__.py
        ├── core/
        │   ├── __init__.py
        │   └── base.py          # 基本的な型定義やインターフェース
        ├── data/
        │   ├── __init__.py
        │   └── splitter.py      # データ分割関連
        ├── evaluation/
        │   ├── __init__.py
        │   └── metrics.py       # 評価指標関連
        ├── models/
        │   ├── __init__.py
        │   └── trainer.py       # LightGBM + Optuna の学習関連
        ├── visualization/
        │   ├── __init__.py
        │   └── plots.py         # 可視化関連
        └── pipeline.py          # メインのパイプライン
```
