import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from lgbopt import DataSplitter, SplitStrategy, ModelConfig
from lgbopt.models import LGBOptimizer

def create_unbalanced_binary_data(n_samples=1000, pos_ratio=0.05):
    np.random.seed(42)
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    # 不均衡な2値分類の目的変数生成
    logits = (
        2 * data['feature1'] + 
        -1.5 * data['feature2'] + 
        0.5 * data['feature3']
    )
    probs = 1 / (1 + np.exp(-logits))
    # 正例の割合を調整
    threshold = np.percentile(probs, (1 - pos_ratio) * 100)
    data['target'] = (probs > threshold).astype(int)
    
    return pd.DataFrame(data)

def main():
    # 不均衡データの生成（正例5%）
    df = create_unbalanced_binary_data(n_samples=10000, pos_ratio=0.05)
    
    print(f"Positive ratio: {df['target'].mean():.3f}")
    
    # 層化抽出でデータ分割
    splitter = DataSplitter(
        strategy=SplitStrategy.STRATIFIED,
        val_size=0.2,
        test_size=0.2,
        target_column="target"
    )
    
    train_df, val_df, test_df = splitter.split(df)
    
    # モデル設定
    config = ModelConfig(
        problem_type="binary",
        target_column="target",
        feature_columns=["feature1", "feature2", "feature3", "categorical1"],
        categorical_columns=["categorical1"],
        beta=2.0,  # F2スコアを使用する場合
        lgb_params={
            "objective": "binary",
            "metric": "None",  # カスタム評価指標を使用するため
            "is_unbalance": True,
        }
    )
    
    # モデルの学習
    optimizer = LGBOptimizer(
        config=config,
        n_trials=20,
        n_jobs=-1,
    )
    
    # scale_pos_weightパラメータを最適化対象に追加
    optimizer.param_distributions.update({
        "scale_pos_weight": {"type": "uniform", "low": 1.0, "high": 5.0}
    })
    
    # 学習の実行
    optimizer.fit(
        train_df=train_df,
        val_df=val_df,
        early_stopping_rounds=50
    )
    
    # 予測
    train_pred_proba = optimizer.predict_proba(train_df)
    train_pred = optimizer.predict(train_df)
    val_pred_proba = optimizer.predict_proba(val_df)
    val_pred = optimizer.predict(val_df)
    test_pred_proba = optimizer.predict_proba(test_df)
    test_pred = optimizer.predict(test_df)
    
    # 評価指標の表示
    print(f"Test Average Precision: {average_precision_score(test_df[config.target_column], test_pred_proba)}")
    
    # Confusion Matrixの表示（訓練、検証、テストデータ）
    metrics = optimizer.plot_confusion_matrix(
        train_df[config.target_column],
        train_pred,
        val_df[config.target_column],
        val_pred,
        test_df[config.target_column],
        test_pred,
        labels=['Negative', 'Positive']
    )
    
    # 特徴量重要度のプロット
    optimizer.plot_feature_importance(plot_type='shap')

if __name__ == "__main__":
    main() 