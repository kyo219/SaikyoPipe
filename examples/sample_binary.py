import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from lgbopt import DataSplitter, SplitStrategy, ModelConfig
from lgbopt.models import LGBOptimizer

def create_binary_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    # 2値分類の目的変数生成（バランスの取れたデータ）
    logits = (
        1.5 * data['feature1'] + 
        -1 * data['feature2'] + 
        0.5 * data['feature3']
    )
    probs = 1 / (1 + np.exp(-logits))
    data['target'] = (probs > 0.5).astype(int)
    
    return pd.DataFrame(data)

def main():
    # データ生成
    df = create_binary_data()
    
    # ランダム分割（時系列性なし）
    splitter = DataSplitter(
        strategy=SplitStrategy.STRATIFIED,  # 層化抽出
        val_size=0.2,
        test_size=0.2,
        target_column="target"  # 層化抽出のために必要
    )
    
    train_df, val_df, test_df = splitter.split(df)
    
    # モデル設定
    config = ModelConfig(
        problem_type="binary",
        target_column="target",
        feature_columns=["feature1", "feature2", "feature3", "categorical1"],
        categorical_columns=["categorical1"]
    )
    
    # モデルの学習
    optimizer = LGBOptimizer(
        config=config,
        n_trials=20,
        n_jobs=-1,
        metric="auc"
    )
    
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
    
    # 性能評価
    print(f"Test Accuracy: {accuracy_score(test_df[config.target_column], test_pred)}")
    print(f"Test AUC: {roc_auc_score(test_df[config.target_column], test_pred_proba)}")
    
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
    
    # 特徴量重要度の表示
    optimizer.plot_feature_importance(plot_type='shap')

if __name__ == "__main__":
    main() 