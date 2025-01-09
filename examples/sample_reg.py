import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lgbopt import DataSplitter, SplitStrategy, ModelConfig
from lgbopt.models import LGBOptimizer
from sklearn.metrics import mean_squared_error

def create_regression_data(n_samples=1000):
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_samples)]
    
    data = {
        'date': dates,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], n_samples),
    }
    
    # 非線形性を持つ目的変数の生成
    y = (
        3 * np.sin(data['feature1']) + 
        2 * data['feature2']**2 + 
        1.5 * data['feature3'] + 
        np.random.normal(0, 0.1, n_samples)
    )
    data['target'] = y
    return pd.DataFrame(data)

def main():
    # データ生成
    df = create_regression_data()
    
    # 時系列を考慮したデータ分割
    splitter = DataSplitter(
        strategy=SplitStrategy.TIME_SERIES,
        val_size=0.2,
        test_size=0.2,
        time_column='date'
    )
    
    train_df, val_df, test_df = splitter.split(df)
    
    # モデル設定
    config = ModelConfig(
        problem_type="regression",
        target_column="target",
        feature_columns=["feature1", "feature2", "feature3"],
        categorical_columns=[]
    )
    
    # モデルの学習
    optimizer = LGBOptimizer(
        config=config,
        n_trials=20,
        n_jobs=-1,
        metric="rmse"  # 回帰の場合はrmseを使用
    )
    
    # 学習の実行
    optimizer.fit(
        train_df=train_df,
        val_df=val_df,
        early_stopping_rounds=50
    )
    
    # 予測と評価
    train_pred = optimizer.predict(train_df)
    val_pred = optimizer.predict(val_df)
    test_pred = optimizer.predict(test_df)
    
    # 結果の表示
    print("\nBest parameters:", optimizer.best_params)
    print(f"Test RMSE: {np.sqrt(mean_squared_error(test_df[config.target_column], test_pred))}")
    
    # 特徴量重要度のプロット
    optimizer.plot_feature_importance(plot_type='shap')
    
    # 予測vs実測値のプロット
    optimizer.plot_prediction_vs_actual(train_df, val_df, test_df)

if __name__ == "__main__":
    main() 