import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lgbopt import DataSplitter, SplitStrategy, ModelConfig
from lgbopt.models import LGBOptimizer
from sklearn.metrics import mean_squared_error

# 時系列データの生成
def create_sample_timeseries_data(n_samples=1000):
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_samples)]
    
    # 特徴量の生成
    data = {
        'date': dates,
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
    }
    
    # 目的変数の生成（時系列性を持たせる）
    y = []
    prev = 0
    for i in range(n_samples):
        noise = np.random.normal(0, 0.1)
        value = (
            0.7 * prev +
            0.3 * data['feature1'][i] +
            0.2 * data['feature2'][i] +
            0.1 * data['feature3'][i] +
            noise
        )
        y.append(value)
        prev = value
    
    data['target'] = y
    return pd.DataFrame(data)

def main():
    # サンプルデータの生成
    df = create_sample_timeseries_data()
    
    # データ分割の設定
    splitter = DataSplitter(
        strategy=SplitStrategy.TIME_SERIES,
        val_size=0.2,
        test_size=0.2,
        time_column='date'
    )
    
    # データの分割
    train_df, val_df, test_df = splitter.split(df)
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
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
        n_trials=20,  # Optunaの試行回数
        n_jobs=-1     # 並列処理数
    )
    
    # モデルの学習と評価
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