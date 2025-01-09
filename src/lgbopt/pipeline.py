from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import lightgbm as lgb

from .core.base import ModelConfig, ProblemType
from .data.splitter import DataSplitter, SplitStrategy
from .models.trainer import LGBOptimizer
from .evaluation.metrics import ModelEvaluator
from .visualization.plots import ModelVisualizer

@dataclass
class Pipeline:
    """Main pipeline class for LightGBM optimization and training"""
    
    problem_type: ProblemType
    target_column: str
    feature_columns: List[str]
    split_strategy: SplitStrategy
    categorical_columns: Optional[List[str]] = None
    n_trials: int = 100
    val_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    time_column: Optional[str] = None

    def __post_init__(self):
        self.config = ModelConfig(
            problem_type=self.problem_type,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            categorical_columns=self.categorical_columns
        )
        
        self.splitter = DataSplitter(
            strategy=self.split_strategy,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state,
            time_column=self.time_column,
            target_column=self.target_column
        )
        
        self.optimizer = LGBOptimizer(
            problem_type=self.problem_type,
            n_trials=self.n_trials,
            random_state=self.random_state
        )
        
        self.evaluator = ModelEvaluator()
        self.visualizer = ModelVisualizer()
        self.best_model = None

    def run(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Run the complete pipeline
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing evaluation metrics for validation and test sets
        """
        # Split data
        train_df, val_df, test_df = self.splitter.split(df)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            train_df[self.feature_columns],
            train_df[self.target_column]
        )
        valid_data = lgb.Dataset(
            val_df[self.feature_columns],
            val_df[self.target_column]
        )
        
        # Optimize hyperparameters
        self.optimizer.optimize(
            train_data,
            valid_data,
            categorical_features=self.categorical_columns
        )
        
        # Train final model
        self.best_model = self.optimizer.train_final_model(
            train_data,
            valid_data,
            categorical_features=self.categorical_columns
        )
        
        # Evaluate results
        results = {}
        for name, dataset in [("validation", val_df), ("test", test_df)]:
            y_true = dataset[self.target_column]
            y_pred = self.best_model.predict(dataset[self.feature_columns])
            
            if self.problem_type == "regression":
                results[name] = self.evaluator.evaluate_regression(y_true, y_pred)
                self.visualizer.plot_regression_results(
                    y_true,
                    y_pred,
                    title=f"Regression Results - {name.capitalize()}"
                )
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
                results[name] = self.evaluator.evaluate_binary(
                    y_true,
                    y_pred_binary,
                    y_pred
                )
                self.visualizer.plot_binary_results(
                    y_true,
                    y_pred,
                    title=f"Binary Classification Results - {name.capitalize()}"
                )
        
        return results