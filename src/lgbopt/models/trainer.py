from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import shap
from sklearn.metrics import fbeta_score

from ..core.base import ModelConfig

class LGBOptimizer:
    """LightGBM trainer with Optuna optimization"""
    
    def __init__(
        self,
        config: ModelConfig,
        n_trials: int = 100,
        n_jobs: int = -1,
        metric: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize LGBOptimizer
        
        Args:
            config: ModelConfig object containing problem settings
            n_trials: Number of Optuna trials
            n_jobs: Number of parallel jobs (-1 for all cores)
            metric: Metric to optimize ('rmse' for regression, 'auc' for binary)
            random_state: Random state for reproducibility
        """
        self.config = config
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.metric = metric
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.study = None
        self.label_encoders = {}
        
        # 問題タイプに応じてパラメータ分布を設定
        if self.config.problem_type == "regression":
            self.param_distributions = {
                "num_leaves": {"type": "int", "low": 31, "high": 255},
                "max_depth": {"type": "int", "low": 5, "high": 15},
                "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
                "min_child_samples": {"type": "int", "low": 5, "high": 100},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "min_split_gain": {"type": "float", "low": 1e-8, "high": 0.1, "log": True},
                "bagging_freq": {"type": "int", "low": 1, "high": 7},
                "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0}
            }
        else:  # binary/multiclass
            self.param_distributions = {
                "num_leaves": {"type": "int", "low": 10, "high": 100},
                "max_depth": {"type": "int", "low": 3, "high": 8},
                "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
                "min_child_samples": {"type": "int", "low": 10, "high": 100},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            }
        
        # 不均衡データの場合はscale_pos_weightも最適化対象に追加
        if self.config.lgb_params.get("is_unbalance", False):
            self.param_distributions["scale_pos_weight"] = {
                "type": "float",
                "low": 1.0,
                "high": 5.0
            }

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization"""
        params = {**self.config.lgb_params} if self.config.lgb_params else {}
        
        # 問題タイプに応じたデフォルトパラメータを設定
        if self.config.problem_type == "regression":
            params.update({
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "boosting_type": "gbdt",
            })
        elif self.config.problem_type == "binary":
            params.update({
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
            })
        
        for param_name, param_config in self.param_distributions.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config["low"], 
                    param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
        
        return params

    def _get_categorical_feature_indices(self, feature_names: List[str]) -> List[int]:
        """Get indices of categorical features"""
        if not self.config.categorical_columns:
            return []
        return [feature_names.index(col) for col in self.config.categorical_columns]

    def _preprocess_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess categorical columns using label encoding
        """
        data = df.copy()
        
        # カテゴリカル変数がない場合はそのまま返す
        if not self.config.categorical_columns:
            return data
        
        for col in self.config.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(data[col])
            data[col] = self.label_encoders[col].transform(data[col])
        
        return data

    def _create_dataset(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        reference: Optional[lgb.Dataset] = None
    ) -> lgb.Dataset:
        """Create LightGBM dataset with proper handling of categorical features"""
        data = self._preprocess_categorical(df)
        X = data[self.config.feature_columns]
        y = data[self.config.target_column] if is_train or self.config.target_column in df.columns else None
        
        # カテゴリカル変数のインデックスを取得
        if self.config.categorical_columns:
            categorical_indices = [
                self.config.feature_columns.index(col) 
                for col in self.config.categorical_columns 
                if col in self.config.feature_columns
            ]
        else:
            categorical_indices = None
        
        return lgb.Dataset(
            data=X,
            label=y,
            reference=reference,
            categorical_feature=categorical_indices,  # インデックスを使用
            free_raw_data=False
        )

    def _create_fbeta_evaluator(self, beta: float):
        """Create F-beta score evaluator function for LightGBM"""
        def fbeta_evaluator(preds, data):
            labels = data.get_label()
            preds = (preds > 0.5).astype(int)
            fbeta = fbeta_score(labels, preds, beta=beta)
            return f'f{beta}', fbeta, True
        return fbeta_evaluator

    def _create_evaluator(self):
        """Create appropriate evaluator function based on problem type"""
        if self.config.problem_type == "binary":
            return self._create_fbeta_evaluator(self.config.beta)
        elif self.config.problem_type == "regression":
            def rmse_evaluator(preds, data):
                labels = data.get_label()
                rmse = np.sqrt(mean_squared_error(labels, preds))
                return 'rmse', rmse, False
            return rmse_evaluator
        else:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        early_stopping_rounds: int = 50
    ) -> 'LGBOptimizer':
        """Train the model with optimized hyperparameters"""
        train_data = self._create_dataset(train_df, is_train=True)
        val_data = self._create_dataset(val_df, is_train=True, reference=train_data)
        
        self._last_train_data = train_df[self.config.feature_columns]
        
        def objective(trial):
            params = self._suggest_params(trial)
            params['verbose'] = -1
            
            try:
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ],
                    feval=self._create_evaluator()
                )
                
                pred = model.predict(val_data.data)
                if self.config.problem_type == "binary":
                    pred_binary = (pred > 0.5).astype(int)
                    return fbeta_score(val_data.label, pred_binary, beta=self.config.beta)
                else:
                    return -np.sqrt(mean_squared_error(val_data.label, pred))  # 回帰の場合はRMSEを最小化
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('-inf')

        # 最適化の方向を問題タイプに応じて設定
        direction = "maximize" if self.config.problem_type == "binary" else "minimize"
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        
        if self.study.best_params:
            self.best_params = self.study.best_params
            final_params = {**self.best_params}
            final_params['verbose'] = -1
            
            self.best_model = lgb.train(
                final_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)
                ],
                feval=self._create_evaluator()
            )
        
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions for new data"""
        if self.best_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        data = self._preprocess_categorical(df)
        X = data[self.config.feature_columns].astype('float32')
        preds = self.best_model.predict(X)
        
        # バイナリ分類の場合は確率値を予測クラスに変換
        if self.config.problem_type == 'binary':
            return (preds > 0.5).astype(int)
        return preds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Make probability predictions for classification"""
        if self.config.problem_type not in ['binary', 'multiclass']:
            raise ValueError("predict_proba() is only available for classification")
        if self.best_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        data = self._preprocess_categorical(df)
        X = data[self.config.feature_columns].astype('float32')
        return self.best_model.predict(X)

    def plot_feature_importance(self, plot_type='gain', max_display=20):
        """
        Plot feature importance
        """
        if plot_type == 'shap':
            # カテゴリカル変数を数値に変換
            train_data = self._last_train_data.copy()
            if self.config.categorical_columns:
                for col in self.config.categorical_columns:
                    if col in train_data.columns and col in self.label_encoders:
                        train_data[col] = self.label_encoders[col].transform(train_data[col])
            
            explainer = shap.TreeExplainer(self.best_model)
            shap_values = explainer.shap_values(train_data)
            
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):  # 二値分類の場合
                shap.summary_plot(
                    shap_values[1],  # 正クラスのSHAP値
                    train_data,
                    feature_names=self.config.feature_columns,
                    max_display=max_display,
                    show=False
                )
            else:  # 回帰の場合
                shap.summary_plot(
                    shap_values,
                    train_data,
                    feature_names=self.config.feature_columns,
                    max_display=max_display,
                    show=False
                )
            plt.tight_layout()
            plt.show()
        else:
            # 従来の特徴量重要度の表示
            importance_type = 'gain'
            importances = pd.DataFrame({
                'feature': self.config.feature_columns,
                'importance': self.best_model.feature_importance(importance_type=importance_type)
            })
            importances = importances.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=importances.head(max_display),
                x='importance',
                y='feature'
            )
            plt.title('Feature Importance (gain)')
            plt.tight_layout()
            plt.show()

    def plot_training_history(self, figsize=(10, 6)):
        """Plot training history"""
        if self.best_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        plt.figure(figsize=figsize)
        self.best_model.plot_metric()
        plt.title('Training History')
        plt.show()

    def plot_roc_curve(self, y_true, y_pred, figsize=(8, 8)):
        """Plot ROC curve for binary classification"""
        if self.config.problem_type != 'binary':
            raise ValueError("ROC curve is only available for binary classification")
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(
        self,
        train_true,
        train_pred,
        val_true,
        val_pred,
        test_true,
        test_pred,
        labels=None,
        figsize=(20, 6)
    ):
        """
        Plot confusion matrices for training, validation, and test data
        
        Parameters:
        -----------
        train_true : array-like
            True labels for training data
        train_pred : array-like
            Predicted labels for training data
        val_true : array-like
            True labels for validation data
        val_pred : array-like
            Predicted labels for validation data
        test_true : array-like
            True labels for test data
        test_pred : array-like
            Predicted labels for test data
        labels : list, optional
            List of label names. If None, will use numerical labels
        figsize : tuple, optional
            Figure size for the plot (width, height)
        """
        if self.config.problem_type not in ['binary', 'multiclass']:
            raise ValueError("Confusion matrix is only available for classification problems")

        # For binary classification, convert probabilities to class labels if needed
        if self.config.problem_type == 'binary':
            for pred in [train_pred, val_pred, test_pred]:
                if pred.ndim > 1 or np.all(np.logical_and(pred >= 0, pred <= 1)):
                    pred = (pred > 0.5).astype(int)

        # Set up labels
        if labels is None:
            if self.config.problem_type == 'binary':
                labels = ['Class 0', 'Class 1']
            else:
                n_classes = len(np.unique(np.concatenate([train_true, val_true, test_true])))
                labels = [f'Class {i}' for i in range(n_classes)]

        # Create confusion matrices
        train_cm = confusion_matrix(train_true, train_pred)
        val_cm = confusion_matrix(val_true, val_pred)
        test_cm = confusion_matrix(test_true, test_pred)

        # Calculate accuracy scores
        train_acc = accuracy_score(train_true, train_pred)
        val_acc = accuracy_score(val_true, val_pred)
        test_acc = accuracy_score(test_true, test_pred)

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Training confusion matrix
        sns.heatmap(
            train_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax1
        )
        ax1.set_title(f'Training Confusion Matrix\nAccuracy: {train_acc:.3f}')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Validation confusion matrix
        sns.heatmap(
            val_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax2
        )
        ax2.set_title(f'Validation Confusion Matrix\nAccuracy: {val_acc:.3f}')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

        # Test confusion matrix
        sns.heatmap(
            test_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax3
        )
        ax3.set_title(f'Test Confusion Matrix\nAccuracy: {test_acc:.3f}')
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.show()

        # Return metrics
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_confusion_matrix': train_cm,
            'val_confusion_matrix': val_cm,
            'test_confusion_matrix': test_cm
        }

    def plot_prediction_vs_actual(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Plot prediction vs actual values for regression problems
        """
        if self.config.problem_type != "regression":
            raise ValueError("This plot is only available for regression problems")
        
        # 全データの最小値と最大値を計算
        all_min = min(
            train_df[self.config.target_column].min(),
            val_df[self.config.target_column].min(),
            test_df[self.config.target_column].min()
        )
        all_max = max(
            train_df[self.config.target_column].max(),
            val_df[self.config.target_column].max(),
            test_df[self.config.target_column].max()
        )
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)  # 2行3列に変更
        
        # Combined plot (larger, top)
        ax1 = fig.add_subplot(gs[0, :])  # 上段全体を使用
        
        # Training data
        train_pred = self.predict(train_df)
        ax1.scatter(
            train_df[self.config.target_column],
            train_pred,
            alpha=0.5,
            label='Train',
            color='blue'
        )
        
        # Validation data
        val_pred = self.predict(val_df)
        ax1.scatter(
            val_df[self.config.target_column],
            val_pred,
            alpha=0.5,
            label='Validation',
            color='green'
        )
        
        # Test data
        test_pred = self.predict(test_df)
        ax1.scatter(
            test_df[self.config.target_column],
            test_pred,
            alpha=0.5,
            label='Test',
            color='red'
        )
        
        ax1.plot([all_min, all_max], [all_min, all_max], 'k--', label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Combined Prediction vs Actual')
        ax1.legend()
        ax1.set_xlim(all_min, all_max)
        ax1.set_ylim(all_min, all_max)
        
        # Individual plots (bottom row)
        # Train
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(
            train_df[self.config.target_column],
            train_pred,
            alpha=0.5,
            color='blue'
        )
        ax2.plot([all_min, all_max], [all_min, all_max], 'k--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Training Data')
        ax2.set_xlim(all_min, all_max)
        ax2.set_ylim(all_min, all_max)
        
        # Validation
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(
            val_df[self.config.target_column],
            val_pred,
            alpha=0.5,
            color='green'
        )
        ax3.plot([all_min, all_max], [all_min, all_max], 'k--')
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title('Validation Data')
        ax3.set_xlim(all_min, all_max)
        ax3.set_ylim(all_min, all_max)
        
        # Test
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(
            test_df[self.config.target_column],
            test_pred,
            alpha=0.5,
            color='red'
        )
        ax4.plot([all_min, all_max], [all_min, all_max], 'k--')
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Test Data')
        ax4.set_xlim(all_min, all_max)
        ax4.set_ylim(all_min, all_max)
        
        plt.tight_layout()
        plt.show()

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted
        
        Returns
        -------
        bool
            True if the model has been fitted, False otherwise
        """
        return self.best_model is not None