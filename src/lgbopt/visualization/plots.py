import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

class ModelVisualizer:
    """Class for visualizing model results"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize ModelVisualizer
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize

    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Regression Results"
    ):
        """
        Plot regression results
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()

    def plot_binary_results(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Binary Classification Results"
    ):
        """
        Plot binary classification results
        
        Args:
            y_true: True values
            y_prob: Predicted probabilities
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], 'r--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        
        # Probability distribution
        sns.histplot(
            data=pd.DataFrame({
                'Probability': y_prob,
                'True Class': y_true
            }),
            x='Probability',
            hue='True Class',
            ax=ax2
        )
        ax2.set_title('Probability Distribution')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig