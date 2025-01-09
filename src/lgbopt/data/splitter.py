from enum import Enum
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

class SplitStrategy(Enum):
    """Data split strategy enumeration"""
    RANDOM = "random"
    TIME_SERIES = "time_series"
    STRATIFIED = "stratified"

class DataSplitter:
    """Class for splitting data into train, validation, and test sets"""
    
    def __init__(
        self,
        strategy: SplitStrategy,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None
    ):
        """
        Initialize DataSplitter
        
        Args:
            strategy: Splitting strategy to use
            val_size: Size of validation set
            test_size: Size of test set
            random_state: Random state for reproducibility
            time_column: Column name for time series data
            target_column: Target column name for stratified split
        """
        self.strategy = strategy
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.time_column = time_column
        self.target_column = target_column

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train, validation, and test sets
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (train, validation, test) dataframes
        """
        if self.strategy == SplitStrategy.TIME_SERIES:
            return self._time_series_split(df)
        elif self.strategy == SplitStrategy.STRATIFIED:
            return self._stratified_split(df)
        else:
            return self._random_split(df)

    def _time_series_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.time_column is None:
            raise ValueError("time_column must be specified for time series split")
        
        df = df.sort_values(self.time_column)
        n = len(df)
        train_end = int(n * (1 - self.val_size - self.test_size))
        val_end = int(n * (1 - self.test_size))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        return train, val, test

    def _stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.target_column is None:
            raise ValueError("target_column must be specified for stratified split")
        
        train, temp = train_test_split(
            df,
            test_size=self.val_size + self.test_size,
            stratify=df[self.target_column],
            random_state=self.random_state
        )
        
        val, test = train_test_split(
            temp,
            test_size=self.test_size / (self.val_size + self.test_size),
            stratify=temp[self.target_column],
            random_state=self.random_state
        )
        
        return train, val, test

    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, temp = train_test_split(
            df,
            test_size=self.val_size + self.test_size,
            random_state=self.random_state
        )
        
        val, test = train_test_split(
            temp,
            test_size=self.test_size / (self.val_size + self.test_size),
            random_state=self.random_state
        )
        
        return train, val, test