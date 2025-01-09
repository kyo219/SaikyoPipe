from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import pandas as pd

ProblemType = Literal["regression", "binary"]

@dataclass
class ModelConfig:
    """Model configuration class"""
    problem_type: ProblemType
    target_column: str
    feature_columns: List[str]
    categorical_columns: Optional[List[str]] = None
    beta: float = 1.0
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {})