# -*- coding: utf-8 -*-
"""
模型基类模块

功能：
    定义所有模型的通用接口，便于后续扩展（XGBoost、Transformer等）

使用示例：
    from models.base_model import BaseModel
    
    class MyModel(BaseModel):
        def fit(self, X_train, y_train, X_valid=None, y_valid=None):
            # 实现训练逻辑
            pass
        
        def predict(self, X_test):
            # 实现预测逻辑
            pass
        
        def get_feature_importance(self):
            # 实现特征重要性
            pass
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class BaseModel(ABC):
    """
    模型基类
    
    所有具体模型（LightGBM、XGBoost等）必须继承此类并实现抽象方法
    """
    
    def __init__(self, config: dict):
        """
        初始化模型
        
        参数：
        ------
        config : dict
            模型配置字典
        """
        self.config = config
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'BaseModel':
        """
        训练模型
        
        参数：
        ------
        X_train : pd.DataFrame
            训练集特征，shape: (n_samples, n_features)
        y_train : pd.Series
            训练集标签，shape: (n_samples,)
        X_valid : pd.DataFrame, optional
            验证集特征，用于早停
        y_valid : pd.Series, optional
            验证集标签
            
        返回：
        ------
        self : BaseModel
            返回自身，支持链式调用
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测得分
        
        参数：
        ------
        X_test : pd.DataFrame
            测试集特征，shape: (n_samples, n_features)
            
        返回：
        ------
        np.ndarray : 预测分数，shape: (n_samples,)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """
        获取特征重要性
        
        返回：
        ------
        pd.Series : index=feature_name, values=importance_score
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型到文件
        
        参数：
        ------
        path : str or Path
            保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        从文件加载模型
        
        参数：
        ------
        path : str or Path
            模型文件路径
            
        返回：
        ------
        BaseModel : 加载的模型实例
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_params(self) -> dict:
        """
        获取模型参数
        
        返回：
        ------
        dict : 模型配置参数
        """
        return self.config.copy()
