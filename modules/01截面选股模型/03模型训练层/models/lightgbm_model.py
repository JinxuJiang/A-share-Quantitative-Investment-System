# -*- coding: utf-8 -*-
"""
LightGBM 模型实现

功能：
    基于LightGBM的回归模型实现，用于截面收益率预测

使用示例：
    from models.lightgbm_model import LightGBMModel
    
    model = LightGBMModel(config)
    model.fit(X_train, y_train, X_valid, y_valid)
    predictions = model.predict(X_test)
    importance = model.get_feature_importance()
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

# 尝试导入lightgbm
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装，请运行: pip install lightgbm")


class LightGBMModel(BaseModel):
    """
    LightGBM 回归模型
    
    参数：
    ------
    config : dict
        必须包含：
        - model.params: LightGBM参数字典
        - model.random_state: 随机种子
    """
    
    def __init__(self, config: dict):
        """
        初始化LightGBM模型
        """
        super().__init__(config)
        
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
        
        # 提取模型参数
        self.model_params = config['model']['params'].copy()
        self.random_state = config['model'].get('random_state', 42)
        
        # 设置随机种子
        self.model_params['random_state'] = self.random_state
        
        # 移除早停参数（在fit中单独处理）
        self.early_stopping_rounds = self.model_params.pop('early_stopping_rounds', 50)
        
        self.model = None
        self.feature_names = None
        
        logger.debug(f"LightGBMModel 初始化完成，参数: {self.model_params}")
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'LightGBMModel':
        """
        训练LightGBM模型
        
        参数：
        ------
        X_train : pd.DataFrame
            训练集特征
        y_train : pd.Series
            训练集标签
        X_valid : pd.DataFrame, optional
            验证集特征（用于早停）
        y_valid : pd.Series, optional
            验证集标签
            
        返回：
        ------
        self : LightGBMModel
        """
        logger.info(f"开始训练LightGBM模型，训练样本: {len(X_train)}")
        
        # 记录特征名
        self.feature_names = X_train.columns.tolist()
        
        # 创建Dataset
        train_data = lgb.Dataset(X_train.values, label=y_train.values)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid.values, label=y_valid.values, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
            logger.info(f"验证样本: {len(X_valid)}")
        
        # 训练模型
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.model_params.get('n_estimators', 500),
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # 关闭日志，避免刷屏
            ]
        )
        
        self.is_fitted = True
        
        # 记录训练信息
        best_iter = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(f"训练完成，最佳迭代: {best_iter}, 最佳分数: {best_score}")
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测得分
        
        参数：
        ------
        X_test : pd.DataFrame
            测试集特征
            
        返回：
        ------
        np.ndarray : 预测分数
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用fit()")
        
        predictions = self.model.predict(
            X_test.values, 
            num_iteration=self.model.best_iteration
        )
        
        return predictions
    
    def get_feature_importance(self) -> pd.Series:
        """
        获取特征重要性
        
        返回：
        ------
        pd.Series : index=feature_name, values=importance (Gain)
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用fit()")
        
        # 获取重要性（按Gain排序）
        importance = self.model.feature_importance(importance_type='gain')
        split_count = self.model.feature_importance(importance_type='split')
        
        # 构建DataFrame
        importance_df = pd.DataFrame({
            'importance': importance,
            'split': split_count
        }, index=self.feature_names)
        
        # 按重要性排序（降序）
        importance_df = importance_df.sort_values(
            by='importance', 
            ascending=False
        )
        
        return importance_df


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    
    print("=" * 80)
    print("测试 LightGBMModel")
    print("=" * 80)
    
    # 检查LightGBM是否安装
    if not HAS_LIGHTGBM:
        print("错误: LightGBM未安装，请运行: pip install lightgbm")
        sys.exit(1)
    
    # 构造测试配置
    config = {
        'model': {
            'params': {
                'objective': 'regression',
                'metric': 'mse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 8,
                'n_estimators': 100,
                'early_stopping_rounds': 10
            },
            'random_state': 42
        }
    }
    
    # 构造测试数据
    np.random.seed(42)
    n_train, n_valid, n_test = 1000, 200, 200
    n_features = 10
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X_train = pd.DataFrame(np.random.randn(n_train, n_features), columns=feature_names)
    y_train = pd.Series(np.random.randn(n_train))
    
    X_valid = pd.DataFrame(np.random.randn(n_valid, n_features), columns=feature_names)
    y_valid = pd.Series(np.random.randn(n_valid))
    
    X_test = pd.DataFrame(np.random.randn(n_test, n_features), columns=feature_names)
    
    print("\n1. 测试模型初始化:")
    model = LightGBMModel(config)
    print("   ✓ 模型初始化成功")
    
    print("\n2. 测试模型训练:")
    model.fit(X_train, y_train, X_valid, y_valid)
    print(f"   ✓ 训练完成，最佳迭代: {model.model.best_iteration}")
    
    print("\n3. 测试模型预测:")
    predictions = model.predict(X_test)
    print(f"   ✓ 预测完成，预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print("\n4. 测试特征重要性:")
    importance = model.get_feature_importance()
    print(f"   ✓ 特征重要性前3名:")
    print(importance.head(3))
    
    print("\n5. 测试模型保存/加载:")
    test_path = Path(__file__).parent / "test_model.pkl"
    model.save(test_path)
    print(f"   ✓ 模型已保存到: {test_path}")
    
    loaded_model = LightGBMModel.load(test_path)
    print(f"   ✓ 模型已加载")
    
    # 验证加载后的模型能正常预测
    loaded_predictions = loaded_model.predict(X_test)
    assert np.allclose(predictions, loaded_predictions), "加载后的模型预测不一致"
    print(f"   ✓ 加载后的模型预测一致")
    
    # 清理测试文件
    test_path.unlink()
    print(f"   ✓ 测试文件已清理")
    
    print("\n✓ LightGBMModel 测试通过！")
