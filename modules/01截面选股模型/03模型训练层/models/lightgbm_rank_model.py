# -*- coding: utf-8 -*-
"""
LightGBM LambdaRank 排序学习模型

功能：
    基于 LightGBM LambdaRank 的排序模型实现，直接优化股票排名
    解决传统回归模型预测值扎堆、导致 Alphalens 分组失败的问题

使用示例：
    from models.lightgbm_rank_model import LightGBMRankModel
    
    config = {
        'model': {
            'params': {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [10, 30, 50],
                'learning_rate': 0.015,
                'num_leaves': 31,
                'max_depth': 5,
                'min_data_in_leaf': 150,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 1.0,
                'verbose': -1,
                'n_estimators': 1500,
                'early_stopping_rounds': 50
            },
            'random_state': 42
        }
    }
    
    model = LightGBMRankModel(config)
    model.fit(X_train, y_train, X_valid, y_valid)
    rankings = model.predict(X_test)  # 输出的是相关性分数，用于排序
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
from pathlib import Path

try:
    from models.base_model import BaseModel
except ImportError:
    # 直接运行时，手动导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from base_model import BaseModel

logger = logging.getLogger(__name__)

# 尝试导入 lightgbm
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM 未安装，请运行: pip install lightgbm")


class LightGBMRankModel(BaseModel):
    """
    LightGBM LambdaRank 排序模型
    
    特点：
    1. 使用 LambdaRank 目标函数，直接优化文档排序（股票排名）
    2. 输出相关性分数，用于截面排序选股
    3. 避免传统回归的预测值扎堆问题
    
    参数：
    ------
    config : dict
        必须包含：
        - model.params: LightGBM 参数字典，注意 objective 应为 'lambdarank'
        - model.random_state: 随机种子
    """
    
    def __init__(self, config: dict):
        """
        初始化 LambdaRank 模型
        """
        super().__init__(config)
        
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        
        # 提取模型参数
        self.model_params = config['model']['params'].copy()
        self.random_state = config['model'].get('random_state', 42)
        
        # 设置随机种子
        self.model_params['random_state'] = self.random_state
        
        # 强制设置 objective 为 lambdarank（如果用户配错了）
        if self.model_params.get('objective') != 'lambdarank':
            logger.warning(f"objective 被强制设置为 'lambdarank'，原值: {self.model_params.get('objective')}")
            self.model_params['objective'] = 'lambdarank'
        
        # 移除早停参数（在 fit 中单独处理）
        self.early_stopping_rounds = self.model_params.pop('early_stopping_rounds', 50)
        
        self.model = None
        self.feature_names = None
        
        logger.debug(f"LightGBMRankModel 初始化完成，参数: {self.model_params}")
    
    def _prepare_group_data(self, X: pd.DataFrame) -> tuple:
        """
        准备 LambdaRank 所需的 group 数据
        
        LambdaRank 需要知道每个 query（这里是每个交易日）有多少个文档（股票）
        
        参数：
        ------
        X : pd.DataFrame with MultiIndex (date, stock_code)
            特征矩阵
            
        返回：
        ------
        X_sorted : pd.DataFrame
            按日期排序后的特征矩阵
        groups : np.ndarray
            每个日期的股票数量，如 [3000, 3000, 2800, ...]
        """
        # 获取日期索引（level 0）
        dates = X.index.get_level_values(0)
        
        # 计算每个日期的股票数量
        # value_counts 返回的是无序的，需要按日期排序
        date_counts = pd.Series(dates).value_counts().sort_index()
        groups = date_counts.values
        
        # 按日期排序 X，确保 groups 和数据对齐
        X_sorted = X.sort_index(level=0)
        
        logger.debug(f"Group 统计: 共 {len(groups)} 个交易日，平均 {groups.mean():.0f} 只股票/天")
        
        return X_sorted, groups
    
    def _discretize_labels(self, y: pd.Series, n_bins: int = 10) -> pd.Series:
        """
        将连续收益率转换为整数等级（分桶）
        
        LambdaRank 要求 label 为整数类型，这里在每个截面（日期）内
        按收益率分位数分桶，转换为 0~(n_bins-1) 的整数等级
        
        参数：
        ------
        y : pd.Series with MultiIndex (date, stock_code)
            连续收益率标签
        n_bins : int
            分桶数量（默认10组）
            
        返回：
        ------
        pd.Series : 整数等级标签 (0 ~ n_bins-1)
        """
        # 按日期分组，在每个截面内独立分桶
        def rank_within_group(group):
            # 使用 qcut 按分位数分桶，labels=False 返回 0~(n_bins-1)
            # duplicates='drop' 处理重复边界的情况
            try:
                return pd.qcut(group, q=n_bins, labels=False, duplicates='drop')
            except ValueError:
                # 如果某组股票太少无法分桶，直接按排名映射到 0~(n_bins-1)
                ranks = group.rank(method='first')
                max_rank = ranks.max()
                if max_rank == 0:
                    return pd.Series(0, index=group.index)
                return ((ranks - 1) / max_rank * n_bins).clip(0, n_bins - 1).astype(int)
        
        # 按日期（level 0）分组应用
        y_discrete = y.groupby(level=0).transform(rank_within_group)
        return y_discrete
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'LightGBMRankModel':
        """
        训练 LambdaRank 模型
        
        参数：
        ------
        X_train : pd.DataFrame with MultiIndex (date, stock_code)
            训练集特征
        y_train : pd.Series with MultiIndex (date, stock_code)
            训练集标签（真实收益，会被转换为整数等级）
        X_valid : pd.DataFrame, optional
            验证集特征（用于早停）
        y_valid : pd.Series, optional
            验证集标签
            
        返回：
        ------
        self : LightGBMRankModel
        """
        logger.info(f"开始训练 LambdaRank 模型，训练样本: {len(X_train)}")
        
        # 记录特征名
        self.feature_names = X_train.columns.tolist()
        
        # 将连续收益率转换为整数等级（每个截面内分10组）
        logger.info("将收益率标签转换为整数等级（10组）...")
        y_train = self._discretize_labels(y_train, n_bins=10)
        logger.info(f"标签分布: {y_train.value_counts().sort_index().to_dict()}")
        
        # 准备训练数据（排序 + 计算 groups）
        X_train_sorted, train_groups = self._prepare_group_data(X_train)
        y_train_sorted = y_train.sort_index(level=0)
        
        # 创建 Dataset
        train_data = lgb.Dataset(
            X_train_sorted.values,
            label=y_train_sorted.values,
            group=train_groups,
            feature_name=self.feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            logger.info(f"验证样本: {len(X_valid)}")
            
            # 验证集同样需要转换标签
            y_valid = self._discretize_labels(y_valid, n_bins=10)
            
            X_valid_sorted, valid_groups = self._prepare_group_data(X_valid)
            y_valid_sorted = y_valid.sort_index(level=0)
            
            valid_data = lgb.Dataset(
                X_valid_sorted.values,
                label=y_valid_sorted.values,
                group=valid_groups,
                reference=train_data,
                feature_name=self.feature_names
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # 训练模型
        logger.info("训练 LambdaRank 中...")
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.model_params.get('n_estimators', 1500),
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)  # 关闭日志，避免刷屏
            ]
        )
        
        self.is_fitted = True
        
        # 记录训练信息
        best_iter = self.model.best_iteration
        best_score = self.model.best_score
        logger.info(f"训练完成，最佳迭代: {best_iter}")
        if best_score:
            for dataset, metrics in best_score.items():
                for metric, value in metrics.items():
                    logger.info(f"  {dataset} {metric}: {value:.6f}")
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测相关性分数（用于排序）
        
        注意：LambdaRank 输出的是相关性分数，不是实际收益！
        分数越高表示模型认为该股票未来表现越好，仅用于截面排序。
        
        参数：
        ------
        X_test : pd.DataFrame with MultiIndex (date, stock_code)
            测试集特征
            
        返回：
        ------
        np.ndarray : 相关性分数，shape: (n_samples,)
                     值越大表示排名越靠前
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # LambdaRank 预测不需要 group 信息（推理时是逐样本打分）
        predictions = self.model.predict(
            X_test.values,
            num_iteration=self.model.best_iteration
        )
        
        return predictions
    
    def predict_rank(self, X_test: pd.DataFrame) -> pd.Series:
        """
        直接输出排名（1 = 最好）
        
        参数：
        ------
        X_test : pd.DataFrame with MultiIndex (date, stock_code)
            测试集特征
            
        返回：
        ------
        pd.Series : 排名，index 与 X_test 相同，值越小表示排名越好
        """
        scores = self.predict(X_test)
        
        # 构造 Series
        scores_series = pd.Series(scores, index=X_test.index, name='rank_score')
        
        # 按日期分组计算排名（每天单独排名）
        ranks = scores_series.groupby(level=0).rank(ascending=False, method='min')
        
        return ranks
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        返回：
        ------
        pd.DataFrame : index=feature_name, columns=['importance', 'split']
                      importance 按 Gain 排序
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # 获取重要性（按 Gain 排序）
        importance = self.model.feature_importance(importance_type='gain')
        split_count = self.model.feature_importance(importance_type='split')
        
        # 构建 DataFrame
        importance_df = pd.DataFrame({
            'importance': importance,
            'split': split_count
        }, index=self.feature_names)
        
        # 按 importance 降序排列
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_ndcg_score(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> float:
        """
        计算 NDCG@k 分数（用于评估排序质量）
        
        参数：
        ------
        X : pd.DataFrame
            特征
        y : pd.Series
            真实收益（相关性标签）
        k : int
            计算 Top-K 的 NDCG
            
        返回：
        ------
        float : 平均 NDCG@k 分数
        """
        from sklearn.metrics import ndcg_score
        
        scores = self.predict(X)
        
        # 构造 DataFrame
        results = pd.DataFrame({
            'score': scores,
            'label': y.values
        }, index=X.index)
        
        # 按日期计算 NDCG
        ndcg_scores = []
        for date, group in results.groupby(level=0):
            if len(group) < k:
                continue
            
            # NDCG 需要 2D 数组
            true_relevance = group['label'].values.reshape(1, -1)
            predicted_scores = group['score'].values.reshape(1, -1)
            
            ndcg = ndcg_score(true_relevance, predicted_scores, k=k)
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0


if __name__ == "__main__":
    # 测试代码
    import sys
    
    print("=" * 80)
    print("测试 LightGBMRankModel (LambdaRank)")
    print("=" * 80)
    
    if not HAS_LIGHTGBM:
        print("错误: LightGBM 未安装")
        sys.exit(1)
    
    # 使用用户提供的 A 股优化配置
    config = {
        'model': {
            'params': {
                # 核心任务设置
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [10, 30, 50],
                
                # 树的生长控制 (防过拟合)
                'boosting_type': 'gbdt',
                'max_depth': 5,
                'num_leaves': 31,
                'min_data_in_leaf': 150,
                
                # 学习节奏
                'learning_rate': 0.015,
                'n_estimators': 200,  # 测试时用少一点
                
                # 随机抽样
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                
                # 正则化
                'lambda_l1': 0.5,
                'lambda_l2': 1.0,
                
                # 工程设置
                'verbose': -1,
                'num_threads': 8,
                'early_stopping_rounds': 20
            },
            'random_state': 42
        }
    }
    
    # 构造测试数据（模拟两个截面）
    np.random.seed(42)
    n_samples = 600
    dates = ['2024-01-01'] * 300 + ['2024-01-02'] * 300
    stocks = [f'stock_{i:03d}' for i in range(300)] * 2
    
    X = pd.DataFrame(
        np.random.randn(n_samples, 10),
        columns=[f'feature_{i}' for i in range(10)],
        index=pd.MultiIndex.from_tuples(
            [(d, s) for d, s in zip(dates, stocks)],
            names=['date', 'stock_code']
        )
    )
    
    # 构造标签（有相关性，不是纯随机）
    # 让 feature_0 和 feature_1 与收益相关
    y = (
        0.3 * X['feature_0'] + 
        0.2 * X['feature_1'] + 
        np.random.randn(n_samples) * 0.5
    )
    y = pd.Series(y, index=X.index, name='return')
    
    # 划分训练/验证/测试
    train_mask = X.index.get_level_values(0) == '2024-01-01'
    valid_mask = X.index.get_level_values(0) == '2024-01-02'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask].iloc[:150], y[valid_mask].iloc[:150]
    X_test, y_test = X[valid_mask].iloc[150:], y[valid_mask].iloc[150:]
    
    print(f"\n数据划分:")
    print(f"  训练: {len(X_train)} 样本 ({X_train.index.get_level_values(0).nunique()} 天)")
    print(f"  验证: {len(X_valid)} 样本 ({X_valid.index.get_level_values(0).nunique()} 天)")
    print(f"  测试: {len(X_test)} 样本 ({X_test.index.get_level_values(0).nunique()} 天)")
    
    # 测试模型
    print("\n1. 测试模型初始化:")
    model = LightGBMRankModel(config)
    print("   ✓ 模型初始化成功")
    print(f"   ✓ Objective: {model.model_params['objective']}")
    
    print("\n2. 测试模型训练:")
    model.fit(X_train, y_train, X_valid, y_valid)
    print(f"   ✓ 训练完成，最佳迭代: {model.model.best_iteration}")
    
    print("\n3. 测试模型预测:")
    predictions = model.predict(X_test)
    print(f"   ✓ 预测完成")
    print(f"   ✓ 预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   ✓ 预测值标准差: {predictions.std():.4f}")
    print(f"   ✓ 唯一值数量: {len(np.unique(predictions))} / {len(predictions)}")
    
    print("\n4. 测试排名预测:")
    ranks = model.predict_rank(X_test)
    print(f"   ✓ 排名范围: [{ranks.min():.0f}, {ranks.max():.0f}]")
    
    print("\n5. 测试特征重要性:")
    importance = model.get_feature_importance()
    print(f"   ✓ Top 3 特征:")
    for feat, row in importance.head(3).iterrows():
        print(f"      {feat}: {row['importance']:.2f}")
    
    print("\n6. 测试 NDCG 分数:")
    ndcg = model.get_ndcg_score(X_test, y_test, k=50)
    print(f"   ✓ NDCG@50: {ndcg:.4f}")
    
    print("\n7. 测试模型保存/加载:")
    test_path = Path(__file__).parent / "test_rank_model.pkl"
    model.save(test_path)
    print(f"   ✓ 模型已保存到: {test_path}")
    
    loaded_model = LightGBMRankModel.load(test_path)
    print(f"   ✓ 模型已加载")
    
    # 验证加载后的模型能正常预测
    loaded_pred = loaded_model.predict(X_test)
    assert np.allclose(predictions, loaded_pred), "加载后的模型预测不一致"
    print(f"   ✓ 加载后的模型预测一致")
    
    # 清理测试文件
    test_path.unlink()
    print(f"   ✓ 测试文件已清理")
    
    print("\n" + "=" * 80)
    print("✓ LightGBMRankModel 测试通过！")
    print("=" * 80)
