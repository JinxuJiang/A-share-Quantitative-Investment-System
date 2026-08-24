from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class AlphaTransformer:
    METHOD = "eligible_rank_normal_score"

    def __init__(self, rank_method: str = "average"):
        self.rank_method = rank_method

    def transform(self, universe: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        result = universe.copy()
        eligible_mask = result["is_eligible"].fillna(False)
        eligible = result.loc[eligible_mask].copy()
        if len(eligible) < top_n:
            raise ValueError(f"可投资股票只有{len(eligible)}只，少于Top{top_n}")

        position = eligible["alpha_score"].rank(method=self.rank_method, ascending=True)
        n_assets = len(eligible)
        probability = (position - 0.5) / n_assets
        eligible["eligible_alpha_rank"] = position / n_assets
        eligible["alpha_zscore"] = norm.ppf(probability.to_numpy())
        eligible["alpha_method"] = self.METHOD
        eligible = eligible.sort_values(["alpha_score", "stock_code"], ascending=[False, True])
        eligible["eligible_selection_rank"] = np.arange(1, len(eligible) + 1, dtype="int32")
        eligible["is_selected"] = eligible["eligible_selection_rank"].le(top_n)

        columns = [
            "stock_code",
            "eligible_alpha_rank",
            "alpha_zscore",
            "alpha_method",
            "eligible_selection_rank",
            "is_selected",
        ]
        result = result.merge(eligible[columns], on="stock_code", how="left", validate="one_to_one")
        result["is_selected"] = result["is_selected"].eq(True)

        selected = result.loc[result["is_selected"]].copy()
        selected = selected.rename(columns={"eligible_selection_rank": "selection_rank"})
        selected = selected.sort_values("selection_rank").reset_index(drop=True)
        selected["selection_rank"] = selected["selection_rank"].astype("int32")
        return result, selected
