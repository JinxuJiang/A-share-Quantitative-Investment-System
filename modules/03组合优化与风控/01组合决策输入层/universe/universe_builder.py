from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")


class UniverseBuilder:
    def __init__(
        self,
        board_scope: str = "main_board",
        min_price_observations: int = 253,
        require_non_st: bool = True,
        require_not_suspended: bool = True,
    ):
        if board_scope not in {"main_board", "all"}:
            raise ValueError(f"不支持的board_scope: {board_scope}")
        self.board_scope = board_scope
        self.min_price_observations = int(min_price_observations)
        self.require_non_st = bool(require_non_st)
        self.require_not_suspended = bool(require_not_suspended)

    def build(
        self,
        alpha: pd.DataFrame,
        stock_info: pd.DataFrame,
        st_status: pd.Series,
        suspend_status: pd.Series,
        price_observations: pd.Series,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        date = pd.Timestamp(signal_date).normalize()
        required = {"stock_code", "alpha_score", "source_alpha_rank"}
        missing = required - set(alpha.columns)
        if missing:
            raise ValueError(f"Alpha截面缺少字段: {sorted(missing)}")

        metadata = stock_info.copy()
        metadata = metadata.drop_duplicates("order_book_id", keep="last")
        metadata = metadata.rename(columns={"order_book_id": "stock_code"})
        keep_meta = [
            column
            for column in ["stock_code", "market", "industry", "list_date", "delist_date", "list_status"]
            if column in metadata.columns
        ]
        result = alpha.merge(metadata[keep_meta], on="stock_code", how="left", validate="one_to_one")

        result["is_main_board"] = result["market"].eq("主板")
        fallback_main = result["stock_code"].str.match(r"^(60|00)\d{4}\.(SH|SZ)$", na=False)
        result["is_main_board"] = result["is_main_board"] | fallback_main

        list_date = _parse_yyyymmdd(result.get("list_date", pd.Series(index=result.index, dtype="string")))
        delist_date = _parse_yyyymmdd(result.get("delist_date", pd.Series(index=result.index, dtype="string")))
        result["is_listed_asof"] = list_date.le(date) & (delist_date.isna() | delist_date.ge(date))

        result["st_status"] = pd.to_numeric(result["stock_code"].map(st_status), errors="coerce")
        result["suspend_status"] = pd.to_numeric(result["stock_code"].map(suspend_status), errors="coerce")
        result["price_observations"] = (
            pd.to_numeric(result["stock_code"].map(price_observations), errors="coerce").fillna(0).astype("int32")
        )
        result["is_non_st"] = result["st_status"].eq(0)
        result["is_not_suspended"] = result["suspend_status"].eq(0)
        result["has_sufficient_history"] = result["price_observations"].ge(self.min_price_observations)

        eligible = result["is_listed_asof"] & result["has_sufficient_history"]
        if self.board_scope == "main_board":
            eligible &= result["is_main_board"]
        if self.require_non_st:
            eligible &= result["is_non_st"]
        if self.require_not_suspended:
            eligible &= result["is_not_suspended"]
        result["is_eligible"] = eligible

        def exclusion_reason(row: pd.Series) -> str:
            reasons = []
            if self.board_scope == "main_board" and not bool(row["is_main_board"]):
                reasons.append("NON_MAIN_BOARD")
            if not bool(row["is_listed_asof"]):
                reasons.append("NOT_LISTED_ASOF")
            if self.require_non_st and not bool(row["is_non_st"]):
                reasons.append("ST_OR_STATUS_MISSING")
            if self.require_not_suspended and not bool(row["is_not_suspended"]):
                reasons.append("SUSPENDED_OR_STATUS_MISSING")
            if not bool(row["has_sufficient_history"]):
                reasons.append("INSUFFICIENT_HISTORY")
            return "|".join(reasons)

        result["exclusion_reason"] = result.apply(exclusion_reason, axis=1)
        return result.sort_values("stock_code").reset_index(drop=True)

