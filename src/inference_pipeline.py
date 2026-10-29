"""Leakage-safe, reusable feature contract for training and online inference.

The pipeline uses fixed, label-free encodings and stateful prior-transaction
summaries. Row order is preserved so training, evaluation, and live scoring use
the same causal feature contract.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd


class FraudFeaturePipeline:
    """Fit-on-train feature preprocessing shared by batch and API scoring."""

    TARGET_COLUMNS = {"isFraud", "TransactionID", "TransactionDT"}
    # IEEE-CIS TransactionDT is a relative time offset, not an absolute
    # timestamp. Only time-of-day survives the train/serve mapping reliably.
    TIME_COLUMNS = {"hour_of_day", "day_of_week", "month_approx", "is_night", "is_weekend"}
    HISTORY_SPECS = {
        "card1": ("card1",),
        "card1_addr1": ("card1", "addr1"),
    }
    HISTORY_WINDOW_SECONDS = 7 * 24 * 60 * 60

    def __init__(self) -> None:
        self.numeric_columns_: list[str] = []
        self.numeric_missing_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.category_levels_: dict[str, list[str]] = {}
        self.category_hash_bins_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.history_state_: dict[str, dict[Any, tuple[deque, float]]] = {}
        self.history_feature_names_: list[str] = []
        self.history_time_end_: float | None = None
        self.history_last_processed_time_: float | None = None
        self.runtime_wall_time_anchor_: float | None = None
        self.runtime_dataset_time_anchor_: float | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def _as_frame(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            # Transform methods only read the input. Avoid copying the full
            # IEEE-CIS frame (hundreds of MB) before every conversion.
            return data
        return pd.DataFrame(data)

    @staticmethod
    def _category_key(value: Any) -> str:
        if pd.isna(value):
            return "__MISSING__"
        return str(value)

    def fit(self, data: Any, y: Any = None) -> "FraudFeaturePipeline":
        frame = self._as_frame(data)
        # Per-entity summaries use only rows already seen by transform().
        # Keeping these summaries (rather than raw transaction histories) makes
        # the same causal feature contract practical for API scoring.
        self.history_state_ = {name: {} for name in self.HISTORY_SPECS}
        self.history_last_processed_time_ = None
        self.runtime_wall_time_anchor_ = None
        self.runtime_dataset_time_anchor_ = None
        if "TransactionDT" in frame:
            fit_times = pd.to_numeric(frame["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
            fit_times = fit_times[np.isfinite(fit_times)]
            self.history_time_end_ = float(fit_times.max()) if len(fit_times) else None
        else:
            self.history_time_end_ = None
        self.history_feature_names_ = [
            feature
            for name in self.HISTORY_SPECS
            for feature in (
                f"{name}_prior_7d_count_log1p",
                f"{name}_prior_7d_mean_amount",
                f"{name}_amount_vs_prior_7d_mean",
            )
        ]
        excluded = self.TARGET_COLUMNS | self.TIME_COLUMNS
        self.numeric_columns_ = [
            col for col in frame.select_dtypes(include=[np.number, "bool"]).columns
            if col not in excluded
        ]
        self.categorical_columns_ = [
            col for col in frame.columns
            if col not in excluded and col not in self.numeric_columns_
        ]
        # Keep a missingness bit for every numeric field, including fields
        # that happened to be complete in this training period.
        self.numeric_missing_columns_ = list(self.numeric_columns_)

        for col in self.numeric_columns_:
            # A fixed sentinel avoids fitting an imputation statistic on
            # future rows inside the chronological OOF training windows.
            self.numeric_medians_[col] = 0.0

        for col in self.categorical_columns_:
            keys = frame[col].map(self._category_key)
            values = sorted(set(keys) - {"__MISSING__"})
            if len(values) <= 32:
                self.category_levels_[col] = values
            else:
                # Hash buckets keep cardinality bounded and are independent of
                # train-period category frequencies.
                self.category_hash_bins_[col] = 64
        self.feature_names_ = list(self.numeric_columns_)
        self.feature_names_.extend(f"{col}__missing" for col in self.numeric_missing_columns_)
        for col in self.categorical_columns_:
            if col in self.category_levels_:
                self.feature_names_.extend(
                    f"{col}__category_{index}"
                    for index in range(len(self.category_levels_[col]))
                )
            else:
                self.feature_names_.extend(
                    f"{col}__hash_{index}"
                    for index in range(self.category_hash_bins_[col])
                )
            self.feature_names_.append(f"{col}__missing")
        self.feature_names_.extend([
            "log_amount", "amount_sqrt", "amount_decimal", "is_round_amount",
            "hour_of_day", "is_night",
        ])
        self.feature_names_.extend(self.history_feature_names_)
        self.is_fitted_ = True
        return self

    def fit_transform(self, data: Any, y: Any = None) -> pd.DataFrame:
        return self.fit(data, y=y).transform(data)

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        features: dict[str, np.ndarray] = {}

        for col in self.numeric_columns_:
            if col in frame:
                vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                vals = pd.Series(np.nan, index=frame.index, dtype=np.float32)
            if col in self.numeric_missing_columns_:
                features[f"{col}__missing"] = vals.isna().to_numpy(dtype=np.float32, copy=False)
            features[col] = vals.fillna(self.numeric_medians_[col]).to_numpy(dtype=np.float32, copy=False)

        for col in self.categorical_columns_:
            values = frame[col] if col in frame else pd.Series(np.nan, index=frame.index)
            keys = values.map(self._category_key)
            if col in self.category_levels_:
                for index, category in enumerate(self.category_levels_[col]):
                    features[f"{col}__category_{index}"] = (keys == category).to_numpy(dtype=np.float32, copy=False)
            else:
                bins = self.category_hash_bins_[col]
                bucket = (pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64) % bins)
                for index in range(bins):
                    features[f"{col}__hash_{index}"] = (bucket == index).astype(np.float32)
            features[f"{col}__missing"] = values.isna().to_numpy(dtype=np.float32, copy=False)

        amount = pd.to_numeric(frame.get("TransactionAmt", pd.Series(0.0, index=frame.index)), errors="coerce")
        amount = amount.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=np.float32, copy=False)
        amount_decimal = amount - np.floor(amount)
        features["log_amount"] = np.log1p(amount)
        features["amount_sqrt"] = np.sqrt(amount)
        features["amount_decimal"] = amount_decimal
        features["is_round_amount"] = ((amount >= 100) & (amount_decimal == 0)).astype(np.float32)

        if "TransactionDT" in frame:
            txn_dt = pd.to_numeric(frame["TransactionDT"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
            hour = ((txn_dt % 86400) // 3600).astype(np.int8)
        else:
            hour = pd.Series(12, index=frame.index, dtype=np.int8)

        if "hour_of_day" in frame:
            hour = pd.to_numeric(frame["hour_of_day"], errors="coerce").fillna(12).clip(0, 23).astype(np.int8)

        features["hour_of_day"] = hour.to_numpy(dtype=np.float32, copy=False)
        features["is_night"] = ((hour <= 6) | (hour >= 23)).to_numpy(dtype=np.float32, copy=False)
        return pd.DataFrame(features, index=frame.index, dtype=np.float32)

    @staticmethod
    def _entity_value(value: Any) -> str | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if not np.isfinite(number):
                return None
            return str(int(number)) if number.is_integer() else format(number, ".12g")
        value = str(value).strip().casefold()
        return value or None

    def _mapped_transaction_times(self, frame: pd.DataFrame, *, update_state: bool) -> np.ndarray:
        if "TransactionDT" in frame:
            times = pd.to_numeric(frame["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(times)
            if not finite.all():
                fallback = self.history_last_processed_time_ or self.history_time_end_ or 0.0
                times[~finite] = fallback
        else:
            start = self.history_last_processed_time_ or self.history_time_end_ or 0.0
            times = start + np.arange(1, len(frame) + 1, dtype=np.float64)

        # Training TransactionDT is relative seconds. API timestamps are Unix
        # seconds, so map their first event to the training endpoint and retain
        # real elapsed time for later rolling-window calculations.
        if len(times) and np.nanmedian(np.abs(times)) > 100_000_000:
            wall_anchor = self.runtime_wall_time_anchor_
            dataset_anchor = self.runtime_dataset_time_anchor_
            if wall_anchor is None:
                wall_anchor = float(np.min(times))
                dataset_anchor = self.history_time_end_ or self.history_last_processed_time_ or 0.0
                if update_state:
                    self.runtime_wall_time_anchor_ = wall_anchor
                    self.runtime_dataset_time_anchor_ = dataset_anchor
            times = float(dataset_anchor) + (times - float(wall_anchor))

        if len(times) > 1 and np.any(np.diff(times) < 0):
            raise ValueError("Transactions must be supplied in chronological order for history features.")
        if update_state and len(times) and self.history_last_processed_time_ is not None:
            if times[0] < self.history_last_processed_time_:
                raise ValueError("Transaction timestamp precedes the last scored transaction.")
        return times

    def _history_features(
        self,
        frame: pd.DataFrame,
        history: dict[str, dict[Any, tuple[deque, float]]],
        transaction_times: np.ndarray,
    ) -> tuple[pd.DataFrame, float | None]:
        """Return per-entity aggregates computed before each transaction.

        Rows are processed in input order. Each feature sees the entity's
        transactions from the previous seven days, then the current event is
        added for later rows. Labels are never read.
        """
        n_rows = len(frame)
        amount_values = pd.to_numeric(
            frame.get("TransactionAmt", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        spec_values: dict[str, tuple[np.ndarray, ...]] = {}
        for name, columns in self.HISTORY_SPECS.items():
            spec_values[name] = tuple(
                frame[col].to_numpy(dtype=object, copy=False)
                if col in frame else np.full(n_rows, None, dtype=object)
                for col in columns
            )

        result = {
            feature: np.zeros(n_rows, dtype=np.float32)
            for feature in self.history_feature_names_
        }
        last_time = self.history_last_processed_time_
        for row_index, amount in enumerate(amount_values):
            current_time = float(transaction_times[row_index])
            for name, columns in self.HISTORY_SPECS.items():
                values = tuple(self._entity_value(col[row_index]) for col in spec_values[name])
                key = values[0] if len(values) == 1 else values
                if any(value is None for value in values):
                    key = None
                events, prior_amount_sum = history[name].get(key, (deque(), 0.0)) if key is not None else (deque(), 0.0)
                oldest_allowed = current_time - self.HISTORY_WINDOW_SECONDS
                while events and events[0][0] < oldest_allowed:
                    _, expired_amount = events.popleft()
                    prior_amount_sum -= expired_amount
                prior_count = len(events)
                prior_mean = prior_amount_sum / prior_count if prior_count else 0.0
                result[f"{name}_prior_7d_count_log1p"][row_index] = np.log1p(prior_count)
                result[f"{name}_prior_7d_mean_amount"][row_index] = prior_mean
                result[f"{name}_amount_vs_prior_7d_mean"][row_index] = (
                    min(float(amount) / prior_mean, 20.0) if prior_mean > 0 else 0.0
                )
                if key is not None:
                    events.append((current_time, float(amount)))
                    history[name][key] = (events, prior_amount_sum + float(amount))
            last_time = current_time

        return pd.DataFrame(result, index=frame.index, dtype=np.float32), last_time

    def transform(self, data: Any, *, update_state: bool = True) -> pd.DataFrame:
        if not self.numeric_columns_ and not self.categorical_columns_ and not self.feature_names_:
            raise RuntimeError("FraudFeaturePipeline must be fitted before transform().")
        frame = self._as_frame(data)
        transformed = self._transform_frame(frame)
        if update_state:
            history = self.history_state_
        else:
            # Holdout evaluation must not become the starting state for a
            # later live request. Tuple values are immutable, so shallow copies
            # of each entity map are sufficient.
            history = {
                name: {key: (deque(events), amount_sum) for key, (events, amount_sum) in values.items()}
                for name, values in self.history_state_.items()
            }
        transaction_times = self._mapped_transaction_times(frame, update_state=update_state)
        historical, last_time = self._history_features(frame, history, transaction_times)
        if update_state:
            self.history_last_processed_time_ = last_time
        transformed = pd.concat([transformed, historical], axis=1)
        return transformed.reindex(columns=self.feature_names_, fill_value=np.float32(0))

    def get_feature_names_out(self) -> list[str]:
        if not self.feature_names_:
            raise RuntimeError("FraudFeaturePipeline must be fitted before requesting feature names.")
        return list(self.feature_names_)
