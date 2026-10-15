"""Dataset validation, schema integrity checks, and leakage detection."""

from __future__ import annotations

import io
import re
from typing import Any
import pandas as pd

from config import MAX_UPLOAD_BYTES


def validate_file_size(payload_bytes: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Validate that the uploaded byte stream is bounded and non-empty."""
    if len(payload_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")
    if len(payload_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        actual_mb = len(payload_bytes) / (1024 * 1024)
        raise ValueError(f"Uploaded file exceeds limit: {actual_mb:.1f} MB > allowed {max_mb} MB.")


def safe_read_csv(source: Any, max_bytes: int = MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Parse CSV safely with defensive encoding and dialect handling."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        if isinstance(source, (bytes, bytearray)):
            payload = bytes(source)
        elif hasattr(source, "getvalue"):
            payload = source.getvalue()
        elif hasattr(source, "read"):
            payload = source.read(max_bytes + 1)
        else:
            raise TypeError("Source must be bytes, a file-like object, or a pandas DataFrame.")

        validate_file_size(payload, max_bytes=max_bytes)

        # Check raw CSV header for duplicate column names before pandas renames them
        try:
            sample_text = payload[:8192].decode("utf-8", errors="ignore")
            first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
            if first_line:
                # Basic delimiter detection
                delim = "," if "," in first_line else ("\t" if "\t" in first_line else ";")
                raw_headers = [h.strip().strip('"').strip("'") for h in first_line.split(delim)]
                if len(raw_headers) != len(set(raw_headers)):
                    seen = set()
                    dups = [h for h in raw_headers if h in seen or seen.add(h)]
                    raise ValueError(f"Duplicate column headers detected: {dups}")
        except ValueError:
            raise
        except Exception:
            pass

        # Attempt utf-8 decoding first, fallback to latin-1 / cp1252
        try:
            df = pd.read_csv(io.BytesIO(payload), low_memory=False, on_bad_lines="error")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(payload), encoding="latin-1", low_memory=False)
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV with alternate encoding: {exc}") from exc
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file contains no data.")
        except pd.errors.ParserError as exc:
            raise ValueError(f"Malformed CSV formatting: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc

    validate_dataframe_integrity(df)
    return df


def validate_dataframe_integrity(df: pd.DataFrame) -> None:
    """Ensure dataframe has valid columns, rows, and unique header names."""
    if df.empty:
        raise ValueError("The dataset contains 0 rows.")
    if len(df.columns) == 0:
        raise ValueError("The dataset contains 0 columns.")
    if not df.columns.is_unique:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column headers detected: {duplicates}")
    for col in df.columns:
        if not isinstance(col, str) or not str(col).strip():
            raise ValueError("All column headers must be non-empty strings.")


def detect_identifier_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify columns likely to be database keys, IDs, or indexing artifacts."""
    id_patterns = re.compile(r"(^id$|_id$|id_|^guid$|^uuid$|^index$|^row_?id)", re.IGNORECASE)
    results = []
    n_rows = len(df)

    for col in df.columns:
        col_str = str(col).strip()
        series = df[col]
        nunique = series.nunique(dropna=True)
        unique_ratio = nunique / n_rows if n_rows > 0 else 0

        is_name_match = bool(id_patterns.search(col_str))
        # Exclude continuous floating point features from pure cardinality check
        is_float = pd.api.types.is_float_dtype(series)
        is_high_cardinality = (not is_float) and (unique_ratio > 0.95 and n_rows >= 20)

        if is_name_match or is_high_cardinality:
            reason = []
            if is_name_match:
                reason.append("Header name matches ID pattern")
            if is_high_cardinality:
                reason.append(f"Near-unique cardinality ({unique_ratio:.1%} unique values)")
            results.append({
                "column": col,
                "unique_count": nunique,
                "unique_ratio": unique_ratio,
                "reason": " & ".join(reason),
            })

    return results


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the chosen target column exists and contains valid values."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in dataset.")
    labeled = df.dropna(subset=[target_column])
    if len(labeled) < 4:
        raise ValueError(f"Target column '{target_column}' has fewer than 4 non-null entries.")
    if labeled[target_column].nunique() < 2:
        raise ValueError(f"Target column '{target_column}' must have at least 2 distinct values.")


def compare_schemas(reference_cols: list[str], current_cols: list[str]) -> dict[str, list[str]]:
    """Compare column sets between reference and current datasets."""
    ref_set = set(reference_cols)
    cur_set = set(current_cols)
    return {
        "missing_in_current": sorted(list(ref_set - cur_set)),
        "extra_in_current": sorted(list(cur_set - ref_set)),
        "common_columns": [c for c in reference_cols if c in cur_set],
    }
