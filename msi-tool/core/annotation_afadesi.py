"""
核心计算：基于本地数据库的 MSI 代谢物注释
基于 AFADESI-MSI 加合物配置（Zhu et al. 2022, Anal Chem, Table S2）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import anndata as ad


# ============================================================
# 加合物定义
# 来源: Zhu Y, Zang Q, Luo Z, He J, Zhang R, Abliz Z.
#   "An Organ-Specific Metabolite Annotation Approach for
#    Ambient Mass Spectrometry Imaging..."
#   Anal Chem, 2022, 94(20), 7286-7294.
#   Table S2 (ac2c00557_si_003.xlsx) 列名
#
# 公式: m/z = (M + delta_mass) / |charge|
#   M = 中性分子单同位素质量
# ============================================================

PROTON  = 1.00727646677
NA      = 22.98922
K       = 38.96316
NH4     = 18.03437   # NH4+ = 14.00307(N) + 4*1.00783(H) - e
H2O     = 18.01056
CL      = 34.96885
ELECTRON = 0.00054858


@dataclass(frozen=True)
class AdductSpec:
    name: str
    delta_mass: float   # 加到中性质量M上的净质量偏移
    charge: int          # 电荷符号（+1或-1）


# ---------- 正离子模式 (6种, Table S2 Sheet1) ----------
POSITIVE_ADDUCTS: list[AdductSpec] = [
    AdductSpec("[M+H]+",       PROTON,              +1),
    AdductSpec("[M+Na]+",      NA,                  +1),
    AdductSpec("[M+K]+",       K,                   +1),
    AdductSpec("[M+NH4]+",     NH4,                 +1),
    AdductSpec("[M+H-H2O]+",   PROTON - H2O,        +1),
    AdductSpec("[M]+",         -ELECTRON,           +1),   # 自由基阳离子，失去一个电子
]

# ---------- 负离子模式 (3种, Table S2 Sheet2) ----------
NEGATIVE_ADDUCTS: list[AdductSpec] = [
    AdductSpec("[M-H]-",       -PROTON,             -1),
    AdductSpec("[M+Cl]-",      CL,                  -1),
    AdductSpec("[M-H-H2O]-",   -PROTON - H2O,       -1),
]

# 按模式索引
_MODE_MAP = {
    'positive': POSITIVE_ADDUCTS,
    'negative': NEGATIVE_ADDUCTS,
}

# 按名称索引（用于手动选择）
ALL_ADDUCTS = {a.name: a for a in POSITIVE_ADDUCTS + NEGATIVE_ADDUCTS}


def _resolve_adducts(
    mode: str | None = None,
    adduct_names: list[str] | None = None,
) -> list[AdductSpec]:
    """
    解析加合物列表。
    - mode='positive'/'negative': 返回该模式全部加合物
    - adduct_names: 手动指定（优先于mode）
    """
    if adduct_names:
        specs = []
        for name in adduct_names:
            if name not in ALL_ADDUCTS:
                raise ValueError(f"不支持的加合物: '{name}'。可选: {list(ALL_ADDUCTS.keys())}")
            specs.append(ALL_ADDUCTS[name])
        return specs

    if mode in _MODE_MAP:
        return _MODE_MAP[mode]

    raise ValueError("请指定 mode='positive'/'negative' 或提供 adduct_names")


def _require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{where} 缺少必要列: {miss}")


def load_db_csv(db_csv_path: str) -> pd.DataFrame:
    """
    读取本地代谢物数据库 CSV。
    必需列: id, name, formula, exact_mass
    """
    db = pd.read_csv(db_csv_path, encoding="utf-8")
    _require_columns(db, ["id", "name", "formula", "exact_mass"], "数据库 CSV")
    db = db.copy()
    db["exact_mass"] = pd.to_numeric(db["exact_mass"], errors="coerce")
    db = db.dropna(subset=["exact_mass"])
    if db.empty:
        raise ValueError("数据库 CSV 中没有可用的 exact_mass。")
    return db


def _mz_to_neutral_mass(mz: float, adduct: AdductSpec) -> float:
    """由观测 m/z 反推中性分子质量 M"""
    z = abs(adduct.charge)
    return mz * z - adduct.delta_mass


def annotate_mz_list(
    mz_values: np.ndarray,
    db_df: pd.DataFrame,
    mode: str | None = None,
    adduct_names: list[str] | None = None,
    ppm_tolerance: float = 10.0,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    对 m/z 列表做精确质量匹配，返回候选注释表。
    """
    if ppm_tolerance <= 0:
        raise ValueError("ppm_tolerance 必须 > 0")
    if top_n <= 0:
        raise ValueError("top_n 必须 > 0")

    adducts = _resolve_adducts(mode=mode, adduct_names=adduct_names)
    db_m = db_df["exact_mass"].to_numpy(dtype=float)
    mz_arr = np.asarray(mz_values, dtype=float)

    rows = []
    for feat_idx, mz in enumerate(mz_arr):
        for spec in adducts:
            neutral = _mz_to_neutral_mass(mz, spec)
            if neutral <= 0:
                continue
            ppm = np.abs((db_m - neutral) / neutral) * 1e6
            hit_idx = np.where(ppm <= ppm_tolerance)[0]
            if hit_idx.size == 0:
                continue
            local = db_df.iloc[hit_idx].copy()
            local["ppm_error"] = ppm[hit_idx]
            local = local.sort_values("ppm_error", ascending=True).head(top_n)
            for _, r in local.iterrows():
                item = {
                    "feature_index": int(feat_idx),
                    "mz_observed": float(mz),
                    "adduct": spec.name,
                    "neutral_mass_est": float(neutral),
                    "ppm_error": float(r["ppm_error"]),
                    "db_id": str(r["id"]),
                    "db_name": str(r["name"]),
                    "formula": str(r["formula"]),
                    "db_exact_mass": float(r["exact_mass"]),
                    "score_mass": max(0.0, 1.0 - float(r["ppm_error"]) / ppm_tolerance),
                }
                for extra_col in ["KEGG", "kegg", "HMDB", "PubChem", "CAS"]:
                    if extra_col in r.index:
                        item[extra_col] = r.get(extra_col)
                rows.append(item)

        if (feat_idx + 1) % 100 == 0:
            print(f"  已注释 {feat_idx + 1}/{len(mz_arr)} 个特征...")

    columns = [
        "feature_index", "mz_observed", "adduct", "neutral_mass_est", "ppm_error",
        "db_id", "db_name", "formula", "db_exact_mass", "score_mass",
        "KEGG", "kegg", "HMDB", "PubChem", "CAS",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    out = out.sort_values(["feature_index", "ppm_error"]).reset_index(drop=True)
    return out


def annotate_h5ad(
    h5ad_path: str,
    db_csv_path: str,
    mode: str | None = None,
    adduct_names: list[str] | None = None,
    ppm_tolerance: float = 10.0,
    top_n: int = 5,
) -> dict:
    """
    从 h5ad 读取 m/z 并执行注释。

    参数:
        h5ad_path: h5ad 文件路径
        db_csv_path: 数据库 CSV 路径 (需含 id, name, formula, exact_mass)
        mode: 'positive' 或 'negative'
        adduct_names: 手动指定加合物列表（优先于 mode）
        ppm_tolerance: 质量偏差容许 ppm，默认10
        top_n: 每个特征每种加合物最多候选数
    """
    adata = ad.read_h5ad(h5ad_path)
    if "m/z" in adata.var.columns:
        mz_raw = adata.var["m/z"].values
    else:
        mz_raw = adata.var_names.values
    mz = pd.to_numeric(pd.Series(mz_raw), errors="coerce").dropna().to_numpy(dtype=float)
    if mz.size == 0:
        raise ValueError("数据中没有可解析的 m/z。")

    db = load_db_csv(db_csv_path)
    adducts = _resolve_adducts(mode=mode, adduct_names=adduct_names)

    ann = annotate_mz_list(
        mz_values=mz,
        db_df=db,
        mode=mode,
        adduct_names=adduct_names,
        ppm_tolerance=ppm_tolerance,
        top_n=top_n,
    )

    n_annotated = ann["feature_index"].nunique() if not ann.empty else 0

    return {
        "results_df": ann,
        "info": {
            "n_features_input": int(mz.size),
            "n_features_annotated": n_annotated,
            "annotation_rate": f"{n_annotated / mz.size * 100:.1f}%",
            "n_db_compounds": int(db.shape[0]),
            "n_hits": int(ann.shape[0]),
            "ppm_tolerance": float(ppm_tolerance),
            "adducts_used": [a.name for a in adducts],
        }
    }
