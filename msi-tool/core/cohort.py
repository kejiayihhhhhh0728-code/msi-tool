"""
跨样本装配层（不依赖 Flask）
==========================

目的：把"批次/样本/ROI"的嵌套目录结构装配成参考 pipeline 期望的格式
       —— 每个 (tissue_name, polarity) 一份 cancer_mat × para_mat（rows=m/z, cols=samples）。

输入：批次目录路径 + 样本元数据（batch_meta.json + 各 sample 的 pseudobulk_all.csv）
输出：可直接喂给 core.differential 的配对矩阵

核心约束
- 每个样本必须同时有 cancer 和 paracancer 才能纳入配对分析
- 一个样本如果有多个 cancer/paracancer 子区域（癌区1/癌区2），按 pixel_count
  加权平均合并成一行 pseudobulk
- 跨样本只保留所有样本共有的 m/z 集合（取交集）
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd


META_COLUMNS = {'pixel_count', 'tissue_name', 'region_name', 'region_type'}


# ─── 读取 batch_meta + 各样本的 pseudobulk_all.csv ─────────────────────

def _load_batch_meta(batch_dir: str) -> dict:
    p = os.path.join(batch_dir, 'batch_meta.json')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_sample_meta(sample_dir: str) -> dict:
    p = os.path.join(sample_dir, 'meta.json')
    if not os.path.exists(p):
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_sample_pseudobulk(sample_dir: str) -> pd.DataFrame | None:
    """
    读取单个样本的 pseudobulk_all.csv。
    返回 None 表示该样本还没跑模块 7 ROI 提取。
    """
    p = os.path.join(sample_dir, 'roi', 'pseudobulk_all.csv')
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _aggregate_within_sample(
    pb_df: pd.DataFrame,
    tissue_name: str,
    region_type: str,
) -> pd.Series | None:
    """
    在单样本内，把同一 (tissue, region_type) 的多行（多个子区域）
    按 pixel_count 加权平均，得到一行 m/z 谱。

    返回 pd.Series（index=m/z 列名），无匹配返回 None。
    """
    sub = pb_df[(pb_df['tissue_name'] == tissue_name)
                & (pb_df['region_type'] == region_type)].copy()
    if sub.empty:
        return None

    mz_cols = [c for c in pb_df.columns if c not in META_COLUMNS]
    if not mz_cols:
        return None

    weights = sub['pixel_count'].astype(float).values
    if weights.sum() <= 0:
        # 退化：等权平均
        return sub[mz_cols].mean(axis=0)

    arr = sub[mz_cols].astype(float).values  # (n_subregion, n_mz)
    weighted = (arr * weights[:, None]).sum(axis=0) / weights.sum()
    return pd.Series(weighted, index=mz_cols)


# ─── 装配主入口 ───────────────────────────────────────────────────────

def assemble_cohort(batch_dir: str) -> dict:
    """
    装配批次内所有样本的配对 cancer/paracancer pseudobulk 表。

    返回
    ----
    {
      tissue_name: {
        'cancer_mat'  : DataFrame  (m/z × sample_ids),
        'para_mat'    : DataFrame  (m/z × sample_ids),
        'sample_ids'  : [str, ...]   (顺序对应 cancer_mat / para_mat 的列)
        'sample_names': [str, ...]   (人类可读名，与 sample_ids 等长)
        'sample_groups':[str, ...]   (用户在样本管理里指定的 group 字段，可空)
        'n_samples'   : int,
      }, ...
    }

    只包含：至少 1 个样本同时有 cancer 和 paracancer 的 tissue。
    跨样本只保留共有 m/z 集合。
    """
    bmeta = _load_batch_meta(batch_dir)
    samples = bmeta.get('samples', [])
    if not samples:
        return {}

    # Step 1: 收集每个样本每个 tissue 的 cancer / paracancer 谱
    # buckets[tissue][sample_id]['cancer'|'paracancer'] = pd.Series(mz)
    buckets: dict[str, dict[str, dict[str, pd.Series]]] = {}
    sample_meta_cache: dict[str, dict] = {}

    for s in samples:
        sid = s['id']
        sdir = os.path.join(batch_dir, 'samples', sid)
        sample_meta_cache[sid] = _load_sample_meta(sdir) or {}
        pb = _load_sample_pseudobulk(sdir)
        if pb is None or pb.empty:
            continue
        if 'tissue_name' not in pb.columns or 'region_type' not in pb.columns:
            # 旧格式（无 tissue 列），跳过
            continue

        tissues_in_sample = pb['tissue_name'].dropna().unique()
        for tname in tissues_in_sample:
            for rtype in ('cancer', 'paracancer'):
                row = _aggregate_within_sample(pb, tname, rtype)
                if row is None:
                    continue
                buckets.setdefault(tname, {}).setdefault(sid, {})[rtype] = row

    # Step 2: 每个 tissue，筛出 "同时有 cancer 和 paracancer" 的样本
    cohorts: dict[str, dict] = {}
    for tname, by_sample in buckets.items():
        paired_sids = [sid for sid, d in by_sample.items()
                       if 'cancer' in d and 'paracancer' in d]
        if not paired_sids:
            continue

        # m/z 取所有样本的交集（既包括 cancer 又包括 paracancer）
        all_indexes = []
        for sid in paired_sids:
            all_indexes.append(set(by_sample[sid]['cancer'].index))
            all_indexes.append(set(by_sample[sid]['paracancer'].index))
        common_mz = sorted(set.intersection(*all_indexes), key=_mz_sort_key)
        if not common_mz:
            continue

        # 显示用名字（meta.json 的 name 字段；缺失则用 sid）
        sample_ids_sorted = sorted(paired_sids)
        sample_names = []
        sample_groups = []
        bmeta_samples = {s['id']: s for s in bmeta.get('samples', [])}
        for sid in sample_ids_sorted:
            entry = bmeta_samples.get(sid, {})
            sample_names.append(entry.get('name') or sid)
            sample_groups.append(entry.get('group') or '')

        cancer_mat = pd.DataFrame(index=common_mz, columns=sample_ids_sorted, dtype=float)
        para_mat = pd.DataFrame(index=common_mz, columns=sample_ids_sorted, dtype=float)
        for sid in sample_ids_sorted:
            cancer_mat[sid] = by_sample[sid]['cancer'].reindex(common_mz).astype(float)
            para_mat[sid] = by_sample[sid]['paracancer'].reindex(common_mz).astype(float)

        cohorts[tname] = {
            'cancer_mat'  : cancer_mat,
            'para_mat'    : para_mat,
            'sample_ids'  : sample_ids_sorted,
            'sample_names': sample_names,
            'sample_groups': sample_groups,
            'n_samples'   : len(sample_ids_sorted),
        }

    return cohorts


def _mz_sort_key(s):
    """m/z 列名按数值排序；非数值时回退字母序。"""
    try:
        return (0, float(str(s).split('|')[0]))
    except (ValueError, TypeError):
        return (1, str(s))


# ─── 状态报告（供 UI 展示"队列里有什么"）────────────────────────────────

def cohort_status(batch_dir: str) -> dict:
    """
    诊断每个样本的就绪状态，便于 UI 提示用户哪些样本还差什么。

    返回
    ----
    {
      'samples': [
        {
          'id'       : sample_id,
          'name'     : 显示名,
          'group'    : 用户标的 cancer/control（可空）,
          'has_pb'   : 是否有 pseudobulk_all.csv,
          'tissues'  : 该样本贡献了哪些 (tissue, types) 组合,
        }, ...
      ],
      'tissues_ready': {
        tissue_name: {
          'paired_samples': N (同时含 cancer+paracancer),
          'total_samples' : 出现该 tissue 的样本总数,
        }
      }
    }
    """
    bmeta = _load_batch_meta(batch_dir)
    samples = bmeta.get('samples', [])
    out_samples = []
    tissues_ready: dict[str, dict[str, set]] = {}

    for s in samples:
        sid = s['id']
        sdir = os.path.join(batch_dir, 'samples', sid)
        pb = _load_sample_pseudobulk(sdir)
        info = {
            'id'    : sid,
            'name'  : s.get('name') or sid,
            'group' : s.get('group') or '',
            'has_pb': pb is not None,
            'tissues': [],
        }
        if pb is not None and 'tissue_name' in pb.columns and 'region_type' in pb.columns:
            grp = pb.groupby(['tissue_name', 'region_type']).size()
            for (tname, rtype), _ in grp.items():
                info['tissues'].append({'tissue': tname, 'type': rtype})
                tissues_ready.setdefault(tname, {'paired_set': set(), 'all_set': set()})
                tissues_ready[tname]['all_set'].add(sid)
                # 标记该样本在该 tissue 下的 type
                tissues_ready[tname].setdefault(sid, set()).add(rtype)
        out_samples.append(info)

    # 推导 paired_samples
    final_tissues = {}
    for tname, d in tissues_ready.items():
        paired = sum(
            1 for k, v in d.items()
            if isinstance(v, set) and {'cancer', 'paracancer'}.issubset(v)
        )
        final_tissues[tname] = {
            'paired_samples': paired,
            'total_samples' : len(d.get('all_set', set())),
        }

    return {
        'samples'       : out_samples,
        'tissues_ready' : final_tissues,
    }
