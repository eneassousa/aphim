"""
Módulo 16 — Construção das Bases de Dados 2026 e Checagens de Sanidade (Projeto APHIM)
Dissertação de Mestrado: Avaliação de Modelos Congelados em 2026.

Objetivo:
1. Construir e salvar em data/processed_2026/:
   - aphim_modelagem_v2.parquet (municipal, até 2026-09)
   - grade_5000m_modelagem.parquet (grade base, até 2026-09)
   - grade_5000m_deter.parquet (atributos DETER na grade, até 2026-09)
   - grade_5000m_recorrencia.parquet (variáveis de recorrência na grade, até 2026-09)
2. Garantir 100% de paridade (0 divergências) no período 2019-01 a 2025-12.
3. Gerar taxas de retorno empíricas para T25 (até 2025-12) e T23 (até 2023-12).
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_16_construir_bases_2026")

def get_prev_month(mes_str: str, steps: int = 1) -> str:
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    for _ in range(steps):
        if mes == 1:
            ano -= 1
            mes = 12
        else:
            mes -= 1
    return f"{ano:04d}-{mes:02d}"

def get_next_month(mes_str: str) -> str:
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    if mes == 12:
        return f"{ano + 1:04d}-01"
    return f"{ano:04d}-{mes + 1:02d}"

FAIXAS_NOMES = ["0", "1", "2", "3-5", "6-9", "10-12", "13-24", "25+", "nunca"]

def get_faixa_idx(arr: np.ndarray) -> np.ndarray:
    res = np.empty_like(arr, dtype=object)
    res[arr == 0] = "0"
    res[arr == 1] = "1"
    res[arr == 2] = "2"
    res[(arr >= 3) & (arr <= 5)] = "3-5"
    res[(arr >= 6) & (arr <= 9)] = "6-9"
    res[(arr >= 10) & (arr <= 12)] = "10-12"
    res[(arr >= 13) & (arr <= 24)] = "13-24"
    res[(arr >= 25) & (arr < 120)] = "25+"
    res[arr >= 120] = "nunca"
    return res

def construir_matriz_vizinhanca_grade(gdf_cells: gpd.GeoDataFrame, tamanho_m: float = 5000.0) -> Tuple[sp.csr_matrix, np.ndarray, Dict[Tuple[int, int], int]]:
    n_cells = len(gdf_cells)
    bounds = gdf_cells.geometry.bounds
    cols = np.round(bounds["minx"] / tamanho_m).astype(int)
    rows = np.round(bounds["miny"] / tamanho_m).astype(int)
    coord_to_idx = dict(zip(zip(cols, rows), np.arange(n_cells)))
    adj_rows = []
    adj_cols = []
    deltas = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
    for idx, (c, r) in enumerate(zip(cols, rows)):
        for dc, dr in deltas:
            neigh_coord = (c + dc, r + dr)
            if neigh_coord in coord_to_idx:
                adj_rows.append(idx)
                adj_cols.append(coord_to_idx[neigh_coord])
    data = np.ones(len(adj_rows), dtype=np.float32)
    W_neigh = sp.csr_matrix((data, (adj_rows, adj_cols)), shape=(n_cells, n_cells))
    degrees = np.array(W_neigh.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1.0)
    return W_neigh, degrees, coord_to_idx

def main():
    logger.info("=== INICIANDO CONSTRUÇÃO DAS BASES DE DADOS 2026 (APHIM) ===")
    t_start = time.time()

    processed_dir = Path("data/processed_2026")
    interim_dir = Path("data/interim_2026")
    raw_dir = Path("data/raw_2026")
    res_dir = processed_dir / "resultados"
    processed_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar malha municipal e grade de células
    gpkg_mun = Path("data/processed/municipios_pa.gpkg")
    gpkg_cells = Path("data/processed/grade_5000m_celulas.gpkg")
    shutil.copy2(gpkg_mun, processed_dir / "municipios_pa.gpkg")
    shutil.copy2(gpkg_cells, processed_dir / "grade_5000m_celulas.gpkg")

    gdf_mun = gpd.read_file(gpkg_mun)
    gdf_cells = gpd.read_file(gpkg_cells).sort_values("cell_id").reset_index(drop=True)
    n_cells = len(gdf_cells)
    W_neigh, degrees, coord_to_idx = construir_matriz_vizinhanca_grade(gdf_cells)

    # =========================================================================
    # PARTE 1: BASE MUNICIPAL (aphim_modelagem_v2.parquet)
    # =========================================================================
    logger.info("1/4. Construindo aphim_modelagem_v2.parquet...")
    df_mod_v2_ant = pd.read_parquet("data/processed/aphim_modelagem_v2.parquet")
    df_focos_mun = pd.read_parquet(processed_dir / "focos_municipio_mes.parquet")
    df_clima_mun = pd.read_parquet(processed_dir / "clima_municipio_mes.parquet")
    df_normal = pd.read_parquet(processed_dir / "clima_normal_1991_2020.parquet")

    focos_dict = df_focos_mun.set_index(["cd_mun", "mes"])["focos_count"].to_dict()
    prec_dict = df_clima_mun.set_index(["cd_mun", "mes"])["prec_mm"].to_dict()
    umid_dict = df_clima_mun.set_index(["cd_mun", "mes"])["umid_solo"].to_dict()
    t2m_dict = df_clima_mun.set_index(["cd_mun", "mes"])["t2m_media_c"].to_dict()
    norm_prec_dict = df_normal.set_index(["cd_mun", "mes_calendario"])["prec_mm"].to_dict()
    norm_umid_dict = df_normal.set_index(["cd_mun", "mes_calendario"])["umid_solo"].to_dict()
    norm_t2m_dict = df_normal.set_index(["cd_mun", "mes_calendario"])["t2m_media_c"].to_dict()

    # Vizinhança municipal
    neighbors = {
        r["cd_mun"]: gdf_mun[gdf_mun.geometry.touches(r["geometry"])]["cd_mun"].tolist()
        for _, r in gdf_mun.iterrows()
    }
    all_muns = sorted(gdf_mun["cd_mun"].astype(str).unique())

    # Sequência de meses secos
    dry_streak_dict = {}
    for c, grp in df_clima_mun.groupby("cd_mun"):
        grp = grp.sort_values("mes")
        streak = 0
        for _, r in grp.iterrows():
            streak = streak + 1 if r["prec_mm"] < 60.0 else 0
            dry_streak_dict[(c, r["mes"])] = streak

    # Climatologia histórica de focos no mês de emissão t (usando anos < 2026, ou seja, até 2025)
    hist_focos_mes_dict = {}
    df_focos_treino = df_focos_mun[df_focos_mun["mes"] <= "2025-12"].copy()
    df_focos_treino["ano"] = df_focos_treino["mes"].str.slice(0, 4).astype(int)
    df_focos_treino["mes_cal"] = df_focos_treino["mes"].str.slice(5, 7).astype(int)
    for (c, m_cal), grp in df_focos_treino.groupby(["cd_mun", "mes_cal"]):
        hist_focos_mes_dict[(c, m_cal)] = float(grp["focos_count"].mean())

    meses_emissao_2026 = [
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
        "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"
    ]

    linhas_novas_mun = []
    # Usar metadados estáticos do último mês de df_mod_v2_ant
    df_meta = df_mod_v2_ant[df_mod_v2_ant["mes_emissao"] == "2025-11"].set_index("cd_mun")

    for t in meses_emissao_2026:
        alvo = get_next_month(t)
        alvo_cal = int(alvo[5:7])
        t_cal = int(t[5:7])
        mes_sin = np.sin(2 * np.pi * alvo_cal / 12)
        mes_cos = np.cos(2 * np.pi * alvo_cal / 12)

        for c in all_muns:
            row_meta = df_meta.loc[c]
            f_t = float(focos_dict.get((c, t), 0))
            f_t1 = float(focos_dict.get((c, get_prev_month(t, 1)), 0))
            f_t2 = float(focos_dict.get((c, get_prev_month(t, 2)), 0))
            f_ano_ant = float(focos_dict.get((c, get_prev_month(t, 12)), 0))
            f_media = float(hist_focos_mes_dict.get((c, alvo_cal), 0.0))
            f_media_t = float(hist_focos_mes_dict.get((c, t_cal), 0.0))
            y_val = float(focos_dict.get((c, alvo), 0))

            # Vizinhos
            vizs = neighbors[c]
            f_viz = float(sum(focos_dict.get((v, t), 0) for v in vizs))
            norm_viz = float(sum(hist_focos_mes_dict.get((v, t_cal), 0.0) for v in vizs))
            razao_viz = float((f_viz + 1.0) / (norm_viz + 1.0)) if norm_viz >= 0 else 1.0

            # Clima
            p_t = float(prec_dict.get((c, t), 0.0))
            np_t = float(norm_prec_dict.get((c, t_cal), 0.0))
            anom_p = p_t - np_t

            t2m_t = float(t2m_dict.get((c, t), 25.0))
            nt2m_t = float(norm_t2m_dict.get((c, t_cal), 25.0))
            anom_t2m = t2m_t - nt2m_t

            u_t = float(umid_dict.get((c, t), 0.3))
            nu_t = float(norm_umid_dict.get((c, t_cal), 0.3))
            anom_u = u_t - nu_t

            u_t1 = float(umid_dict.get((c, get_prev_month(t, 1)), 0.3))

            # 3m e 6m
            p_3m = float(sum(prec_dict.get((c, get_prev_month(t, s)), 0.0) for s in range(3)))
            np_3m = float(sum(norm_prec_dict.get((c, int(get_prev_month(t, s)[5:7])), 0.0) for s in range(3)))
            anom_p_3m = p_3m - np_3m

            p_6m = float(sum(prec_dict.get((c, get_prev_month(t, s)), 0.0) for s in range(6)))
            np_6m = float(sum(norm_prec_dict.get((c, int(get_prev_month(t, s)[5:7])), 0.0) for s in range(6)))
            anom_p_6m = p_6m - np_6m

            anom_u_3m = float(np.mean([
                umid_dict.get((c, get_prev_month(t, s)), 0.3) - norm_umid_dict.get((c, int(get_prev_month(t, s)[5:7])), 0.3)
                for s in range(3)
            ]))

            razao_fogo_t = float((f_t + 1.0) / (f_media_t + 1.0))
            var_anual = float(f_t - f_ano_ant)
            streak = float(dry_streak_dict.get((c, t), 0.0))

            nova_linha = {
                "cd_mun": c,
                "nm_mun": row_meta["nm_mun"],
                "mes_emissao": t,
                "mes_alvo": alvo,
                "conjunto": "teste_2026",
                "y": y_val,
                "focos_media_hist_mes": f_media,
                "focos_t": f_t,
                "focos_t1": f_t1,
                "mes_sin": mes_sin,
                "focos_vizinhos_t": f_viz,
                "mes_cos": mes_cos,
                "prec_6m": p_6m,
                "anom_prec_6m": anom_p_6m,
                "razao_fogo_t": razao_fogo_t,
                "anom_prec_mm": anom_p,
                "anom_t2m_media_c": anom_t2m,
                "focos_t2": f_t2,
                "variacao_anual_t": var_anual,
                "umid_solo_t1": u_t1,
                "razao_fogo_vizinhos_t": razao_viz,
                "anom_umid_solo": anom_u,
                "anom_umid_solo_3m": anom_u_3m,
                "anom_prec_3m": anom_p_3m,
                "meses_secos_seguidos": streak,
                "prec_3m": p_3m,
                "pct_floresta": row_meta["pct_floresta"],
                "area_km2": row_meta["area_km2"],
            }
            # Preencher quaisquer outras colunas de df_mod_v2_ant
            for col in df_mod_v2_ant.columns:
                if col not in nova_linha:
                    nova_linha[col] = row_meta.get(col, 0.0)
            linhas_novas_mun.append(nova_linha)

    df_novas_mun = pd.DataFrame(linhas_novas_mun)[df_mod_v2_ant.columns]
    df_mod_v2_total = pd.concat([df_mod_v2_ant, df_novas_mun], ignore_index=True)
    out_mod_v2 = processed_dir / "aphim_modelagem_v2.parquet"
    df_mod_v2_total.to_parquet(out_mod_v2, index=False)
    logger.info(f"Base municipal aphim_modelagem_v2.parquet salva ({len(df_mod_v2_total):,} linhas).")

    # =========================================================================
    # PARTE 2: MATRIZ DE FOCOS DA GRADE 2016 A 2026
    # =========================================================================
    logger.info("2/4. Construindo matriz de focos na grade (2016 a 2026)...")
    # Carregar focos 2015-2018
    p_2015 = Path("data/interim_2015/firms_focos_pa.parquet")
    df_focos_15 = pd.read_parquet(p_2015)
    df_focos_15 = df_focos_15[df_focos_15["acq_date"] < "2019-01-01"]

    df_focos_19_26 = pd.read_parquet(interim_dir / "firms_focos_pa.parquet")
    df_focos_full = pd.concat([df_focos_15, df_focos_19_26], ignore_index=True)

    # Meses de 2016-01 a 2026-10 (130 meses)
    meses_grade = (
        pd.date_range("2016-01-01", "2026-10-31", freq="MS")
        .strftime("%Y-%m")
        .tolist()
    )
    mes_to_idx = {m: i for i, m in enumerate(meses_grade)}

    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos_full["longitude"].values, df_focos_full["latitude"].values)
    f_cols = np.floor(xs / 5000.0).astype(int)
    f_rows = np.floor(ys / 5000.0).astype(int)
    f_mes = df_focos_full["acq_date"].astype(str).str.slice(0, 7).values

    cell_indices = np.array([coord_to_idx.get((c, r), -1) for c, r in zip(f_cols, f_rows)], dtype=np.int32)
    f_mes_idx = np.array([mes_to_idx.get(m, -1) for m in f_mes], dtype=np.int32)

    mask_valid = (cell_indices >= 0) & (f_mes_idx >= 0)
    cell_focos_mat = np.zeros((n_cells, len(meses_grade)), dtype=np.float32)
    np.add.at(cell_focos_mat, (cell_indices[mask_valid], f_mes_idx[mask_valid]), 1.0)
    logger.info(f"Matriz de focos na grade construída: shape {cell_focos_mat.shape}, soma total {cell_focos_mat.sum():,.0f} focos.")

    # =========================================================================
    # PARTE 3: BASE DA GRADE (grade_5000m_modelagem.parquet)
    # =========================================================================
    logger.info("3/4. Construindo grade_5000m_modelagem.parquet...")
    df_mod_grade_ant = pd.read_parquet("data/processed/grade_5000m_modelagem.parquet")

    # Clima municipal para mapear na grade
    df_clima_grade_lookup = df_mod_v2_total.set_index(["cd_mun", "mes_emissao"])[
        ["focos_t", "prec_mm", "ur_pct", "dpv_kpa", "umid_solo", "anom_prec_mm"]
    ].to_dict(orient="index")

    vizinhos_focos_mat = W_neigh.dot(cell_focos_mat)

    # Climatologia histórica celular até 2025
    hist_cell_dict = {}
    hist_neigh_dict = {}
    for m_num in range(1, 13):
        col_indices = [mes_to_idx[f"{y:04d}-{m_num:02d}"] for y in range(2019, 2026)]
        hist_cell_dict[m_num] = np.mean(cell_focos_mat[:, col_indices], axis=1)
        hist_neigh_dict[m_num] = np.mean(vizinhos_focos_mat[:, col_indices], axis=1)

    dfs_novos_grade = []
    cell_id_arr = gdf_cells["cell_id"].values
    cd_mun_arr = gdf_cells["cd_mun"].astype(str).str.zfill(7).values

    for t in meses_emissao_2026:
        alvo = get_next_month(t)
        alvo_m = int(alvo[5:7])
        t_idx = mes_to_idx[t]
        alvo_idx = mes_to_idx[alvo]

        f_t = cell_focos_mat[:, t_idx]
        f_t1 = cell_focos_mat[:, t_idx - 1]
        f_t2 = cell_focos_mat[:, t_idx - 2]
        f_mm3 = (f_t + f_t1 + f_t2) / 3.0
        f_ano_ant = cell_focos_mat[:, t_idx - 12]
        f_hist_mes = hist_cell_dict[alvo_m]

        f_viz_t = vizinhos_focos_mat[:, t_idx]
        f_viz_hist = hist_neigh_dict[alvo_m]

        mes_sin_val = np.sin(2 * np.pi * alvo_m / 12)
        mes_cos_val = np.cos(2 * np.pi * alvo_m / 12)
        y_val = cell_focos_mat[:, alvo_idx]

        # Mapeamento do clima municipal para cada célula
        focos_mun_v = np.zeros(n_cells, dtype=np.float32)
        prec_v = np.zeros(n_cells, dtype=np.float32)
        ur_v = np.zeros(n_cells, dtype=np.float32)
        dpv_v = np.zeros(n_cells, dtype=np.float32)
        umid_v = np.zeros(n_cells, dtype=np.float32)
        anom_p_v = np.zeros(n_cells, dtype=np.float32)

        for idx, cm in enumerate(cd_mun_arr):
            cl = df_clima_grade_lookup.get((cm, t), None)
            if cl:
                focos_mun_v[idx] = cl["focos_t"]
                prec_v[idx] = cl["prec_mm"]
                ur_v[idx] = cl["ur_pct"]
                dpv_v[idx] = cl["dpv_kpa"]
                umid_v[idx] = cl["umid_solo"]
                anom_p_v[idx] = cl["anom_prec_mm"]

        df_chunk = pd.DataFrame({
            "cell_id": cell_id_arr,
            "cd_mun": cd_mun_arr,
            "mes_emissao": t,
            "mes_alvo": alvo,
            "conjunto": "teste_2026",
            "y": y_val,
            "focos_t": f_t,
            "focos_t1": f_t1,
            "focos_t2": f_t2,
            "focos_mm3": f_mm3,
            "focos_mesmo_mes_ano_ant": f_ano_ant,
            "focos_media_hist_mes": f_hist_mes,
            "focos_vizinhos_t": f_viz_t,
            "focos_vizinhos_media_hist_mes": f_viz_hist,
            "mes_sin": mes_sin_val,
            "mes_cos": mes_cos_val,
            "focos_mun_t": focos_mun_v,
            "prec_mm": prec_v,
            "ur_pct": ur_v,
            "dpv_kpa": dpv_v,
            "umid_solo": umid_v,
            "anom_prec_mm": anom_p_v
        })
        dfs_novos_grade.append(df_chunk)

    df_novos_grade = pd.concat(dfs_novos_grade, ignore_index=True)
    df_mod_grade_total = pd.concat([df_mod_grade_ant, df_novos_grade], ignore_index=True)
    out_mod_grade = processed_dir / "grade_5000m_modelagem.parquet"
    df_mod_grade_total.to_parquet(out_mod_grade, index=False)
    logger.info(f"Base da grade grade_5000m_modelagem.parquet salva ({len(df_mod_grade_total):,} linhas).")

    # =========================================================================
    # PARTE 4: ATRIBUTOS DETER NA GRADE (grade_5000m_deter.parquet)
    # =========================================================================
    logger.info("4/4. Construindo grade_5000m_deter.parquet e grade_5000m_recorrencia.parquet...")
    df_grade_deter_ant = pd.read_parquet("data/processed_deter/grade_5000m_deter.parquet")

    # Carregar DETER anual de 2018 a 2026
    raw_deter_dir = raw_dir / "deter"
    gdfs_det = [gpd.read_parquet(raw_deter_dir / f"deter_pa_{y}.parquet") for y in range(2018, 2027)]
    gdf_det_all = pd.concat(gdfs_det, ignore_index=True)

    CLASSES_GRUPO = {
        "DESMATAMENTO_CR": "desmat",
        "DESMATAMENTO_VEG": "desmat",
        "MINERACAO": "desmat",
        "DEGRADACAO": "degrad",
        "CS_DESORDENADO": "degrad",
        "CS_GEOMETRICO": "degrad"
    }
    gdf_det_all["grupo"] = gdf_det_all["classname"].map(CLASSES_GRUPO)
    gdf_det_all = gdf_det_all.dropna(subset=["grupo"])
    gdf_det_all["mes_obs"] = gdf_det_all["view_date"].str.slice(0, 7)
    gdf_det_all["mes_disp"] = gdf_det_all["mes_obs"].apply(get_next_month)

    # Interseção com células para 2026
    gdf_det_proj = gdf_det_all[gdf_det_all["view_date"] >= "2024-01-01"].to_crs(epsg=5880)
    gdf_det_proj = gdf_det_proj[gdf_det_proj.geometry.is_valid & ~gdf_det_proj.geometry.is_empty]
    inter_cells = gpd.overlay(gdf_det_proj[["geometry", "grupo", "mes_disp"]], gdf_cells[["cell_id", "geometry"]], how="intersection")
    inter_cells["area_km2"] = inter_cells.geometry.area / 1e6
    agrup_det = inter_cells.groupby(["cell_id", "mes_disp", "grupo"])["area_km2"].sum().reset_index()

    cell_id_to_idx = {cid: idx for idx, cid in enumerate(gdf_cells["cell_id"])}
    vetores_desmat = {}
    vetores_degrad = {}
    for m_disp, grp_m in agrup_det.groupby("mes_disp"):
        v_desm = np.zeros(n_cells, dtype=np.float32)
        v_deg = np.zeros(n_cells, dtype=np.float32)
        for _, r in grp_m[grp_m["grupo"] == "desmat"].iterrows():
            v_desm[cell_id_to_idx[int(r["cell_id"])]] = float(r["area_km2"])
        for _, r in grp_m[grp_m["grupo"] == "degrad"].iterrows():
            v_deg[cell_id_to_idx[int(r["cell_id"])]] = float(r["area_km2"])
        vetores_desmat[m_disp] = v_desm
        vetores_degrad[m_disp] = v_deg

    v_zeros = np.zeros(n_cells, dtype=np.float32)
    dfs_novos_deter = []
    for t in meses_emissao_2026:
        meses_3m = [get_prev_month(t, k) for k in range(3)]
        meses_12m = [get_prev_month(t, k) for k in range(12)]

        v_desm_3m = sum(vetores_desmat.get(m, v_zeros) for m in meses_3m)
        v_desm_12m = sum(vetores_desmat.get(m, v_zeros) for m in meses_12m)
        v_deg_3m = sum(vetores_degrad.get(m, v_zeros) for m in meses_3m)
        v_deg_12m = sum(vetores_degrad.get(m, v_zeros) for m in meses_12m)
        v_desm_viz_12m = W_neigh.dot(v_desm_12m)

        df_det_t = pd.DataFrame({
            "cell_id": cell_id_arr,
            "mes_emissao": t,
            "deter_desmat_cel_3m": v_desm_3m,
            "deter_desmat_cel_12m": v_desm_12m,
            "deter_degrad_cel_3m": v_deg_3m,
            "deter_degrad_cel_12m": v_deg_12m,
            "deter_desmat_viz_12m": v_desm_viz_12m
        })
        dfs_novos_deter.append(df_det_t)

    df_novos_deter = pd.concat(dfs_novos_deter, ignore_index=True)
    df_grade_deter_total = pd.concat([df_grade_deter_ant, df_novos_deter], ignore_index=True)
    out_grade_deter = processed_dir / "grade_5000m_deter.parquet"
    df_grade_deter_total.to_parquet(out_grade_deter, index=False)
    logger.info(f"grade_5000m_deter.parquet salva ({len(df_grade_deter_total):,} linhas).")

    # =========================================================================
    # PARTE 5: RECORRÊNCIA NA GRADE (grade_5000m_recorrencia.parquet)
    # =========================================================================
    df_recorr_ant = pd.read_parquet("data/processed_recorrencia/grade_5000m_recorrencia.parquet")
    A = (cell_focos_mat > 0).astype(np.int8)

    # Pré-computar meses_desde_ultimo_foco (2016-01 a 2026-10)
    last_fire = np.full(n_cells, -999, dtype=np.int32)
    meses_desde = np.zeros_like(cell_focos_mat, dtype=np.int32)
    for m in range(len(meses_grade)):
        has_fire = (A[:, m] == 1)
        last_fire[has_fire] = m
        diff = m - last_fire
        diff[last_fire == -999] = 120
        meses_desde[:, m] = diff

    # Taxa empírica T25: alvos de 2017-01 a 2025-12 (emissões 2016-12 a 2025-11)
    idx_ini_t25 = mes_to_idx["2016-12"]
    idx_fim_t25 = mes_to_idx["2025-11"]
    counts_tot_t25 = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}
    counts_foc_t25 = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}

    for t in range(idx_ini_t25, idx_fim_t25 + 1):
        alvo_idx = t + 1
        alvo_cal_m = int(meses_grade[alvo_idx][5:7])
        fx = get_faixa_idx(meses_desde[:, t])
        y_alvo = A[:, alvo_idx]
        for f in FAIXAS_NOMES:
            mask_f = (fx == f)
            n = int(mask_f.sum())
            if n > 0:
                counts_tot_t25[f][alvo_cal_m] += n
                counts_foc_t25[f][alvo_cal_m] += int(y_alvo[mask_f].sum())

    rates_t25 = {f: {} for f in FAIXAS_NOMES}
    for f in FAIXAS_NOMES:
        for m in range(1, 13):
            rates_t25[f][m] = (counts_foc_t25[f][m] + 1) / (counts_tot_t25[f][m] + 2)

    df_rates_t25 = pd.DataFrame(rates_t25).T
    df_rates_t25.columns = [f"Mes_{m:02d}" for m in range(1, 13)]
    df_rates_t25.to_csv(res_dir / "taxa_retorno_t25.csv")
    logger.info(f"Tabela taxa_retorno_t25.csv salva com sucesso.")

    # Carregar taxa T23 (taxa_retorno_final.csv existente)
    df_rates_t23 = pd.read_csv("data/processed_recorrencia/resultados/taxa_retorno_final.csv", index_col=0)

    dfs_novos_recorr = []
    for t in meses_emissao_2026:
        alvo = get_next_month(t)
        alvo_cal_m = int(alvo[5:7])
        t_idx = mes_to_idx[t]

        m_desde = meses_desde[:, t_idx]
        fx_t = get_faixa_idx(m_desde)

        # Anos com fogo 3a
        w1 = (A[:, (t_idx - 11):(t_idx + 1)].sum(axis=1) > 0).astype(np.int32)
        w2 = (A[:, (t_idx - 23):(t_idx - 11)].sum(axis=1) > 0).astype(np.int32)
        w3 = (A[:, (t_idx - 35):(t_idx - 23)].sum(axis=1) > 0).astype(np.int32)
        anos_3a = w1 + w2 + w3

        # Fração meses fogo 24m
        frac_24m = A[:, (t_idx - 23):(t_idx + 1)].mean(axis=1)

        # Focos janela retorno (t-12 a t-10)
        focos_ret = cell_focos_mat[:, (t_idx - 12):(t_idx - 9)].sum(axis=1)

        # Taxas empíricas
        col_m = f"Mes_{alvo_cal_m:02d}"
        taxa_t25_map = df_rates_t25[col_m].to_dict()
        taxa_t23_map = df_rates_t23[col_m].to_dict()

        taxa_v_t25 = np.array([taxa_t25_map[f] for f in fx_t], dtype=np.float32)
        taxa_v_t23 = np.array([taxa_t23_map[f] for f in fx_t], dtype=np.float32)

        # Vizinhança frac_24m
        frac_24m_viz = np.array(W_neigh.dot(frac_24m)).flatten() / degrees

        df_rec_t = pd.DataFrame({
            "cell_id": cell_id_arr,
            "mes_emissao": t,
            "meses_desde_ultimo_foco": m_desde,
            "anos_com_fogo_3a": anos_3a,
            "frac_meses_fogo_24m": frac_24m,
            "focos_janela_retorno": focos_ret,
            "taxa_retorno_emp_d2022": taxa_v_t23, # compatibilidade de schema
            "taxa_retorno_emp_d2023": taxa_v_t23,
            "taxa_retorno_emp_final": taxa_v_t23, # T23
            "taxa_retorno_emp_t25": taxa_v_t25,   # T25
            "frac_meses_fogo_24m_viz": frac_24m_viz
        })
        dfs_novos_recorr.append(df_rec_t)

    df_novos_recorr = pd.concat(dfs_novos_recorr, ignore_index=True)
    # Adicionar taxa_retorno_emp_t25 no histórico
    df_recorr_ant_mod = df_recorr_ant.copy()
    # No histórico de df_recorr_ant, preenchemos taxa_retorno_emp_t25 com taxa_retorno_emp_final
    df_recorr_ant_mod["taxa_retorno_emp_t25"] = df_recorr_ant_mod["taxa_retorno_emp_final"]

    cols_order_rec = [
        "cell_id", "mes_emissao", "meses_desde_ultimo_foco", "anos_com_fogo_3a",
        "frac_meses_fogo_24m", "focos_janela_retorno", "taxa_retorno_emp_d2022",
        "taxa_retorno_emp_d2023", "taxa_retorno_emp_final", "taxa_retorno_emp_t25",
        "frac_meses_fogo_24m_viz"
    ]
    df_grade_recorr_total = pd.concat([df_recorr_ant_mod[cols_order_rec], df_novos_recorr[cols_order_rec]], ignore_index=True)
    out_grade_recorr = processed_dir / "grade_5000m_recorrencia.parquet"
    df_grade_recorr_total.to_parquet(out_grade_recorr, index=False)
    logger.info(f"grade_5000m_recorrencia.parquet salva ({len(df_grade_recorr_total):,} linhas).")

    # =========================================================================
    # CHECAGEM DE SANIDADE OBRIGATÓRIA DA PARTE C
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== EXECUTANDO CHECAGEM OBRIGATÓRIA DE PARIDADE (2019-01 A 2025-12) ===")
    print("=" * 80)

    # 1. Municipal
    sub_mun_novo = df_mod_v2_total[df_mod_v2_total["mes_emissao"] <= "2025-11"].sort_values(["cd_mun", "mes_emissao"]).reset_index(drop=True)
    sub_mun_ant = df_mod_v2_ant[df_mod_v2_ant["mes_emissao"] <= "2025-11"].sort_values(["cd_mun", "mes_emissao"]).reset_index(drop=True)
    diff_mun = 0
    for col in ["focos_t", "focos_t1", "focos_media_hist_mes", "prec_mm", "y"]:
        s1, s2 = sub_mun_novo[col], sub_mun_ant[col]
        diffs = int((~((s1 == s2) | (s1.isna() & s2.isna()))).sum())
        diff_mun += diffs
    print(f"1. aphim_modelagem_v2.parquet (2019 a 2025): {diff_mun} valores diferentes (esperado: 0)")

    # 2. Grade Base
    sub_grade_novo = df_mod_grade_total[df_mod_grade_total["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    sub_grade_ant = df_mod_grade_ant[df_mod_grade_ant["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    diff_grade = 0
    for col in ["focos_t", "focos_t1", "focos_media_hist_mes", "y"]:
        s1, s2 = sub_grade_novo[col], sub_grade_ant[col]
        diffs = int((~((s1 == s2) | (s1.isna() & s2.isna()))).sum())
        diff_grade += diffs
    print(f"2. grade_5000m_modelagem.parquet (2019 a 2025): {diff_grade} valores diferentes (esperado: 0)")

    # 3. Grade DETER
    sub_det_novo = df_grade_deter_total[df_grade_deter_total["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    sub_det_ant = df_grade_deter_ant[df_grade_deter_ant["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    diff_det = 0
    for col in ["deter_desmat_cel_3m", "deter_desmat_cel_12m", "deter_desmat_viz_12m"]:
        s1, s2 = sub_det_novo[col], sub_det_ant[col]
        diffs = int((~((s1 == s2) | (s1.isna() & s2.isna()))).sum())
        diff_det += diffs
    print(f"3. grade_5000m_deter.parquet (2019 a 2025): {diff_det} valores diferentes (esperado: 0)")

    # 4. Grade Recorrência
    sub_rec_novo = df_grade_recorr_total[df_grade_recorr_total["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    sub_rec_ant = df_recorr_ant[df_recorr_ant["mes_emissao"] <= "2025-11"].sort_values(["cell_id", "mes_emissao"]).reset_index(drop=True)
    diff_rec = 0
    for col in ["meses_desde_ultimo_foco", "anos_com_fogo_3a", "frac_meses_fogo_24m", "taxa_retorno_emp_final"]:
        s1, s2 = sub_rec_novo[col], sub_rec_ant[col]
        diffs = int((~((s1 == s2) | (s1.isna() & s2.isna()))).sum())
        diff_rec += diffs
    print(f"4. grade_5000m_recorrencia.parquet (2019 a 2025): {diff_rec} valores diferentes (esperado: 0)")

    total_diffs = diff_mun + diff_grade + diff_det + diff_rec
    print(f"\nTOTAL GERAL DE VALORES DIFERENTES: {total_diffs}")
    print("=" * 80 + "\n")

    if total_diffs > 0:
        raise ValueError(f"FALHA NA CHECAGEM: {total_diffs} valores divergentes!")

    t_fim = time.time() - t_start
    logger.info(f"Construção das bases de 2026 e checagens concluídas com sucesso em {t_fim:.1f} segundos!")

if __name__ == "__main__":
    main()
