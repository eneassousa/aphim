"""
Módulo 11 — Construção dos Tensores Espaço-Temporais da Grade de 5 km (Projeto APHIM)
Dissertação de Mestrado: Comparação com Modelo Espaço-Temporal Profundo (ConvLSTM).

Objetivos:
1. Reconstruir a matriz bidimensional da grade regular de 5 km a partir de grade_5000m_celulas.gpkg:
   - col = round(minx / 5000) e row = round(miny / 5000)
   - Normalizar col e row para começar em 0
   - Validar dimensões (H=277, W=287) e contagem exata de 50.736 células ativas no Pará.
2. Montar para cada mês de emissão t (2019-01 a 2025-11, 83 meses) um tensor [10, H, W]:
   - Canal 0: focos_t (log1p)
   - Canal 1: focos_media_hist_mes (log1p)
   - Canal 2: deter_desmat_cel_12m (log1p)
   - Canal 3: deter_degrad_cel_12m (log1p)
   - Canal 4: prec_mm (padronizado com média e desvio padrão do conjunto de treino)
   - Canal 5: umid_solo (padronizado com conjunto de treino)
   - Canal 6: anom_prec_mm (padronizado com conjunto de treino)
   - Canal 7: mes_sin do mês-alvo (constante no mapa)
   - Canal 8: mes_cos do mês-alvo (constante no mapa)
   - Canal 9: máscara do Pará (1 para célula pertencente ao Pará, 0 fora)
   - Alvo y: mapa [H, W] com contagem real de focos em t+1 (coluna y).
3. Checagem de Sanidade:
   - Para 3 meses aleatórios, somar y por município a partir do tensor e comparar com a soma direta
     em grade_5000m_modelagem.parquet. A diferença deve ser exatamente zero.
4. Salvar os tensores compactados em data/processed_convlstm/tensores_5000m.npz.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_11_convlstm_tensores")

CANAIS_NOMES = [
    "focos_t_log1p",
    "focos_media_hist_mes_log1p",
    "deter_desmat_cel_12m_log1p",
    "deter_degrad_cel_12m_log1p",
    "prec_mm_norm",
    "umid_solo_norm",
    "anom_prec_mm_norm",
    "mes_sin",
    "mes_cos",
    "mascara_pa"
]

def main():
    logger.info("=== CONSTRUINDO TENSORES ESPAÇO-TEMPORAIS DA GRADE (PARTE A) ===")
    
    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = Path("data/processed_convlstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"
    grade_mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    grade_det_path = processed_deter_dir / "grade_5000m_deter.parquet"

    # 1. Carregar geometrias e reconstruir coordenadas de grade (row, col)
    logger.info(f"Carregando geometrias de '{gpkg_path}'...")
    gdf_cells = gpd.read_file(gpkg_path)
    gdf_cells["cell_id"] = gdf_cells["cell_id"].astype(np.int32)
    gdf_cells["cd_mun"] = gdf_cells["cd_mun"].astype(str).str.zfill(7)

    bounds = gdf_cells.bounds
    cols_raw = np.round(bounds["minx"] / 5000.0).astype(int)
    rows_raw = np.round(bounds["miny"] / 5000.0).astype(int)

    col_min = cols_raw.min()
    row_min = rows_raw.min()

    gdf_cells["col_norm"] = cols_raw - col_min
    gdf_cells["row_norm"] = rows_raw - row_min

    H = int(gdf_cells["row_norm"].max() + 1)
    W = int(gdf_cells["col_norm"].max() + 1)
    n_cells = len(gdf_cells)

    logger.info(f"Dimensões do mapa: H = {H} linhas, W = {W} colunas.")
    logger.info(f"Células do Pará preenchidas: {n_cells:,} (esperado: 50.736).")
    assert n_cells == 50736, f"Esperado 50.736 células, obtido {n_cells}."
    assert H == 277 and W == 287, f"Esperado H=277, W=287, obtido H={H}, W={W}."

    # Matriz 2D de mapeamento (row, col) -> cell_id (-1 para fora do Pará)
    grid_cell_id = np.full((H, W), -1, dtype=np.int32)
    grid_mask_pa = np.zeros((H, W), dtype=np.float32)

    for _, row in gdf_cells.iterrows():
        r = int(row["row_norm"])
        c = int(row["col_norm"])
        cid = int(row["cell_id"])
        grid_cell_id[r, c] = cid
        grid_mask_pa[r, c] = 1.0

    # 2. Carregar base de modelagem da grade e atributos DETER
    logger.info(f"Carregando base de modelagem de '{grade_mod_path}'...")
    df_mod = pd.read_parquet(grade_mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_mod["cell_id"] = df_mod["cell_id"].astype(np.int32)

    logger.info(f"Carregando atributos DETER de '{grade_det_path}'...")
    df_det = pd.read_parquet(grade_det_path)
    df_det["cell_id"] = df_det["cell_id"].astype(np.int32)

    logger.info("Mesclando atributos DETER...")
    df_mod = df_mod.merge(
        df_det[["cell_id", "mes_emissao", "deter_desmat_cel_12m", "deter_degrad_cel_12m"]],
        on=["cell_id", "mes_emissao"],
        how="left"
    )
    df_mod["deter_desmat_cel_12m"] = df_mod["deter_desmat_cel_12m"].fillna(0.0).astype(np.float32)
    df_mod["deter_degrad_cel_12m"] = df_mod["deter_degrad_cel_12m"].fillna(0.0).astype(np.float32)

    # 3. Estatísticas de normalização exclusivamente no treino (anti-vazamento)
    logger.info("Calculando médias e desvios-padrão das variáveis meteorológicas no treino...")
    df_tr = df_mod[df_mod["conjunto"] == "treino"]
    prec_mean = float(df_tr["prec_mm"].mean())
    prec_std = float(df_tr["prec_mm"].std())
    umid_mean = float(df_tr["umid_solo"].mean())
    umid_std = float(df_tr["umid_solo"].std())
    anom_prec_mean = float(df_tr["anom_prec_mm"].mean())
    anom_prec_std = float(df_tr["anom_prec_mm"].std())

    logger.info(f"  prec_mm: média = {prec_mean:.3f}, desvio = {prec_std:.3f}")
    logger.info(f"  umid_solo: média = {umid_mean:.3f}, desvio = {umid_std:.3f}")
    logger.info(f"  anom_prec_mm: média = {anom_prec_mean:.3f}, desvio = {anom_prec_std:.3f}")

    # Mesclar coordenadas (row_norm, col_norm) no dataframe
    df_coords = gdf_cells[["cell_id", "row_norm", "col_norm"]].copy()
    df_mod = df_mod.merge(df_coords, on="cell_id", how="left")

    # 4. Construção dos Tensores
    meses_emissao = sorted(df_mod["mes_emissao"].unique())
    n_meses = len(meses_emissao)
    logger.info(f"Montando tensores para {n_meses} meses ({meses_emissao[0]} a {meses_emissao[-1]})...")

    # Tensores X: [N_meses, 10, H, W] e Y: [N_meses, H, W]
    X_all = np.zeros((n_meses, len(CANAIS_NOMES), H, W), dtype=np.float32)
    Y_all = np.zeros((n_meses, H, W), dtype=np.float32)
    meses_alvo = []

    for idx, mes_e in enumerate(meses_emissao):
        sub_m = df_mod[df_mod["mes_emissao"] == mes_e]
        mes_a = sub_m["mes_alvo"].iloc[0]
        meses_alvo.append(mes_a)

        r_idx = sub_m["row_norm"].values.astype(int)
        c_idx = sub_m["col_norm"].values.astype(int)

        # Canal 0: focos_t (log1p)
        X_all[idx, 0, r_idx, c_idx] = np.log1p(sub_m["focos_t"].values.astype(np.float32))

        # Canal 1: focos_media_hist_mes (log1p)
        X_all[idx, 1, r_idx, c_idx] = np.log1p(sub_m["focos_media_hist_mes"].values.astype(np.float32))

        # Canal 2: deter_desmat_cel_12m (log1p)
        X_all[idx, 2, r_idx, c_idx] = np.log1p(sub_m["deter_desmat_cel_12m"].values.astype(np.float32))

        # Canal 3: deter_degrad_cel_12m (log1p)
        X_all[idx, 3, r_idx, c_idx] = np.log1p(sub_m["deter_degrad_cel_12m"].values.astype(np.float32))

        # Canal 4: prec_mm (padronizado)
        X_all[idx, 4, r_idx, c_idx] = ((sub_m["prec_mm"].values.astype(np.float32) - prec_mean) / prec_std)

        # Canal 5: umid_solo (padronizado)
        X_all[idx, 5, r_idx, c_idx] = ((sub_m["umid_solo"].values.astype(np.float32) - umid_mean) / umid_std)

        # Canal 6: anom_prec_mm (padronizado)
        X_all[idx, 6, r_idx, c_idx] = ((sub_m["anom_prec_mm"].values.astype(np.float32) - anom_prec_mean) / anom_prec_std)

        # Canal 7: mes_sin (constante nas células do Pará)
        sin_val = float(sub_m["mes_sin"].iloc[0])
        X_all[idx, 7, r_idx, c_idx] = sin_val

        # Canal 8: mes_cos (constante nas células do Pará)
        cos_val = float(sub_m["mes_cos"].iloc[0])
        X_all[idx, 8, r_idx, c_idx] = cos_val

        # Canal 9: máscara do Pará
        X_all[idx, 9] = grid_mask_pa

        # Alvo y: focos reais em t+1
        Y_all[idx, r_idx, c_idx] = sub_m["y"].values.astype(np.float32)

    logger.info(f"Tensores X montados com formato {X_all.shape} e tipo {X_all.dtype}.")
    logger.info(f"Tensores Y montados com formato {Y_all.shape} e tipo {Y_all.dtype}.")

    # 5. Checagem de Sanidade
    logger.info("Executando checagem de sanidade: somas de y por município em 3 meses aleatórios...")
    rng = np.random.RandomState(42)
    meses_checagem = rng.choice(meses_emissao, size=3, replace=False)

    # Mapeamento cell_id -> cd_mun
    cell_to_mun = dict(zip(gdf_cells["cell_id"], gdf_cells["cd_mun"]))
    active_mask = grid_cell_id > 0
    active_rows, active_cols = np.where(active_mask)
    active_cids = grid_cell_id[active_rows, active_cols]
    active_muns = np.array([cell_to_mun[cid] for cid in active_cids])

    for mes_c in meses_checagem:
        idx_c = meses_emissao.index(mes_c)
        y_map = Y_all[idx_c]
        y_cells = y_map[active_rows, active_cols]

        # Soma a partir do tensor
        df_tensor_check = pd.DataFrame({"cd_mun": active_muns, "y": y_cells})
        soma_tensor = df_tensor_check.groupby("cd_mun")["y"].sum().sort_index()

        # Soma direta do parquet original
        df_orig_check = df_mod[df_mod["mes_emissao"] == mes_c]
        soma_orig = df_orig_check.groupby("cd_mun")["y"].sum().sort_index()

        diff_total = float((soma_tensor - soma_orig).abs().sum())
        logger.info(f"  Mês {mes_c}: soma total = {soma_orig.sum():.0f} focos | Diferença absoluta = {diff_total:.6f}")
        if diff_total > 1e-4:
            raise ValueError(f"FALHA NA CHECAGEM DE SANIDADE: Soma de y difere no mês {mes_c}!")

    logger.info("CHECAGEM DE SANIDADE APROVADA: Somas por município são idênticas!")

    # 6. Salvar em tensores_5000m.npz
    out_npz = out_dir / "tensores_5000m.npz"
    logger.info(f"Salvando tensores compactados em '{out_npz}'...")
    np.savez_compressed(
        out_npz,
        X=X_all,
        Y=Y_all,
        meses_emissao=np.array(meses_emissao),
        meses_alvo=np.array(meses_alvo),
        canais=np.array(CANAIS_NOMES),
        grid_cell_id=grid_cell_id,
        grid_mask_pa=grid_mask_pa,
        cell_id=gdf_cells["cell_id"].values,
        row_norm=gdf_cells["row_norm"].values,
        col_norm=gdf_cells["col_norm"].values,
        cd_mun=gdf_cells["cd_mun"].values
    )
    tamanho_mb = out_npz.stat().st_size / (1024 * 1024)
    logger.info(f"Arquivo '{out_npz}' salvo com sucesso! Tamanho: {tamanho_mb:.2f} MB.")

if __name__ == "__main__":
    main()
