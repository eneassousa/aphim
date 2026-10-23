"""
Módulo 08b — Atributos DETER: Recortes Espaciais no Município e na Grade de 5 km (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal para Ação contra Queimadas no Pará.

Objetivos:
1. Recorte Municipal (EPSG:5880):
   - Interseção espacial estrita dos polígonos brutos do DETER (2018–2025) com a malha de municípios.
   - Cálculo de áreas em km² para os grupos 'desmat', 'degrad' e 'cicatriz'.
   - Atribuição do mês de disponibilidade temporal (anti-vazamento: view_date no mês m -> disponível em m+1).
   - Saída: data/processed_deter/deter_municipio_mes.parquet.

2. Engenharia de Atributos Municipais:
   - Para cada município e mês de emissão t (2019-01 a 2025-11):
     * deter_<g>_1m: km² disponíveis em t
     * deter_<g>_3m: soma de t-2 a t
     * deter_<g>_12m: soma de t-11 a t
     * deter_<g>_3m_pct: deter_<g>_3m dividido pela área municipal em %
     (8 atributos para desmat e degrad; cicatriz não entra como feature preditiva).
   - Merge com data/processed/aphim_modelagem_v2.parquet por cd_mun e mes_emissao (zeros explícitos).
   - Saída: data/processed_deter/aphim_modelagem_v2_deter.parquet (exatamente 11.952 linhas).

3. Recorte e Atributos na Grade de 5 km:
   - Interseção com data/processed/grade_5000m_celulas.gpkg (50.736 células).
   - Para cada célula e mês de emissão t:
     * deter_desmat_cel_3m e deter_desmat_cel_12m
     * deter_degrad_cel_3m e deter_degrad_cel_12m
     * deter_desmat_viz_12m: soma em 12 meses nos 8 vizinhos (matriz de adjacência de Moore).
   - Saída: data/processed_deter/grade_5000m_deter.parquet (cell_id, mes_emissao e 5 atributos).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import scipy.sparse as sp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_08b_deter_atributos")

CLASSES_GRUPO = {
    "DESMATAMENTO_CR": "desmat",
    "DESMATAMENTO_VEG": "desmat",
    "MINERACAO": "desmat",
    "DEGRADACAO": "degrad",
    "CS_DESORDENADO": "degrad",
    "CS_GEOMETRICO": "degrad",
    "CICATRIZ_DE_QUEIMADA": "cicatriz"
}

def get_prev_month(mes_str: str, steps: int = 1) -> str:
    """Calcula o mês t - steps no formato AAAA-MM."""
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    for _ in range(steps):
        if mes == 1:
            ano -= 1
            mes = 12
        else:
            mes -= 1
    return f"{ano:04d}-{mes:02d}"

def get_next_month(mes_str: str) -> str:
    """Calcula o mês t + 1 no formato AAAA-MM."""
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    if mes == 12:
        return f"{ano + 1:04d}-01"
    return f"{ano:04d}-{mes + 1:02d}"

def carregar_deter_bruto(raw_dir: Path) -> gpd.GeoDataFrame:
    """Carrega os arquivos anuais do DETER (2018–2025) e garante consistência de CRS e grupos."""
    anos = range(2018, 2026)
    gdfs = []
    for ano in anos:
        p = raw_dir / f"deter_pa_{ano}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Arquivo anual '{p}' não encontrado. Execute coleta 11_deter.py primeiro.")
        logger.info(f"Carregando polígonos brutos de {ano} ('{p}')...")
        gdf_ano = gpd.read_parquet(p)
        if not gdf_ano.empty:
            gdfs.append(gdf_ano)
            
    gdf_all = pd.concat(gdfs, ignore_index=True)
    gdf_all["grupo"] = gdf_all["classname"].map(CLASSES_GRUPO)
    gdf_all["mes_obs"] = gdf_all["view_date"].str.slice(0, 7)
    gdf_all["mes_disp"] = gdf_all["mes_obs"].apply(get_next_month)
    logger.info(f"Total consolidado: {len(gdf_all):,} polígonos DETER carregados.")
    return gdf_all

# ==============================================================================
# PARTE 1: RECORTE MUNICIPAL E GERAÇÃO DE ATRIBUTOS MUNICIPAIS
# ==============================================================================

def processar_recorte_municipal(
    gdf_deter: gpd.GeoDataFrame,
    processed_dir: Path,
    out_dir: Path,
    proj_epsg: int = 5880
) -> pd.DataFrame:
    """
    Recorta cada polígono pela malha municipal (municipios_pa.gpkg) em EPSG:5880,
    calcula as áreas em km² para cada grupo e mês de disponibilidade, e salva o parquet.
    """
    out_parquet = out_dir / "deter_municipio_mes.parquet"
    if out_parquet.exists():
        logger.info(f"Arquivo '{out_parquet}' já existe. Carregando...")
        return pd.read_parquet(out_parquet)

    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada.")

    logger.info(f"Carregando malha municipal de '{gpkg_path}'...")
    gdf_mun = gpd.read_file(gpkg_path).to_crs(epsg=proj_epsg)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)

    logger.info(f"Reprojetando polígonos DETER para EPSG:{proj_epsg}...")
    gdf_det_proj = gdf_deter.to_crs(epsg=proj_epsg)
    gdf_det_proj = gdf_det_proj[gdf_det_proj.geometry.is_valid & ~gdf_det_proj.geometry.is_empty]

    logger.info("Executando interseção geométrica estrita (overlay) com a malha municipal...")
    cols_det = ["geometry", "grupo", "mes_disp"]
    cols_mun = ["cd_mun", "area_km2", "geometry"]
    inter = gpd.overlay(gdf_det_proj[cols_det], gdf_mun[cols_mun], how="intersection")
    
    inter["area_km2_calc"] = inter.geometry.area / 1e6

    logger.info("Agrupando áreas por município, mês de disponibilidade e grupo...")
    agrupado = inter.groupby(["cd_mun", "mes_disp", "grupo"])["area_km2_calc"].sum().reset_index()

    # Pivotar para colunas: desmat, degrad, cicatriz
    pivot = agrupado.pivot(index=["cd_mun", "mes_disp"], columns="grupo", values="area_km2_calc").reset_index()
    for col in ["desmat", "degrad", "cicatriz"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
        else:
            pivot[col] = pivot[col].fillna(0.0).round(4)

    pivot.rename(columns={
        "mes_disp": "mes",
        "desmat": "desmat_km2",
        "degrad": "degrad_km2",
        "cicatriz": "cicatriz_km2"
    }, inplace=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_parquet(out_parquet, index=False)
    logger.info(f"Tabela deter_municipio_mes salva em '{out_parquet}' ({len(pivot):,} linhas com alertas).")
    return pivot

def construir_atributos_municipio(
    df_deter_mun: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path
) -> pd.DataFrame:
    """
    Calcula os 8 atributos municipais de desmatamento e degradação para cada mês de emissão t:
    - deter_<g>_1m: km² disponíveis em t
    - deter_<g>_3m: soma de t-2 a t
    - deter_<g>_12m: soma de t-11 a t
    - deter_<g>_3m_pct: deter_<g>_3m / area_km2 * 100
    Mescla com aphim_modelagem_v2.parquet por cd_mun e mes_emissao.
    """
    out_v2_deter = out_dir / "aphim_modelagem_v2_deter.parquet"
    v2_path = processed_dir / "aphim_modelagem_v2.parquet"
    if not v2_path.exists():
        raise FileNotFoundError(f"Arquivo '{v2_path}' não encontrado.")

    logger.info(f"Carregando base de modelagem v2 de '{v2_path}'...")
    df_v2 = pd.read_parquet(v2_path)
    df_v2["cd_mun"] = df_v2["cd_mun"].astype(str).str.zfill(7)
    n_linhas_original = len(df_v2)

    logger.info("Estruturando dicionários de busca rápida para agregações móveis...")
    # Mapas: (cd_mun, mes) -> desmat_km2 / degrad_km2
    map_desmat = df_deter_mun.set_index(["cd_mun", "mes"])["desmat_km2"].to_dict()
    map_degrad = df_deter_mun.set_index(["cd_mun", "mes"])["degrad_km2"].to_dict()

    logger.info("Calculando os 8 atributos DETER para cada linha de aphim_modelagem_v2...")
    registros = []
    for _, row in tqdm(df_v2.iterrows(), total=len(df_v2), desc="Atributos Município DETER"):
        c = row["cd_mun"]
        t = row["mes_emissao"]
        area_m = float(row["area_km2"])

        # Desmatamento
        d_1m = float(map_desmat.get((c, t), 0.0))
        d_3m = float(sum(map_desmat.get((c, get_prev_month(t, s)), 0.0) for s in range(3)))
        d_12m = float(sum(map_desmat.get((c, get_prev_month(t, s)), 0.0) for s in range(12)))
        d_3m_pct = float((d_3m / area_m) * 100.0) if area_m > 0 else 0.0

        # Degradação
        deg_1m = float(map_degrad.get((c, t), 0.0))
        deg_3m = float(sum(map_degrad.get((c, get_prev_month(t, s)), 0.0) for s in range(3)))
        deg_12m = float(sum(map_degrad.get((c, get_prev_month(t, s)), 0.0) for s in range(12)))
        deg_3m_pct = float((deg_3m / area_m) * 100.0) if area_m > 0 else 0.0

        registros.append({
            "deter_desmat_1m": d_1m,
            "deter_desmat_3m": d_3m,
            "deter_desmat_12m": d_12m,
            "deter_desmat_3m_pct": d_3m_pct,
            "deter_degrad_1m": deg_1m,
            "deter_degrad_3m": deg_3m,
            "deter_degrad_12m": deg_12m,
            "deter_degrad_3m_pct": deg_3m_pct
        })

    df_deter_attrs = pd.DataFrame(registros)
    df_final = pd.concat([df_v2, df_deter_attrs], axis=1)

    n_linhas_final = len(df_final)
    logger.info(f"Contagem de linhas: Base v2 original = {n_linhas_original:,} | Base v2 + DETER = {n_linhas_final:,}")
    if n_linhas_original != n_linhas_final:
        raise ValueError(f"Discrepância no número de linhas: esperado {n_linhas_original}, obtido {n_linhas_final}.")

    # Validação de nulos
    cols_deter = list(df_deter_attrs.columns)
    nulos = df_final[cols_deter].isnull().sum().sum()
    if nulos > 0:
        raise ValueError(f"Valores nulos proibidos encontrados nos atributos DETER municipais ({nulos}).")

    df_final.to_parquet(out_v2_deter, index=False)
    logger.info(f"Base aphim_modelagem_v2_deter salva com sucesso em '{out_v2_deter}'.")
    return df_final

# ==============================================================================
# PARTE 2: RECORTE DA GRADE DE 5 KM E ATRIBUTOS ESPACIAIS
# ==============================================================================

def construir_matriz_vizinhanca_grade(gdf_cells: gpd.GeoDataFrame) -> sp.csr_matrix:
    """
    Constrói a matriz de adjacência de 8 vizinhos (Moore) alinhada à grade de 5.000 m.
    Exatamente idêntica à de 06_grade_base.py.
    """
    logger.info("Construindo matriz de vizinhança de 8 células (Moore) para a grade 5 km...")
    bounds = gdf_cells.geometry.bounds
    cols = np.round(bounds["minx"] / 5000.0).astype(int)
    rows = np.round(bounds["miny"] / 5000.0).astype(int)

    coord_to_idx = dict(zip(zip(cols, rows), np.arange(len(gdf_cells))))
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

    W_neigh = sp.csr_matrix(
        (np.ones(len(adj_rows), dtype=np.float32), (adj_rows, adj_cols)),
        shape=(len(gdf_cells), len(gdf_cells))
    )
    return W_neigh

def processar_atributos_grade(
    gdf_deter: gpd.GeoDataFrame,
    processed_dir: Path,
    out_dir: Path,
    proj_epsg: int = 5880
) -> pd.DataFrame:
    """
    Recorta os polígonos pelas células de data/processed/grade_5000m_celulas.gpkg.
    Gera para cada célula e mês de emissão t:
    - deter_desmat_cel_3m e deter_desmat_cel_12m
    - deter_degrad_cel_3m e deter_degrad_cel_12m
    - deter_desmat_viz_12m (vizinhança de 8 células)
    Salva em data/processed_deter/grade_5000m_deter.parquet.
    """
    out_grade_deter = out_dir / "grade_5000m_deter.parquet"
    if out_grade_deter.exists():
        logger.info(f"Arquivo '{out_grade_deter}' já existe. Carregando...")
        return pd.read_parquet(out_grade_deter)

    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage de células '{gpkg_path}' não encontrado.")

    logger.info(f"Carregando células de '{gpkg_path}'...")
    gdf_cells = gpd.read_file(gpkg_path).to_crs(epsg=proj_epsg)
    gdf_cells["cell_id"] = gdf_cells["cell_id"].astype(np.int32)
    gdf_cells = gdf_cells.sort_values("cell_id").reset_index(drop=True)
    n_cells = len(gdf_cells)
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(gdf_cells["cell_id"])}

    W_neigh = construir_matriz_vizinhanca_grade(gdf_cells)

    # Filtrar apenas desmat e degrad para a grade
    gdf_det_filtro = gdf_deter[gdf_deter["grupo"].isin(["desmat", "degrad"])].copy()
    logger.info(f"Reprojetando {len(gdf_det_filtro):,} polígonos de desmat/degrad para EPSG:{proj_epsg}...")
    gdf_det_proj = gdf_det_filtro.to_crs(epsg=proj_epsg)
    gdf_det_proj = gdf_det_proj[gdf_det_proj.geometry.is_valid & ~gdf_det_proj.geometry.is_empty]

    logger.info("Executando interseção espacial (overlay) dos polígonos com as 50.736 células...")
    cols_det = ["geometry", "grupo", "mes_disp"]
    cols_cell = ["cell_id", "geometry"]
    inter_cells = gpd.overlay(gdf_det_proj[cols_det], gdf_cells[cols_cell], how="intersection")
    inter_cells["area_km2"] = inter_cells.geometry.area / 1e6

    logger.info("Agrupando áreas de desmat/degrad por célula e mês de disponibilidade...")
    agrup = inter_cells.groupby(["cell_id", "mes_disp", "grupo"])["area_km2"].sum().reset_index()

    # Mapeamento para matrizes esparsas por mês
    meses_emissao = [
        f"{ano:04d}-{mes:02d}"
        for ano in range(2019, 2026)
        for mes in range(1, 13)
        if not (ano == 2025 and mes == 12)
    ] # 2019-01 a 2025-11 (83 meses)

    # Coletar todos os meses de disponibilidade necessários: de 2018-02 até 2025-11
    todos_meses_disp = sorted(agrup["mes_disp"].unique())
    logger.info(f"Meses de disponibilidade mapeados no DETER: de {todos_meses_disp[0]} a {todos_meses_disp[-1]}")

    # Montar dicionário de vetores de área por mês: mes -> np.ndarray de tamanho n_cells
    vetores_desmat = {}
    vetores_degrad = {}

    for m_disp, grp_m in agrup.groupby("mes_disp"):
        v_desmat = np.zeros(n_cells, dtype=np.float32)
        v_degrad = np.zeros(n_cells, dtype=np.float32)

        grp_desm = grp_m[grp_m["grupo"] == "desmat"]
        for _, r in grp_desm.iterrows():
            idx = cell_id_to_idx[int(r["cell_id"])]
            v_desmat[idx] = float(r["area_km2"])

        grp_deg = grp_m[grp_m["grupo"] == "degrad"]
        for _, r in grp_deg.iterrows():
            idx = cell_id_to_idx[int(r["cell_id"])]
            v_degrad[idx] = float(r["area_km2"])

        vetores_desmat[m_disp] = v_desmat
        vetores_degrad[m_disp] = v_degrad

    logger.info("Calculando atributos móveis e espaciais para todas as 50.736 células ao longo dos 83 meses...")
    v_zeros = np.zeros(n_cells, dtype=np.float32)

    dfs_mensais = []
    for t in tqdm(meses_emissao, desc="Atributos Grade DETER"):
        # Janelas móveis para t:
        meses_3m = [get_prev_month(t, s) for s in range(3)]
        meses_12m = [get_prev_month(t, s) for s in range(12)]

        # 3m e 12m desmat
        s_desmat_3m = np.sum([vetores_desmat.get(m, v_zeros) for m in meses_3m], axis=0)
        s_desmat_12m = np.sum([vetores_desmat.get(m, v_zeros) for m in meses_12m], axis=0)

        # 3m e 12m degrad
        s_degrad_3m = np.sum([vetores_degrad.get(m, v_zeros) for m in meses_3m], axis=0)
        s_degrad_12m = np.sum([vetores_degrad.get(m, v_zeros) for m in meses_12m], axis=0)

        # Vizinhança de 8 células (desmat 12m nos vizinhos)
        s_desmat_viz_12m = W_neigh.dot(s_desmat_12m)

        df_m = pd.DataFrame({
            "cell_id": gdf_cells["cell_id"].values,
            "mes_emissao": t,
            "deter_desmat_cel_3m": s_desmat_3m.astype(np.float32),
            "deter_desmat_cel_12m": s_desmat_12m.astype(np.float32),
            "deter_degrad_cel_3m": s_degrad_3m.astype(np.float32),
            "deter_degrad_cel_12m": s_degrad_12m.astype(np.float32),
            "deter_desmat_viz_12m": s_desmat_viz_12m.astype(np.float32)
        })
        dfs_mensais.append(df_m)

    df_grade_final = pd.concat(dfs_mensais, ignore_index=True)
    logger.info(f"Tabela final da grade gerada com {len(df_grade_final):,} linhas.")

    out_grade_deter.parent.mkdir(parents=True, exist_ok=True)
    df_grade_final.to_parquet(out_grade_deter, index=False)
    logger.info(f"Atributos da grade salvos com sucesso em '{out_grade_deter}'.")
    return df_grade_final

def main():
    logger.info("=== INICIANDO PROCESSAMENTO DE ATRIBUTOS DETER (MUNICÍPIO E GRADE) ===")
    t0 = time.time()
    raw_dir = Path("data/raw/deter")
    processed_dir = Path("data/processed")
    out_dir = Path("data/processed_deter")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar polígonos consolidados
    gdf_deter = carregar_deter_bruto(raw_dir)

    # 2. Recorte municipal e atributos municipais
    df_deter_mun = processar_recorte_municipal(gdf_deter, processed_dir, out_dir)
    df_v2_deter = construir_atributos_municipio(df_deter_mun, processed_dir, out_dir)

    # 3. Recorte na grade de 5 km e atributos celulares
    df_grade_deter = processar_atributos_grade(gdf_deter, processed_dir, out_dir)

    dt = time.time() - t0
    logger.info(f"=== PROCESSAMENTO DE ATRIBUTOS DETER CONCLUÍDO COM SUCESSO EM {dt:.1f} SEGUNDOS ===")

if __name__ == "__main__":
    main()
