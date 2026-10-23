"""
Módulo 06 — Grade Espacial Base e Modelagem (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Objetivo:
Construir a grade quadrada de células (parametrizada em config.yaml, ex: 10 km / 10.000 m),
alinhada na origem em EPSG:5880 cobrindo o Estado do Pará.
Atribuir cada célula ao município correspondente e agregar o histórico de focos (2019-2025)
com zeros explícitos.
Gerar os atributos temporais, espaciais (vizinhança) e contextuais (clima do município)
para a priorização em Nível 2.
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import scipy.sparse as sp

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_06_grade_base")

def carregar_config() -> dict:
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    config_path = Path(__file__).resolve().parents[2] / config_file if not Path(config_file).is_absolute() else Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração '{config_path}' não encontrado.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_next_month(mes_str: str) -> str:
    ano, mes = map(int, mes_str.split("-"))
    if mes == 12:
        return f"{ano + 1:04d}-01"
    return f"{ano:04d}-{mes + 1:02d}"

def classificar_conjunto(mes_emissao: str) -> str:
    if "2019-01" <= mes_emissao <= "2019-12":
        return "aquecimento"
    elif "2020-01" <= mes_emissao <= "2023-11":
        return "treino"
    elif "2023-12" <= mes_emissao <= "2025-11":
        return "teste"
    elif mes_emissao >= "2025-12":
        return "teste_2026"
    raise ValueError(f"Mês de emissão {mes_emissao} fora do escopo de modelagem.")

def construir_grade_celulas(gdf_mun: gpd.GeoDataFrame, tamanho_m: float) -> Tuple[gpd.GeoDataFrame, Dict[Tuple[int, int], int], sp.csr_matrix]:
    """
    Cria a grade quadrada alinhada em (0,0) no CRS EPSG:5880 cobrindo o Pará.
    Retorna o GeoDataFrame das células que tocam o Pará, dicionário (col, row) -> cell_id
    e matriz de adjacência esparsa de 8 vizinhos.
    """
    logger.info(f"Construindo grade quadrada de tamanho {tamanho_m:.0f} m (EPSG:5880)...")
    pa_geom = gdf_mun.union_all()
    minx, miny, maxx, maxy = pa_geom.bounds

    gx0 = np.floor(minx / tamanho_m) * tamanho_m
    gy0 = np.floor(miny / tamanho_m) * tamanho_m
    gx1 = np.ceil(maxx / tamanho_m) * tamanho_m
    gy1 = np.ceil(maxy / tamanho_m) * tamanho_m

    xs = np.arange(gx0, gx1, tamanho_m)
    ys = np.arange(gy0, gy1, tamanho_m)

    boxes = []
    cols = []
    rows = []
    for x in xs:
        for y in ys:
            boxes.append(box(x, y, x + tamanho_m, y + tamanho_m))
            cols.append(int(round(x / tamanho_m)))
            rows.append(int(round(y / tamanho_m)))

    gdf_all = gpd.GeoDataFrame({"col": cols, "row": rows}, geometry=boxes, crs="EPSG:5880")
    
    # Manter apenas células que tocam o Pará
    logger.info("Filtrando células que interceptam o Estado do Pará...")
    touches_mask = gdf_all.intersects(pa_geom)
    gdf_cells = gdf_all[touches_mask].copy().reset_index(drop=True)
    gdf_cells["cell_id"] = np.arange(1, len(gdf_cells) + 1, dtype=np.int32)

    logger.info(f"Total de células geradas que tocam o Pará: {len(gdf_cells):,}")

    # Calcular área dentro do Pará em km²
    logger.info("Calculando área de interseção de cada célula com o Pará (km²)...")
    gdf_cells["area_km2_dentro_pa"] = gdf_cells.intersection(pa_geom).area / 1e6

    # Calcular centroide da célula
    centroids = gdf_cells.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame({"cell_id": gdf_cells["cell_id"]}, geometry=centroids, crs="EPSG:5880")

    # Atribuir município via centroide (predicate='within')
    logger.info("Atribuindo cd_mun às células via centroide...")
    sj = gpd.sjoin(centroids_gdf, gdf_mun[["cd_mun", "nm_mun", "geometry"]], how="left", predicate="within")

    # Para centroides fora do estado, usar município com maior área de interseção
    missing_mask = sj["cd_mun"].isna()
    if missing_mask.sum() > 0:
        logger.info(f"Tratando {missing_mask.sum()} células cujo centroide caiu fora do estado...")
        missing_ids = sj.loc[missing_mask, "cell_id"].values
        missing_cells = gdf_cells[gdf_cells["cell_id"].isin(missing_ids)][["cell_id", "geometry"]].copy()
        inter = gpd.overlay(missing_cells, gdf_mun[["cd_mun", "nm_mun", "geometry"]], how="intersection")
        inter["inter_area"] = inter.geometry.area
        best_mun = inter.sort_values("inter_area", ascending=False).groupby("cell_id").first()
        
        for cid in missing_ids:
            if cid in best_mun.index:
                sj.loc[sj["cell_id"] == cid, "cd_mun"] = best_mun.loc[cid, "cd_mun"]
                sj.loc[sj["cell_id"] == cid, "nm_mun"] = best_mun.loc[cid, "nm_mun"]

    gdf_cells["cd_mun"] = sj["cd_mun"].values
    gdf_cells["nm_mun"] = sj["nm_mun"].values

    # Centroides em EPSG:4674 (SIRGAS 2000 geográfico)
    logger.info("Projetando centroides para EPSG:4674 (lat/lon)...")
    centroids_4674 = centroids.to_crs(epsg=4674)
    gdf_cells["centroid_lat"] = centroids_4674.y
    gdf_cells["centroid_lon"] = centroids_4674.x

    # Dicionário de busca (col, row) -> cell_id (0-indexed para matrizes)
    coord_to_idx = dict(zip(zip(gdf_cells["col"], gdf_cells["row"]), np.arange(len(gdf_cells))))

    # Construir matriz de vizinhança de 8 células
    logger.info("Construindo matriz de adjacência de 8 vizinhos (Moore)...")
    adj_rows = []
    adj_cols = []
    deltas = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

    for idx, (c, r) in enumerate(zip(gdf_cells["col"], gdf_cells["row"])):
        for dc, dr in deltas:
            neigh_coord = (c + dc, r + dr)
            if neigh_coord in coord_to_idx:
                adj_rows.append(idx)
                adj_cols.append(coord_to_idx[neigh_coord])

    n_cells = len(gdf_cells)
    W_neigh = sp.csr_matrix(
        (np.ones(len(adj_rows), dtype=np.float32), (adj_rows, adj_cols)),
        shape=(n_cells, n_cells)
    )

    return gdf_cells, coord_to_idx, W_neigh

def agregar_focos_grade(
    focos_parquet_path: Path,
    coord_to_idx: Dict[Tuple[int, int], int],
    tamanho_m: float,
    n_cells: int
) -> Tuple[np.ndarray, list]:
    """
    Projeta todos os focos para EPSG:5880 e agrega na matriz [n_cells, 84 meses]
    cobrindo 2019-01 a 2025-12.
    """
    logger.info(f"Carregando pontos de focos de '{focos_parquet_path}'...")
    df_focos = pd.read_parquet(
        focos_parquet_path,
        columns=["latitude", "longitude", "acq_date", "cd_mun"]
    )
    total_focos = len(df_focos)
    logger.info(f"Total de focos carregados: {total_focos:,}")

    # Meses de 2019-01 a 2025-12 (84 meses)
    meses_todos = []
    for ano in range(2019, 2026):
        for m in range(1, 13):
            meses_todos.append(f"{ano:04d}-{m:02d}")
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}

    # Transformação de coordenadas EPSG:4326 -> EPSG:5880
    logger.info("Transformando coordenadas para EPSG:5880 e identificando células...")
    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos["longitude"].values, df_focos["latitude"].values)

    f_cols = np.floor(xs / tamanho_m).astype(int)
    f_rows = np.floor(ys / tamanho_m).astype(int)

    f_mes = df_focos["acq_date"].astype(str).str.slice(0, 7).values
    f_mes_idx = np.array([mes_to_idx.get(m, -1) for m in f_mes], dtype=np.int32)

    # Identificar cell index
    cell_indices = np.array([coord_to_idx.get((c, r), -1) for c, r in zip(f_cols, f_rows)], dtype=np.int32)

    # Validação de alocação de focos
    mask_valid = (cell_indices >= 0) & (f_mes_idx >= 0)
    focos_fora = total_focos - mask_valid.sum()
    if focos_fora > 0:
        logger.warning(f"Atenção: {focos_fora} focos não caíram dentro das células da grade.")
    else:
        logger.info(f"Validação: Todos os {total_focos:,} focos foram atribuídos com sucesso à grade!")

    # Montar matriz célula x mês (com zeros explícitos)
    logger.info("Agregando contagem de focos na matriz esparsa/densa célula x mês...")
    cell_focos_mat = np.zeros((n_cells, len(meses_todos)), dtype=np.float32)
    valid_c_idx = cell_indices[mask_valid]
    valid_m_idx = f_mes_idx[mask_valid]

    np.add.at(cell_focos_mat, (valid_c_idx, valid_m_idx), 1.0)
    logger.info(f"Soma total da matriz célula x mês: {cell_focos_mat.sum():,.0f}")

    return cell_focos_mat, meses_todos

def montar_base_modelagem(
    gdf_cells: gpd.GeoDataFrame,
    cell_focos_mat: np.ndarray,
    meses_todos: list,
    W_neigh: sp.csr_matrix,
    processed_dir: Path,
    out_parquet_path: Path
) -> int:
    """
    Calcula os atributos temporais, de vizinhança e contextuais municipais
    para os 83 meses de emissão (2019-01 a 2025-11), gravando via streaming
    diretamente no arquivo Parquet para manter uso de memória sob controle.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("Iniciando engenharia de atributos temporais e espaciais da grade...")
    n_cells = len(gdf_cells)
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}

    # Meses de emissão t: 2019-01 a 2025-11 (83 meses)
    meses_emissao = meses_todos[:-1]
    n_emissao = len(meses_emissao)

    # Carregar dados climáticos e focos municipais de aphim_modelagem.parquet
    mod_mun_path = processed_dir / "aphim_modelagem.parquet"
    if not mod_mun_path.exists():
        raise FileNotFoundError(f"Base '{mod_mun_path}' não encontrada.")
    df_mod_mun = pd.read_parquet(mod_mun_path)
    df_mod_mun["cd_mun"] = df_mod_mun["cd_mun"].astype(str).str.zfill(7)
    df_mod_mun["mes_emissao"] = df_mod_mun["mes_emissao"].astype(str)

    mun_clima_lookup = df_mod_mun.set_index(["cd_mun", "mes_emissao"])[
        ["focos_t", "prec_mm", "ur_pct", "dpv_kpa", "umid_solo", "anom_prec_mm"]
    ].to_dict(orient="index")

    # Pré-computar matriz de vizinhos para todos os meses: W_neigh * cell_focos_mat
    logger.info("Calculando matriz de focos dos 8 vizinhos para todos os meses...")
    vizinhos_focos_mat = W_neigh.dot(cell_focos_mat)  # shape (n_cells, 84)

    # Pré-computar climatologia histórica por célula e mês de calendário
    logger.info("Pré-computando média histórica de focos da célula por mês de calendário...")
    hist_cell_dict = {}
    hist_neigh_dict = {}

    for target_year in range(2019, 2026):
        anos_disponiveis = [y for y in range(2019, target_year)]
        for m_num in range(1, 13):
            if len(anos_disponiveis) == 0:
                hist_cell_dict[(m_num, target_year)] = np.zeros(n_cells, dtype=np.float32)
                hist_neigh_dict[(m_num, target_year)] = np.zeros(n_cells, dtype=np.float32)
            else:
                col_indices = [mes_to_idx[f"{y:04d}-{m_num:02d}"] for y in anos_disponiveis]
                mean_cell = np.mean(cell_focos_mat[:, col_indices], axis=1)
                mean_neigh = np.mean(vizinhos_focos_mat[:, col_indices], axis=1)
                hist_cell_dict[(m_num, target_year)] = mean_cell
                hist_neigh_dict[(m_num, target_year)] = mean_neigh

    logger.info(f"Gravando base de modelagem via streaming em '{out_parquet_path}'...")
    writer = None
    total_linhas = 0

    for t_str in meses_emissao:
        t_idx = mes_to_idx[t_str]
        alvo_str = get_next_month(t_str)
        alvo_idx = mes_to_idx[alvo_str]
        alvo_ano = int(alvo_str[:4])
        alvo_mes_num = int(alvo_str[5:7])

        conjunto = classificar_conjunto(t_str)

        # 1. Alvo y (focos da célula em t+1)
        y = cell_focos_mat[:, alvo_idx]

        # 2. Histórico da célula
        focos_t = cell_focos_mat[:, t_idx]
        focos_t1 = cell_focos_mat[:, t_idx - 1] if t_idx >= 1 else np.zeros(n_cells, dtype=np.float32)
        focos_t2 = cell_focos_mat[:, t_idx - 2] if t_idx >= 2 else np.zeros(n_cells, dtype=np.float32)
        focos_mm3 = (focos_t + focos_t1 + focos_t2) / 3.0

        # t - 11 (mesmo mês do ano anterior ao alvo)
        focos_mesmo_mes_ano_ant = cell_focos_mat[:, t_idx - 11] if t_idx >= 11 else np.zeros(n_cells, dtype=np.float32)

        # Média histórica do mês de calendário do alvo (apenas anos < ano_alvo)
        focos_media_hist_mes = hist_cell_dict[(alvo_mes_num, alvo_ano)]

        # 3. Vizinhança
        focos_vizinhos_t = vizinhos_focos_mat[:, t_idx]
        focos_vizinhos_media_hist_mes = hist_neigh_dict[(alvo_mes_num, alvo_ano)]

        # 4. Sazonalidade
        mes_sin = np.sin(2.0 * np.pi * alvo_mes_num / 12.0)
        mes_cos = np.cos(2.0 * np.pi * alvo_mes_num / 12.0)

        df_t = pd.DataFrame({
            "cell_id": gdf_cells["cell_id"].values,
            "cd_mun": gdf_cells["cd_mun"].values,
            "mes_emissao": t_str,
            "mes_alvo": alvo_str,
            "conjunto": conjunto,
            "y": y.astype(np.int32),
            "focos_t": focos_t,
            "focos_t1": focos_t1,
            "focos_t2": focos_t2,
            "focos_mm3": focos_mm3,
            "focos_mesmo_mes_ano_ant": focos_mesmo_mes_ano_ant,
            "focos_media_hist_mes": focos_media_hist_mes,
            "focos_vizinhos_t": focos_vizinhos_t,
            "focos_vizinhos_media_hist_mes": focos_vizinhos_media_hist_mes,
            "mes_sin": np.float32(mes_sin),
            "mes_cos": np.float32(mes_cos)
        })

        # 5. Contexto municipal em t
        clima_tuples = [mun_clima_lookup.get((c, t_str), {}) for c in gdf_cells["cd_mun"].values]
        df_t["focos_mun_t"] = np.array([c.get("focos_t", 0.0) for c in clima_tuples], dtype=np.float32)
        df_t["prec_mm"] = np.array([c.get("prec_mm", 0.0) for c in clima_tuples], dtype=np.float32)
        df_t["ur_pct"] = np.array([c.get("ur_pct", 0.0) for c in clima_tuples], dtype=np.float32)
        df_t["dpv_kpa"] = np.array([c.get("dpv_kpa", 0.0) for c in clima_tuples], dtype=np.float32)
        df_t["umid_solo"] = np.array([c.get("umid_solo", 0.0) for c in clima_tuples], dtype=np.float32)
        df_t["anom_prec_mm"] = np.array([c.get("anom_prec_mm", 0.0) for c in clima_tuples], dtype=np.float32)

        table = pa.Table.from_pandas(df_t, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet_path, table.schema, compression="snappy")
        writer.write_table(table)
        total_linhas += len(df_t)

    if writer is not None:
        writer.close()

    logger.info(f"Base de modelagem concluída com {total_linhas:,} registros.")
    return total_linhas

def main():
    t_start = time.time()
    logger.info("=== INICIANDO SCRIPT 06: GRADE ESPACIAL BASE E MODELAGEM (APHIM) ===")

    parser = argparse.ArgumentParser(description="Geração de Grade Espacial e Atributos de Modelagem.")
    parser.add_argument("--tamanho_m", type=float, default=None, help="Tamanho da célula em metros.")
    args = parser.parse_args()

    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    interim_dir = Path(config["caminhos"]["interim"])

    # Obter tamanho da célula do config se não passado como argumento
    tamanho_m = args.tamanho_m
    if tamanho_m is None:
        tamanho_m = float(config.get("grade", {}).get("tamanho_m", 10000))
    tamanho_str = f"{int(tamanho_m)}"

    logger.info(f"Tamanho de célula configurado: {tamanho_m:,.0f} metros ({tamanho_m/1000:.1f} km).")

    # 1. Carregar malha municipal
    mun_gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not mun_gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{mun_gpkg_path}' não encontrada.")
    gdf_mun = gpd.read_file(mun_gpkg_path).to_crs(epsg=5880)
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)

    # 2. Construir grade de células
    gdf_cells, coord_to_idx, W_neigh = construir_grade_celulas(gdf_mun, tamanho_m)

    # 3. Agregar focos 2019-2025
    focos_parquet_path = interim_dir / "firms_focos_pa.parquet"
    if not focos_parquet_path.exists():
        raise FileNotFoundError(f"Arquivo de focos '{focos_parquet_path}' não encontrado.")
    cell_focos_mat, meses_todos = agregar_focos_grade(
        focos_parquet_path=focos_parquet_path,
        coord_to_idx=coord_to_idx,
        tamanho_m=tamanho_m,
        n_cells=len(gdf_cells)
    )

    # 4. Salvar GPKG da grade de células
    out_gpkg = processed_dir / f"grade_{tamanho_str}m_celulas.gpkg"
    logger.info(f"Salvando GeoPackage da grade de células em '{out_gpkg}'...")
    cols_gpkg = ["cell_id", "cd_mun", "nm_mun", "area_km2_dentro_pa", "centroid_lat", "centroid_lon", "geometry"]
    gdf_cells[cols_gpkg].to_file(out_gpkg, driver="GPKG")
    gpkg_size_mb = out_gpkg.stat().st_size / (1024 * 1024)
    logger.info(f"GeoPackage salvo com sucesso! Tamanho: {gpkg_size_mb:.2f} MB.")

    # 5. Montar e salvar tabela de modelagem via streaming
    out_parquet = processed_dir / f"grade_{tamanho_str}m_modelagem.parquet"
    total_registros = montar_base_modelagem(
        gdf_cells=gdf_cells,
        cell_focos_mat=cell_focos_mat,
        meses_todos=meses_todos,
        W_neigh=W_neigh,
        processed_dir=processed_dir,
        out_parquet_path=out_parquet
    )
    parquet_size_mb = out_parquet.stat().st_size / (1024 * 1024)
    logger.info(f"Parquet salvo com sucesso! Tamanho: {parquet_size_mb:.2f} MB.")

    # 6. Validações e Relatório
    logger.info("Executando validações da etapa 06...")
    soma_total_focos = cell_focos_mat.sum()
    n_celulas = len(gdf_cells)
    area_media_km2 = gdf_cells["area_km2_dentro_pa"].mean()

    # Fração de células que nunca tiveram foco em 2019-2025
    focos_totais_por_celula = cell_focos_mat.sum(axis=1)
    celulas_sem_foco = (focos_totais_por_celula == 0).sum()
    pct_sem_foco = (celulas_sem_foco / n_celulas) * 100.0

    # Diferença célula x município
    df_base_mun = pd.read_parquet(processed_dir / "aphim_base_municipio_mes.parquet")
    df_base_mun["cd_mun"] = df_base_mun["cd_mun"].astype(str).str.zfill(7)
    df_base_mun["mes"] = df_base_mun["mes"].astype(str)
    mun_focos_dict = df_base_mun.set_index(["cd_mun", "mes"])["focos_count"].to_dict()

    # Mapeamento rápido de índices de células por município
    cd_mun_arr = gdf_cells["cd_mun"].values
    unique_muns = np.unique(cd_mun_arr)
    mun_cell_indices = {m: np.where(cd_mun_arr == m)[0] for m in unique_muns}
    diffs_pct = []

    for m_idx, m_str in enumerate(meses_todos):
        col_f = cell_focos_mat[:, m_idx]
        for cd, indices in mun_cell_indices.items():
            f_cel = col_f[indices].sum()
            f_mun = mun_focos_dict.get((cd, m_str), 0)
            if f_mun > 0:
                diff_pct = abs(f_cel - f_mun) / f_mun * 100.0
                diffs_pct.append(diff_pct)


    dif_media_pct = float(np.mean(diffs_pct)) if len(diffs_pct) > 0 else 0.0
    dif_max_pct = float(np.max(diffs_pct)) if len(diffs_pct) > 0 else 0.0

    t_total = time.time() - t_start

    print("\n" + "=" * 80)
    print("           RELATÓRIO DE VALIDAÇÃO DA ETAPA 06: GRADE ESPACIAL BASE")
    print("=" * 80)
    print(f"• Tamanho da célula:                 {tamanho_m:,.0f} m ({tamanho_m/1000:.1f} km)")
    print(f"• Total de células que tocam o PA:   {n_celulas:,}")
    print(f"• Área média das células:            {area_media_km2:.2f} km²")
    print(f"• Células sem nenhum foco 2019-2025: {celulas_sem_foco:,} ({pct_sem_foco:.2f}%)")
    print(f"• Soma de focos na grade:            {soma_total_focos:,.0f} (esperado: 1.695.702)")
    print(f"• Diferença célula x município:      Média = {dif_media_pct:.2f}% | Máxima = {dif_max_pct:.2f}%")
    print(f"• Tamanho do arquivo .gpkg:          {gpkg_size_mb:.2f} MB")
    print(f"• Tamanho do arquivo .parquet:       {parquet_size_mb:.2f} MB")
    print(f"• Tempo total de execução:           {t_total:.1f} segundos")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
