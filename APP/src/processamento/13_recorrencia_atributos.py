"""
Módulo 13 — Variáveis de Recorrência do Fogo na Grade de 5 km e Análise Descritiva (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Etapas:
- Parte A: Matriz histórica de focos por célula [50.736 x 120 meses] (2016-01 a 2025-12).
  Checagem estrita de igualdade contra a base atual (2019-01 a 2025-12).
- Parte B: Criação das 6 variáveis de recorrência na grade de 5 km:
  1. meses_desde_ultimo_foco (120 se sem foco desde 2016-01)
  2. anos_com_fogo_3a (0 a 3)
  3. frac_meses_fogo_24m
  4. focos_janela_retorno (t-12 a t-10)
  5. taxa_retorno_emp (com isolamento estrito: _d2022, _d2023, _final)
  6. frac_meses_fogo_24m_viz (média na vizinhança Moore)
  Salva data/processed_recorrencia/grade_5000m_recorrencia.parquet e taxas empíricas.
- Parte F: Análise descritiva de recorrência no período de treino (2017 a 2023)
  e gráfico recorrencia_tempo_retorno.png no estilo acadêmico.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_13_recorrencia_atributos")

FAIXAS_NOMES = ["0", "1", "2", "3-5", "6-9", "10-12", "13-24", "25+", "nunca"]

def get_faixa_idx(arr: np.ndarray) -> np.ndarray:
    """Mapeia o array de meses_desde_ultimo_foco para as categorias nominais padronizadas."""
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

def construir_matriz_focos_grade(
    gdf_cells: gpd.GeoDataFrame,
    focos_parquet_path: Path,
    tamanho_m: float = 5000.0
) -> Tuple[np.ndarray, List[str], Dict[Tuple[int, int], int], sp.csr_matrix, np.ndarray]:
    """
    Constrói a matriz [50.736 células x 120 meses] de 2016-01 a 2025-12.
    Utiliza estritamente a mesma regra de agregação de 06_grade_base.py.
    """
    logger.info("Mapeando coordenadas da grade para índices...")
    bounds = gdf_cells.geometry.bounds
    cols = np.round(bounds["minx"] / tamanho_m).astype(int)
    rows = np.round(bounds["miny"] / tamanho_m).astype(int)
    coord_to_idx = dict(zip(zip(cols, rows), np.arange(len(gdf_cells))))

    # Matriz de adjacência de 8 vizinhos (Moore)
    logger.info("Construindo matriz de adjacência de 8 vizinhos (Moore)...")
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
    degrees = np.maximum(1.0, np.array(W_neigh.sum(axis=1)).flatten())

    # Carregar focos de 2016-01 a 2025-12
    logger.info(f"Carregando pontos de focos de '{focos_parquet_path}'...")
    df_focos = pd.read_parquet(
        focos_parquet_path,
        columns=["latitude", "longitude", "acq_date"]
    )
    df_focos = df_focos[df_focos["acq_date"] >= "2016-01-01"].copy()
    logger.info(f"Total de focos carregados (>= 2016-01-01): {len(df_focos):,}")

    meses_todos = [f"{ano:04d}-{m:02d}" for ano in range(2016, 2026) for m in range(1, 13)]
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}

    # Transformação de coordenadas EPSG:4326 -> EPSG:5880
    logger.info("Projetando focos para EPSG:5880 e agregando na grade celular...")
    trans = Transformer.from_crs("EPSG:4326", "EPSG:5880", always_xy=True)
    xs, ys = trans.transform(df_focos["longitude"].values, df_focos["latitude"].values)

    f_cols = np.floor(xs / tamanho_m).astype(int)
    f_rows = np.floor(ys / tamanho_m).astype(int)
    f_mes = df_focos["acq_date"].astype(str).str.slice(0, 7).values
    f_mes_idx = np.array([mes_to_idx.get(m, -1) for m in f_mes], dtype=np.int32)

    cell_indices = np.array([coord_to_idx.get((c, r), -1) for c, r in zip(f_cols, f_rows)], dtype=np.int32)
    mask_valid = (cell_indices >= 0) & (f_mes_idx >= 0)

    cell_focos_mat = np.zeros((len(gdf_cells), len(meses_todos)), dtype=np.float32)
    np.add.at(cell_focos_mat, (cell_indices[mask_valid], f_mes_idx[mask_valid]), 1.0)
    logger.info(f"Matriz de focos gerada com sucesso! Shape: {cell_focos_mat.shape} | Soma total: {cell_focos_mat.sum():,.0f}")

    return cell_focos_mat, meses_todos, coord_to_idx, W_neigh, degrees

def executar_checagem_parte_a(
    cell_focos_mat: np.ndarray,
    meses_todos: List[str],
    processed_dir: Path
) -> int:
    """
    Executa a checagem obrigatória da Parte A:
    A matriz gerada de 2019-01 a 2025-12 deve ser rigorosamente idêntica à usada hoje na grade.
    """
    logger.info("=== EXECUTANDO CHECAGEM DA PARTE A: IGUALDADE COM A BASE ATUAL (2019-2025) ===")
    mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    if not mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{mod_path}' não encontrado.")

    df_mod = pd.read_parquet(mod_path, columns=["cell_id", "mes_emissao", "focos_t", "y"])
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}

    diff_count = 0
    # Comparar focos_t para mes_emissao de 2019-01 a 2025-11
    for m in sorted(df_mod["mes_emissao"].unique()):
        m_idx = mes_to_idx[m]
        sub = df_mod[df_mod["mes_emissao"] == m].sort_values("cell_id")
        f_mod = sub["focos_t"].values
        f_mat = cell_focos_mat[:, m_idx]
        diffs = int(np.sum(f_mod != f_mat))
        if diffs > 0:
            logger.error(f"Divergência detectada no mês {m}: {diffs} células diferentes!")
            diff_count += diffs

    # Comparar o último mês 2025-12 (que é o y de mes_emissao = 2025-11)
    sub_y = df_mod[df_mod["mes_emissao"] == "2025-11"].sort_values("cell_id")
    diff_2025_12 = int(np.sum(sub_y["y"].values != cell_focos_mat[:, mes_to_idx["2025-12"]]))
    if diff_2025_12 > 0:
        logger.error(f"Divergência detectada em 2025-12 (alvo de 2025-11): {diff_2025_12} células diferentes!")
        diff_count += diff_2025_12

    print("\n" + "=" * 80)
    print(f"CHECAGEM DA PARTE A — COMPARAÇÃO COM GRADE_5000M_MODELAGEM:")
    print(f"• Total de células-mês comparadas (2019-01 a 2025-12): {50736 * 84:,}")
    print(f"• Número de células-mês diferentes: {diff_count}")
    print("=" * 80 + "\n")

    if diff_count != 0:
        msg = f"FALHA CRÍTICA NA PARTE A: Matriz diverge da base atual em {diff_count} células-mês!"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("PARTE A APROVADA COM SUCESSO: Matriz idêntica (zero divergências) à base atual!")
    return diff_count

def calcular_taxas_empiricas_recortes(
    A_mat: np.ndarray,
    meses_desde_mat: np.ndarray,
    meses_todos: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Calcula a probabilidade empírica de fogo em t+1 dado (faixa, mês do calendário do alvo)
    para cada recorte temporal (dobra 2022, dobra 2023, modelo final).
    Suavização de Laplace: (focos + 1) / (n + 2).
    """
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}
    recortes = {
        "d2022": ("2016-12", "2021-11"), # alvos 2017-01 a 2021-12
        "d2023": ("2016-12", "2022-11"), # alvos 2017-01 a 2022-12
        "final": ("2016-12", "2023-11")  # alvos 2017-01 a 2023-12
    }

    tabelas_taxas = {}

    for nome_recorte, (t_inicio, t_fim) in recortes.items():
        idx_ini = mes_to_idx[t_inicio]
        idx_fim = mes_to_idx[t_fim]

        counts_total = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}
        counts_focos = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}

        for t in range(idx_ini, idx_fim + 1):
            alvo_idx = t + 1
            alvo_cal_m = int(meses_todos[alvo_idx][5:7])

            fx = get_faixa_idx(meses_desde_mat[:, t])
            y_alvo = A_mat[:, alvo_idx]

            for f in FAIXAS_NOMES:
                mask_f = (fx == f)
                n = int(mask_f.sum())
                if n > 0:
                    k = int(y_alvo[mask_f].sum())
                    counts_total[f][alvo_cal_m] += n
                    counts_focos[f][alvo_cal_m] += k

        rates_dict = {f: {} for f in FAIXAS_NOMES}
        for f in FAIXAS_NOMES:
            for m in range(1, 13):
                n = counts_total[f][m]
                k = counts_focos[f][m]
                rates_dict[f][m] = (k + 1) / (n + 2)

        df_rates = pd.DataFrame(rates_dict).T
        df_rates.columns = [f"Mes_{m:02d}" for m in range(1, 13)]
        tabelas_taxas[nome_recorte] = df_rates

    return tabelas_taxas

def gerar_base_recorrencia_grade(
    gdf_cells: gpd.GeoDataFrame,
    cell_focos_mat: np.ndarray,
    meses_todos: List[str],
    W_neigh: sp.csr_matrix,
    degrees: np.ndarray,
    out_dir: Path
) -> pd.DataFrame:
    """
    Gera as 6 variáveis de recorrência para os 83 meses de emissão (2019-01 a 2025-11).
    Salva grade_5000m_recorrencia.parquet.
    """
    logger.info("=== GERANDO VARIÁVEIS DE RECORRÊNCIA NA GRADE (PARTE B) ===")
    n_cells = len(gdf_cells)
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}
    A = (cell_focos_mat > 0).astype(np.int8)

    # 1. Pré-computar meses_desde_ultimo_foco para todos os 120 meses
    logger.info("Calculando matriz meses_desde_ultimo_foco (2016-01 a 2025-12)...")
    last_fire = np.full(n_cells, -999, dtype=np.int32)
    meses_desde = np.zeros_like(cell_focos_mat, dtype=np.int32)

    for m in range(len(meses_todos)):
        has_fire = (A[:, m] == 1)
        last_fire[has_fire] = m
        diff = m - last_fire
        diff[last_fire == -999] = 120
        meses_desde[:, m] = diff

    # 2. Calcular taxas empíricas de retorno para os 3 recortes
    logger.info("Calculando taxas de retorno empíricas com isolamento temporal estrito...")
    tabelas_taxas = calcular_taxas_empiricas_recortes(A, meses_desde, meses_todos)

    # Salvar as tabelas de taxas em CSV
    resultados_dir = out_dir / "resultados"
    for recorte, df_t in tabelas_taxas.items():
        p1 = out_dir / f"taxa_retorno_{recorte}.csv"
        p2 = resultados_dir / f"taxa_retorno_{recorte}.csv"
        df_t.to_csv(p1, encoding="utf-8")
        df_t.to_csv(p2, encoding="utf-8")
        logger.info(f"Tabela taxa_retorno_{recorte}.csv salva com sucesso.")

    # 3. Meses de emissão t de 2019-01 a 2025-11 (83 meses)
    meses_emissao = [f"{ano:04d}-{m:02d}" for ano in range(2019, 2026) for m in range(1, 13)][:-1]
    logger.info(f"Gerando registros para {len(meses_emissao)} meses de emissão (2019-01 a 2025-11)...")

    # Arrays temporários
    cell_ids_arr = gdf_cells["cell_id"].values.astype(np.int32)
    n_emissao = len(meses_emissao)
    total_linhas = n_cells * n_emissao

    out_cell_id = np.empty(total_linhas, dtype=np.int32)
    out_mes_emissao = []
    out_meses_desde = np.empty(total_linhas, dtype=np.int16)
    out_anos_3a = np.empty(total_linhas, dtype=np.int8)
    out_frac_24m = np.empty(total_linhas, dtype=np.float32)
    out_focos_retorno = np.empty(total_linhas, dtype=np.float32)
    out_taxa_d22 = np.empty(total_linhas, dtype=np.float32)
    out_taxa_d23 = np.empty(total_linhas, dtype=np.float32)
    out_taxa_fin = np.empty(total_linhas, dtype=np.float32)
    out_frac_24m_viz = np.empty(total_linhas, dtype=np.float32)

    t0 = time.time()
    for idx_t, t_str in enumerate(meses_emissao):
        t = mes_to_idx[t_str]
        alvo_cal_m = int(meses_todos[t + 1][5:7])
        col_name = f"Mes_{alvo_cal_m:02d}"

        start_row = idx_t * n_cells
        end_row = start_row + n_cells

        out_cell_id[start_row:end_row] = cell_ids_arr
        out_mes_emissao.extend([t_str] * n_cells)

        # 1. meses_desde_ultimo_foco
        md = meses_desde[:, t]
        out_meses_desde[start_row:end_row] = md

        # 2. anos_com_fogo_3a
        w1 = (A[:, t-11 : t+1].sum(axis=1) > 0).astype(np.int8)
        w2 = (A[:, t-23 : t-11].sum(axis=1) > 0).astype(np.int8)
        w3 = (A[:, t-35 : t-23].sum(axis=1) > 0).astype(np.int8)
        out_anos_3a[start_row:end_row] = w1 + w2 + w3

        # 3. frac_meses_fogo_24m
        f24 = A[:, t-23 : t+1].sum(axis=1) / 24.0
        out_frac_24m[start_row:end_row] = f24

        # 4. focos_janela_retorno (t-12 a t-10)
        f_ret = cell_focos_mat[:, t-12 : t-9].sum(axis=1)
        out_focos_retorno[start_row:end_row] = f_ret

        # 5. taxa_retorno_emp
        fx = get_faixa_idx(md)
        out_taxa_d22[start_row:end_row] = tabelas_taxas["d2022"][col_name].loc[fx].values.astype(np.float32)
        out_taxa_d23[start_row:end_row] = tabelas_taxas["d2023"][col_name].loc[fx].values.astype(np.float32)
        out_taxa_fin[start_row:end_row] = tabelas_taxas["final"][col_name].loc[fx].values.astype(np.float32)

        # 6. frac_meses_fogo_24m_viz (média das 8 vizinhas)
        f24_viz = W_neigh.dot(f24) / degrees
        out_frac_24m_viz[start_row:end_row] = f24_viz

    df_recorrencia = pd.DataFrame({
        "cell_id": out_cell_id,
        "mes_emissao": out_mes_emissao,
        "meses_desde_ultimo_foco": out_meses_desde,
        "anos_com_fogo_3a": out_anos_3a,
        "frac_meses_fogo_24m": out_frac_24m,
        "focos_janela_retorno": out_focos_retorno,
        "taxa_retorno_emp_d2022": out_taxa_d22,
        "taxa_retorno_emp_d2023": out_taxa_d23,
        "taxa_retorno_emp_final": out_taxa_fin,
        "frac_meses_fogo_24m_viz": out_frac_24m_viz
    })

    out_parquet = out_dir / "grade_5000m_recorrencia.parquet"
    logger.info(f"Salvando DataFrame com {len(df_recorrencia):,} linhas em '{out_parquet}'...")
    df_recorrencia.to_parquet(out_parquet, index=False, compression="snappy")
    size_mb = out_parquet.stat().st_size / (1024 * 1024)
    logger.info(f"Parquet salvo com sucesso! Tamanho: {size_mb:.2f} MB ({time.time()-t0:.1f} s).")

    # Checagens da Parte B (impressas)
    print("\n" + "=" * 80)
    print("           CHECAGENS DA PARTE B (VARIÁVEIS DE RECORRÊNCIA NA GRADE)")
    print("=" * 80)
    t_1912 = mes_to_idx["2019-12"]
    md_1912 = meses_desde[:, t_1912]
    c_com_fogo = np.where(A[:, t_1912] == 1)[0][:5]
    c_nunca = np.where(md_1912 == 120)[0][:5]

    print("• Em 2019-12, 5 células com foco em t (esperado: meses_desde = 0):")
    for idx_c in c_com_fogo:
        print(f"  - Célula {gdf_cells.iloc[idx_c]['cell_id']}: meses_desde = {md_1912[idx_c]}")

    print("\n• Em 2019-12, 5 células sem foco desde 2016 (esperado: meses_desde = 120):")
    for idx_c in c_nunca:
        print(f"  - Célula {gdf_cells.iloc[idx_c]['cell_id']}: meses_desde = {md_1912[idx_c]}")

    print("\n• TABELA TAXA_RETORNO_EMP DO RECORTE FINAL (faixas nas linhas x meses nas colunas) (%):")
    print((tabelas_taxas["final"] * 100).round(2).to_string())
    print("=" * 80 + "\n")

    return df_recorrencia

def executar_analise_descritiva_parte_f(
    cell_focos_mat: np.ndarray,
    meses_todos: List[str],
    out_dir: Path
):
    """
    Executa a análise descritiva da Parte F:
    Com os alvos de treino (2017 a 2023, nunca o teste), calcula:
    - Fração que teve fogo de novo em até 1, 3, 6, 12, 24 e 36 meses.
    - Distribuição do tempo até o próximo foco (1 a 36 meses).
    - O mesmo separado por estação do primeiro foco (Jan-Jun e Jul-Dez).
    Salva recorrencia_descritiva_treino.csv e o gráfico acadêmico recorrencia_tempo_retorno.png.
    """
    logger.info("=== EXECUTANDO ANÁLISE DESCRITIVA DA RECORRÊNCIA NO TREINO (PARTE F) ===")
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}
    A = (cell_focos_mat > 0).astype(np.int8)

    # Para acompanhar o retorno em até 36 meses sem tocar no teste (teste inicia em 2024-01):
    # Eventos de fogo ocorridos entre 2017-01 e 2020-12 possuem janela completa de 36 meses dentro do treino (<= 2023-12).
    idx_ini = mes_to_idx["2017-01"]
    idx_fim = mes_to_idx["2020-12"]

    events_c, events_m = np.where(A[:, idx_ini : idx_fim + 1] == 1)
    events_m = events_m + idx_ini

    tempos_retorno = []
    estacoes = []

    for c, m in zip(events_c, events_m):
        fut = A[c, m + 1 : m + 37]
        idxs = np.where(fut == 1)[0]
        m_num = int(meses_todos[m][5:7])
        est = "Jan-Jun" if m_num <= 6 else "Jul-Dez"
        estacoes.append(est)
        if len(idxs) > 0:
            tempos_retorno.append(idxs[0] + 1)
        else:
            tempos_retorno.append(np.nan)

    tempos_retorno = np.array(tempos_retorno)
    estacoes = np.array(estacoes)

    n_total = len(tempos_retorno)
    n_jj = int(np.sum(estacoes == "Jan-Jun"))
    n_jd = int(np.sum(estacoes == "Jul-Dez"))

    logger.info(f"Total de eventos de fogo analisados (2017-2020): {n_total:,} (Jan-Jun: {n_jj:,}, Jul-Dez: {n_jd:,})")

    # 1. Tabela descritiva acumulada
    janelas = [1, 3, 6, 12, 24, 36]
    registros_descritivos = []

    for k in janelas:
        pct_geral = float(np.nanmean(tempos_retorno <= k) * 100.0)
        pct_jj = float(np.nanmean(tempos_retorno[estacoes == "Jan-Jun"] <= k) * 100.0)
        pct_jd = float(np.nanmean(tempos_retorno[estacoes == "Jul-Dez"] <= k) * 100.0)

        registros_descritivos.append({
            "janela_meses": f"Ate {k} meses",
            "fracao_retorno_geral_pct": round(pct_geral, 2),
            "fracao_retorno_jan_jun_pct": round(pct_jj, 2),
            "fracao_retorno_jul_dez_pct": round(pct_jd, 2)
        })

    df_desc = pd.DataFrame(registros_descritivos)
    out_csv = out_dir / "resultados" / "recorrencia_descritiva_treino.csv"
    df_desc.to_csv(out_csv, index=False, encoding="utf-8")
    df_desc.to_csv(out_dir / "recorrencia_descritiva_treino.csv", index=False, encoding="utf-8")
    logger.info(f"Tabela recorrencia_descritiva_treino.csv salva em '{out_csv}'.")

    print("\n" + "=" * 80)
    print("      TABELA DESCRITIVA DA RECORRÊNCIA DO FOGO NO TREINO (PARTE F)")
    print("=" * 80)
    print(df_desc.to_string(index=False))
    print("=" * 80 + "\n")

    # 2. Distribuição mensal exata de 1 a 36 meses
    meses_x = np.arange(1, 37)
    freq_geral = np.array([np.sum(tempos_retorno == k) for k in meses_x]) / n_total * 100.0
    freq_jj = np.array([np.sum(tempos_retorno[estacoes == "Jan-Jun"] == k) for k in meses_x]) / n_jj * 100.0
    freq_jd = np.array([np.sum(tempos_retorno[estacoes == "Jul-Dez"] == k) for k in meses_x]) / n_jd * 100.0

    # 3. Gráfico recorrencia_tempo_retorno.png (Estilo acadêmico, sem título interno)
    logger.info("Gerando gráfico acadêmico 'recorrencia_tempo_retorno.png'...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Estilo refinado
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#888888")

    width = 0.8
    bars = ax.bar(meses_x, freq_geral, width=width, color="#1B365D", edgecolor="#0D1B2A", linewidth=0.8, alpha=0.9)

    # Destaque nos picos: mês 1 e mês 12
    bars[0].set_color("#8B0000")  # Vermelho escuro para mês 1
    bars[11].set_color("#B8860B") # Dourado/ocre para mês 12

    ax.set_xlabel("Tempo até o próximo foco na mesma célula (meses)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Frequência de retorno (% de todos os eventos)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xticks(np.arange(1, 37, 2))
    ax.set_xlim(0, 37)
    ax.set_ylim(0, max(freq_geral) * 1.15)

    # Anotações acadêmicas nos picos
    ax.annotate(
        f"Mês seguinte\n({freq_geral[0]:.1f}%)",
        xy=(1, freq_geral[0]),
        xytext=(3, freq_geral[0] + 1.5),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.0),
        fontsize=9, fontweight="bold", color="#8B0000"
    )

    ax.annotate(
        f"Seca seguinte\n(~12 meses: {freq_geral[11]:.1f}%)",
        xy=(12, freq_geral[11]),
        xytext=(14, freq_geral[11] + 2.0),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.0),
        fontsize=9, fontweight="bold", color="#B8860B"
    )

    plt.tight_layout()
    out_img = out_dir / "resultados" / "recorrencia_tempo_retorno.png"
    plt.savefig(out_img, dpi=300)
    plt.savefig(out_dir / "recorrencia_tempo_retorno.png", dpi=300)
    plt.close()
    logger.info(f"Gráfico salvo com sucesso em '{out_img}'.")

def main():
    t_start = time.time()
    logger.info("=== INICIANDO SCRIPT 13: VARIÁVEIS DE RECORRÊNCIA E ANÁLISE DESCRITIVA ===")

    processed_dir = Path("data/processed")
    interim_2015_dir = Path("data/interim_2015")
    out_dir = Path("data/processed_recorrencia")
    resultados_dir = out_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar malha da grade 5 km
    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Arquivo '{gpkg_path}' não encontrado.")
    logger.info(f"Carregando células de '{gpkg_path}'...")
    gdf_cells = gpd.read_file(gpkg_path)

    # 2. Construir matriz histórica de focos (Parte A)
    focos_parquet_path = interim_2015_dir / "firms_focos_pa.parquet"
    if not focos_parquet_path.exists():
        raise FileNotFoundError(f"Arquivo '{focos_parquet_path}' não encontrado.")

    cell_focos_mat, meses_todos, coord_to_idx, W_neigh, degrees = construir_matriz_focos_grade(
        gdf_cells=gdf_cells,
        focos_parquet_path=focos_parquet_path,
        tamanho_m=5000.0
    )

    # 3. Checagem obrigatória da Parte A
    executar_checagem_parte_a(
        cell_focos_mat=cell_focos_mat,
        meses_todos=meses_todos,
        processed_dir=processed_dir
    )

    # 4. Gerar base de modelagem com as 6 variáveis de recorrência (Parte B)
    gerar_base_recorrencia_grade(
        gdf_cells=gdf_cells,
        cell_focos_mat=cell_focos_mat,
        meses_todos=meses_todos,
        W_neigh=W_neigh,
        degrees=degrees,
        out_dir=out_dir
    )

    # 5. Executar análise descritiva no treino e gerar gráfico acadêmico (Parte F)
    executar_analise_descritiva_parte_f(
        cell_focos_mat=cell_focos_mat,
        meses_todos=meses_todos,
        out_dir=out_dir
    )

    logger.info(f"Script 13 concluído com sucesso em {time.time()-t_start:.1f} segundos!")

if __name__ == "__main__":
    main()
