"""
Módulo 08 — Fogo em Floresta e Avaliação pelo Dano (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Objetivo:
Avaliar se os métodos multicritério (Híbrido B) e o perigo priorizam focos em floresta
e vegetação nativa vs áreas antrópicas no período de teste (2024–2025).

Etapas:
1. Extração da classe MapBiomas Coleção 10 para cada foco de teste (492.267 pontos)
   usando o mapa de t-1 (2024 usa 2023; 2025 usa 2024) via Google Earth Engine (scale=30).
2. Agregação mensal por município (focos_floresta, focos_veg_nativa, focos_antropico, focos_outros).
3. Avaliação dos rankings (CapturaFloresta@15, CapturaNativa@15, Captura@15)
   para hibrido_B, hibrido_B_abs, perigo_x_floresta, perigo_isolado, baselines e oráculos.
4. Testes estatísticos pareados (Wilcoxon + Bootstrap BCa + Holm) e composição de dano.
"""

import os
import sys
import time
import math
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import ee

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_08_fogo_floresta")

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    adj_pvals = np.empty(m)
    for i in range(m):
        adj_pvals[i] = (m - i) * sorted_pvals[i]
    for i in range(1, m):
        adj_pvals[i] = max(adj_pvals[i], adj_pvals[i-1])
    adj_pvals = np.minimum(adj_pvals, 1.0)
    rev_indices = np.empty(m, dtype=int)
    rev_indices[sorted_indices] = np.arange(m)
    return adj_pvals[rev_indices]

# Definição dos grupos MapBiomas conforme config.yaml
GRUPOS_MAPBIOMAS = {
    "floresta": [3, 5, 6, 49],
    "veg_nativa_nao_florestal": [4, 11, 12, 29, 32, 50],
    "antropico": [15, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62, 21, 9, 24, 30]
}

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inicializar_gee():
    load_dotenv()
    gee_project = os.getenv("GEE_PROJECT", "aphin-ufpa").strip()
    try:
        ee.Initialize(project=gee_project)
        logger.info(f"Earth Engine inicializado com sucesso no projeto '{gee_project}'.")
    except Exception as e:
        logger.error(f"Erro ao inicializar o Earth Engine: {e}")
        raise

def categorizar_classe(cod: int) -> str:
    if cod in GRUPOS_MAPBIOMAS["floresta"]:
        return "floresta"
    elif cod in GRUPOS_MAPBIOMAS["veg_nativa_nao_florestal"]:
        return "veg_nativa_nao_florestal"
    elif cod in GRUPOS_MAPBIOMAS["antropico"]:
        return "antropico"
    else:
        return "outros"

# ==============================================================================
# PARTE 1 — EXTRAÇÃO DA CLASSE MAPBIOMAS PARA OS FOCOS DE TESTE
# ==============================================================================

def processar_lote_gee(batch_df: pd.DataFrame, lote_id: int, ano_mapa: int, lote_path: Path) -> pd.DataFrame:
    """
    Envia um lote de pontos (até 5.000) para o Earth Engine e amostra a classe do MapBiomas.
    """
    if lote_path.exists():
        return pd.read_parquet(lote_path)

    asset_mapa = f"projects/mapbiomas-public/assets/brazil/lulc/v1/collection_10_{ano_mapa}_v1"
    img = ee.Image(asset_mapa).select("classification")

    features = [
        ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]), {"ponto_id": int(r.ponto_id)})
        for _, r in batch_df.iterrows()
    ]
    fc = ee.FeatureCollection(features)

    max_tentativas = 4
    res = None
    for tentativa in range(1, max_tentativas + 1):
        try:
            sampled = img.sampleRegions(collection=fc, scale=30, geometries=False)
            res = sampled.getInfo()
            break
        except Exception as e:
            if tentativa == max_tentativas:
                logger.error(f"Lote {lote_id:04d} (ano {ano_mapa}) falhou após {max_tentativas} tentativas: {e}")
                raise e
            wait_sec = 2 ** tentativa * 2
            logger.warning(f"Lote {lote_id:04d} tentativa {tentativa} falhou ({e}). Aguardando {wait_sec}s...")
            time.sleep(wait_sec)

    amostrados_dict = {}
    for feat in res.get("features", []):
        props = feat.get("properties", {})
        pid = props.get("ponto_id")
        classe_val = props.get("classification", 0)
        amostrados_dict[pid] = int(classe_val) if classe_val is not None else 0

    classes = [amostrados_dict.get(int(pid), 0) for pid in batch_df["ponto_id"]]
    grupos = [categorizar_classe(c) for c in classes]

    res_df = batch_df.copy()
    res_df["mapbiomas_classe"] = classes
    res_df["grupo"] = grupos

    res_df.to_parquet(lote_path, index=False)
    return res_df

def extrair_classes_focos(firms_path: Path, interim_dir: Path) -> pd.DataFrame:
    """
    Coordena a extração em lotes de 5.000 pontos do MapBiomas para o teste (2024-2025).
    """
    out_final_path = interim_dir / "focos_teste_mapbiomas.parquet"
    if out_final_path.exists():
        logger.info(f"Base de focos classificados já existe em '{out_final_path}'. Carregando...")
        df_final = pd.read_parquet(out_final_path)
        if len(df_final) == 492267:
            return df_final

    lotes_dir = interim_dir / "focos_mapbiomas"
    lotes_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Carregando focos FIRMS de '{firms_path}'...")
    df_firms = pd.read_parquet(firms_path)
    df_firms["acq_date"] = pd.to_datetime(df_firms["acq_date"])
    
    # Filtro estrito: período de teste 2024-01-01 a 2025-12-31
    mask_teste = (df_firms["acq_date"] >= "2024-01-01") & (df_firms["acq_date"] <= "2025-12-31")
    df_teste = df_firms[mask_teste].copy().sort_values(by=["acq_date", "cd_mun"]).reset_index(drop=True)
    n_pontos = len(df_teste)
    logger.info(f"Total de focos no período de teste (2024-2025): {n_pontos}")
    assert n_pontos == 492267, f"Esperado 492.267 focos no teste, obtido {n_pontos}."

    df_teste["ponto_id"] = np.arange(n_pontos)
    df_teste["cd_mun"] = df_teste["cd_mun"].astype(str).str.zfill(7)
    df_teste["ano"] = df_teste["acq_date"].dt.year
    df_teste["mes"] = df_teste["acq_date"].dt.strftime("%Y-%m")

    # Mapeamento do ano anterior para MapBiomas
    # 2024 usa 2023; 2025 usa 2024
    df_teste["ano_mapa"] = df_teste["ano"] - 1

    tamanho_lote = 5000
    lotes_info = []
    lote_counter = 0

    for ano_foco, sub_ano in df_teste.groupby("ano", sort=True):
        ano_mapa = ano_foco - 1
        sub_ano = sub_ano.copy()
        n_sub = len(sub_ano)
        n_lotes = math.ceil(n_sub / tamanho_lote)
        for i in range(n_lotes):
            b_df = sub_ano.iloc[i * tamanho_lote : (i + 1) * tamanho_lote].copy()
            lote_p = lotes_dir / f"lote_{lote_counter:04d}.parquet"
            lotes_info.append((b_df, lote_counter, ano_mapa, lote_p))
            lote_counter += 1

    logger.info(f"Total de lotes a processar: {len(lotes_info)} (lotes de até {tamanho_lote} pontos).")
    inicializar_gee()

    # Processamento paralelo dos lotes via ThreadPoolExecutor
    lotes_resultados = [None] * len(lotes_info)
    pendentes = [info for info in lotes_info if not info[3].exists()]
    logger.info(f"Lotes já computados no disco: {len(lotes_info) - len(pendentes)} / {len(lotes_info)}.")

    if pendentes:
        logger.info(f"Extraindo {len(pendentes)} lotes pendentes no Earth Engine com 6 workers...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            fut_map = {
                executor.submit(processar_lote_gee, b_df, lid, amapa, lp): lid
                for b_df, lid, amapa, lp in pendentes
            }
            concluidos = 0
            for fut in as_completed(fut_map):
                lid = fut_map[fut]
                res_lote = fut.result()
                concluidos += 1
                if concluidos % 10 == 0 or concluidos == len(pendentes):
                    logger.info(f"Progresso da extração GEE: {concluidos}/{len(pendentes)} lotes concluídos.")

    # Carregar todos os lotes ordenados
    dfs_lotes = []
    for _, lid, _, lp in lotes_info:
        dfs_lotes.append(pd.read_parquet(lp))

    df_final = pd.concat(dfs_lotes, ignore_index=True)
    df_final = df_final.sort_values(by="ponto_id").reset_index(drop=True)

    # Validações estritas da Parte 1
    assert len(df_final) == 492267, f"Erro: Esperado 492.267 linhas, obtido {len(df_final)}."
    
    # Proporção sem classe (fora do mapa ou classe 0)
    sem_classe_count = (df_final["mapbiomas_classe"] == 0).sum()
    pct_sem_classe = (sem_classe_count / len(df_final)) * 100.0
    logger.info(f"Focos sem classe (fora do mapa / classe 0): {sem_classe_count} ({pct_sem_classe:.3f}%)")
    assert pct_sem_classe < 0.5, f"Erro: Mais de 0,5% dos focos sem classe ({pct_sem_classe:.3f}%)."

    # Salvar consolidado final
    colunas_salvar = [
        "latitude", "longitude", "acq_date", "ano", "mes", "cd_mun",
        "mapbiomas_classe", "grupo"
    ]
    df_final[colunas_salvar].to_parquet(out_final_path, index=False)
    logger.info(f"Base de focos de teste classificados salva com sucesso em '{out_final_path}'.")

    # Exibir distribuição por grupo e ano
    logger.info("=== DISTRIBUIÇÃO DE FOCOS POR GRUPO E ANO (TESTE) ===")
    for ano_foco, sub in df_final.groupby("ano"):
        logger.info(f"--- Ano {ano_foco} (Total: {len(sub):,} focos) ---")
        g_counts = sub["grupo"].value_counts(normalize=True) * 100.0
        for g_nome, pct in g_counts.items():
            logger.info(f"  {g_nome:<25}: {pct:.2f}%")

    return df_final

# ==============================================================================
# PARTE 2 — AGREGAÇÃO POR MUNICÍPIO × MÊS
# ==============================================================================

def agregar_focos_municipio_mes(
    df_focos: pd.DataFrame,
    modelagem_path: Path,
    processed_dir: Path
) -> pd.DataFrame:
    """
    Agrega os focos de calor por município × mês no período de teste e valida
    a soma dos grupos contra a contagem oficial (y).
    """
    out_dano_path = processed_dir / "focos_dano_municipio_mes.parquet"
    logger.info(f"Agregando focos classificados por município × mês...")

    # Pivot / agregação por município × mês
    agg_df = (
        df_focos.groupby(["cd_mun", "mes", "grupo"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["floresta", "veg_nativa_nao_florestal", "antropico", "outros"]:
        if col not in agg_df.columns:
            agg_df[col] = 0

    agg_df = agg_df.rename(columns={
        "floresta": "focos_floresta",
        "veg_nativa_nao_florestal": "focos_veg_nativa_nao_florestal",
        "antropico": "focos_antropico",
        "outros": "focos_outros"
    })
    agg_df["focos_veg_nativa"] = agg_df["focos_floresta"] + agg_df["focos_veg_nativa_nao_florestal"]

    # Carregar a base de modelagem de teste para garantir grade completa de 3.456 linhas
    df_mod = pd.read_parquet(modelagem_path)
    df_teste_mod = df_mod[df_mod["conjunto"] == "teste"][["cd_mun", "mes_alvo", "y"]].copy()
    df_teste_mod["cd_mun"] = df_teste_mod["cd_mun"].astype(str).str.zfill(7)
    df_teste_mod = df_teste_mod.rename(columns={"mes_alvo": "mes", "y": "focos_count"})

    # Mesclar com a grade completa (3.456 linhas: 144 municípios x 24 meses)
    df_dano = pd.merge(df_teste_mod, agg_df, on=["cd_mun", "mes"], how="left")
    for col in ["focos_floresta", "focos_veg_nativa_nao_florestal", "focos_veg_nativa", "focos_antropico", "focos_outros"]:
        df_dano[col] = df_dano[col].fillna(0).astype(int)

    # Validação estrita: soma dos 3 grupos + outros deve bater exatamente com focos_count
    soma_grupos = (
        df_dano["focos_floresta"] +
        df_dano["focos_veg_nativa_nao_florestal"] +
        df_dano["focos_antropico"] +
        df_dano["focos_outros"]
    )
    diff = np.abs(soma_grupos.values - df_dano["focos_count"].values)
    max_diff = diff.max()
    soma_diff = diff.sum()
    logger.info(f"Validação da soma de focos por município × mês: diferença máxima = {max_diff}, soma das diferenças = {soma_diff}.")
    assert max_diff == 0, f"Erro: A soma dos grupos não bateu com focos_count em {np.sum(diff > 0)} linhas."

    # Salvar resultado
    df_dano.to_parquet(out_dano_path, index=False)
    logger.info(f"Tabela de dano agregada salva com sucesso em '{out_dano_path}' ({len(df_dano)} linhas).")
    return df_dano

# ==============================================================================
# PARTE 3 — AVALIAÇÃO DOS RANKINGS PELO DANO
# ==============================================================================

def criar_ecdf(valores_treino: np.ndarray):
    v_sorted = np.sort(valores_treino)
    n = len(v_sorted)
    def ecdf_func(x):
        return np.searchsorted(v_sorted, x, side="right") / n
    return ecdf_func

def concordancia_parcial(ga: float, gb: float, q: float, p: float) -> float:
    diff = ga - gb
    if diff >= -q:
        return 1.0
    elif diff <= -p:
        return 0.0
    else:
        return float((diff + p) / (p - q))

def classificar_alternativa_electre(
    crit_vals: Dict[str, float],
    weights: Dict[str, float],
    perfis: Dict[str, Tuple[float, float, float]],
    q: float,
    p: float,
    classes_info: List[Tuple[int, str, int]],
    lambda_cut: float = 0.75
) -> Tuple[str, int]:
    for h_idx, cl_nome, cl_ord in classes_info:
        sigma = 0.0
        for c_nome, w in weights.items():
            ga = crit_vals[c_nome]
            gb = perfis[c_nome][h_idx]
            c_val = concordancia_parcial(ga, gb, q, p)
            sigma += w * c_val
        if sigma >= lambda_cut:
            return cl_nome, cl_ord
    return "Baixo", 1

def calcular_hibrido_b_abs(df_treino: pd.DataFrame, df_teste: pd.DataFrame, col_perigo: str, config_mc: dict) -> pd.DataFrame:
    """
    Calcula a fusão ELECTRE Tri-B utilizando desmatamento absoluto em km² (desmat_km2_prodes).
    A ECDF e os perfis são aprendidos exclusivamente no conjunto de treino.
    """
    logger.info("Computando Híbrido B Absoluto (desmat_km2_prodes) no espaço de percentis...")
    df_teste_res = df_teste.copy()

    # ECDFs com dados de treino
    ecdfs = {
        "perigo": criar_ecdf(df_treino["y"].values.astype(float)),
        "desmat_km2_prodes": criar_ecdf(df_treino["desmat_km2_prodes"].values.astype(float)),
        "pct_floresta": criar_ecdf(df_treino["pct_floresta"].values.astype(float)),
        "renda_pc": criar_ecdf((-df_treino["renda_pc"].values).astype(float))
    }

    # Pesos e perfis
    weights = {
        "perigo": 0.50,
        "desmat_km2_prodes": 0.25,
        "pct_floresta": 0.125,
        "renda_pc": 0.125
    }
    quantis = [0.50, 0.75, 0.90]
    perfis = {k: (0.50, 0.75, 0.90) for k in weights}
    limiar_q = float(config_mc["electre"]["limiar_q_percentil"])
    limiar_p = float(config_mc["electre"]["limiar_p_percentil"])
    lambda_cut = float(config_mc["electre"]["corte_lambda"])
    classes_info = [(2, "Critico", 4), (1, "Alto", 3), (0, "Moderado", 2)]

    classes = []
    classes_ordem = []
    for _, r in df_teste_res.iterrows():
        c_vals = {
            "perigo": float(ecdfs["perigo"](r[col_perigo])),
            "desmat_km2_prodes": float(ecdfs["desmat_km2_prodes"](r["desmat_km2_prodes"])),
            "pct_floresta": float(ecdfs["pct_floresta"](r["pct_floresta"])),
            "renda_pc": float(ecdfs["renda_pc"](-r["renda_pc"]))
        }
        cl_nome, cl_ord = classificar_alternativa_electre(
            c_vals, weights, perfis, limiar_q, limiar_p, classes_info, lambda_cut
        )
        classes.append(cl_nome)
        classes_ordem.append(cl_ord)

    df_teste_res["classe_hibrido_abs"] = classes
    df_teste_res["classe_ordem_abs"] = classes_ordem
    df_teste_res["score_hibrido_B_abs"] = df_teste_res["classe_ordem_abs"] * 1e9 + df_teste_res[col_perigo]
    return df_teste_res[["cd_mun", "mes_alvo", "score_hibrido_B_abs"]]

def avaliar_rankings_dano(
    df_dano: pd.DataFrame,
    df_mod: pd.DataFrame,
    resultados_dir: Path,
    config_mc: dict
):
    """
    Avalia os métodos quanto à captura de focos em floresta, vegetação nativa e total.
    Gera tabelas dano_mensal.csv, dano_resumo.csv, dano_testes.csv e dano_composicao_focos.csv.
    """
    logger.info("Iniciando avaliação dos rankings pelo dano ecológico...")

    # Carregar previsões e rankings existentes
    df_el = pd.read_parquet(resultados_dir / "electre_classes_teste.parquet")
    df_v2 = pd.read_parquet(resultados_dir / "features_v2_predicoes_teste.parquet")
    df_base = pd.read_parquet("data/processed/aphim_base_municipio_mes.parquet")

    # Mapeamento do Híbrido B
    classe_ord_map = {"Critico": 4, "Alto": 3, "Moderado": 2, "Baixo": 1}
    df_el["classe_ordem"] = df_el["classe"].map(classe_ord_map)
    df_el["score_hibrido_B"] = df_el["classe_ordem"] * 1e9 + df_el["perigo"]

    # Histórico fixo (total de focos 2019-2023 por município)
    df_base["cd_mun"] = df_base["cd_mun"].astype(str).str.zfill(7)
    mask_hist = (df_base["ano"] >= 2019) & (df_base["ano"] <= 2023)
    hist_fixo_dict = df_base[mask_hist].groupby("cd_mun")["focos_count"].sum().to_dict()

    # Preparar DataFrame de teste integrado
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_t = df_mod[df_mod["conjunto"] == "teste"].copy()
    df_t["cd_mun"] = df_t["cd_mun"].astype(str).str.zfill(7)
    df_t["pred_perigo"] = df_v2["pred_vencedor"].values

    # Adicionar scores existentes
    df_t["score_hibrido_B"] = df_el["score_hibrido_B"].values
    df_t["V"] = df_el["V"].values
    df_t["score_hist_fixo"] = df_t["cd_mun"].map(hist_fixo_dict).fillna(0)

    # Novo método 1: perigo_x_floresta (perigo * pct_floresta / 100)
    df_t["score_perigo_x_floresta"] = df_t["pred_perigo"] * (df_t["pct_floresta"] / 100.0)

    # Novo método 2: hibrido_B_abs
    df_habs = calcular_hibrido_b_abs(df_treino, df_t, "pred_perigo", config_mc)
    df_t["score_hibrido_B_abs"] = df_habs["score_hibrido_B_abs"].values

    # Merge com os dados de dano (focos_floresta, focos_veg_nativa, etc.)
    df_eval = pd.merge(
        df_t,
        df_dano[["cd_mun", "mes", "focos_floresta", "focos_veg_nativa", "focos_antropico", "focos_outros"]],
        left_on=["cd_mun", "mes_alvo"],
        right_on=["cd_mun", "mes"],
        how="inner"
    )

    meses_alvo = sorted(df_eval["mes_alvo"].unique())
    meses_seca = [m for m in meses_alvo if m[5:7] in ["07", "08", "09", "10", "11"]]

    # Totais no período de teste
    total_floresta_periodo = float(df_eval["focos_floresta"].sum())
    total_nativa_periodo = float(df_eval["focos_veg_nativa"].sum())
    total_focos_periodo = float(df_eval["y"].sum())

    metodos_avaliar = [
        ("oraculo_floresta", "focos_floresta"),
        ("oraculo", "y"),
        ("perigo_x_floresta", "score_perigo_x_floresta"),
        ("hibrido_B_abs", "score_hibrido_B_abs"),
        ("hibrido_B", "score_hibrido_B"),
        ("perigo_isolado", "pred_perigo"),
        ("climatologia_focos", "focos_media_hist_mes"),
        ("persistencia", "focos_t"),
        ("historico_fixo", "score_hist_fixo"),
        ("vulnerabilidade_isolada", "V")
    ]

    linhas_mensais = []
    linhas_resumo = []

    for nome_m, col_score in metodos_avaliar:
        acum_floresta_top15 = 0.0
        acum_nativa_top15 = 0.0
        acum_focos_top15 = 0.0

        caps_floresta_mes = []
        caps_nativa_mes = []
        caps_focos_mes = []

        for mes in meses_alvo:
            sub = df_eval[df_eval["mes_alvo"] == mes].copy()
            sub_s = sub.sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            top15 = sub_s.iloc[:15]

            tot_f = float(sub["focos_floresta"].sum())
            tot_n = float(sub["focos_veg_nativa"].sum())
            tot_y = float(sub["y"].sum())

            top_f = float(top15["focos_floresta"].sum())
            top_n = float(top15["focos_veg_nativa"].sum())
            top_y = float(top15["y"].sum())

            cap_f = float(top_f / tot_f) if tot_f > 0 else np.nan
            cap_n = float(top_n / tot_n) if tot_n > 0 else np.nan
            cap_y = float(top_y / tot_y) if tot_y > 0 else np.nan

            caps_floresta_mes.append(cap_f)
            caps_nativa_mes.append(cap_n)
            caps_focos_mes.append(cap_y)

            acum_floresta_top15 += top_f
            acum_nativa_top15 += top_n
            acum_focos_top15 += top_y

            linhas_mensais.append({
                "mes_alvo": mes,
                "metodo": nome_m,
                "captura_floresta_15": cap_f,
                "captura_nativa_15": cap_n,
                "captura_15": cap_y
            })

        df_sub_m = pd.DataFrame([r for r in linhas_mensais if r["metodo"] == nome_m])
        df_sub_seca = df_sub_m[df_sub_m["mes_alvo"].isin(meses_seca)]

        linhas_resumo.append({
            "metodo": nome_m,
            "captura_floresta_15": float(np.nanmean(caps_floresta_mes)),
            "captura_nativa_15": float(np.nanmean(caps_nativa_mes)),
            "captura_15": float(np.nanmean(caps_focos_mes)),
            "captura_agregada_floresta_15": float(acum_floresta_top15 / total_floresta_periodo),
            "captura_agregada_nativa_15": float(acum_nativa_top15 / total_nativa_periodo),
            "captura_agregada_15": float(acum_focos_top15 / total_focos_periodo),
            "captura_floresta_15_seca": float(df_sub_seca["captura_floresta_15"].mean()),
            "captura_nativa_15_seca": float(df_sub_seca["captura_nativa_15"].mean()),
            "captura_15_seca": float(df_sub_seca["captura_15"].mean())
        })

    df_dano_mensal = pd.DataFrame(linhas_mensais)
    df_dano_resumo = pd.DataFrame(linhas_resumo)

    df_dano_mensal.to_csv(resultados_dir / "dano_mensal.csv", index=False, encoding="utf-8")
    df_dano_resumo.to_csv(resultados_dir / "dano_resumo.csv", index=False, encoding="utf-8")
    logger.info("Tabelas dano_mensal.csv e dano_resumo.csv salvas com sucesso.")

    # 4. Testes Estatísticos Pareados (Wilcoxon + Bootstrap + Holm)
    logger.info("Executando testes pareados na CapturaFloresta@15 contra o perigo isolado...")
    rng_boot = np.random.RandomState(42)

    s_perigo = df_dano_mensal[df_dano_mensal["metodo"] == "perigo_isolado"].set_index("mes_alvo")["captura_floresta_15"].loc[meses_alvo].values

    comparacoes_dano = [
        ("hibrido_B", "perigo_isolado"),
        ("hibrido_B_abs", "perigo_isolado"),
        ("perigo_x_floresta", "perigo_isolado")
    ]

    testes_res = []
    p_brutos = []

    for mod, comp in comparacoes_dano:
        s_mod = df_dano_mensal[df_dano_mensal["metodo"] == mod].set_index("mes_alvo")["captura_floresta_15"].loc[meses_alvo].values
        diffs = s_mod - s_perigo
        dif_media = float(np.mean(diffs))

        indices = rng_boot.randint(0, len(diffs), size=(10000, len(diffs)))
        boot_means = np.mean(diffs[indices], axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        res_w = wilcoxon(s_mod, s_perigo, alternative="two-sided")
        pval = float(res_w.pvalue)
        p_brutos.append(pval)

        testes_res.append({
            "modelo": mod,
            "comparado_com": comp,
            "metrica": "captura_floresta_15",
            "diferenca_media": dif_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": pval
        })

    p_holm = holm_bonferroni(np.array(p_brutos))
    for i, p_h in enumerate(p_holm):
        testes_res[i]["p_holm"] = float(p_h)

    df_testes = pd.DataFrame(testes_res)
    df_testes.to_csv(resultados_dir / "dano_testes.csv", index=False, encoding="utf-8")
    logger.info("Tabela dano_testes.csv salva com sucesso.")

    # 5. Composição dos Focos: dano_composicao_focos.csv
    logger.info("Gerando análise de composição de focos (dano_composicao_focos.csv)...")
    
    # Composição por ano
    comp_linhas = []
    for ano_val in [2024, 2025]:
        sub_ano = df_eval[df_eval["mes_alvo"].str.startswith(str(ano_val))]
        f_tot = float(sub_ano["y"].sum())
        f_flor = float(sub_ano["focos_floresta"].sum())
        f_nat = float(sub_ano["focos_veg_nativa"].sum())
        f_ant = float(sub_ano["focos_antropico"].sum())
        f_out = float(sub_ano["focos_outros"].sum())
        comp_linhas.append({
            "categoria": f"Ano {ano_val}",
            "cd_mun": "TODOS",
            "nm_mun": "ESTADO DO PARÁ",
            "focos_total": int(f_tot),
            "pct_floresta": (f_flor / f_tot) * 100.0,
            "pct_veg_nativa": (f_nat / f_tot) * 100.0,
            "pct_antropico": (f_ant / f_tot) * 100.0,
            "pct_outros": (f_out / f_tot) * 100.0
        })

    # Total teste
    f_tot = float(df_eval["y"].sum())
    f_flor = float(df_eval["focos_floresta"].sum())
    f_nat = float(df_eval["focos_veg_nativa"].sum())
    f_ant = float(df_eval["focos_antropico"].sum())
    f_out = float(df_eval["focos_outros"].sum())
    comp_linhas.append({
        "categoria": "Total Teste (2024-2025)",
        "cd_mun": "TODOS",
        "nm_mun": "ESTADO DO PARÁ",
        "focos_total": int(f_tot),
        "pct_floresta": (f_flor / f_tot) * 100.0,
        "pct_veg_nativa": (f_nat / f_tot) * 100.0,
        "pct_antropico": (f_ant / f_tot) * 100.0,
        "pct_outros": (f_out / f_tot) * 100.0
    })

    # Top 15 municípios com mais focos no total do teste
    top15_muns = df_eval.groupby("cd_mun")["y"].sum().sort_values(ascending=False).iloc[:15].index
    mun_nomes_dict = df_base.groupby("cd_mun")["nm_mun"].first().to_dict()

    for cd in top15_muns:
        sub_m = df_eval[df_eval["cd_mun"] == cd]
        f_tot_m = float(sub_m["y"].sum())
        f_flor_m = float(sub_m["focos_floresta"].sum())
        f_nat_m = float(sub_m["focos_veg_nativa"].sum())
        f_ant_m = float(sub_m["focos_antropico"].sum())
        f_out_m = float(sub_m["focos_outros"].sum())
        comp_linhas.append({
            "categoria": "Top 15 Municípios com Mais Focos",
            "cd_mun": cd,
            "nm_mun": mun_nomes_dict.get(cd, cd),
            "focos_total": int(f_tot_m),
            "pct_floresta": (f_flor_m / f_tot_m) * 100.0 if f_tot_m > 0 else 0.0,
            "pct_veg_nativa": (f_nat_m / f_tot_m) * 100.0 if f_tot_m > 0 else 0.0,
            "pct_antropico": (f_ant_m / f_tot_m) * 100.0 if f_tot_m > 0 else 0.0,
            "pct_outros": (f_out_m / f_tot_m) * 100.0 if f_tot_m > 0 else 0.0
        })

    df_composicao = pd.DataFrame(comp_linhas)
    df_composicao.to_csv(resultados_dir / "dano_composicao_focos.csv", index=False, encoding="utf-8")
    logger.info("Tabela dano_composicao_focos.csv salva com sucesso.")

    return df_dano_resumo, df_testes, df_composicao

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    logger.info("==========================================================================================")
    logger.info(" INICIANDO ETAPA 08 — FOGO EM FLORESTA E AVALIAÇÃO PELO DANO ECOLÓGICO (APHIM)")
    logger.info("==========================================================================================")

    config = carregar_config()
    interim_dir = Path(config["caminhos"]["interim"])
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    firms_path = interim_dir / "firms_focos_pa.parquet"
    modelagem_path = processed_dir / "aphim_modelagem.parquet"

    # 1. Extração do MapBiomas para os focos de teste
    df_focos_teste = extrair_classes_focos(firms_path, interim_dir)

    # 2. Agregação por município × mês
    df_dano = agregar_focos_municipio_mes(df_focos_teste, modelagem_path, processed_dir)

    # 3. Avaliação dos rankings pelo dano e testes
    df_resumo, df_testes, df_comp = avaliar_rankings_dano(
        df_dano,
        pd.read_parquet(modelagem_path),
        resultados_dir,
        config.get("multicriterio", {})
    )

    logger.info("\n==========================================================================================")
    logger.info(" ETAPA 08 CONCLUÍDA COM SUCESSO!")
    logger.info("==========================================================================================")

if __name__ == "__main__":
    main()
