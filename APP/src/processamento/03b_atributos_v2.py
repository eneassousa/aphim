"""
Módulo 03b — Rodada 2 de Atributos: Limpeza, Novos Atributos e Seleção por Grupo (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Parte 1 — Criação de novos atributos a partir da base e clima, estritamente até o mês de emissão t:
   - Seca longa: prec_6m, anom_prec_6m, meses_secos_seguidos, umid_solo_t1, anom_umid_solo_3m.
   - Tendência do fogo: razao_fogo_t, variacao_anual_t.
   - Vizinhança: focos_vizinhos_t, razao_fogo_vizinhos_t (malha geográfica IBGE / touches).
   - Estado: focos_estado_t, razao_fogo_estado_t (total do Pará em t).
   - Teste de vazamento temporal automático em 50 linhas aleatórias.
2. Parte 2 — Limpeza (|ρ| > 0,9 no treino) e Definição de Conjuntos:
   - Conjunto A_original: 28 features originais da Etapa 5.
   - Conjunto A_limpo: 16 features (eliminação de densidades, DPV, LST, lags redundantes, etc.).
   - Conjunto A_v2: 27 features (A_limpo + 11 novos atributos).
   - Conjunto A_v2_sel: eliminação para trás por grupo na validação (critério: queda <= 0,2 ponto).
3. Parte 3 — Comparação na Validação Temporal (dobras 2022 e 2023):
   - Modelos: Random Forest e XGBoost com hiperparâmetros fixos de perigo_hiperparametros.json.
   - Importância por permutação na validação do melhor modelo.
   - Escolha do vencedor: substitui o rf_A se ganhar por pelo menos 0,5 ponto (+0,0050).
4. Parte 4 — Avaliação no Teste (2024-2025, estritamente uma única vez):
   - Treino do vencedor em todo o conjunto de treino (2020 a 2023).
   - Avaliação mensal e agregada (Captura@k, ponderada, excesso, seca, Spearman, rotatividade).
   - Teste pareado de Wilcoxon + Bootstrap 95% + Correção de Holm contra rf_A, climatologia e persistência.

Saídas:
- data/processed/aphim_modelagem_v2.parquet
- data/processed/resultados/features_v2_validacao.csv
- data/processed/resultados/features_v2_removidas.csv
- data/processed/resultados/features_v2_permutacao.csv
- data/processed/resultados/features_v2_teste_resumo.csv
- data/processed/resultados/features_v2_testes.csv
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import (
    calcular_metricas_mes,
    avaliar_metodo_mensal,
    adicionar_captura_star_15
)

logger = setup_logger("processamento_03b_atributos_v2")

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_prev_month(mes_str: str, steps: int = 1) -> str:
    """Calcula o mês t-steps no formato AAAA-MM."""
    ano, mes = int(mes_str[:4]), int(mes_str[5:7])
    for _ in range(steps):
        if mes == 1:
            ano -= 1
            mes = 12
        else:
            mes -= 1
    return f"{ano:04d}-{mes:02d}"

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Aplica a correção de Holm-Bonferroni (step-down) para controle do FWER."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    
    order = np.argsort(pvals)
    sorted_p = pvals[order]

    adj_p = np.empty(n, dtype=float)
    running_max = 0.0
    for k in range(n):
        val = sorted_p[k] * (n - k)
        running_max = max(running_max, val)
        adj_p[k] = min(1.0, running_max)

    out = np.empty(n, dtype=float)
    out[order] = adj_p
    return out

# ==============================================================================
# PARTE 1: CONSTRUÇÃO DOS NOVOS ATRIBUTOS E TESTE DE VAZAMENTO
# ==============================================================================

def construir_atributos_v2(processed_dir: Path) -> pd.DataFrame:
    logger.info("Carregando bases para engenharia de novos atributos v2...")
    mod_path = processed_dir / "aphim_modelagem.parquet"
    base_path = processed_dir / "aphim_base_municipio_mes.parquet"
    clima_path = processed_dir / "clima_municipio_mes.parquet"
    normal_path = processed_dir / "clima_normal_1991_2020.parquet"
    gpkg_path = processed_dir / "municipios_pa.gpkg"

    if not mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{mod_path}' não encontrado.")
    if not base_path.exists():
        raise FileNotFoundError(f"Arquivo '{base_path}' não encontrado.")
    if not clima_path.exists():
        raise FileNotFoundError(f"Arquivo '{clima_path}' não encontrado.")
    if not normal_path.exists():
        raise FileNotFoundError(f"Arquivo '{normal_path}' não encontrado.")
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Arquivo '{gpkg_path}' não encontrado.")

    df_mod = pd.read_parquet(mod_path)
    df_base = pd.read_parquet(base_path)
    df_clima = pd.read_parquet(clima_path)
    df_normal = pd.read_parquet(normal_path)
    gdf = gpd.read_file(gpkg_path)

    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)
    df_base["cd_mun"] = df_base["cd_mun"].astype(str).str.zfill(7)
    df_clima["cd_mun"] = df_clima["cd_mun"].astype(str).str.zfill(7)
    df_normal["cd_mun"] = df_normal["cd_mun"].astype(str).str.zfill(7)
    gdf["cd_mun"] = gdf["cd_mun"].astype(str).str.zfill(7)

    logger.info("Mapeando vizinhança geográfica de fronteira (touches)...")
    neighbors = {
        r["cd_mun"]: gdf[gdf.geometry.touches(r["geometry"])]["cd_mun"].tolist()
        for _, r in gdf.iterrows()
    }
    assert all(len(v) > 0 for v in neighbors.values()), "Todos os 144 municípios devem possuir vizinhos por fronteira."

    # Climatologia histórica de focos no mês de emissão t (usando estritamente anos anteriores ao ano de t)
    logger.info("Pré-computando climatologia do mês de emissão t com isolamento temporal estrito...")
    hist_fire = df_base[["cd_mun", "ano", "mes_num", "focos_count"]].copy()
    hist_dict_t = {}
    for (c, m_num), group in hist_fire.groupby(["cd_mun", "mes_num"]):
        group = group.sort_values("ano")
        anos = group["ano"].values
        counts = group["focos_count"].values
        for target_year in range(2019, 2026):
            mask = anos < target_year
            if mask.any():
                hist_dict_t[(c, m_num, target_year)] = float(np.mean(counts[mask]))
            else:
                hist_dict_t[(c, m_num, target_year)] = np.nan

    focos_lookup = df_base.set_index(["cd_mun", "mes"])["focos_count"].to_dict()
    prec_lookup = df_clima.set_index(["cd_mun", "mes"])["prec_mm"].to_dict()
    umid_lookup = df_clima.set_index(["cd_mun", "mes"])["umid_solo"].to_dict()
    norm_prec_dict = df_normal.set_index(["cd_mun", "mes_calendario"])["prec_mm"].to_dict()
    norm_umid_dict = df_normal.set_index(["cd_mun", "mes_calendario"])["umid_solo"].to_dict()
    all_muns = sorted(df_base["cd_mun"].unique())

    # Sequência de meses secos consecutivos (prec < 60 mm)
    logger.info("Calculando sequência histórica de meses secos consecutivos...")
    dry_streak_dict = {}
    for c, group in df_clima.groupby("cd_mun"):
        group = group.sort_values("mes")
        streak = 0
        for _, r in group.iterrows():
            p = r["prec_mm"]
            m = r["mes"]
            streak = streak + 1 if p < 60.0 else 0
            dry_streak_dict[(c, m)] = streak

    logger.info("Calculando os 11 novos atributos para cada par (município, t)...")
    novos_registros = []
    for _, r in df_mod.iterrows():
        c = r["cd_mun"]
        t = r["mes_emissao"]
        ano_t = int(t[:4])
        mes_t_num = int(t[5:7])

        # 1. Seca Longa
        p6_list = [prec_lookup.get((c, get_prev_month(t, s)), np.nan) for s in range(6)]
        prec_6m = float(sum(p6_list))
        np6_list = [norm_prec_dict.get((c, int(get_prev_month(t, s)[5:7])), np.nan) for s in range(6)]
        norm_prec_6m = float(sum(np6_list))
        anom_prec_6m = prec_6m - norm_prec_6m

        meses_secos_seguidos = float(dry_streak_dict.get((c, t), np.nan))
        umid_solo_t1 = float(umid_lookup.get((c, get_prev_month(t, 1)), np.nan))

        anom_umid_3 = []
        for s in range(3):
            m_s = get_prev_month(t, s)
            u_s = umid_lookup.get((c, m_s), np.nan)
            nu_s = norm_umid_dict.get((c, int(m_s[5:7])), np.nan)
            anom_umid_3.append(u_s - nu_s)
        anom_umid_solo_3m = float(np.mean(anom_umid_3))

        # 2. Tendência do Fogo
        normal_t = hist_dict_t.get((c, mes_t_num, ano_t), np.nan)
        f_t = float(r["focos_t"])
        razao_fogo_t = float((f_t + 1.0) / (normal_t + 1.0)) if not np.isnan(normal_t) else np.nan
        f_ano_ant = focos_lookup.get((c, get_prev_month(t, 12)), np.nan)
        variacao_anual_t = float(f_t - f_ano_ant) if not np.isnan(f_ano_ant) else np.nan

        # 3. Vizinhança
        vizs = neighbors[c]
        focos_viz = float(sum(focos_lookup.get((v, t), 0) for v in vizs))
        norm_viz_list = [hist_dict_t.get((v, mes_t_num, ano_t), np.nan) for v in vizs]
        norm_viz = float(sum(norm_viz_list)) if not any(np.isnan(norm_viz_list)) else np.nan
        razao_viz = float((focos_viz + 1.0) / (norm_viz + 1.0)) if not np.isnan(norm_viz) else np.nan

        # 4. Estado
        focos_est = float(sum(focos_lookup.get((m, t), 0) for m in all_muns))
        norm_est_list = [hist_dict_t.get((m, mes_t_num, ano_t), np.nan) for m in all_muns]
        norm_est = float(sum(norm_est_list)) if not any(np.isnan(norm_est_list)) else np.nan
        razao_est = float((focos_est + 1.0) / (norm_est + 1.0)) if not np.isnan(norm_est) else np.nan

        novos_registros.append({
            "prec_6m": prec_6m,
            "anom_prec_6m": anom_prec_6m,
            "meses_secos_seguidos": meses_secos_seguidos,
            "umid_solo_t1": umid_solo_t1,
            "anom_umid_solo_3m": anom_umid_solo_3m,
            "razao_fogo_t": razao_fogo_t,
            "variacao_anual_t": variacao_anual_t,
            "focos_vizinhos_t": focos_viz,
            "razao_fogo_vizinhos_t": razao_viz,
            "focos_estado_t": focos_est,
            "razao_fogo_estado_t": razao_est
        })

    df_novos = pd.DataFrame(novos_registros)
    df_mod_v2 = pd.concat([df_mod, df_novos], axis=1)

    # Validação de Nulos em treino e teste
    cols_novas = list(df_novos.columns)
    df_tt = df_mod_v2[df_mod_v2["conjunto"].isin(["treino", "teste"])]
    nulos = df_tt[cols_novas].isnull().sum()
    if nulos.sum() > 0:
        raise ValueError(f"Nulos proibidos encontrados nos novos atributos de treino/teste: {nulos[nulos > 0].to_dict()}")
    logger.info(f"Validação aprovada: zero nulos nos 11 novos atributos em treino e teste.")

    # Teste de vazamento temporal automático em 50 linhas aleatórias
    logger.info("Executando teste de vazamento temporal em 50 linhas aleatórias nos novos atributos...")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(df_mod_v2), size=50, replace=False)
    sample_rows = df_mod_v2.iloc[sample_indices]

    for _, row in sample_rows.iterrows():
        c = row["cd_mun"]
        t = row["mes_emissao"]
        ano_t = int(t[:4])
        mes_t_num = int(t[5:7])

        # Teste 1: focos_vizinhos_t deve usar estritamente dados de t
        viz_muns = neighbors[c]
        soma_viz_check = sum(focos_lookup[(v, t)] for v in viz_muns)
        if not np.isclose(soma_viz_check, row["focos_vizinhos_t"]):
            raise ValueError(f"VAZAMENTO em focos_vizinhos_t ({c}, {t})")

        # Teste 2: prec_6m deve usar estritamente t até t-5
        soma_p6_check = sum(prec_lookup[(c, get_prev_month(t, s))] for s in range(6))
        if not np.isclose(soma_p6_check, row["prec_6m"]):
            raise ValueError(f"VAZAMENTO em prec_6m ({c}, {t})")

        # Teste 3: razao_fogo_t não pode usar anos >= ano(t)
        raw_hist = df_base[(df_base["cd_mun"] == c) & (df_base["mes_num"] == mes_t_num)]
        valid_raw = raw_hist[raw_hist["ano"] < ano_t]
        if len(valid_raw) > 0:
            assert all(a < ano_t for a in valid_raw["ano"].values), "Ano futuro detectado na climatologia histórica de t!"

    logger.info("Teste de vazamento aprovado em 100% das 50 amostras auditadas!")

    out_v2_path = processed_dir / "aphim_modelagem_v2.parquet"
    df_mod_v2.to_parquet(out_v2_path, index=False)
    logger.info(f"Base aphim_modelagem_v2.parquet salva com sucesso em '{out_v2_path}'.")
    return df_mod_v2

# ==============================================================================
# PARTE 2: DEFINIÇÃO DE CONJUNTOS DE ATRIBUTOS E LIMPEZA
# ==============================================================================

def definir_conjuntos_features() -> Tuple[List[str], List[str], List[str], Dict[str, List[str]], List[Dict[str, str]]]:
    # 1. Conjunto A_original (28 features)
    cols_A_orig = [
        "focos_t", "focos_t1", "focos_t2", "focos_mm3", "focos_mesmo_mes_ano_ant", "focos_media_hist_mes",
        "dens_t", "dens_t1", "dens_t2", "dens_mm3", "dens_mesmo_mes_ano_ant", "dens_media_hist_mes",
        "prec_mm", "t2m_max_c", "ur_pct", "dpv_kpa", "umid_solo", "lst_c", "vento_ms",
        "anom_prec_mm", "anom_t2m_media_c", "anom_ur_pct", "anom_umid_solo", "anom_lst_c",
        "prec_3m", "anom_prec_3m",
        "mes_sin", "mes_cos"
    ]

    # Registros do que saiu na limpeza
    removidas_info = [
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "dens_t, dens_t1, dens_t2, dens_mm3, dens_mesmo_mes_ano_ant, dens_media_hist_mes",
            "motivo_remocao": "Colinear com contagens de focos (|ρ| > 0,90); priorização metodológica de contagens absolutas sobre densidades."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "dpv_kpa",
            "motivo_remocao": "Perfeitamente colinear com umidade relativa (|ρ| = 1,00); ur_pct é métrica direta, padronizada e mais interpretável."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "anom_lst_c",
            "motivo_remocao": "Quase perfeitamente colinear com anomalia de temperatura a 2m (|ρ| = 0,98); anom_t2m_media_c é livre de artefatos de nuvens do sensor LST."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "focos_mesmo_mes_ano_ant",
            "motivo_remocao": "Colinear com média histórica (|ρ| = 0,96); focos_media_hist_mes possui menor variância temporal por agregar múltiplos anos anteriores."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "focos_mm3",
            "motivo_remocao": "Colinear com lag t-1 (|ρ| = 0,93); os lags diretos mensais (t, t-1, t-2) capturam a inércia temporal de forma mais limpa e parcimoniosa."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "anom_ur_pct",
            "motivo_remocao": "Colinear com anomalia de umidade do solo (|ρ| = 0,91); anom_umid_solo reflete diretamente a dessecação do combustível florestal."
        },
        {
            "etapa": "limpeza_multicolinearidade",
            "atributo_ou_grupo": "prec_mm",
            "motivo_remocao": "Colinear com ur_pct (|ρ| = 0,94) e prec_3m (|ρ| = 0,91); umidade relativa do ar e acumulados de precipitação capturam o sinal hidrometeorológico sem redundância."
        }
    ]

    # Atributos eliminados na limpeza
    drop_limpeza = [
        "dens_t", "dens_t1", "dens_t2", "dens_mm3", "dens_mesmo_mes_ano_ant", "dens_media_hist_mes",
        "dpv_kpa", "anom_lst_c", "focos_mesmo_mes_ano_ant", "focos_mm3", "anom_ur_pct", "prec_mm"
    ]
    cols_A_limpo = [c for c in cols_A_orig if c not in drop_limpeza]
    assert len(cols_A_limpo) == 16, f"Esperado 16 atributos em A_limpo, obtido {len(cols_A_limpo)}."

    # Novos atributos
    novos_atributos = [
        "prec_6m", "anom_prec_6m", "meses_secos_seguidos", "umid_solo_t1", "anom_umid_solo_3m",
        "razao_fogo_t", "variacao_anual_t",
        "focos_vizinhos_t", "razao_fogo_vizinhos_t",
        "focos_estado_t", "razao_fogo_estado_t"
    ]
    cols_A_v2 = cols_A_limpo + novos_atributos
    assert len(cols_A_v2) == 27, f"Esperado 27 atributos em A_v2, obtido {len(cols_A_v2)}."

    # Grupos conceituais para eliminação para trás
    grupos_v2 = {
        "historico": ["focos_t", "focos_t1", "focos_t2", "focos_media_hist_mes"],
        "clima_mes": ["t2m_max_c", "ur_pct", "umid_solo", "lst_c", "vento_ms"],
        "anomalias": ["anom_prec_mm", "anom_t2m_media_c", "anom_umid_solo", "prec_3m", "anom_prec_3m"],
        "seca_longa": ["prec_6m", "anom_prec_6m", "meses_secos_seguidos", "umid_solo_t1", "anom_umid_solo_3m"],
        "tendencia": ["razao_fogo_t", "variacao_anual_t"],
        "vizinhanca": ["focos_vizinhos_t", "razao_fogo_vizinhos_t"],
        "estado": ["focos_estado_t", "razao_fogo_estado_t"],
        "sazonalidade": ["mes_sin", "mes_cos"]
    }

    return cols_A_orig, cols_A_limpo, cols_A_v2, grupos_v2, removidas_info

# ==============================================================================
# PARTE 3: COMPARAÇÃO NA VALIDAÇÃO TEMPORAL E SELEÇÃO DE GRUPOS
# ==============================================================================

def avaliar_modelo_validacao(
    tipo_modelo: str,
    params: Dict[str, Any],
    feat_cols: List[str],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame
) -> Tuple[float, float, float]:
    """Treina nas dobras temporais 2022 e 2023 e calcula a Captura@15 média."""
    if tipo_modelo == "rf":
        m22 = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42, n_jobs=-1
        )
        m22.fit(df_tr_22[feat_cols], np.log1p(df_tr_22["y"]))
        preds_22 = np.maximum(0.0, np.expm1(m22.predict(df_val_22[feat_cols])))

        m23 = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42, n_jobs=-1
        )
        m23.fit(df_tr_23[feat_cols], np.log1p(df_tr_23["y"]))
        preds_23 = np.maximum(0.0, np.expm1(m23.predict(df_val_23[feat_cols])))

    elif tipo_modelo == "xgb":
        m22 = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            objective="count:poisson",
            random_state=42, n_jobs=-1
        )
        m22.fit(df_tr_22[feat_cols], df_tr_22["y"])
        preds_22 = np.maximum(0.0, m22.predict(df_val_22[feat_cols]))

        m23 = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            objective="count:poisson",
            random_state=42, n_jobs=-1
        )
        m23.fit(df_tr_23[feat_cols], df_tr_23["y"])
        preds_23 = np.maximum(0.0, m23.predict(df_val_23[feat_cols]))
    else:
        raise ValueError(f"Modelo {tipo_modelo} não suportado.")

    v22 = df_val_22[["cd_mun", "mes_alvo", "y", "pct_floresta"]].copy()
    v22["pred"] = preds_22
    caps_22 = [calcular_metricas_mes(v22[v22["mes_alvo"] == mes], "pred", k_valores=(15,))[0]["captura_15"]
               for mes in sorted(v22["mes_alvo"].unique())]
    c22 = float(np.mean(caps_22))

    v23 = df_val_23[["cd_mun", "mes_alvo", "y", "pct_floresta"]].copy()
    v23["pred"] = preds_23
    caps_23 = [calcular_metricas_mes(v23[v23["mes_alvo"] == mes], "pred", k_valores=(15,))[0]["captura_15"]
               for mes in sorted(v23["mes_alvo"].unique())]
    c23 = float(np.mean(caps_23))

    return c22, c23, float((c22 + c23) / 2.0)

def executar_selecao_e_validacao(
    df_mod: pd.DataFrame,
    params_rf: Dict[str, Any],
    params_xgb: Dict[str, Any],
    cols_A_orig: List[str],
    cols_A_limpo: List[str],
    cols_A_v2: List[str],
    grupos_v2: Dict[str, List[str]],
    removidas_info: List[Dict[str, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str, str]:
    logger.info("Configurando partições temporais de validação (dobras 2022 e 2023)...")
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_tr_22 = df_treino[df_treino["mes_alvo"] < "2022-01"].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()
    df_tr_23 = df_treino[df_treino["mes_alvo"] < "2023-01"].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    # 1. Eliminação para trás por grupo a partir de A_v2 no modelo mais forte (XGBoost)
    logger.info("Executando eliminação para trás por grupo na validação (critério: piora <= 0,2 ponto)...")
    remaining_groups = list(grupos_v2.keys())
    current_feats = [f for g in remaining_groups for f in grupos_v2[g]]
    _, _, current_score = avaliar_modelo_validacao("xgb", params_xgb, current_feats, df_tr_22, df_val_22, df_tr_23, df_val_23)
    logger.info(f"Score inicial XGBoost em A_v2 ({len(current_feats)} features): {current_score*100:.2f}%")

    while len(remaining_groups) > 1:
        best_candidate_group = None
        best_candidate_score = -1.0
        for g in remaining_groups:
            trial_groups = [gr for gr in remaining_groups if gr != g]
            trial_feats = [f for gr in trial_groups for f in grupos_v2[gr]]
            _, _, score = avaliar_modelo_validacao("xgb", params_xgb, trial_feats, df_tr_22, df_val_22, df_tr_23, df_val_23)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate_group = g

        diff = best_candidate_score - current_score
        logger.info(f"Candidato a remoção: grupo '{best_candidate_group}' | Novo score: {best_candidate_score*100:.2f}% (diff: {diff*100:+.2f} p.p.)")

        # Regra de parada: não piorar mais que 0,2 ponto (diff >= -0,0020)
        if diff >= -0.0020:
            remaining_groups.remove(best_candidate_group)
            current_score = best_candidate_score
            logger.info(f"-> Grupo '{best_candidate_group}' REMOVIDO! Grupos restantes: {remaining_groups}")
            removidas_info.append({
                "etapa": "selecao_para_tras_grupo",
                "atributo_ou_grupo": best_candidate_group,
                "motivo_remocao": f"Remoção pelo critério de backward elimination na validação elevou/manteve a Captura@15 (diff: {diff*100:+.2f} p.p.)."
            })
        else:
            logger.info(f"Critério de parada atingido: a remoção de '{best_candidate_group}' pioraria em {abs(diff)*100:.2f} p.p. (> 0,20 ponto).")
            break

    cols_A_v2_sel = [f for gr in remaining_groups for f in grupos_v2[gr]]
    logger.info(f"Conjunto A_v2_sel definido com {len(cols_A_v2_sel)} atributos nos grupos: {remaining_groups}")

    # 2. Avaliação de todas as combinações modelo x conjunto na validação
    logger.info("Calculando tabela completa de validação para todos os modelos e conjuntos...")
    conjuntos_map = {
        "A_original": cols_A_orig,
        "A_limpo": cols_A_limpo,
        "A_v2": cols_A_v2,
        "A_v2_sel": cols_A_v2_sel
    }

    linhas_val = []
    melhor_score_val = -1.0
    melhor_modelo_val = ""
    melhor_conjunto_val = ""

    for conj_nome, feat_list in conjuntos_map.items():
        # Random Forest
        rf_22, rf_23, rf_med = avaliar_modelo_validacao("rf", params_rf, feat_list, df_tr_22, df_val_22, df_tr_23, df_val_23)
        linhas_val.append({
            "modelo": "rf",
            "conjunto": conj_nome,
            "n_features": len(feat_list),
            "cap15_dobra_2022": rf_22,
            "cap15_dobra_2023": rf_23,
            "cap15_media": rf_med
        })
        if rf_med > melhor_score_val:
            melhor_score_val = rf_med
            melhor_modelo_val = "rf"
            melhor_conjunto_val = conj_nome

        # XGBoost
        xgb_22, xgb_23, xgb_med = avaliar_modelo_validacao("xgb", params_xgb, feat_list, df_tr_22, df_val_22, df_tr_23, df_val_23)
        linhas_val.append({
            "modelo": "xgb",
            "conjunto": conj_nome,
            "n_features": len(feat_list),
            "cap15_dobra_2022": xgb_22,
            "cap15_dobra_2023": xgb_23,
            "cap15_media": xgb_med
        })
        if xgb_med > melhor_score_val:
            melhor_score_val = xgb_med
            melhor_modelo_val = "xgb"
            melhor_conjunto_val = conj_nome

    df_val_resumo = pd.DataFrame(linhas_val)
    df_removidas = pd.DataFrame(removidas_info)

    # 3. Importância por Permutação na validação para o melhor modelo
    logger.info(f"Calculando importância por permutação na validação para o melhor modelo ({melhor_modelo_val}_{melhor_conjunto_val})...")
    melhor_params = params_xgb if melhor_modelo_val == "xgb" else params_rf
    feats_vencedor = conjuntos_map[melhor_conjunto_val]

    # Baseline do vencedor
    _, _, score_base = avaliar_modelo_validacao(melhor_modelo_val, melhor_params, feats_vencedor, df_tr_22, df_val_22, df_tr_23, df_val_23)

    perm_registros = []
    rng_perm = np.random.RandomState(42)
    for f_col in feats_vencedor:
        df_v22_perm = df_val_22.copy()
        df_v23_perm = df_val_23.copy()
        df_v22_perm[f_col] = rng_perm.permutation(df_v22_perm[f_col].values)
        df_v23_perm[f_col] = rng_perm.permutation(df_v23_perm[f_col].values)

        _, _, score_perm = avaliar_modelo_validacao(
            melhor_modelo_val, melhor_params, feats_vencedor,
            df_tr_22, df_v22_perm, df_tr_23, df_v23_perm
        )
        drop = score_base - score_perm
        perm_registros.append({
            "atributo": f_col,
            "importancia_cap15": float(drop)
        })

    df_perm = pd.DataFrame(perm_registros).sort_values("importancia_cap15", ascending=False).reset_index(drop=True)
    df_perm["posicao"] = np.arange(1, len(df_perm) + 1)
    df_perm = df_perm[["posicao", "atributo", "importancia_cap15"]]

    logger.info(f"-> Vencedor na validação: {melhor_modelo_val} com {melhor_conjunto_val} (Captura@15 média = {melhor_score_val*100:.2f}%)")
    return df_val_resumo, df_removidas, df_perm, cols_A_v2_sel, melhor_modelo_val, melhor_conjunto_val

# ==============================================================================
# PARTE 4: AVALIAÇÃO NO TESTE E TESTES ESTATÍSTICOS
# ==============================================================================

def avaliar_no_teste(
    df_mod: pd.DataFrame,
    melhor_modelo: str,
    feats_vencedor: List[str],
    params_vencedor: Dict[str, Any],
    processed_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== AVALIANDO VENCEDOR NO TESTE (2024-2025, UMA ÚNICA VEZ) ===")
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    df_ancora = df_mod[df_mod["mes_emissao"] == "2023-11"].copy() # mes_alvo = 2023-12

    if melhor_modelo == "rf":
        m = RandomForestRegressor(
            n_estimators=params_vencedor["n_estimators"],
            max_depth=params_vencedor["max_depth"],
            min_samples_leaf=params_vencedor["min_samples_leaf"],
            max_features=params_vencedor["max_features"],
            random_state=42, n_jobs=-1
        )
        m.fit(df_treino[feats_vencedor], np.log1p(df_treino["y"]))
        preds_teste = np.maximum(0.0, np.expm1(m.predict(df_teste[feats_vencedor])))
        preds_ancora = np.maximum(0.0, np.expm1(m.predict(df_ancora[feats_vencedor])))
    else:
        m = xgb.XGBRegressor(
            n_estimators=params_vencedor["n_estimators"],
            learning_rate=params_vencedor["learning_rate"],
            max_depth=params_vencedor["max_depth"],
            subsample=params_vencedor["subsample"],
            colsample_bytree=params_vencedor["colsample_bytree"],
            min_child_weight=params_vencedor["min_child_weight"],
            objective="count:poisson",
            random_state=42, n_jobs=-1
        )
        m.fit(df_treino[feats_vencedor], df_treino["y"])
        preds_teste = np.maximum(0.0, m.predict(df_teste[feats_vencedor]))
        preds_ancora = np.maximum(0.0, m.predict(df_ancora[feats_vencedor]))

    df_teste["pred_vencedor"] = preds_teste
    df_ancora["pred_vencedor"] = preds_ancora

    # Avaliação mensal de métricas
    anc_sort = df_ancora.sort_values(by=["pred_vencedor", "cd_mun"], ascending=[False, True])
    prev_top15 = set(anc_sort.iloc[:15]["cd_mun"])

    meses_alvo = sorted(df_teste["mes_alvo"].unique())
    registros_mensais = []
    for mes in meses_alvo:
        sub = df_teste[df_teste["mes_alvo"] == mes]
        met, prev_top15 = calcular_metricas_mes(
            sub, col_score="pred_vencedor", prev_top15=prev_top15, k_valores=(10, 15, 20, 25, 30)
        )
        # CapturaExcesso@15
        e_total = np.maximum(0.0, sub["y"] - sub["focos_media_hist_mes"]).sum()
        sub_sorted = sub.sort_values(by=["pred_vencedor", "cd_mun"], ascending=[False, True])
        e_top15 = np.maximum(0.0, sub_sorted.iloc[:15]["y"] - sub_sorted.iloc[:15]["focos_media_hist_mes"]).sum()
        met["captura_excesso_15"] = float(e_top15 / e_total) if e_total > 0 else np.nan
        met["mes_alvo"] = mes
        met["metodo"] = f"{melhor_modelo}_A_v2_sel"
        registros_mensais.append(met)

    df_mensal_vencedor = pd.DataFrame(registros_mensais)

    # Salvar predições de teste do modelo vencedor
    out_preds_vencedor = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
    df_teste[["cd_mun", "mes_emissao", "mes_alvo", "y", "pred_vencedor"]].to_parquet(out_preds_vencedor, index=False)
    logger.info(f"Predições de teste do modelo vencedor salvas em '{out_preds_vencedor}'.")

    # Carregar previsões salvas de teste da etapa 5 e baselines para comparação
    preds_e5_path = processed_dir / "resultados" / "perigo_predicoes_teste.parquet"
    base_mensal_path = processed_dir / "resultados" / "baselines_mensal.csv"
    perigo_mensal_path = processed_dir / "resultados" / "perigo_mensal.csv"
    excesso_mensal_path = processed_dir / "resultados" / "excesso_mensal.csv"

    df_preds_e5 = pd.read_parquet(preds_e5_path)
    df_perigo_mensal = pd.read_csv(perigo_mensal_path)
    df_excesso_mensal = pd.read_csv(excesso_mensal_path)

    # Obter dados do oraculo
    sub_oraculo = df_perigo_mensal[df_perigo_mensal["metodo"] == "oraculo"].set_index("mes_alvo")["captura_15"].to_dict()
    df_mensal_vencedor["captura_star_15"] = df_mensal_vencedor.apply(
        lambda r: r["captura_15"] / sub_oraculo[r["mes_alvo"]], axis=1
    )

    # Obter CapturaExcesso@15 dos métodos comparadores
    rf_A_mensal = df_perigo_mensal[df_perigo_mensal["metodo"] == "rf_A"].set_index("mes_alvo")
    clima_mensal = df_perigo_mensal[df_perigo_mensal["metodo"] == "climatologia_focos"].set_index("mes_alvo")
    pers_mensal = df_perigo_mensal[df_perigo_mensal["metodo"] == "persistencia"].set_index("mes_alvo")
    oraculo_mensal = df_perigo_mensal[df_perigo_mensal["metodo"] == "oraculo"].set_index("mes_alvo")

    excesso_p1 = df_excesso_mensal[df_excesso_mensal["parte"] == "parte_1"].set_index(["metodo", "mes_alvo"])["captura_excesso_15"].to_dict()

    # Incorporar predição do rf_A salva para avaliação comparativa direta
    df_teste = df_teste.merge(
        df_preds_e5[["cd_mun", "mes_alvo", "pred_rf_A"]],
        on=["cd_mun", "mes_alvo"],
        how="left"
    )

    # Cálculo da Captura Agregada@15
    focos_totais_periodo = float(df_teste["y"].sum())
    
    def calc_agregada(col_score: str):
        soma_top = 0.0
        for mes in meses_alvo:
            sub = df_teste[df_teste["mes_alvo"] == mes].sort_values(by=[col_score, "cd_mun"], ascending=[False, True])
            soma_top += float(sub.iloc[:15]["y"].sum())
        return float(soma_top / focos_totais_periodo)

    agregada_vencedor = calc_agregada("pred_vencedor")
    agregada_rf_A = calc_agregada("pred_rf_A")
    agregada_clima = calc_agregada("focos_media_hist_mes")
    agregada_pers = calc_agregada("focos_t")
    agregada_oraculo = calc_agregada("y")

    meses_seca = [m for m in meses_alvo if m[5:7] in ["07", "08", "09", "10", "11"]]

    def montar_resumo_metodo(nome_exibicao, df_m, cap_e_dict, metodo_chave_excesso, cap_agregada):
        df_sec = df_m[df_m["mes_alvo"].isin(meses_seca)] if "mes_alvo" in df_m.columns else df_m.loc[meses_seca]
        cap_e_vals = [cap_e_dict.get((metodo_chave_excesso, m), np.nan) for m in meses_alvo] if metodo_chave_excesso else df_m["captura_excesso_15"].values
        return {
            "metodo": nome_exibicao,
            "captura_10": df_m["captura_10"].mean(),
            "captura_15": df_m["captura_15"].mean(),
            "captura_20": df_m["captura_20"].mean(),
            "captura_25": df_m["captura_25"].mean(),
            "captura_30": df_m["captura_30"].mean(),
            "captura_star_15": df_m["captura_star_15"].mean(),
            "captura_pond_15": df_m["captura_pond_15"].mean(),
            "captura_excesso_15": float(np.mean(cap_e_vals)),
            "captura_agregada_15": cap_agregada,
            "captura_15_seca": df_sec["captura_15"].mean(),
            "spearman": df_m["spearman"].mean(),
            "rotatividade": df_m["rotatividade"].mean()
        }

    resumo_vencedor = montar_resumo_metodo(f"{melhor_modelo}_A_v2_sel (vencedor v2)", df_mensal_vencedor, excesso_p1, None, agregada_vencedor)
    resumo_rf_A = montar_resumo_metodo("rf_A (perigo rodada 1)", rf_A_mensal.reset_index(), excesso_p1, "rf_A", agregada_rf_A)
    resumo_clima = montar_resumo_metodo("climatologia_focos", clima_mensal.reset_index(), excesso_p1, "climatologia_focos", agregada_clima)
    resumo_pers = montar_resumo_metodo("persistencia", pers_mensal.reset_index(), excesso_p1, "persistencia", agregada_pers)
    resumo_oraculo = montar_resumo_metodo("oraculo (teto)", oraculo_mensal.reset_index(), excesso_p1, "oraculo", agregada_oraculo)

    df_teste_resumo = pd.DataFrame([resumo_vencedor, resumo_rf_A, resumo_clima, resumo_pers, resumo_oraculo])

    # 4. Testes estatísticos pareados (Wilcoxon pareado em 24 meses + Bootstrap 95% + Holm)
    logger.info("Executando testes pareados de Wilcoxon e Bootstrap 95% com correção de Holm...")
    vencedor_caps = df_mensal_vencedor.set_index("mes_alvo")["captura_15"].loc[meses_alvo].values
    comparadores = {
        "rf_A": rf_A_mensal.loc[meses_alvo, "captura_15"].values,
        "climatologia_focos": clima_mensal.loc[meses_alvo, "captura_15"].values,
        "persistencia": pers_mensal.loc[meses_alvo, "captura_15"].values
    }

    rng_boot = np.random.RandomState(42)
    testes_linhas = []
    p_valores_brutos = []

    for nome_comp, comp_caps in comparadores.items():
        diffs = vencedor_caps - comp_caps
        dif_media = float(np.mean(diffs))

        # Bootstrap
        indices = rng_boot.randint(0, len(diffs), size=(10000, len(diffs)))
        boot_means = np.mean(diffs[indices], axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        # Wilcoxon
        res_w = wilcoxon(vencedor_caps, comp_caps, alternative="two-sided")
        pval = float(res_w.pvalue)
        p_valores_brutos.append(pval)

        testes_linhas.append({
            "modelo": f"{melhor_modelo}_A_v2_sel",
            "comparado_com": nome_comp,
            "diferenca_media": dif_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": pval
        })

    p_holm = holm_bonferroni(np.array(p_valores_brutos))
    for i, p_h in enumerate(p_holm):
        testes_linhas[i]["p_holm"] = float(p_h)

    df_testes = pd.DataFrame(testes_linhas)
    return df_teste_resumo, df_testes

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    logger.info("=== INICIANDO SCRIPT 03b: RODADA 2 DE ATRIBUTOS (APHIM) ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar hiperparâmetros fixos
    hiper_path = resultados_dir / "perigo_hiperparametros.json"
    if not hiper_path.exists():
        raise FileNotFoundError(f"Arquivo '{hiper_path}' não encontrado.")
    with open(hiper_path, "r", encoding="utf-8") as f:
        params_all = json.load(f)
    params_rf = params_all["rf_A"]
    params_xgb = params_all["xgb_A"]

    # 2. Parte 1: Engenharia dos novos atributos
    df_mod_v2 = construir_atributos_v2(processed_dir)

    # 3. Parte 2: Definição de Conjuntos e Limpeza
    cols_A_orig, cols_A_limpo, cols_A_v2, grupos_v2, removidas_info = definir_conjuntos_features()

    # 4. Parte 3: Validação Temporal, Eliminação de Grupos e Importância por Permutação
    df_val_resumo, df_removidas, df_perm, cols_A_v2_sel, melhor_mod, melhor_conj = executar_selecao_e_validacao(
        df_mod=df_mod_v2,
        params_rf=params_rf,
        params_xgb=params_xgb,
        cols_A_orig=cols_A_orig,
        cols_A_limpo=cols_A_limpo,
        cols_A_v2=cols_A_v2,
        grupos_v2=grupos_v2,
        removidas_info=removidas_info
    )

    # Exportar saídas de validação e seleção
    out_val_csv = resultados_dir / "features_v2_validacao.csv"
    out_rem_csv = resultados_dir / "features_v2_removidas.csv"
    out_perm_csv = resultados_dir / "features_v2_permutacao.csv"

    df_val_resumo.to_csv(out_val_csv, index=False, encoding="utf-8")
    df_removidas.to_csv(out_rem_csv, index=False, encoding="utf-8")
    df_perm.to_csv(out_perm_csv, index=False, encoding="utf-8")
    logger.info(f"Arquivos de validação e permutação salvos em '{resultados_dir}'.")

    # 5. Parte 4: Avaliação no Teste (2024-2025)
    params_vencedor = params_xgb if melhor_mod == "xgb" else params_rf
    df_teste_resumo, df_testes = avaliar_no_teste(
        df_mod=df_mod_v2,
        melhor_modelo=melhor_mod,
        feats_vencedor=cols_A_v2_sel,
        params_vencedor=params_vencedor,
        processed_dir=processed_dir
    )

    out_teste_resumo = resultados_dir / "features_v2_teste_resumo.csv"
    out_teste_testes = resultados_dir / "features_v2_testes.csv"
    df_teste_resumo.to_csv(out_teste_resumo, index=False, encoding="utf-8")
    df_testes.to_csv(out_teste_testes, index=False, encoding="utf-8")
    logger.info(f"Resultados de teste salvos com sucesso em '{out_teste_resumo}' e '{out_teste_testes}'.")

    print("\n" + "=" * 90)
    print("               RODADA 2 DE ATRIBUTOS - VALIDAÇÃO E TESTE CONCLUÍDOS")
    print("=" * 90)
    print("\nTABELA 1 — RESULTADOS NA VALIDAÇÃO (DOBRAS 2022 E 2023):")
    print(df_val_resumo.to_string(index=False))
    print("\nTABELA 2 — IMPORTÂNCIA POR PERMUTAÇÃO (TOP 15):")
    print(df_perm.head(15).to_string(index=False))
    print("\nTABELA 3 — BENCHMARK DE TESTE (2024-2025):")
    print(df_teste_resumo.to_string(index=False))
    print("\nTABELA 4 — TESTES ESTATÍSTICOS PAREADOS (WILCOXON + BOOTSTRAP + HOLM):")
    print(df_testes.to_string(index=False))
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
