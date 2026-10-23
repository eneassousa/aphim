"""
Módulo 10 — Validação e Teste do DETER no Nível 2 (Grade de 5 km) (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Objetivos:
1. Comparar na Validação Temporal (dobras 2022 e 2023):
   - G0: FEATURES_CELULA atuais (16 atributos de 07_grade_priorizacao.py)
   - G1: FEATURES_CELULA + 5 atributos de célula do DETER (21 atributos)
   - Mesma busca em grade de 16 combinações de hiperparâmetros (max_depth, lr, n_est, mcw)
   - Critério: maior CapturaÁrea@10% média nas dobras 2022 e 2023 nos 15 municípios com mais focos observados em t.
2. Checagem de Sanidade:
   - G0 deve reproduzir o melhor score de grade_5000m_hiperparametros.json (0.51082) com diferença <= 0,1 ponto (|diff| <= 0.0010).
3. Regra de Escolha do Vencedor:
   - G1 só substitui G0 se ganhar pelo menos +0,5 ponto (+0.0050) de CapturaÁrea@10% média de validação.
4. Importância por Permutação:
   - Importância por permutação das features de G1 na validação.
5. Avaliação no Teste (executada estritamente se G1 vencer):
   - Se G1 vencer: retreina G1 em todo o treino da grade e avalia no teste (2024-2025).
   - Os 15 municípios do topo vêm estritamente de features_v2_predicoes_teste.parquet (isolando o efeito do DETER).
   - Comparações com XGBoost atual da grade, histórico fixo e climatologia celular.
   - Wilcoxon pareado, Bootstrap 95% (10.000 reamostragens, semente 42) e correção de Holm.
   - Se G1 não vencer: não executa o teste; registra continuidade de G0.

Saídas em data/processed_deter/resultados/:
- deter_validacao_grade.csv
- deter_permutacao_grade.csv
- deter_grade_mensal.csv (se G1 vencer)
- deter_grade_resumo.csv (se G1 vencer)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import wilcoxon
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("processamento_10_deter_grade")

FEATURES_CELULA_BASE = [
    "focos_t",
    "focos_t1",
    "focos_t2",
    "focos_mm3",
    "focos_mesmo_mes_ano_ant",
    "focos_media_hist_mes",
    "focos_vizinhos_t",
    "focos_vizinhos_media_hist_mes",
    "mes_sin",
    "mes_cos",
    "focos_mun_t",
    "prec_mm",
    "ur_pct",
    "dpv_kpa",
    "umid_solo",
    "anom_prec_mm"
]

FEATURES_CELULA_DETER = [
    "deter_desmat_cel_3m",
    "deter_desmat_cel_12m",
    "deter_degrad_cel_3m",
    "deter_degrad_cel_12m",
    "deter_desmat_viz_12m"
]

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

def get_top15_obs_emissao(df_subset: pd.DataFrame) -> Dict[str, List[str]]:
    """Identifica os 15 municípios com mais focos observados em t para cada mês."""
    top15_dict = {}
    for m_alvo, grp in df_subset.groupby("mes_alvo"):
        mun_focos_t = grp.groupby("cd_mun")["focos_mun_t"].first()
        top15_muns = mun_focos_t.sort_values(ascending=False).head(15).index.tolist()
        top15_dict[m_alvo] = top15_muns
    return top15_dict

def calcular_captura_area_mes(
    df_mes: pd.DataFrame,
    col_score: str,
    top15_muns: List[str],
    pct_area_k: float = 10.0
) -> float:
    """Calcula a CapturaÁrea@k% para um único mês dentro dos 15 municípios."""
    sub_top15 = df_mes[df_mes["cd_mun"].isin(top15_muns)]
    focos_capturados = 0.0
    focos_top15 = 0.0

    for _, grp in sub_top15.groupby("cd_mun"):
        f_mun = grp["y"].sum()
        focos_top15 += f_mun

        grp_sort = grp.sort_values(by=[col_score, "cell_id"], ascending=[False, True])
        cell_areas = grp_sort["area_km2"].values
        cum_areas = np.cumsum(cell_areas)
        total_mun_area = cum_areas[-1] if len(cum_areas) > 0 else 0.0
        target_area = (pct_area_k / 100.0) * total_mun_area

        mask = (cum_areas - cell_areas) < target_area
        focos_capturados += grp_sort.iloc[mask]["y"].sum()

    cap = (focos_capturados / focos_top15) if focos_top15 > 0 else 0.0
    return float(cap)

def avaliar_modelo_grade_val(
    model: xgb.XGBRegressor,
    features: List[str],
    df_val_22: pd.DataFrame,
    df_val_23: pd.DataFrame,
    top15_val_22: Dict[str, List[str]],
    top15_val_23: Dict[str, List[str]]
) -> Tuple[float, float, float]:
    """Avalia o modelo nas dobras 2022 e 2023 e retorna a CapturaÁrea@10% média."""
    # Dobra 2022
    df_v22 = df_val_22[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
    df_v22["pred"] = np.maximum(0.0, model.predict(df_val_22[features]))
    scores_22 = [
        calcular_captura_area_mes(df_v22[df_v22["mes_alvo"] == m], "pred", top15_val_22[m])
        for m in sorted(df_v22["mes_alvo"].unique())
    ]
    s22 = float(np.mean(scores_22))

    # Dobra 2023
    df_v23 = df_val_23[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
    df_v23["pred"] = np.maximum(0.0, model.predict(df_val_23[features]))
    scores_23 = [
        calcular_captura_area_mes(df_v23[df_v23["mes_alvo"] == m], "pred", top15_val_23[m])
        for m in sorted(df_v23["mes_alvo"].unique())
    ]
    s23 = float(np.mean(scores_23))

    return s22, s23, float((s22 + s23) / 2.0)

def buscar_hiperparametros_grade(
    config_nome: str,
    features: List[str],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    top15_val_22: Dict[str, List[str]],
    top15_val_23: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], float, float, float, List[Dict[str, Any]]]:
    """Executa a busca nas 16 combinações de hiperparâmetros de 07_grade_priorizacao.py."""
    grid_params = []
    for md in [4, 6]:
        for lr in [0.05, 0.1]:
            for ne in [300, 600]:
                for mcw in [1, 10]:
                    grid_params.append({
                        "max_depth": md,
                        "learning_rate": lr,
                        "n_estimators": ne,
                        "min_child_weight": mcw
                    })

    logger.info(f"Iniciando sintonia em grade ({len(grid_params)} combinações) para {config_nome}...")
    melhor_params = None
    melhor_media = -1.0
    melhor_s22 = -1.0
    melhor_s23 = -1.0
    todas_combinacoes = []

    for idx, p in enumerate(grid_params):
        t0 = time.time()
        # Treinar na dobra 2022
        m22 = xgb.XGBRegressor(
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            min_child_weight=p["min_child_weight"],
            objective="count:poisson",
            tree_method="hist",
            random_state=42, n_jobs=-1
        )
        m22.fit(df_tr_22[features], df_tr_22["y"])

        # Treinar na dobra 2023
        m23 = xgb.XGBRegressor(
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"],
            min_child_weight=p["min_child_weight"],
            objective="count:poisson",
            tree_method="hist",
            random_state=42, n_jobs=-1
        )
        m23.fit(df_tr_23[features], df_tr_23["y"])

        # Avaliar
        df_v22 = df_val_22[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
        df_v22["pred"] = np.maximum(0.0, m22.predict(df_val_22[features]))
        s22 = float(np.mean([calcular_captura_area_mes(df_v22[df_v22["mes_alvo"] == m], "pred", top15_val_22[m])
                             for m in sorted(df_v22["mes_alvo"].unique())]))

        df_v23 = df_val_23[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
        df_v23["pred"] = np.maximum(0.0, m23.predict(df_val_23[features]))
        s23 = float(np.mean([calcular_captura_area_mes(df_v23[df_v23["mes_alvo"] == m], "pred", top15_val_23[m])
                             for m in sorted(df_v23["mes_alvo"].unique())]))

        s_med = float((s22 + s23) / 2.0)
        dt = time.time() - t0

        todas_combinacoes.append({
            "parametros": p,
            "score_2022": s22,
            "score_2023": s23,
            "score_medio": s_med,
            "tempo_s": round(dt, 2)
        })

        if s_med > melhor_media:
            melhor_media = s_med
            melhor_s22 = s22
            melhor_s23 = s23
            melhor_params = p

    logger.info(
        f"-> Melhor {config_nome}: CapturaÁrea@10% média = {melhor_media*100:.3f}% "
        f"(2022: {melhor_s22*100:.3f}%, 2023: {melhor_s23*100:.3f}%) | "
        f"Params: {melhor_params}"
    )
    return melhor_params, melhor_s22, melhor_s23, melhor_media, todas_combinacoes

def calcular_permutacao_grade(
    features: List[str],
    params: Dict[str, Any],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    top15_val_22: Dict[str, List[str]],
    top15_val_23: Dict[str, List[str]],
    score_baseline: float
) -> pd.DataFrame:
    """Calcula a importância por permutação na validação para as features de G1."""
    logger.info("Ajustando modelos base nas dobras 2022 e 2023 para importância por permutação na grade...")
    m22 = xgb.XGBRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_weight=params["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        random_state=42, n_jobs=-1
    )
    m22.fit(df_tr_22[features], df_tr_22["y"])

    m23 = xgb.XGBRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_weight=params["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        random_state=42, n_jobs=-1
    )
    m23.fit(df_tr_23[features], df_tr_23["y"])

    rng = np.random.RandomState(42)
    records = []

    for f_col in features:
        # Permutar em 2022
        val_22_p = df_val_22[features].copy()
        val_22_p[f_col] = rng.permutation(val_22_p[f_col].values)
        df_v22 = df_val_22[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
        df_v22["pred"] = np.maximum(0.0, m22.predict(val_22_p))
        s22_p = float(np.mean([calcular_captura_area_mes(df_v22[df_v22["mes_alvo"] == m], "pred", top15_val_22[m])
                               for m in sorted(df_v22["mes_alvo"].unique())]))

        # Permutar em 2023
        val_23_p = df_val_23[features].copy()
        val_23_p[f_col] = rng.permutation(val_23_p[f_col].values)
        df_v23 = df_val_23[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
        df_v23["pred"] = np.maximum(0.0, m23.predict(val_23_p))
        s23_p = float(np.mean([calcular_captura_area_mes(df_v23[df_v23["mes_alvo"] == m], "pred", top15_val_23[m])
                               for m in sorted(df_v23["mes_alvo"].unique())]))

        s_med_p = (s22_p + s23_p) / 2.0
        importancia = score_baseline - s_med_p
        records.append({"atributo": f_col, "importancia_cap_area_10": importancia})

    df_perm = pd.DataFrame(records).sort_values("importancia_cap_area_10", ascending=False).reset_index(drop=True)
    df_perm.insert(0, "posicao", range(1, len(df_perm) + 1))
    return df_perm

def avaliar_teste_g1(
    features_g1: List[str],
    params_g1: Dict[str, Any],
    df_mod_merged: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path
):
    """
    Avalia G1 no teste (2024-2025) estritamente se G1 vencer.
    Top 15 municípios por mês vêm das previsões do nível 1 (features_v2_predicoes_teste.parquet).
    """
    logger.info("=== AVALIANDO G1 NO TESTE (2024–2025) ===")
    df_treino = df_mod_merged[df_mod_merged["conjunto"] == "treino"].copy()
    df_teste = df_mod_merged[df_mod_merged["conjunto"] == "teste"].copy()

    out_mensal = out_dir / "deter_grade_mensal.csv"
    if out_mensal.exists():
        logger.info(f"Carregando resultados mensais de teste já gerados de '{out_mensal}'...")
        df_mensal = pd.read_csv(out_mensal)
    else:
        logger.info("Ajustando modelo final G1 em todo o treino da grade...")
        m_final = xgb.XGBRegressor(
            max_depth=params_g1["max_depth"],
            learning_rate=params_g1["learning_rate"],
            n_estimators=params_g1["n_estimators"],
            min_child_weight=params_g1["min_child_weight"],
            objective="count:poisson",
            tree_method="hist",
            random_state=42, n_jobs=-1
        )
        m_final.fit(df_treino[features_g1], df_treino["y"])
        df_teste["pred_g1"] = np.maximum(0.0, m_final.predict(df_teste[features_g1]))

        # Carregar top 15 municípios oficiais do teste de nível 1
        l1_path = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
        if not l1_path.exists():
            raise FileNotFoundError(f"Arquivo de predições de nível 1 '{l1_path}' não encontrado.")
        df_pred_l1 = pd.read_parquet(l1_path)
        df_pred_l1["cd_mun"] = df_pred_l1["cd_mun"].astype(str).str.zfill(7)

        # Baselines celulares existentes na própria base da grade
        metodos = [
            ("G1_xgb_deter", "pred_g1"),
            ("historico_fixo_celula", "focos_fixo_2019_2023"),
            ("climatologia_celula", "focos_media_hist_mes")
        ]

        meses_alvo = sorted(df_teste["mes_alvo"].unique())
        registros_mensais = []

        for m_alvo in meses_alvo:
            sub_l1 = df_pred_l1[df_pred_l1["mes_alvo"] == m_alvo].sort_values(
                by=["pred_vencedor", "cd_mun"], ascending=[False, True]
            )
            top15_muns = sub_l1["cd_mun"].head(15).tolist()

            df_m = df_teste[df_teste["mes_alvo"] == m_alvo].copy()
            focos_total_pa = float(df_m["y"].sum())

            for nome_met, col_sc in metodos:
                sub_top15 = df_m[df_m["cd_mun"].isin(top15_muns)]
                focos_top15_tot = 0.0
                focos_cap_10 = 0.0

                for _, grp in sub_top15.groupby("cd_mun"):
                    f_mun = grp["y"].sum()
                    focos_top15_tot += f_mun

                    grp_s = grp.sort_values(by=[col_sc, "cell_id"], ascending=[False, True])
                    c_areas = grp_s["area_km2"].values
                    cum_a = np.cumsum(c_areas)
                    tot_a = cum_a[-1] if len(cum_a) > 0 else 0.0

                    m_10 = (cum_a - c_areas) < (0.10 * tot_a)
                    sel_10 = grp_s.iloc[m_10]
                    focos_cap_10 += sel_10["y"].sum()

                cap_area_10 = (focos_cap_10 / focos_top15_tot) if focos_top15_tot > 0 else 0.0
                registros_mensais.append({
                    "mes_alvo": m_alvo,
                    "metodo": nome_met,
                    "captura_area_10": cap_area_10,
                    "focos_capturados_10": focos_cap_10,
                    "focos_top15": focos_top15_tot,
                    "focos_pa": focos_total_pa
                })

        df_mensal = pd.DataFrame(registros_mensais)
        df_mensal.to_csv(out_mensal, index=False, encoding="utf-8")
        logger.info(f"Resultados mensais de teste salvos em '{out_mensal}'.")

    # Carregar resultados mensais da grade atual (para comparativo)
    grade_ref_path = processed_dir / "resultados" / "grade_5000m_mensal.csv"
    if not grade_ref_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_ref_path}' não encontrado.")
    df_grade_ref = pd.read_csv(grade_ref_path)

    # Comparativo estatístico contra o XGBoost atual (G0), histórico fixo e climatologia
    caps_g1 = df_mensal[df_mensal["metodo"] == "G1_xgb_deter"].sort_values("mes_alvo")["captura_area_10"].values
    caps_g0 = df_grade_ref[df_grade_ref["metodo"].isin(["xgboost", "xgb_celula"])].sort_values("mes_alvo")["captura_area_10"].values
    caps_fixo = df_grade_ref[df_grade_ref["metodo"] == "historico_fixo_celula"].sort_values("mes_alvo")["captura_area_10"].values
    caps_clim = df_grade_ref[df_grade_ref["metodo"] == "climatologia_celula"].sort_values("mes_alvo")["captura_area_10"].values

    assert len(caps_g1) == 24, f"Esperado 24 meses para G1, obtido {len(caps_g1)}"
    assert len(caps_g0) == 24, f"Esperado 24 meses para G0, obtido {len(caps_g0)}"
    assert len(caps_fixo) == 24, f"Esperado 24 meses para historico_fixo, obtido {len(caps_fixo)}"
    assert len(caps_clim) == 24, f"Esperado 24 meses para climatologia, obtido {len(caps_clim)}"

    def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
        rng = np.random.RandomState(seed)
        diffs = a - b
        mean_diff = float(np.mean(diffs))
        boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        stat, pval = wilcoxon(a, b)
        return mean_diff, ci_lower, ci_upper, float(pval)

    diff_g0, ci_l_g0, ci_u_g0, p_g0 = bootstrap_diff(caps_g1, caps_g0)
    diff_fixo, ci_l_f, ci_u_f, p_fixo = bootstrap_diff(caps_g1, caps_fixo)
    diff_clim, ci_l_c, ci_u_c, p_clim = bootstrap_diff(caps_g1, caps_clim)

    pvals_raw = np.array([p_g0, p_fixo, p_clim])
    pvals_holm = holm_bonferroni(pvals_raw)

    df_resumo_grade = pd.DataFrame([
        {
            "comparador": "xgb_grade_atual_G0",
            "cap_area_10_g1": float(np.mean(caps_g1)),
            "cap_area_10_comparador": float(np.mean(caps_g0)),
            "diff_media": diff_g0,
            "ic95_inferior": ci_l_g0,
            "ic95_superior": ci_u_g0,
            "p_wilcoxon": p_g0,
            "p_holm": pvals_holm[0]
        },
        {
            "comparador": "historico_fixo_celula",
            "cap_area_10_g1": float(np.mean(caps_g1)),
            "cap_area_10_comparador": float(np.mean(caps_fixo)),
            "diff_media": diff_fixo,
            "ic95_inferior": ci_l_f,
            "ic95_superior": ci_u_f,
            "p_wilcoxon": p_fixo,
            "p_holm": pvals_holm[1]
        },
        {
            "comparador": "climatologia_celula",
            "cap_area_10_g1": float(np.mean(caps_g1)),
            "cap_area_10_comparador": float(np.mean(caps_clim)),
            "diff_media": diff_clim,
            "ic95_inferior": ci_l_c,
            "ic95_superior": ci_u_c,
            "p_wilcoxon": p_clim,
            "p_holm": pvals_holm[2]
        }
    ])

    out_resumo = out_dir / "deter_grade_resumo.csv"
    df_resumo_grade.to_csv(out_resumo, index=False, encoding="utf-8")
    logger.info(f"Resumo estatístico do teste na grade salvo em '{out_resumo}'.")
    print("\nBENCHMARK DE TESTE DA GRADE (G1):")
    print(df_resumo_grade.to_string(index=False))

def main():
    logger.info("=== INICIANDO EXPERIMENTO DE VALIDAÇÃO G0 vs G1 (GRADE 5 KM) ===")
    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = processed_deter_dir / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem da grade e atributos DETER da grade
    grade_mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    grade_det_path = processed_deter_dir / "grade_5000m_deter.parquet"
    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"

    if not grade_mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_mod_path}' não encontrado.")
    if not grade_det_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_det_path}' não encontrado. Execute 08b_deter_atributos.py primeiro.")
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Arquivo '{gpkg_path}' não encontrado.")

    logger.info("Carregando base de modelagem da grade (4,2 milhões de linhas)...")
    df_mod = pd.read_parquet(grade_mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    logger.info("Carregando atributos DETER da grade de 5 km...")
    df_deter_cells = pd.read_parquet(grade_det_path)

    # Mesclar apenas as 5 colunas do DETER
    logger.info("Mesclando atributos DETER na base da grade em tempo de execução...")
    df_mod_merged = df_mod.merge(df_deter_cells, on=["cell_id", "mes_emissao"], how="left")
    for col in FEATURES_CELULA_DETER:
        df_mod_merged[col] = df_mod_merged[col].fillna(0.0).astype(np.float32)

    # Carregar áreas de célula para a métrica de priorização
    gdf_cells = gpd.read_file(gpkg_path)
    cell_area_map = dict(zip(gdf_cells["cell_id"].astype(np.int32), gdf_cells["area_km2_dentro_pa"].astype(float)))
    df_mod_merged["area_km2"] = df_mod_merged["cell_id"].map(cell_area_map).astype(np.float32)

    # Calcular histórico fixo da célula (2019-2023)
    logger.info("Calculando total de focos por célula de 2019 a 2023 (historico_fixo_celula)...")
    df_hist_alvos = df_mod_merged[df_mod_merged["mes_alvo"] <= "2023-12"]
    s_alvos = df_hist_alvos.groupby("cell_id")["y"].sum()
    df_jan19 = df_mod_merged[df_mod_merged["mes_emissao"] == "2019-01"]
    s_jan19 = df_jan19.groupby("cell_id")["focos_t"].sum()
    s_hist_fixo = (s_alvos + s_jan19).to_dict()
    df_mod_merged["focos_fixo_2019_2023"] = df_mod_merged["cell_id"].map(s_hist_fixo).fillna(0.0).astype(np.float32)

    # 2. Partições de Validação
    df_treino = df_mod_merged[df_mod_merged["conjunto"] == "treino"].copy()
    df_tr_22 = df_treino[df_treino["mes_alvo"] < "2022-01"].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()
    df_tr_23 = df_treino[df_treino["mes_alvo"] < "2023-01"].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    top15_val_22 = get_top15_obs_emissao(df_val_22)
    top15_val_23 = get_top15_obs_emissao(df_val_23)

    # 3. Validação e Busca de Hiperparâmetros para G0 e G1
    feats_g0 = FEATURES_CELULA_BASE
    feats_g1 = FEATURES_CELULA_BASE + FEATURES_CELULA_DETER
    out_val_csv = out_dir / "deter_validacao_grade.csv"
    out_perm_csv = out_dir / "deter_permutacao_grade.csv"

    if out_val_csv.exists() and out_perm_csv.exists() and "--recomputar" not in sys.argv:
        logger.info(f"Carregando resultados de validação pré-calculados de '{out_val_csv}'...")
        df_val_grade = pd.read_csv(out_val_csv)
        row_g0 = df_val_grade[df_val_grade["configuracao"] == "G0"].iloc[0]
        row_g1 = df_val_grade[df_val_grade["configuracao"] == "G1"].iloc[0]
        p_g0 = json.loads(row_g0["hiperparametros"])
        med_g0 = float(row_g0["cap_area_10_media"])
        p_g1 = json.loads(row_g1["hiperparametros"])
        med_g1 = float(row_g1["cap_area_10_media"])

        # Checagem de Sanidade de G0 contra grade_5000m_hiperparametros.json
        hiper_ref_path = processed_dir / "resultados" / "grade_5000m_hiperparametros.json"
        with open(hiper_ref_path, "r", encoding="utf-8") as f:
            ref_json = json.load(f)
        score_esperado_g0 = float(ref_json["melhor_score_validacao"])
        diff_sanidade_g = abs(med_g0 - score_esperado_g0)
        logger.info(
            f"Checagem de Sanidade Nível 2: G0 reproduzido = {med_g0*100:.4f}% | "
            f"Esperado = {score_esperado_g0*100:.4f}% | Diferença = {diff_sanidade_g*100:.4f} p.p."
        )
        if diff_sanidade_g > 0.0010:
            raise ValueError(f"FALHA NA CHECAGEM DE SANIDADE DA GRADE: Diferença {diff_sanidade_g*100:.4f} p.p. excede 0,10 p.p.")
        logger.info("CHECAGEM DE SANIDADE NÍVEL 2 APROVADA: G0 reproduziu a validação da grade perfeitamente!")

        df_perm_g1 = pd.read_csv(out_perm_csv)
    else:
        logger.info("Executando busca de hiperparâmetros para G0...")
        p_g0, s22_g0, s23_g0, med_g0, combs_g0 = buscar_hiperparametros_grade(
            "G0 (FEATURES_CELULA)", feats_g0,
            df_tr_22, df_val_22, df_tr_23, df_val_23,
            top15_val_22, top15_val_23
        )

        # Checagem de Sanidade de G0 contra grade_5000m_hiperparametros.json
        hiper_ref_path = processed_dir / "resultados" / "grade_5000m_hiperparametros.json"
        with open(hiper_ref_path, "r", encoding="utf-8") as f:
            ref_json = json.load(f)
        score_esperado_g0 = float(ref_json["melhor_score_validacao"])

        diff_sanidade_g = abs(med_g0 - score_esperado_g0)
        logger.info(
            f"Checagem de Sanidade Nível 2: G0 reproduzido = {med_g0*100:.4f}% | "
            f"Esperado = {score_esperado_g0*100:.4f}% | Diferença = {diff_sanidade_g*100:.4f} p.p."
        )
        if diff_sanidade_g > 0.0010: # > 0,10 ponto
            msg = f"FALHA NA CHECAGEM DE SANIDADE DA GRADE: Diferença {diff_sanidade_g*100:.4f} p.p. excede 0,10 p.p."
            logger.error(msg)
            raise ValueError(msg)
        logger.info("CHECAGEM DE SANIDADE NÍVEL 2 APROVADA: G0 reproduziu a validação da grade perfeitamente!")

        # 4. Busca de Hiperparâmetros para G1 (FEATURES_CELULA + 5 DETER)
        p_g1, s22_g1, s23_g1, med_g1, combs_g1 = buscar_hiperparametros_grade(
            "G1 (FEATURES_CELULA + 5 DETER)", feats_g1,
            df_tr_22, df_val_22, df_tr_23, df_val_23,
            top15_val_22, top15_val_23
        )

        # Tabela de Validação da Grade
        df_val_grade = pd.DataFrame([
            {
                "configuracao": "G0",
                "n_atributos": len(feats_g0),
                "hiperparametros": json.dumps(p_g0),
                "cap_area_10_2022": s22_g0,
                "cap_area_10_2023": s23_g0,
                "cap_area_10_media": med_g0
            },
            {
                "configuracao": "G1",
                "n_atributos": len(feats_g1),
                "hiperparametros": json.dumps(p_g1),
                "cap_area_10_2022": s22_g1,
                "cap_area_10_2023": s23_g1,
                "cap_area_10_media": med_g1
            }
        ])
        df_val_grade.to_csv(out_val_csv, index=False, encoding="utf-8")
        logger.info(f"Tabela deter_validacao_grade salva em '{out_val_csv}'.")

        # 5. Importância por Permutação de G1
        logger.info("Calculando importância por permutação na validação para G1 na grade...")
        df_perm_g1 = calcular_permutacao_grade(
            feats_g1, p_g1,
            df_tr_22, df_val_22, df_tr_23, df_val_23,
            top15_val_22, top15_val_23,
            med_g1
        )
        df_perm_g1.to_csv(out_perm_csv, index=False, encoding="utf-8")
        logger.info(f"Importância por permutação da grade salva em '{out_perm_csv}'.")

    print("\n" + "=" * 80)
    print("      TABELA DE VALIDAÇÃO TEMPORAL - NÍVEL 2 (GRADE 5 KM)")
    print("=" * 80)
    print(df_val_grade.to_string(index=False))

    print("\nIMPORTÂNCIA POR PERMUTAÇÃO G1 NA GRADE (TOP 15):")
    print(df_perm_g1.head(15).to_string(index=False))

    # 6. Decisão do Vencedor
    diff_g1_g0 = med_g1 - med_g0
    logger.info(f"\nDiferença G1 vs G0: {diff_g1_g0*100:+.3f} p.p. (limiar de vitória: +0.500 p.p.)")
    g1_venceu = diff_g1_g0 >= 0.0050

    if g1_venceu:
        logger.info(f"-> VENCEDOR: G1 VENCEU! Superou G0 em {diff_g1_g0*100:+.3f} p.p. (>= +0.500 p.p.).")
        print(f"\nDECISÃO: G1 VENCEU! (+{diff_g1_g0*100:.3f} p.p. >= +0.500 p.p.) -> Executando avaliação no teste.")
        avaliar_teste_g1(feats_g1, p_g1, df_mod_merged, processed_dir, out_dir)
    else:
        logger.info(f"-> VENCEDOR: G0 CONTINUA! G1 não atingiu o ganho mínimo de +0.500 p.p. (ganho = {diff_g1_g0*100:+.3f} p.p.).")
        print(f"\nDECISÃO: G0 CONTINUA COMO VENCEDOR! (G1 obteve {diff_g1_g0*100:+.3f} p.p., abaixo do limiar de +0.500 p.p.).")
        print("Regra de teste respeitada: TESTE NÃO EXECUTADO para G1.")

if __name__ == "__main__":
    main()
