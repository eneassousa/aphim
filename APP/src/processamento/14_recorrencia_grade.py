"""
Módulo 14 — Validação e Teste Único da Recorrência no Nível 2 (Grade de 5 km) (Projeto APHIM)
Dissertação de Mestrado: Priorização Espacial Mensal para Ação contra Queimadas no Pará.

Objetivos:
1. Comparar na Validação Temporal (dobras 2022 e 2023):
   - G1: 21 atributos atuais (16 base + 5 DETER)
   - G2: 21 atributos atuais + 6 atributos de recorrência (27 atributos)
   - Mesma busca em grade de 16 combinações de hiperparâmetros (max_depth, lr, n_est, mcw)
   - Critério: maior CapturaÁrea@10% média nas dobras 2022 e 2023 nos 15 municípios com mais focos observados em t.
2. Checagem de Sanidade:
   - G1 deve reproduzir 52,23% de média de validação (|diff| <= 0,10 p.p. / 0.0010).
3. Regra de Escolha do Vencedor:
   - G2 só substitui G1 se ganhar pelo menos +0,5 ponto (+0.0050) de CapturaÁrea@10% média de validação.
4. Importância por Permutação:
   - Importância por permutação das features de G2 na validação.
5. Avaliação no Teste (executada estritamente uma única vez se G2 vencer):
   - Retreina G2 em todo o treino da grade (alvos <= 2023-12) e avalia no teste (2024-2025).
   - Top 15 municípios vêm estritamente de features_v2_predicoes_teste.parquet.
   - Comparação pareada contra G1 (g1_predicoes_teste.parquet), histórico fixo celular e climatologia celular.
   - Wilcoxon pareado, Bootstrap 95% (10.000 reamostragens, semente 42) e correção de Holm.
   - Se G2 não vencer: não executa o teste.

Saídas em data/processed_recorrencia/resultados/:
- recorrencia_validacao_grade.csv
- recorrencia_permutacao_grade.csv
- recorrencia_predicoes_teste_grade.parquet (se G2 vencer)
- recorrencia_grade_mensal.csv (se G2 vencer)
- recorrencia_grade_resumo.csv (se G2 vencer)
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

logger = setup_logger("processamento_14_recorrencia_grade")

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

FEATURES_RECORRENCIA_BASE = [
    "meses_desde_ultimo_foco",
    "anos_com_fogo_3a",
    "frac_meses_fogo_24m",
    "focos_janela_retorno",
    "frac_meses_fogo_24m_viz"
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
    df_v22 = df_val_22[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
    df_v22["pred"] = np.maximum(0.0, model.predict(df_val_22[features]))
    scores_22 = [
        calcular_captura_area_mes(df_v22[df_v22["mes_alvo"] == m], "pred", top15_val_22[m])
        for m in sorted(df_v22["mes_alvo"].unique())
    ]
    s22 = float(np.mean(scores_22))

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
    """Executa a busca nas 16 combinações de hiperparâmetros de 10_deter_grade.py."""
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
            device="cuda",
            random_state=42
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
            device="cuda",
            random_state=42
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
    """Calcula a importância por permutação na validação para as features de G2."""
    logger.info("Ajustando modelos base nas dobras 2022 e 2023 para importância por permutação na grade...")
    m22 = xgb.XGBRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_weight=params["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    m22.fit(df_tr_22[features], df_tr_22["y"])

    m23 = xgb.XGBRegressor(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        min_child_weight=params["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        device="cuda",
        random_state=42
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

def avaliar_teste_g2(
    features_g2: List[str],
    params_g2: Dict[str, Any],
    df_mod_merged: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path
):
    """
    Avalia G2 no teste (2024–2025) estritamente se G2 vencer.
    Top 15 municípios por mês vêm das previsões do nível 1 (features_v2_predicoes_teste.parquet).
    Comparação pareada contra G1 (g1_predicoes_teste.parquet), histórico fixo celular e climatologia celular.
    """
    logger.info("=== AVALIANDO G2 NO TESTE (2024–2025) ===")
    df_treino = df_mod_merged[df_mod_merged["conjunto"] == "treino"].copy()
    df_teste = df_mod_merged[df_mod_merged["conjunto"] == "teste"].copy()

    # Garantir que a feature taxa_retorno_emp use a coluna final no treino e teste
    df_treino["taxa_retorno_emp"] = df_treino["taxa_retorno_emp_final"]
    df_teste["taxa_retorno_emp"] = df_teste["taxa_retorno_emp_final"]

    logger.info("Ajustando modelo final G2 em todo o treino da grade (alvos <= 2023-12)...")
    m_final = xgb.XGBRegressor(
        max_depth=params_g2["max_depth"],
        learning_rate=params_g2["learning_rate"],
        n_estimators=params_g2["n_estimators"],
        min_child_weight=params_g2["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    m_final.fit(df_treino[features_g2], df_treino["y"])
    df_teste["pred_g2"] = np.maximum(0.0, m_final.predict(df_teste[features_g2]))

    # Salvar predições de teste
    out_preds_parquet = out_dir / "resultados" / "recorrencia_predicoes_teste_grade.parquet"
    df_teste[["cell_id", "cd_mun", "mes_emissao", "mes_alvo", "y", "pred_g2"]].to_parquet(out_preds_parquet, index=False)
    logger.info(f"Predições de teste salvas em '{out_preds_parquet}'.")

    # Carregar predições do G1 oficial (data/processed_convlstm/g1_predicoes_teste.parquet)
    g1_path = Path("data/processed_convlstm/g1_predicoes_teste.parquet")
    if not g1_path.exists():
        raise FileNotFoundError(f"Arquivo de predições de teste do G1 '{g1_path}' não encontrado.")
    df_g1 = pd.read_parquet(g1_path)
    df_teste["pred_g1"] = df_g1["pred_g1"].values

    # Carregar top 15 municípios oficiais do teste de nível 1
    l1_path = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
    if not l1_path.exists():
        raise FileNotFoundError(f"Arquivo de predições de nível 1 '{l1_path}' não encontrado.")
    df_pred_l1 = pd.read_parquet(l1_path)
    df_pred_l1["cd_mun"] = df_pred_l1["cd_mun"].astype(str).str.zfill(7)

    metodos = [
        ("G2_xgb_recorrencia", "pred_g2"),
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
    out_mensal = out_dir / "resultados" / "recorrencia_grade_mensal.csv"
    df_mensal.to_csv(out_mensal, index=False, encoding="utf-8")
    logger.info(f"Resultados mensais de teste salvos em '{out_mensal}'.")

    # Comparativo estatístico contra G1, histórico fixo e climatologia
    caps_g2 = df_mensal[df_mensal["metodo"] == "G2_xgb_recorrencia"].sort_values("mes_alvo")["captura_area_10"].values
    caps_g1 = df_mensal[df_mensal["metodo"] == "G1_xgb_deter"].sort_values("mes_alvo")["captura_area_10"].values
    caps_fixo = df_mensal[df_mensal["metodo"] == "historico_fixo_celula"].sort_values("mes_alvo")["captura_area_10"].values
    caps_clim = df_mensal[df_mensal["metodo"] == "climatologia_celula"].sort_values("mes_alvo")["captura_area_10"].values

    assert len(caps_g2) == 24, f"Esperado 24 meses para G2, obtido {len(caps_g2)}"
    assert len(caps_g1) == 24, f"Esperado 24 meses para G1, obtido {len(caps_g1)}"

    def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
        rng = np.random.RandomState(seed)
        diffs = a - b
        mean_diff = float(np.mean(diffs))
        boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        stat, pval = wilcoxon(a, b)
        meses_venceu = int(np.sum(diffs > 0))
        return mean_diff, ci_lower, ci_upper, float(pval), meses_venceu

    diff_g1, ci_l_g1, ci_u_g1, p_g1, v_g1 = bootstrap_diff(caps_g2, caps_g1)
    diff_fixo, ci_l_f, ci_u_f, p_fixo, v_f = bootstrap_diff(caps_g2, caps_fixo)
    diff_clim, ci_l_c, ci_u_c, p_clim, v_c = bootstrap_diff(caps_g2, caps_clim)

    pvals_raw = np.array([p_g1, p_fixo, p_clim])
    pvals_holm = holm_bonferroni(pvals_raw)

    df_resumo_grade = pd.DataFrame([
        {
            "comparador": "G1_xgb_deter",
            "cap_area_10_g2": float(np.mean(caps_g2)),
            "cap_area_10_comparador": float(np.mean(caps_g1)),
            "diff_media": diff_g1,
            "ic95_inferior": ci_l_g1,
            "ic95_superior": ci_u_g1,
            "p_wilcoxon": p_g1,
            "p_holm": pvals_holm[0],
            "meses_venceu": v_g1
        },
        {
            "comparador": "historico_fixo_celula",
            "cap_area_10_g2": float(np.mean(caps_g2)),
            "cap_area_10_comparador": float(np.mean(caps_fixo)),
            "diff_media": diff_fixo,
            "ic95_inferior": ci_l_f,
            "ic95_superior": ci_u_f,
            "p_wilcoxon": p_fixo,
            "p_holm": pvals_holm[1],
            "meses_venceu": v_f
        },
        {
            "comparador": "climatologia_celula",
            "cap_area_10_g2": float(np.mean(caps_g2)),
            "cap_area_10_comparador": float(np.mean(caps_clim)),
            "diff_media": diff_clim,
            "ic95_inferior": ci_l_c,
            "ic95_superior": ci_u_c,
            "p_wilcoxon": p_clim,
            "p_holm": pvals_holm[2],
            "meses_venceu": v_c
        }
    ])

    out_resumo = out_dir / "resultados" / "recorrencia_grade_resumo.csv"
    df_resumo_grade.to_csv(out_resumo, index=False, encoding="utf-8")
    logger.info(f"Resumo estatístico do teste na grade salvo em '{out_resumo}'.")

    print("\n" + "=" * 95)
    print("           BENCHMARK DE TESTE DA GRADE — MODELO G2 (2024–2025)")
    print("=" * 95)
    print(df_resumo_grade.to_string(index=False))
    print("=" * 95 + "\n")

def main():
    t_start = time.time()
    logger.info("=== INICIANDO SCRIPT 14: VALIDAÇÃO E TESTE DA RECORRÊNCIA NA GRADE DE 5 KM ===")

    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = Path("data/processed_recorrencia")
    resultados_dir = out_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem da grade
    grade_mod_path = processed_dir / "grade_5000m_modelagem.parquet"
    grade_det_path = processed_deter_dir / "grade_5000m_deter.parquet"
    grade_rec_path = out_dir / "grade_5000m_recorrencia.parquet"
    gpkg_path = processed_dir / "grade_5000m_celulas.gpkg"

    if not grade_mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_mod_path}' não encontrado.")
    if not grade_det_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_det_path}' não encontrado.")
    if not grade_rec_path.exists():
        raise FileNotFoundError(f"Arquivo '{grade_rec_path}' não encontrado. Execute script 13 primeiro.")

    logger.info("Carregando base de modelagem da grade (4,2 milhões de linhas)...")
    df_mod = pd.read_parquet(grade_mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    logger.info("Carregando atributos DETER da grade de 5 km...")
    df_deter_cells = pd.read_parquet(grade_det_path)

    logger.info("Carregando variáveis de recorrência da grade de 5 km...")
    df_rec_cells = pd.read_parquet(grade_rec_path)

    # Mesclar atributos DETER e de Recorrência
    logger.info("Mesclando atributos DETER e de Recorrência na base da grade em tempo de execução...")
    df_mod_merged = df_mod.merge(df_deter_cells, on=["cell_id", "mes_emissao"], how="left")
    for col in FEATURES_CELULA_DETER:
        df_mod_merged[col] = df_mod_merged[col].fillna(0.0).astype(np.float32)

    df_mod_merged = df_mod_merged.merge(df_rec_cells, on=["cell_id", "mes_emissao"], how="left")

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

    # Configurar taxas empíricas estritas por dobra
    df_tr_22["taxa_retorno_emp"] = df_tr_22["taxa_retorno_emp_d2022"]
    df_val_22["taxa_retorno_emp"] = df_val_22["taxa_retorno_emp_d2022"]
    df_tr_23["taxa_retorno_emp"] = df_tr_23["taxa_retorno_emp_d2023"]
    df_val_23["taxa_retorno_emp"] = df_val_23["taxa_retorno_emp_d2023"]

    top15_val_22 = get_top15_obs_emissao(df_val_22)
    top15_val_23 = get_top15_obs_emissao(df_val_23)

    # 3. Definição de atributos G1 e G2
    feats_g1 = FEATURES_CELULA_BASE + FEATURES_CELULA_DETER
    feats_g2 = feats_g1 + FEATURES_RECORRENCIA_BASE + ["taxa_retorno_emp"]

    assert len(feats_g1) == 21, f"Esperado 21 atributos em G1, obtido {len(feats_g1)}"
    assert len(feats_g2) == 27, f"Esperado 27 atributos em G2, obtido {len(feats_g2)}"

    # 4. Avaliação e Checagem de Sanidade de G1 contra deter_validacao_grade.csv (52,23%)
    deter_val_path = processed_deter_dir / "resultados" / "deter_validacao_grade.csv"
    if not deter_val_path.exists():
        raise FileNotFoundError(f"Arquivo '{deter_val_path}' não encontrado.")
    df_deter_val = pd.read_csv(deter_val_path)
    row_g1_ref = df_deter_val[df_deter_val["configuracao"] == "G1"].iloc[0]
    p_g1_ref = json.loads(row_g1_ref["hiperparametros"])
    score_esperado_g1 = float(row_g1_ref["cap_area_10_media"])

    logger.info("Avaliando G1 (21 atributos atuais) na validação para checagem de sanidade...")
    m22_g1 = xgb.XGBRegressor(
        max_depth=p_g1_ref["max_depth"],
        learning_rate=p_g1_ref["learning_rate"],
        n_estimators=p_g1_ref["n_estimators"],
        min_child_weight=p_g1_ref["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    m22_g1.fit(df_tr_22[feats_g1], df_tr_22["y"])

    m23_g1 = xgb.XGBRegressor(
        max_depth=p_g1_ref["max_depth"],
        learning_rate=p_g1_ref["learning_rate"],
        n_estimators=p_g1_ref["n_estimators"],
        min_child_weight=p_g1_ref["min_child_weight"],
        objective="count:poisson",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    m23_g1.fit(df_tr_23[feats_g1], df_tr_23["y"])

    s22_g1, s23_g1, med_g1 = avaliar_modelo_grade_val(
        m22_g1, feats_g1, df_val_22, df_val_23, top15_val_22, top15_val_23
    )
    # Note: s23 uses m23
    df_v23 = df_val_23[["cell_id", "cd_mun", "mes_alvo", "y", "area_km2"]].copy()
    df_v23["pred"] = np.maximum(0.0, m23_g1.predict(df_val_23[feats_g1]))
    s23_g1 = float(np.mean([calcular_captura_area_mes(df_v23[df_v23["mes_alvo"] == m], "pred", top15_val_23[m])
                            for m in sorted(df_v23["mes_alvo"].unique())]))
    med_g1 = float((s22_g1 + s23_g1) / 2.0)

    diff_sanidade_g1 = abs(med_g1 - score_esperado_g1)
    logger.info(
        f"Checagem de Sanidade G1: Reproduzido = {med_g1*100:.4f}% | "
        f"Esperado = {score_esperado_g1*100:.4f}% | Diferença = {diff_sanidade_g1*100:.4f} p.p."
    )
    if diff_sanidade_g1 > 0.0010: # > 0,10 ponto
        msg = f"FALHA CRÍTICA NA CHECAGEM DE SANIDADE DE G1: Diferença {diff_sanidade_g1*100:.4f} p.p. excede 0,10 p.p."
        logger.error(msg)
        raise ValueError(msg)
    logger.info("CHECAGEM DE SANIDADE DE G1 APROVADA: G1 reproduziu 52,23% perfeitamente!")

    # 5. Busca de Hiperparâmetros para G2 (21 atributos + 6 recorrência)
    logger.info("Iniciando busca de hiperparâmetros para G2 (27 atributos)...")
    p_g2, s22_g2, s23_g2, med_g2, combs_g2 = buscar_hiperparametros_grade(
        "G2 (FEATURES_CELULA + 5 DETER + 6 RECORRENCIA)", feats_g2,
        df_tr_22, df_val_22, df_tr_23, df_val_23,
        top15_val_22, top15_val_23
    )

    # 6. Salvar Tabela de Validação
    df_val_grade = pd.DataFrame([
        {
            "configuracao": "G1",
            "n_atributos": len(feats_g1),
            "hiperparametros": json.dumps(p_g1_ref),
            "cap_area_10_2022": s22_g1,
            "cap_area_10_2023": s23_g1,
            "cap_area_10_media": med_g1
        },
        {
            "configuracao": "G2",
            "n_atributos": len(feats_g2),
            "hiperparametros": json.dumps(p_g2),
            "cap_area_10_2022": s22_g2,
            "cap_area_10_2023": s23_g2,
            "cap_area_10_media": med_g2
        }
    ])
    out_val_csv = resultados_dir / "recorrencia_validacao_grade.csv"
    df_val_grade.to_csv(out_val_csv, index=False, encoding="utf-8")
    logger.info(f"Tabela recorrencia_validacao_grade.csv salva em '{out_val_csv}'.")

    # 7. Importância por Permutação de G2
    logger.info("Calculando importância por permutação na validação para G2 na grade...")
    df_perm_g2 = calcular_permutacao_grade(
        feats_g2, p_g2,
        df_tr_22, df_val_22, df_tr_23, df_val_23,
        top15_val_22, top15_val_23,
        med_g2
    )
    out_perm_csv = resultados_dir / "recorrencia_permutacao_grade.csv"
    df_perm_g2.to_csv(out_perm_csv, index=False, encoding="utf-8")
    logger.info(f"Importância por permutação da grade salva em '{out_perm_csv}'.")

    print("\n" + "=" * 80)
    print("      TABELA DE VALIDAÇÃO TEMPORAL - NÍVEL 2 (GRADE 5 KM)")
    print("=" * 80)
    print(df_val_grade.to_string(index=False))

    print("\nIMPORTÂNCIA POR PERMUTAÇÃO G2 NA GRADE (TOP 15):")
    print(df_perm_g2.head(15).to_string(index=False))

    # 8. Regra de Decisão do Vencedor (+0,5 p.p.)
    diff_g2_g1 = med_g2 - med_g1
    logger.info(f"\nDiferença G2 vs G1: {diff_g2_g1*100:+.3f} p.p. (limiar de vitória: +0.500 p.p.)")
    g2_venceu = diff_g2_g1 >= 0.0050

    if g2_venceu:
        logger.info(f"-> VENCEDOR: G2 VENCEU! Superou G1 em {diff_g2_g1*100:+.3f} p.p. (>= +0.500 p.p.).")
        print(f"\nDECISÃO: G2 VENCEU! (+{diff_g2_g1*100:.3f} p.p. >= +0.500 p.p.) -> Executando avaliação no teste.")
        avaliar_teste_g2(feats_g2, p_g2, df_mod_merged, processed_dir, out_dir)
    else:
        logger.info(f"-> VENCEDOR: G1 CONTINUA! G2 não atingiu o ganho mínimo de +0.500 p.p. (ganho = {diff_g2_g1*100:+.3f} p.p.).")
        print(f"\nDECISÃO: G1 CONTINUA COMO VENCEDOR! (G2 obteve {diff_g2_g1*100:+.3f} p.p., abaixo do limiar de +0.500 p.p.).")
        print("Regra de teste respeitada: TESTE NÃO EXECUTADO para G2.")

    logger.info(f"Script 14 concluído em {time.time()-t_start:.1f} segundos!")

if __name__ == "__main__":
    main()
