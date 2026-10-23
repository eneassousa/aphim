"""
Módulo 09 — Validação e Teste do DETER no Nível 1 (Município) (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Comparar na Validação Temporal (dobras 2022 e 2023):
   - D0: 20 atributos do modelo vencedor atual (features_v2_permutacao.csv)
   - D1: 20 atributos do D0 + 8 novos atributos do DETER (28 atributos)
   - Modelo: XGBoost count:poisson
   - Treino: alvos a partir de 2020-02 (como hoje)
2. Checagem de Sanidade:
   - A Captura@15 média de validação do D0 deve reproduzir a do vencedor atual em features_v2_validacao.csv
     com diferença máxima de até 0,1 ponto (|diff| <= 0.0010).
3. Regra de Escolha do Vencedor:
   - D1 só substitui D0 se a Captura@15 média de validação for pelo menos 0,5 ponto maior (+0.0050).
   - Sem eliminação para trás.
4. Importância por Permutação:
   - Importância por permutação do D1 na validação nas dobras 2022 e 2023.
5. Avaliação no Teste (executada estritamente uma única vez se D1 vencer):
   - Se D1 vencer: retreina D1 em todo o treino (2020-02 a 2023-12) e prevê o teste (2024-01 a 2025-12).
   - Comparação pareada contra vencedor atual (features_v2_predicoes_teste.parquet), climatologia e persistência.
   - Teste de Wilcoxon, Bootstrap 95% (10.000 reamostragens, semente 42) e correção de Holm.
   - Se D1 não vencer: não executa o teste; registra continuidade do D0.

Saídas em data/processed_deter/resultados/:
- deter_validacao_municipio.csv
- deter_permutacao_municipio.csv
- deter_predicoes_teste_municipio.parquet (se D1 vencer)
- deter_teste_resumo_municipio.csv (se D1 vencer)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import calcular_metricas_mes, avaliar_metodo_mensal

logger = setup_logger("processamento_09_deter_validacao")

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

def avaliar_predicoes_val(df_val: pd.DataFrame, preds: np.ndarray) -> float:
    """Calcula a Captura@15 média mensal no conjunto de validação."""
    df_eval = df_val[["cd_mun", "mes_alvo", "y", "pct_floresta"]].copy()
    df_eval["pred"] = preds
    caps = []
    for mes in sorted(df_eval["mes_alvo"].unique()):
        sub = df_eval[df_eval["mes_alvo"] == mes]
        met, _ = calcular_metricas_mes(sub, col_score="pred", k_valores=(15,))
        caps.append(met["captura_15"])
    return float(np.mean(caps))

def treinar_e_avaliar_dobras(
    feats: List[str],
    params: Dict[str, Any],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame
) -> Tuple[float, float, float]:
    """Treina o modelo XGBoost Poisson nas dobras temporais 2022 e 2023 e retorna as Capturas@15."""
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
    m22.fit(df_tr_22[feats], df_tr_22["y"])
    preds_22 = np.maximum(0.0, m22.predict(df_val_22[feats]))
    c22 = avaliar_predicoes_val(df_val_22, preds_22)

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
    m23.fit(df_tr_23[feats], df_tr_23["y"])
    preds_23 = np.maximum(0.0, m23.predict(df_val_23[feats]))
    c23 = avaliar_predicoes_val(df_val_23, preds_23)

    c_med = (c22 + c23) / 2.0
    return c22, c23, c_med

def calcular_permutacao(
    feats: List[str],
    params: Dict[str, Any],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    score_baseline: float
) -> pd.DataFrame:
    """Calcula a importância por permutação na validação para a lista de atributos de D1."""
    logger.info("Ajustando modelos base nas dobras 2022 e 2023 para importância por permutação...")
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
    m22.fit(df_tr_22[feats], df_tr_22["y"])

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
    m23.fit(df_tr_23[feats], df_tr_23["y"])

    rng = np.random.RandomState(42)
    records = []

    for f_col in feats:
        # Dobra 2022
        val_22_p = df_val_22[feats].copy()
        val_22_p[f_col] = rng.permutation(val_22_p[f_col].values)
        p22 = np.maximum(0.0, m22.predict(val_22_p))
        c22 = avaliar_predicoes_val(df_val_22, p22)

        # Dobra 2023
        val_23_p = df_val_23[feats].copy()
        val_23_p[f_col] = rng.permutation(val_23_p[f_col].values)
        p23 = np.maximum(0.0, m23.predict(val_23_p))
        c23 = avaliar_predicoes_val(df_val_23, p23)

        score_p = (c22 + c23) / 2.0
        importancia = score_baseline - score_p
        records.append({"atributo": f_col, "importancia_cap15": importancia})

    df_perm = pd.DataFrame(records).sort_values("importancia_cap15", ascending=False).reset_index(drop=True)
    df_perm.insert(0, "posicao", range(1, len(df_perm) + 1))
    return df_perm

def avaliar_teste_d1(
    feats_d1: List[str],
    params: Dict[str, Any],
    df_mod: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path
):
    """
    Treina o modelo D1 em todo o conjunto de treino (2020-02 a 2023-12)
    e prevê o conjunto de teste (2024-01 a 2025-12). Executado estritamente se D1 vencer.
    """
    logger.info("=== RETREINANDO MODELO D1 NO TREINO COMPLETO E PREVENDO NO TESTE ===")
    df_treino = df_mod[(df_mod["conjunto"] == "treino") & (df_mod["mes_alvo"] >= "2020-02")].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()

    m_final = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        objective="count:poisson",
        random_state=42, n_jobs=-1
    )
    m_final.fit(df_treino[feats_d1], df_treino["y"])
    preds_teste = np.maximum(0.0, m_final.predict(df_teste[feats_d1]))

    df_pred_out = df_teste[["cd_mun", "mes_emissao", "mes_alvo", "y"]].copy()
    df_pred_out["pred_vencedor"] = preds_teste

    out_parquet = out_dir / "deter_predicoes_teste_municipio.parquet"
    df_pred_out.to_parquet(out_parquet, index=False)
    logger.info(f"Predições de teste salvas em '{out_parquet}'.")

    # Avaliação mensal do D1
    df_m_d1 = avaliar_metodo_mensal(
        df_periodo=df_pred_out,
        col_score="pred_vencedor",
        nome_metodo="D1_xgb_deter",
        df_ancora=None,
        col_target="y"
    )

    # Carregar predições do vencedor atual
    venc_path = processed_dir / "resultados" / "features_v2_predicoes_teste.parquet"
    if not venc_path.exists():
        raise FileNotFoundError(f"Arquivo '{venc_path}' não encontrado.")
    df_venc = pd.read_parquet(venc_path)

    df_m_venc = avaliar_metodo_mensal(
        df_periodo=df_venc,
        col_score="pred_vencedor",
        nome_metodo="vencedor_v2",
        df_ancora=None,
        col_target="y"
    )

    # Carregar baselines
    base_path = processed_dir / "resultados" / "baselines_mensal.csv"
    df_base = pd.read_csv(base_path)
    df_clim = df_base[df_base["baseline"] == "climatologia_focos"].sort_values("mes_alvo")
    df_pers = df_base[df_base["baseline"] == "persistencia"].sort_values("mes_alvo")

    # Comparação pareada de Captura@15 nos 24 meses
    caps_d1 = df_m_d1.sort_values("mes_alvo")["captura_15"].values
    caps_venc = df_m_venc.sort_values("mes_alvo")["captura_15"].values
    caps_clim = df_clim["captura_15"].values
    caps_pers = df_pers["captura_15"].values

    assert len(caps_d1) == 24, "Esperado exatamente 24 meses de teste."

    # Bootstrap 95% (10.000 reamostragens, semente 42)
    def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
        rng = np.random.RandomState(seed)
        diffs = a - b
        mean_diff = float(np.mean(diffs))
        boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        stat, pval = wilcoxon(a, b)
        return mean_diff, ci_lower, ci_upper, float(pval)

    diff_venc, ci_l_v, ci_u_v, p_venc = bootstrap_diff(caps_d1, caps_venc)
    diff_clim, ci_l_c, ci_u_c, p_clim = bootstrap_diff(caps_d1, caps_clim)
    diff_pers, ci_l_p, ci_u_p, p_pers = bootstrap_diff(caps_d1, caps_pers)

    # Correção de Holm
    pvals_raw = np.array([p_venc, p_clim, p_pers])
    pvals_holm = holm_bonferroni(pvals_raw)

    resumo_teste = pd.DataFrame([
        {
            "comparador": "vencedor_v2",
            "cap15_d1": float(np.mean(caps_d1)),
            "cap15_comparador": float(np.mean(caps_venc)),
            "diff_media": diff_venc,
            "ic95_inferior": ci_l_v,
            "ic95_superior": ci_u_v,
            "p_wilcoxon": p_venc,
            "p_holm": pvals_holm[0]
        },
        {
            "comparador": "climatologia",
            "cap15_d1": float(np.mean(caps_d1)),
            "cap15_comparador": float(np.mean(caps_clim)),
            "diff_media": diff_clim,
            "ic95_inferior": ci_l_c,
            "ic95_superior": ci_u_c,
            "p_wilcoxon": p_clim,
            "p_holm": pvals_holm[1]
        },
        {
            "comparador": "persistencia",
            "cap15_d1": float(np.mean(caps_d1)),
            "cap15_comparador": float(np.mean(caps_pers)),
            "diff_media": diff_pers,
            "ic95_inferior": ci_l_p,
            "ic95_superior": ci_u_p,
            "p_wilcoxon": p_pers,
            "p_holm": pvals_holm[2]
        }
    ])

    out_csv = out_dir / "deter_teste_resumo_municipio.csv"
    resumo_teste.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Resumo do benchmark de teste salvo em '{out_csv}'.")
    print("\nBENCHMARK DE TESTE DO MODELO D1:")
    print(resumo_teste.to_string(index=False))

def main():
    logger.info("=== INICIANDO EXPERIMENTO DE VALIDAÇÃO D0 vs D1 (MUNICÍPIO) ===")
    processed_dir = Path("data/processed")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = processed_deter_dir / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem com atributos DETER
    deter_mod_path = processed_deter_dir / "aphim_modelagem_v2_deter.parquet"
    if not deter_mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{deter_mod_path}' não encontrado. Execute 08b_deter_atributos.py primeiro.")

    logger.info(f"Carregando base com atributos DETER de '{deter_mod_path}'...")
    df_mod = pd.read_parquet(deter_mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    # 2. Definir atributos de D0 e D1
    perm_path = processed_dir / "resultados" / "features_v2_permutacao.csv"
    if not perm_path.exists():
        raise FileNotFoundError(f"Arquivo '{perm_path}' não encontrado.")
    df_perm_ref = pd.read_csv(perm_path)
    feats_d0_set = set(df_perm_ref["atributo"].tolist())
    assert len(feats_d0_set) == 20, f"Esperado 20 atributos em D0, obtido {len(feats_d0_set)}."

    # Ordem canônica de A_v2_sel de 03b_atributos_v2.py (garante reprodutibilidade com colsample_bytree)
    ordem_canonica = [
        "focos_t", "focos_t1", "focos_t2", "focos_media_hist_mes",
        "anom_prec_mm", "anom_t2m_media_c", "anom_umid_solo", "prec_3m", "anom_prec_3m",
        "prec_6m", "anom_prec_6m", "meses_secos_seguidos", "umid_solo_t1", "anom_umid_solo_3m",
        "razao_fogo_t", "variacao_anual_t",
        "focos_vizinhos_t", "razao_fogo_vizinhos_t",
        "mes_sin", "mes_cos"
    ]
    feats_d0 = [f for f in ordem_canonica if f in feats_d0_set]
    assert len(feats_d0) == 20, f"Esperado 20 atributos na ordem canônica, obtido {len(feats_d0)}."

    feats_deter_8 = [
        "deter_desmat_1m", "deter_desmat_3m", "deter_desmat_12m", "deter_desmat_3m_pct",
        "deter_degrad_1m", "deter_degrad_3m", "deter_degrad_12m", "deter_degrad_3m_pct"
    ]
    feats_d1 = feats_d0 + feats_deter_8
    assert len(feats_d1) == 28, f"Esperado 28 atributos em D1, obtido {len(feats_d1)}."

    # 3. Carregar hiperparâmetros do modelo vencedor
    hiper_path = processed_dir / "resultados" / "perigo_hiperparametros.json"
    with open(hiper_path, "r", encoding="utf-8") as f:
        params_all = json.load(f)
    params_xgb = params_all["xgb_A"]
    logger.info(f"Hiperparâmetros fixos XGBoost de 03b: {params_xgb}")

    # 4. Partições de Validação
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    # Dobra 2022: treino alvos 2020-02 a 2021-12, validação 2022
    df_tr_22 = df_treino[(df_treino["mes_alvo"] >= "2020-02") & (df_treino["mes_alvo"] <= "2021-12")].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()

    # Dobra 2023: treino alvos 2020-02 a 2022-12, validação 2023
    df_tr_23 = df_treino[(df_treino["mes_alvo"] >= "2020-02") & (df_treino["mes_alvo"] <= "2022-12")].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    logger.info(f"Treino 2022: {len(df_tr_22)} linhas | Validação 2022: {len(df_val_22)} linhas")
    logger.info(f"Treino 2023: {len(df_tr_23)} linhas | Validação 2023: {len(df_val_23)} linhas")

    # 5. Avaliação de D0 na validação
    logger.info("Avaliando D0 (20 atributos atuais) na validação temporal...")
    c22_d0, c23_d0, med_d0 = treinar_e_avaliar_dobras(feats_d0, params_xgb, df_tr_22, df_val_22, df_tr_23, df_val_23)
    logger.info(f"-> D0: Dobra 2022 = {c22_d0*100:.3f}% | Dobra 2023 = {c23_d0*100:.3f}% | Média = {med_d0*100:.3f}%")

    # Checagem de Sanidade de D0 contra features_v2_validacao.csv
    val_ref_path = processed_dir / "resultados" / "features_v2_validacao.csv"
    df_val_ref = pd.read_csv(val_ref_path)
    score_esperado_d0 = float(df_val_ref[(df_val_ref["modelo"] == "xgb") & (df_val_ref["conjunto"] == "A_v2_sel")]["cap15_media"].iloc[0])
    
    diff_sanidade = abs(med_d0 - score_esperado_d0)
    logger.info(f"Checagem de Sanidade: D0 reproduzido = {med_d0*100:.4f}% | Esperado = {score_esperado_d0*100:.4f}% | Diferença = {diff_sanidade*100:.4f} p.p.")
    if diff_sanidade > 0.0010: # mais que 0,1 ponto (0,0010)
        msg = f"FALHA NA CHECAGEM DE SANIDADE: Diferença {diff_sanidade*100:.4f} p.p. excede o limite estrito de 0,10 p.p."
        logger.error(msg)
        raise ValueError(msg)
    logger.info("CHECAGEM DE SANIDADE APROVADA: D0 reproduziu o vencedor atual com perfeição!")

    # 6. Avaliação de D1 na validação
    logger.info("Avaliando D1 (20 atributos + 8 atributos DETER) na validação temporal...")
    c22_d1, c23_d1, med_d1 = treinar_e_avaliar_dobras(feats_d1, params_xgb, df_tr_22, df_val_22, df_tr_23, df_val_23)
    logger.info(f"-> D1: Dobra 2022 = {c22_d1*100:.3f}% | Dobra 2023 = {c23_d1*100:.3f}% | Média = {med_d1*100:.3f}%")

    # Tabela de Validação
    df_validacao_out = pd.DataFrame([
        {
            "configuracao": "D0",
            "n_atributos": len(feats_d0),
            "hiperparametros": json.dumps(params_xgb),
            "cap15_dobra_2022": c22_d0,
            "cap15_dobra_2023": c23_d0,
            "cap15_media": med_d0
        },
        {
            "configuracao": "D1",
            "n_atributos": len(feats_d1),
            "hiperparametros": json.dumps(params_xgb),
            "cap15_dobra_2022": c22_d1,
            "cap15_dobra_2023": c23_d1,
            "cap15_media": med_d1
        }
    ])
    out_val_csv = out_dir / "deter_validacao_municipio.csv"
    df_validacao_out.to_csv(out_val_csv, index=False, encoding="utf-8")
    logger.info(f"Tabela deter_validacao_municipio salva em '{out_val_csv}'.")

    print("\n" + "=" * 80)
    print("        TABELA DE VALIDAÇÃO TEMPORAL - NÍVEL 1 (MUNICÍPIO)")
    print("=" * 80)
    print(df_validacao_out.to_string(index=False))

    # 7. Importância por Permutação de D1 na validação
    logger.info("Calculando importância por permutação na validação para D1...")
    df_perm_d1 = calcular_permutacao(feats_d1, params_xgb, df_tr_22, df_val_22, df_tr_23, df_val_23, med_d1)
    out_perm_csv = out_dir / "deter_permutacao_municipio.csv"
    df_perm_d1.to_csv(out_perm_csv, index=False, encoding="utf-8")
    logger.info(f"Importância por permutação salva em '{out_perm_csv}'.")

    print("\nIMPORTÂNCIA POR PERMUTAÇÃO D1 (TOP 15):")
    print(df_perm_d1.head(15).to_string(index=False))

    # 8. Decisão do Vencedor
    diff_d1_d0 = med_d1 - med_d0
    logger.info(f"\nDiferença D1 vs D0: {diff_d1_d0*100:+.3f} p.p. (limiar de vitória: +0.500 p.p.)")
    d1_venceu = diff_d1_d0 >= 0.0050

    if d1_venceu:
        logger.info(f"-> VENCEDOR: D1 VENCEU! Superou D0 em {diff_d1_d0*100:+.3f} p.p. (>= +0.500 p.p.).")
        print(f"\nDECISÃO: D1 VENCEU! (+{diff_d1_d0*100:.3f} p.p. >= +0.500 p.p.) -> Executando avaliação no teste.")
        avaliar_teste_d1(feats_d1, params_xgb, df_mod, processed_dir, out_dir)
    else:
        logger.info(f"-> VENCEDOR: D0 CONTINUA! D1 não atingiu o ganho mínimo de +0.500 p.p. (ganho = {diff_d1_d0*100:+.3f} p.p.).")
        print(f"\nDECISÃO: D0 CONTINUA COMO VENCEDOR! (D1 obteve {diff_d1_d0*100:+.3f} p.p., abaixo do limiar de +0.500 p.p.).")
        print("Regra de teste respeitada: TESTE NÃO EXECUTADO para D1.")

if __name__ == "__main__":
    main()
