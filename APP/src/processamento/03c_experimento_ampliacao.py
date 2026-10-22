"""
Módulo 03c — Experimento de Validação R0/R1/R2 e Teste Único (Projeto APHIM - Período 2015–2025)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Configurações Avaliadas na Validação (XGBoost count:poisson):
- R0: Treino alvos 2020-02 em diante (como hoje), 20 atributos atuais
- R1: Treino alvos 2016-02 em diante (novo), 20 atributos atuais
- R2: Treino alvos 2016-02 em diante (novo), 20 atributos atuais + 4 índices oceânicos (24 atributos)

Grade de Hiperparâmetros (24 combinações):
- n_estimators: [300, 800]
- learning_rate: [0.03, 0.1]
- max_depth: [3, 5, 7]
- min_child_weight: [1, 10]
- subsample: 0.8, colsample_bytree: 0.8, random_state: 42, n_jobs: -1

Regra de Decisão na Validação (dobras 2022 e 2023):
- R1 ou R2 substitui R0 se Captura@15 média for >= R0 + 0,5 p.p. (+0.0050).
- Se ambos passarem, vence o de maior média.
- Se nenhum passar, vence R0.

Saídas em {processed_dir}/resultados/:
- ampliacao_validacao.csv
- ampliacao_permutacao_R1.csv
- ampliacao_permutacao_R2.csv
- ampliacao_predicoes_teste.parquet
- ampliacao_teste_mensal.csv
- ampliacao_teste_resumo.csv
"""

import os
import sys
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import calcular_metricas_mes

logger = setup_logger("processamento_03c_experimento_ampliacao")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config_2015.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def buscar_hiperparametros(
    config_nome: str,
    feats: List[str],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    grade: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float, float, float]:
    """Varre a grade de hiperparâmetros nas dobras 2022 e 2023 e seleciona a melhor média."""
    logger.info(f"Iniciando busca em grade ({len(grade)} combinações) para {config_nome}...")
    
    melhor_params = None
    melhor_media = -1.0
    melhor_c22 = -1.0
    melhor_c23 = -1.0

    for i, p in enumerate(grade):
        # Dobra 2022
        m22 = xgb.XGBRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            objective="count:poisson",
            random_state=42, n_jobs=-1
        )
        m22.fit(df_tr_22[feats], df_tr_22["y"])
        preds_22 = np.maximum(0.0, m22.predict(df_val_22[feats]))
        c22 = avaliar_predicoes_val(df_val_22, preds_22)

        # Dobra 2023
        m23 = xgb.XGBRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"],
            objective="count:poisson",
            random_state=42, n_jobs=-1
        )
        m23.fit(df_tr_23[feats], df_tr_23["y"])
        preds_23 = np.maximum(0.0, m23.predict(df_val_23[feats]))
        c23 = avaliar_predicoes_val(df_val_23, preds_23)

        media = (c22 + c23) / 2.0
        if media > melhor_media:
            melhor_media = media
            melhor_c22 = c22
            melhor_c23 = c23
            melhor_params = p

    logger.info(
        f"-> Melhor {config_nome}: Média = {melhor_media*100:.3f}% "
        f"(2022: {melhor_c22*100:.3f}%, 2023: {melhor_c23*100:.3f}%) | "
        f"Params: {melhor_params}"
    )
    return melhor_params, melhor_c22, melhor_c23, melhor_media

def calcular_permutacao_validacao(
    feats: List[str],
    params: Dict[str, Any],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    baseline_score: float
) -> pd.DataFrame:
    """Calcula a importância por permutação na validação para a lista de atributos."""
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
        # Permutar em 2022
        val_22_perm = df_val_22[feats].copy()
        val_22_perm[f_col] = rng.permutation(val_22_perm[f_col].values)
        preds_22_perm = np.maximum(0.0, m22.predict(val_22_perm))
        c22_perm = avaliar_predicoes_val(df_val_22, preds_22_perm)

        # Permutar em 2023
        val_23_perm = df_val_23[feats].copy()
        val_23_perm[f_col] = rng.permutation(val_23_perm[f_col].values)
        preds_23_perm = np.maximum(0.0, m23.predict(val_23_perm))
        c23_perm = avaliar_predicoes_val(df_val_23, preds_23_perm)

        score_perm = (c22_perm + c23_perm) / 2.0
        importancia = baseline_score - score_perm
        records.append({"atributo": f_col, "importancia_cap15": importancia})

    df_perm = pd.DataFrame(records).sort_values("importancia_cap15", ascending=False).reset_index(drop=True)
    df_perm.insert(0, "posicao", range(1, len(df_perm) + 1))
    return df_perm

def executar_experimento():
    logger.info("=== INICIANDO ETAPA 2: EXPERIMENTO DE AMPLIAÇÃO 2015-2025 ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem v2
    mod_v2_path = processed_dir / "aphim_modelagem_v2.parquet"
    if not mod_v2_path.exists():
        raise FileNotFoundError(f"Arquivo '{mod_v2_path}' não encontrado. Execute construir_atributos_v2 primeiro.")
    
    df_mod = pd.read_parquet(mod_v2_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    # 2. Definição dos atributos fixos (os 20 atributos do vencedor atual)
    feats_20 = [
        "focos_media_hist_mes", "focos_t", "focos_t1", "mes_sin", "focos_vizinhos_t",
        "mes_cos", "prec_6m", "anom_prec_6m", "razao_fogo_t", "anom_prec_mm",
        "anom_t2m_media_c", "focos_t2", "variacao_anual_t", "umid_solo_t1",
        "razao_fogo_vizinhos_t", "anom_umid_solo", "anom_umid_solo_3m", "anom_prec_3m",
        "meses_secos_seguidos", "prec_3m"
    ]
    feats_ocean = ["nino34_t", "nino34_3m", "tna_t", "tna_3m"]
    feats_24 = feats_20 + feats_ocean

    for f in feats_24:
        if f not in df_mod.columns:
            raise KeyError(f"Atributo '{f}' não encontrado nas colunas da base.")

    # 3. Grade de Hiperparâmetros (mesma do XGBoost)
    grade_xgb = [
        {
            "n_estimators": ne,
            "learning_rate": lr,
            "max_depth": md,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": mcw
        }
        for ne, lr, md, mcw in itertools.product(
            [300, 800],
            [0.03, 0.1],
            [3, 5, 7],
            [1, 10]
        )
    ]

    # Partição de Validação
    # Dobra 2022
    val_22 = df_mod[df_mod["mes_alvo"].str.startswith("2022-")].copy()
    # Dobra 2023
    val_23 = df_mod[df_mod["mes_alvo"].str.startswith("2023-")].copy()

    # R0: Treino 2020-02 em diante (como hoje)
    tr_r0_22 = df_mod[(df_mod["mes_alvo"] >= "2020-02") & (df_mod["mes_alvo"] <= "2021-12")].copy()
    tr_r0_23 = df_mod[(df_mod["mes_alvo"] >= "2020-02") & (df_mod["mes_alvo"] <= "2022-12")].copy()

    # R1 e R2: Treino 2016-02 em diante (novo período ampliado)
    tr_novo_22 = df_mod[(df_mod["mes_alvo"] >= "2016-02") & (df_mod["mes_alvo"] <= "2021-12")].copy()
    tr_novo_23 = df_mod[(df_mod["mes_alvo"] >= "2016-02") & (df_mod["mes_alvo"] <= "2022-12")].copy()

    logger.info(f"Linhas Treino R0 dobra 2022: {len(tr_r0_22)} | dobra 2023: {len(tr_r0_23)}")
    logger.info(f"Linhas Treino R1/R2 dobra 2022: {len(tr_novo_22)} | dobra 2023: {len(tr_novo_23)}")

    # 4. Execução da Busca na Validação (com cache se já gerada)
    out_val_csv = resultados_dir / "ampliacao_validacao.csv"
    if out_val_csv.exists():
        logger.info(f"Carregando tabela de validacao existente de '{out_val_csv}'...")
        df_val_resumo = pd.read_csv(out_val_csv)
        row_r0 = df_val_resumo[df_val_resumo["configuracao"] == "R0"].iloc[0]
        row_r1 = df_val_resumo[df_val_resumo["configuracao"] == "R1"].iloc[0]
        row_r2 = df_val_resumo[df_val_resumo["configuracao"] == "R2"].iloc[0]
        p_r0 = json.loads(row_r0["hiperparametros"])
        med_r0 = float(row_r0["cap15_media"])
        p_r1 = json.loads(row_r1["hiperparametros"])
        med_r1 = float(row_r1["cap15_media"])
        p_r2 = json.loads(row_r2["hiperparametros"])
        med_r2 = float(row_r2["cap15_media"])
    else:
        # R0
        p_r0, c22_r0, c23_r0, med_r0 = buscar_hiperparametros("R0", feats_20, tr_r0_22, val_22, tr_r0_23, val_23, grade_xgb)
        
        # Checagem de Sanidade de R0
        val_esperado_r0 = 0.56409
        diff_sanidade = med_r0 - val_esperado_r0
        logger.info(f"Checagem de Sanidade R0: obtido = {med_r0:.5f} | esperado ~ {val_esperado_r0:.5f} | diff = {diff_sanidade*100:+.3f} p.p.")
        if abs(diff_sanidade) > 0.0150:
            raise ValueError(f"FALHA NA CHECAGEM DE SANIDADE: Diferenca do R0 ({diff_sanidade*100:+.2f} p.p.) excedeu 1.5 ponto percentual!")

        # R1
        p_r1, c22_r1, c23_r1, med_r1 = buscar_hiperparametros("R1", feats_20, tr_novo_22, val_22, tr_novo_23, val_23, grade_xgb)

        # R2
        p_r2, c22_r2, c23_r2, med_r2 = buscar_hiperparametros("R2", feats_24, tr_novo_22, val_22, tr_novo_23, val_23, grade_xgb)

        # Tabela de Validação
        df_val_resumo = pd.DataFrame([
            {
                "configuracao": "R0",
                "treino_inicio": "2020-02",
                "n_atributos": len(feats_20),
                "hiperparametros": json.dumps(p_r0),
                "cap15_dobra_2022": c22_r0,
                "cap15_dobra_2023": c23_r0,
                "cap15_media": med_r0
            },
            {
                "configuracao": "R1",
                "treino_inicio": "2016-02",
                "n_atributos": len(feats_20),
                "hiperparametros": json.dumps(p_r1),
                "cap15_dobra_2022": c22_r1,
                "cap15_dobra_2023": c23_r1,
                "cap15_media": med_r1
            },
            {
                "configuracao": "R2",
                "treino_inicio": "2016-02",
                "n_atributos": len(feats_24),
                "hiperparametros": json.dumps(p_r2),
                "cap15_dobra_2022": c22_r2,
                "cap15_dobra_2023": c23_r2,
                "cap15_media": med_r2
            }
        ])
        df_val_resumo.to_csv(out_val_csv, index=False, encoding="utf-8")
        logger.info(f"Tabela de validacao salva em '{out_val_csv}'.")

    # Importância por Permutação para R1 e R2
    out_perm_r1 = resultados_dir / "ampliacao_permutacao_R1.csv"
    if not out_perm_r1.exists():
        logger.info("Calculando importancia por permutacao na validacao para R1...")
        df_perm_r1 = calcular_permutacao_validacao(feats_20, p_r1, tr_novo_22, val_22, tr_novo_23, val_23, med_r1)
        df_perm_r1.to_csv(out_perm_r1, index=False, encoding="utf-8")

    out_perm_r2 = resultados_dir / "ampliacao_permutacao_R2.csv"
    if not out_perm_r2.exists():
        logger.info("Calculando importancia por permutacao na validacao para R2...")
        df_perm_r2 = calcular_permutacao_validacao(feats_24, p_r2, tr_novo_22, val_22, tr_novo_23, val_23, med_r2)
        df_perm_r2.to_csv(out_perm_r2, index=False, encoding="utf-8")


    # 5. Regra de Escolha do Vencedor
    candidatos_passaram = []
    diff_r1 = med_r1 - med_r0
    diff_r2 = med_r2 - med_r0
    logger.info(f"Diferença R1 vs R0: {diff_r1*100:+.3f} p.p. (limiar: +0.500 p.p.)")
    logger.info(f"Diferença R2 vs R0: {diff_r2*100:+.3f} p.p. (limiar: +0.500 p.p.)")

    if diff_r1 >= 0.0050:
        candidatos_passaram.append(("R1", med_r1, p_r1, feats_20, "2016-02"))
    if diff_r2 >= 0.0050:
        candidatos_passaram.append(("R2", med_r2, p_r2, feats_24, "2016-02"))

    if len(candidatos_passaram) > 0:
        # Vence o de maior média
        candidatos_passaram.sort(key=lambda x: x[1], reverse=True)
        vencedor_cod, vencedor_media, vencedor_params, vencedor_feats, vencedor_inicio = candidatos_passaram[0]
        logger.info(f"-> VENCEDOR: {vencedor_cod} com Captura@15 média de {vencedor_media*100:.3f}% (superou o limiar de +0.5 p.p.)")
    else:
        vencedor_cod = "R0"
        vencedor_media = med_r0
        vencedor_params = p_r0
        vencedor_feats = feats_20
        vencedor_inicio = "2020-02"
        logger.info(f"-> VENCEDOR: R0 (nem R1 nem R2 superaram R0 por pelo menos +0.5 p.p.). R0 mantido com {med_r0*100:.3f}%.")

    # ==============================================================================
    # PARTE D: TESTE, UMA ÚNICA VEZ
    # ==============================================================================
    logger.info(f"=== PARTE D: RETREINO DO VENCEDOR ({vencedor_cod}) E AVALIAÇÃO NO TESTE ===")

    # Treino completo do vencedor (até 2023-12, mes_emissao até 2023-11)
    df_treino_full = df_mod[(df_mod["mes_alvo"] >= vencedor_inicio) & (df_mod["mes_alvo"] <= "2023-12")].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()
    meses_alvo_teste = sorted(df_teste["mes_alvo"].unique())

    logger.info(f"Retreinando {vencedor_cod} em {len(df_treino_full)} linhas (alvos {vencedor_inicio} a 2023-12)...")
    m_final = xgb.XGBRegressor(
        n_estimators=vencedor_params["n_estimators"],
        learning_rate=vencedor_params["learning_rate"],
        max_depth=vencedor_params["max_depth"],
        subsample=vencedor_params["subsample"],
        colsample_bytree=vencedor_params["colsample_bytree"],
        min_child_weight=vencedor_params["min_child_weight"],
        objective="count:poisson",
        random_state=42, n_jobs=-1
    )
    m_final.fit(df_treino_full[vencedor_feats], df_treino_full["y"])

    preds_teste = np.maximum(0.0, m_final.predict(df_teste[vencedor_feats]))
    df_teste["pred_vencedor"] = preds_teste

    # Salvar ampliacao_predicoes_teste.parquet
    out_preds_teste = resultados_dir / "ampliacao_predicoes_teste.parquet"
    df_teste[["cd_mun", "mes_emissao", "mes_alvo", "y", "pred_vencedor"]].to_parquet(out_preds_teste, index=False)
    logger.info(f"Predições salvas em '{out_preds_teste}'.")

    # Mês a mês: métricas, Captura@15, soma de focos previstos vs reais e correlação de Spearman
    print("\n" + "=" * 95)
    print(f"             AVALIACAO MENSAL NO CONJUNTO DE TESTE - MODELO {vencedor_cod} (2024-2025)")
    print("=" * 95)
    print(f"{'Mes Alvo':<10} | {'Captura@15':<12} | {'Focos Reais':<14} | {'Focos Previstos':<16} | {'Spearman':<12}")
    print("-" * 95)


    mensal_records = []
    for mes in meses_alvo_teste:
        sub = df_teste[df_teste["mes_alvo"] == mes]
        met, _ = calcular_metricas_mes(sub, col_score="pred_vencedor", k_valores=(15,))
        cap15 = met["captura_15"]
        soma_real = float(sub["y"].sum())
        soma_pred = float(sub["pred_vencedor"].sum())
        sp_corr, _ = spearmanr(sub["pred_vencedor"], sub["y"])

        print(f"{mes:<10} | {cap15*100:>10.2f}% | {soma_real:>14,.0f} | {soma_pred:>16,.0f} | {sp_corr:>12.4f}")
        mensal_records.append({
            "mes_alvo": mes,
            "captura_15": cap15,
            "focos_real": soma_real,
            "focos_prev": soma_pred,
            "spearman": sp_corr
        })
    print("=" * 95)

    df_mensal_out = pd.DataFrame(mensal_records)
    out_mensal_csv = resultados_dir / "ampliacao_teste_mensal.csv"
    df_mensal_out.to_csv(out_mensal_csv, index=False, encoding="utf-8")

    # Cálculo da Captura Agregada
    total_focos_teste = float(df_teste["y"].sum())
    top15_focos_soma = 0.0
    for mes in meses_alvo_teste:
        sub = df_teste[df_teste["mes_alvo"] == mes].sort_values(by=["pred_vencedor", "cd_mun"], ascending=[False, True])
        top15_focos_soma += float(sub.iloc[:15]["y"].sum())
    cap_agregada_vencedor = top15_focos_soma / total_focos_teste
    cap_media_mensal = float(df_mensal_out["captura_15"].mean())

    # Comparação com o vencedor atual
    # Ler vencedor atual de data/processed/resultados/features_v2_predicoes_teste.parquet
    old_preds_path = Path("data/processed/resultados/features_v2_predicoes_teste.parquet")
    if not old_preds_path.exists():
        raise FileNotFoundError(f"Arquivo '{old_preds_path}' não encontrado para comparação de teste.")
    df_old_preds = pd.read_parquet(old_preds_path)
    df_old_preds["cd_mun"] = df_old_preds["cd_mun"].astype(str).str.zfill(7)
    df_old_preds = df_old_preds.merge(df_teste[["cd_mun", "mes_alvo", "pct_floresta"]], on=["cd_mun", "mes_alvo"], how="left")


    # Obter Captura@15 mensal do vencedor atual
    old_caps_dict = {}
    for mes in meses_alvo_teste:
        sub_old = df_old_preds[df_old_preds["mes_alvo"] == mes]
        met_old, _ = calcular_metricas_mes(sub_old, col_score="pred_vencedor", k_valores=(15,))
        old_caps_dict[mes] = met_old["captura_15"]

    # Obter Captura@15 mensal dos baselines recalculados
    base_mensal_path = resultados_dir / "baselines_mensal.csv"
    if not base_mensal_path.exists():
        raise FileNotFoundError(f"Arquivo '{base_mensal_path}' não encontrado.")
    df_base_mensal = pd.read_csv(base_mensal_path)
    col_base = "baseline" if "baseline" in df_base_mensal.columns else "metodo"
    clima_caps = df_base_mensal[df_base_mensal[col_base] == "climatologia_focos"].set_index("mes_alvo")["captura_15"].to_dict()
    pers_caps = df_base_mensal[df_base_mensal[col_base] == "persistencia"].set_index("mes_alvo")["captura_15"].to_dict()



    # Montar séries alinhadas de 24 meses
    v_cap = np.array([df_mensal_out[df_mensal_out["mes_alvo"] == m]["captura_15"].values[0] for m in meses_alvo_teste])
    old_cap = np.array([old_caps_dict[m] for m in meses_alvo_teste])
    clima_cap = np.array([clima_caps[m] for m in meses_alvo_teste])
    pers_cap = np.array([pers_caps[m] for m in meses_alvo_teste])

    comparadores = [
        ("vencedor_atual_features_v2", old_cap),
        ("climatologia_focos", clima_cap),
        ("persistencia", pers_cap)
    ]

    rng = np.random.RandomState(42)
    n_boot = 10000

    testes_resumo = []
    pvals_raw = []

    for comp_nome, comp_vals in comparadores:
        diff = v_cap - comp_vals
        diff_media = float(np.mean(diff))

        # Bootstrap 95% CI
        boot_idx = rng.choice(len(diff), size=(n_boot, len(diff)), replace=True)
        boot_means = diff[boot_idx].mean(axis=1)
        ic_inf = float(np.percentile(boot_means, 2.5))
        ic_sup = float(np.percentile(boot_means, 97.5))

        # Wilcoxon
        if np.all(diff == 0):
            p_val = 1.0
        else:
            try:
                res_w = wilcoxon(v_cap, comp_vals, alternative="two-sided")
                p_val = float(res_w.pvalue)
            except Exception as e:
                logger.warning(f"Erro no Wilcoxon com {comp_nome}: {e}")
                p_val = 1.0

        pvals_raw.append(p_val)
        testes_resumo.append({
            "modelo_vencedor": vencedor_cod,
            "captura_15_media_mensal": cap_media_mensal,
            "captura_15_agregada": cap_agregada_vencedor,
            "comparado_com": comp_nome,
            "captura_15_comparador": float(np.mean(comp_vals)),
            "diferenca_media": diff_media,
            "ic_95_inf": ic_inf,
            "ic_95_sup": ic_sup,
            "p_valor": p_val
        })

    # Correção de Holm
    adj_p = holm_bonferroni(np.array(pvals_raw))
    for i in range(len(testes_resumo)):
        testes_resumo[i]["p_valor_holm"] = float(adj_p[i])

    df_testes_resumo = pd.DataFrame(testes_resumo)
    out_teste_resumo_csv = resultados_dir / "ampliacao_teste_resumo.csv"
    df_testes_resumo.to_csv(out_teste_resumo_csv, index=False, encoding="utf-8")
    logger.info(f"Resumo do teste salvo em '{out_teste_resumo_csv}'.")

    print("\n" + "=" * 105)
    print("                    RESUMO DO TESTE E COMPARACOES ESTATISTICAS")
    print("=" * 105)

    print(df_testes_resumo.to_string(index=False))
    print("=" * 105 + "\n")

    return df_val_resumo, vencedor_cod, df_testes_resumo

if __name__ == "__main__":
    executar_experimento()
