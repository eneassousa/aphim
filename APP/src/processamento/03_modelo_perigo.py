"""
Módulo 03 — Módulo de Perigo e Modelagem Preditiva (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Treinar modelos de aprendizado de máquina para predição da contagem de focos no mês alvo (y).
2. Avaliar o ranking de priorização municipal gerado pelos modelos, utilizando as mesmas métricas
   padronizadas dos baselines (Captura@k, Captura*@15, Captura Ponderada@15, Spearman, Rotatividade).
3. Comparar dois conjuntos de atributos:
   - Conjunto A (perigo puro): histórico de focos, clima, anomalias, acumulados e sazonalidade (28 features).
   - Conjunto C (híbrido C): Conjunto A + atributos estruturais (área, uso do solo, desmatamento PRODES, renda) (36 features).
4. Modelos avaliados (cada um com A e C):
   - GLM Poisson (PoissonRegressor com StandardScaler e log1p nos termos de contagem/densidade).
   - Random Forest (RandomForestRegressor com alvo log1p(y)).
   - XGBoost (XGBRegressor com objective='count:poisson').
5. Seleção temporal de hiperparâmetros por origem móvel (dobras 2022 e 2023) com foco em Captura@15.
6. Retreino no conjunto de treino completo e predição única no conjunto de teste (2024-2025).
7. Testes estatísticos pareados (Wilcoxon e Bootstrap 95%) contra os melhores baselines.
8. Explicabilidade global via SHAP (TreeExplainer).

Saídas:
- data/processed/resultados/perigo_validacao.csv
- data/processed/resultados/perigo_hiperparametros.json
- data/processed/resultados/perigo_predicoes_teste.parquet
- data/processed/resultados/perigo_mensal.csv
- data/processed/resultados/perigo_resumo.csv
- data/processed/resultados/perigo_testes_estatisticos.csv
- data/processed/resultados/shap_xgb_A.png
- data/processed/resultados/shap_xgb_C.png
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
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
import xgboost as xgb
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import (
    calcular_metricas_mes,
    avaliar_metodo_mensal,
    adicionar_captura_star_15,
    gerar_tabela_resumo
)

logger = setup_logger("processamento_03_modelo_perigo")

# Definição dos conjuntos de atributos
COLS_HIST_FOCOS = [
    "focos_t", "focos_t1", "focos_t2", "focos_mm3", "focos_mesmo_mes_ano_ant", "focos_media_hist_mes",
    "dens_t", "dens_t1", "dens_t2", "dens_mm3", "dens_mesmo_mes_ano_ant", "dens_media_hist_mes"
]

COLS_CLIMA = [
    "prec_mm", "t2m_max_c", "ur_pct", "dpv_kpa", "umid_solo", "lst_c", "vento_ms"
]

COLS_ANOMALIAS = [
    "anom_prec_mm", "anom_t2m_media_c", "anom_ur_pct", "anom_umid_solo", "anom_lst_c"
]

COLS_ACUMULADOS = [
    "prec_3m", "anom_prec_3m"
]

COLS_SAZONALIDADE = [
    "mes_sin", "mes_cos"
]

COLS_ESTRUTURAIS = [
    "area_km2", "pct_floresta", "pct_pastagem", "pct_agricultura", "pct_solo_exposto",
    "desmat_km2_prodes", "desmat_pct_area_prodes", "renda_pc"
]

# Conjunto A (28 features)
COLS_A = COLS_HIST_FOCOS + COLS_CLIMA + COLS_ANOMALIAS + COLS_ACUMULADOS + COLS_SAZONALIDADE
# Conjunto C (36 features)
COLS_C = COLS_A + COLS_ESTRUTURAIS

# Colunas de contagem e densidade que recebem log1p no GLM
COLS_LOG1P_GLM = COLS_HIST_FOCOS

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE TREINAMENTO E PREDIÇÃO POR MODELO
# ==============================================================================

def preparar_features_glm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Aplica log1p nas variáveis de contagem/densidade e padroniza com StandardScaler."""
    X_tr = X_train[feature_cols].copy()
    X_v = X_val[feature_cols].copy()

    log_cols = [c for c in COLS_LOG1P_GLM if c in feature_cols]
    X_tr[log_cols] = np.log1p(np.maximum(0, X_tr[log_cols]))
    X_v[log_cols] = np.log1p(np.maximum(0, X_v[log_cols]))

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_v_scaled = scaler.transform(X_v)
    return X_tr_scaled, X_v_scaled, scaler

def treinar_e_prever(
    tipo_modelo: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Treina o modelo nos dados de treino e retorna previsões para o conjunto de validação/teste.
    Para o Random Forest, treina com log1p(y) e retorna previsão na escala original (expm1).
    """
    if tipo_modelo == "glm":
        X_tr_s, X_v_s, _ = preparar_features_glm(X_train, X_val, feature_cols)
        reg = PoissonRegressor(alpha=params["alpha"], max_iter=1000)
        reg.fit(X_tr_s, y_train)
        preds = reg.predict(X_v_s)
        return np.maximum(0.0, preds)

    elif tipo_modelo == "rf":
        y_train_log = np.log1p(y_train)
        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train[feature_cols], y_train_log)
        preds_log = rf.predict(X_val[feature_cols])
        # expm1 para converter de volta para a escala de contagem esperada
        preds = np.expm1(preds_log)
        return np.maximum(0.0, preds)

    elif tipo_modelo == "xgb":
        xgb_reg = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            objective="count:poisson",
            random_state=42,
            n_jobs=-1
        )
        xgb_reg.fit(X_train[feature_cols], y_train)
        preds = xgb_reg.predict(X_val[feature_cols])
        return np.maximum(0.0, preds)

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {tipo_modelo}")

# ==============================================================================
# 2. SELEÇÃO DE HIPERPARÂMETROS POR VALIDAÇÃO TEMPORAL (ORIGEM MÓVEL)
# ==============================================================================

def avaliar_grade_temporal(
    df_treino: pd.DataFrame,
    tipo_modelo: str,
    grade_params: List[Dict[str, Any]],
    feature_cols: List[str],
    nome_conjunto: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executa a validação temporal por origem móvel em duas dobras:
    - Dobra 2022: treina em mes_alvo < 2022-01, valida em 2022.
    - Dobra 2023: treina em mes_alvo < 2023-01, valida em 2023.
    Critério: maior Captura@15 média nas duas dobras.
    """
    logger.info(f"Iniciando busca em grade para {tipo_modelo.upper()} com Conjunto {nome_conjunto} ({len(grade_params)} combinações)...")

    # Dobra 2022
    df_tr_22 = df_treino[df_treino["mes_alvo"] < "2022-01"].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()

    # Dobra 2023
    df_tr_23 = df_treino[df_treino["mes_alvo"] < "2023-01"].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    resultados_grade = []
    melhor_score = -1.0
    melhor_param = grade_params[0]

    for params in grade_params:
        # 1. Dobra 2022
        preds_22 = treinar_e_prever(tipo_modelo, params, df_tr_22, df_tr_22["y"], df_val_22, feature_cols)
        df_val_22_eval = df_val_22[["cd_mun", "mes_alvo", "y", "pct_floresta"]].copy()
        df_val_22_eval["score_pred"] = preds_22

        caps_22 = []
        for m in sorted(df_val_22_eval["mes_alvo"].unique()):
            sub = df_val_22_eval[df_val_22_eval["mes_alvo"] == m]
            met, _ = calcular_metricas_mes(sub, col_score="score_pred", k_valores=(15,))
            caps_22.append(met["captura_15"])
        score_22 = float(np.mean(caps_22))

        # 2. Dobra 2023
        preds_23 = treinar_e_prever(tipo_modelo, params, df_tr_23, df_tr_23["y"], df_val_23, feature_cols)
        df_val_23_eval = df_val_23[["cd_mun", "mes_alvo", "y", "pct_floresta"]].copy()
        df_val_23_eval["score_pred"] = preds_23

        caps_23 = []
        for m in sorted(df_val_23_eval["mes_alvo"].unique()):
            sub = df_val_23_eval[df_val_23_eval["mes_alvo"] == m]
            met, _ = calcular_metricas_mes(sub, col_score="score_pred", k_valores=(15,))
            caps_23.append(met["captura_15"])
        score_23 = float(np.mean(caps_23))

        score_medio = (score_22 + score_23) / 2.0

        reg = {
            "modelo": tipo_modelo,
            "conjunto_features": nome_conjunto,
            "hiperparametros": json.dumps(params),
            "cap15_dobra_2022": score_22,
            "cap15_dobra_2023": score_23,
            "cap15_media": score_medio
        }
        resultados_grade.append(reg)

        if score_medio > melhor_score:
            melhor_score = score_medio
            melhor_param = params

    logger.info(f"-> Melhor configuração {tipo_modelo.upper()} ({nome_conjunto}): Captura@15 média = {melhor_score*100:.2f}% | Params: {melhor_param}")
    return melhor_param, resultados_grade

# ==============================================================================
# 3. TESTES ESTATÍSTICOS (WILCOXON E BOOTSTRAP)
# ==============================================================================

def calcular_testes_estatisticos(
    df_mensal: pd.DataFrame,
    modelos: List[str],
    baselines_ref: List[str] = ["climatologia_focos", "persistencia"]
) -> pd.DataFrame:
    """
    Realiza o teste de Wilcoxon pareado e calcula o intervalo de confiança de 95%
    por bootstrap (10.000 reamostragens) para a diferença na Captura@15 mensal.
    """
    logger.info("Executando testes pareados de Wilcoxon e Bootstrap (10.000 amostras, semente 42)...")
    rng = np.random.RandomState(42)
    n_boot = 10000

    linhas_testes = []
    for mod in modelos:
        sub_mod = df_mensal[df_mensal["metodo"] == mod].sort_values("mes_alvo")
        c_mod = sub_mod["captura_15"].values

        for b_ref in baselines_ref:
            sub_base = df_mensal[df_mensal["metodo"] == b_ref].sort_values("mes_alvo")
            c_base = sub_base["captura_15"].values

            diff = c_mod - c_base
            diff_media = float(np.mean(diff))

            # Wilcoxon signed-rank test
            # Se todas as diferenças forem idênticas a zero, p-valor é 1.0
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    res_wilc = wilcoxon(c_mod, c_base, alternative="two-sided")
                    p_val = float(res_wilc.pvalue)
                except Exception as e:
                    logger.warning(f"Erro no Wilcoxon entre {mod} e {b_ref}: {e}")
                    p_val = 1.0

            # Bootstrap 95% CI
            boot_idx = rng.choice(len(diff), size=(n_boot, len(diff)), replace=True)
            boot_means = diff[boot_idx].mean(axis=1)
            ic_inf = float(np.percentile(boot_means, 2.5))
            ic_sup = float(np.percentile(boot_means, 97.5))

            linhas_testes.append({
                "modelo": mod,
                "comparado_com": b_ref,
                "diferenca_media": diff_media,
                "ic_95_inf": ic_inf,
                "ic_95_sup": ic_sup,
                "p_valor": p_val
            })

    return pd.DataFrame(linhas_testes)

# ==============================================================================
# 4. PIPELINE PRINCIPAL DO MÓDULO 03
# ==============================================================================

def executar_modulo_perigo():
    logger.info("=== INICIANDO MÓDULO 03: MODELAGEM PREDITIVA E PERIGO (APHIM) ===")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    resultados_dir = processed_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem
    mod_path = processed_dir / "aphim_modelagem.parquet"
    if not mod_path.exists():
        raise FileNotFoundError(f"Base de modelagem '{mod_path}' não encontrada.")
    df_mod = pd.read_parquet(mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    # Separação cronológica
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()

    # Validação estrita de isolamento temporal
    assert (df_treino["mes_alvo"] <= "2023-12").all(), "ERRO CRÍTICO: Vazamento! Treino contém meses alvo posteriores a 2023-12."
    assert (df_teste["mes_alvo"] >= "2024-01").all(), "ERRO CRÍTICO: Teste contém meses alvo anteriores a 2024-01."
    assert len(df_teste) == 3456, f"ERRO: Conjunto de teste deve ter exatamente 3.456 linhas, obtido {len(df_teste)}."
    logger.info(f"Isolamento temporal estrito verificado com sucesso: Treino ({len(df_treino)} linhas), Teste ({len(df_teste)} linhas).")

    # Obter linha âncora 2023-12 (mes_emissao = 2023-11) para cálculo contínuo de rotatividade
    df_ancora_2312 = df_mod[df_mod["mes_alvo"] == "2023-12"].copy()

    # 2. Definição das grades de hiperparâmetros
    grade_glm = [{"alpha": a} for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]]

    grade_rf = [
        {
            "n_estimators": 500,
            "max_depth": md,
            "min_samples_leaf": msl,
            "max_features": mf
        }
        for md, msl, mf in itertools.product(
            [None, 10, 20],
            [1, 5, 20],
            ["sqrt", 0.5]
        )
    ]

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

    # 3. Busca de hiperparâmetros por validação temporal nas dobras 2022 e 2023
    logger.info("Iniciando processo de validação temporal por origem móvel...")
    todos_resultados_validacao = []
    melhores_params = {}

    config_modelos = [
        ("glm", "A", COLS_A, grade_glm),
        ("glm", "C", COLS_C, grade_glm),
        ("rf", "A", COLS_A, grade_rf),
        ("rf", "C", COLS_C, grade_rf),
        ("xgb", "A", COLS_A, grade_xgb),
        ("xgb", "C", COLS_C, grade_xgb),
    ]

    for tipo_mod, conj_nome, f_cols, g_params in config_modelos:
        chave = f"{tipo_mod}_{conj_nome}"
        best_p, res_list = avaliar_grade_temporal(
            df_treino=df_treino,
            tipo_modelo=tipo_mod,
            grade_params=g_params,
            feature_cols=f_cols,
            nome_conjunto=conj_nome
        )
        melhores_params[chave] = best_p
        todos_resultados_validacao.extend(res_list)

    # Salvar resultados de validação e hiperparâmetros escolhidos
    df_validacao = pd.DataFrame(todos_resultados_validacao)
    out_val_csv = resultados_dir / "perigo_validacao.csv"
    df_validacao.to_csv(out_val_csv, index=False, encoding="utf-8")
    logger.info(f"Resultados de validação temporal salvos em '{out_val_csv}'.")

    out_params_json = resultados_dir / "perigo_hiperparametros.json"
    with open(out_params_json, "w", encoding="utf-8") as f:
        json.dump(melhores_params, f, indent=2)
    logger.info(f"Melhores hiperparâmetros salvos em '{out_params_json}'.")

    # 4. Retreinamento em TODO o conjunto de treino e predição no teste (executado UMA ÚNICA VEZ)
    logger.info("=== RETREINANDO MODELOS NO CONJUNTO DE TREINO COMPLETO (2020-2023) ===")
    df_predicoes_teste = df_teste[["cd_mun", "mes_emissao", "mes_alvo", "y"]].copy()

    # Previsões âncora em 2023-12 para rotatividade
    df_predicoes_ancora = df_ancora_2312[["cd_mun", "mes_emissao", "mes_alvo", "y"]].copy()

    modelos_treinados = {}

    for tipo_mod, conj_nome, f_cols, _ in config_modelos:
        chave = f"{tipo_mod}_{conj_nome}"
        col_pred = f"pred_{chave}"
        params = melhores_params[chave]
        logger.info(f"Ajustando modelo final {chave.upper()} com parâmetros ótimos...")

        # Treinar modelo final
        if tipo_mod == "glm":
            X_tr_s, X_te_s, scaler = preparar_features_glm(df_treino, df_teste, f_cols)
            reg = PoissonRegressor(alpha=params["alpha"], max_iter=1000)
            reg.fit(X_tr_s, df_treino["y"])
            preds_te = np.maximum(0.0, reg.predict(X_te_s))

            # Previsão na âncora
            _, X_anc_s, _ = preparar_features_glm(df_treino, df_ancora_2312, f_cols)
            preds_anc = np.maximum(0.0, reg.predict(X_anc_s))
            modelos_treinados[chave] = reg

        elif tipo_mod == "rf":
            y_tr_log = np.log1p(df_treino["y"])
            rf = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(df_treino[f_cols], y_tr_log)
            preds_te = np.maximum(0.0, np.expm1(rf.predict(df_teste[f_cols])))
            preds_anc = np.maximum(0.0, np.expm1(rf.predict(df_ancora_2312[f_cols])))
            modelos_treinados[chave] = rf

        elif tipo_mod == "xgb":
            xgb_reg = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                min_child_weight=params["min_child_weight"],
                objective="count:poisson",
                random_state=42,
                n_jobs=-1
            )
            xgb_reg.fit(df_treino[f_cols], df_treino["y"])
            preds_te = np.maximum(0.0, xgb_reg.predict(df_teste[f_cols]))
            preds_anc = np.maximum(0.0, xgb_reg.predict(df_ancora_2312[f_cols]))
            modelos_treinados[chave] = xgb_reg

        # Validações estritas das previsões
        assert len(preds_te) == 3456, f"Esperado 3456 previsões para {chave}, obtido {len(preds_te)}."
        assert np.all(preds_te >= 0.0), f"Previsões negativas detectadas em {chave}!"
        assert not np.isnan(preds_te).any(), f"Valores NaN detectados em {chave}!"

        df_predicoes_teste[col_pred] = preds_te
        df_predicoes_ancora[col_pred] = preds_anc

    out_preds_parquet = resultados_dir / "perigo_predicoes_teste.parquet"
    df_predicoes_teste.to_parquet(out_preds_parquet, index=False)
    logger.info(f"Previsões de teste exportadas com sucesso para '{out_preds_parquet}'.")

    # 5. Avaliação do Ranking no Teste e Métricas de Erro
    logger.info("=== AVALIANDO RANKINGS E MÉTRICAS DE PRIORIZAÇÃO NO TESTE (2024-2025) ===")
    # Mesclar com colunas estruturais para ponderação de floresta
    df_teste_eval = df_predicoes_teste.merge(
        df_teste[["cd_mun", "mes_alvo", "pct_floresta"]],
        on=["cd_mun", "mes_alvo"],
        how="left"
    )
    df_ancora_eval = df_predicoes_ancora.merge(
        df_ancora_2312[["cd_mun", "mes_alvo", "pct_floresta"]],
        on=["cd_mun", "mes_alvo"],
        how="left"
    )

    lista_modelos_chaves = [f"{m}_{c}" for m, c in itertools.product(["glm", "rf", "xgb"], ["A", "C"])]
    dfs_mensais_modelos = []

    for chave in lista_modelos_chaves:
        col_pred = f"pred_{chave}"
        df_m = avaliar_metodo_mensal(
            df_periodo=df_teste_eval,
            col_score=col_pred,
            nome_metodo=chave,
            df_ancora=df_ancora_eval,
            col_target="y",
            col_weight="pct_floresta"
        )
        dfs_mensais_modelos.append(df_m)

    df_mensal_modelos = pd.concat(dfs_mensais_modelos, ignore_index=True)

    # Carregar baselines existentes para unificação das tabelas
    base_mensal_path = resultados_dir / "baselines_mensal.csv"
    if not base_mensal_path.exists():
        raise FileNotFoundError(f"Arquivo '{base_mensal_path}' não encontrado.")
    df_mensal_baselines = pd.read_csv(base_mensal_path)
    df_mensal_baselines = df_mensal_baselines.rename(columns={"baseline": "metodo"})

    # Adicionar Captura*@15 para os modelos utilizando o oráculo dos baselines
    df_mensal_unificado = pd.concat([df_mensal_baselines, df_mensal_modelos], ignore_index=True)
    df_mensal_unificado = adicionar_captura_star_15(df_mensal_unificado, col_metodo="metodo", nome_oraculo="oraculo")

    colunas_ordem_mensal = [
        "metodo", "mes_alvo",
        "captura_10", "captura_15", "captura_20", "captura_25", "captura_30",
        "captura_star_15", "captura_pond_15",
        "spearman", "rotatividade"
    ]
    df_mensal_unificado = df_mensal_unificado[colunas_ordem_mensal].sort_values(["metodo", "mes_alvo"]).reset_index(drop=True)

    out_perigo_mensal = resultados_dir / "perigo_mensal.csv"
    df_mensal_unificado.to_csv(out_perigo_mensal, index=False, encoding="utf-8")
    logger.info(f"Tabela mensal unificada salva em '{out_perigo_mensal}'.")

    # 6. Tabela Resumo (Médias de 24 meses + Estação Seca + MAE + Poisson Deviance)
    meses_alvo_teste = sorted(df_teste["mes_alvo"].unique())
    meses_seca = [m for m in meses_alvo_teste if m[5:7] in ["07", "08", "09", "10", "11"]]
    ordem_apresentacao = [
        "oraculo", "climatologia_focos", "persistencia", "historico_fixo", "rf_inpe",
        "xgb_C", "xgb_A", "rf_C", "rf_A", "glm_C", "glm_A"
    ]

    df_resumo = gerar_tabela_resumo(
        df_mensal=df_mensal_unificado,
        col_metodo="metodo",
        meses_seca=meses_seca,
        ordem_metodos=ordem_apresentacao
    )

    # Calcular MAE e Desvio de Poisson médio no conjunto de teste
    y_real_teste = df_predicoes_teste["y"].values
    mae_dict = {}
    pdev_dict = {}

    for chave in lista_modelos_chaves:
        col_pred = f"pred_{chave}"
        y_pred = df_predicoes_teste[col_pred].values
        mae_val = float(mean_absolute_error(y_real_teste, y_pred))
        # Poisson deviance requer previsões estritamente positivas (> 0)
        pdev_val = float(mean_poisson_deviance(y_real_teste, np.clip(y_pred, 1e-6, None)))
        mae_dict[chave] = mae_val
        pdev_dict[chave] = pdev_val

    # Calcular também para os baselines onde a contagem é diretamente expressa
    mae_dict["oraculo"] = 0.0
    pdev_dict["oraculo"] = 0.0

    # Persistência prevê focos_t
    df_teste_pers = df_teste.copy()
    mae_dict["persistencia"] = float(mean_absolute_error(y_real_teste, df_teste_pers["focos_t"].values))
    pdev_dict["persistencia"] = float(mean_poisson_deviance(y_real_teste, np.clip(df_teste_pers["focos_t"].values, 1e-6, None)))

    # Climatologia prevê focos_media_hist_mes
    mae_dict["climatologia_focos"] = float(mean_absolute_error(y_real_teste, df_teste_pers["focos_media_hist_mes"].values))
    pdev_dict["climatologia_focos"] = float(mean_poisson_deviance(y_real_teste, np.clip(df_teste_pers["focos_media_hist_mes"].values, 1e-6, None)))

    # Histórico fixo e rf_inpe não produzem contagem mensal compatível direta
    mae_dict["historico_fixo"] = np.nan
    pdev_dict["historico_fixo"] = np.nan
    mae_dict["rf_inpe"] = np.nan
    pdev_dict["rf_inpe"] = np.nan

    df_resumo["mae"] = df_resumo["metodo"].map(mae_dict)
    df_resumo["poisson_deviance"] = df_resumo["metodo"].map(pdev_dict)

    out_perigo_resumo = resultados_dir / "perigo_resumo.csv"
    df_resumo.to_csv(out_perigo_resumo, index=False, encoding="utf-8")
    logger.info(f"Tabela resumo unificada salva em '{out_perigo_resumo}'.")

    # 7. Testes Estatísticos Pareados
    df_testes_stat = calcular_testes_estatisticos(
        df_mensal=df_mensal_unificado,
        modelos=lista_modelos_chaves,
        baselines_ref=["climatologia_focos", "persistencia"]
    )
    out_testes_csv = resultados_dir / "perigo_testes_estatisticos.csv"
    df_testes_stat.to_csv(out_testes_csv, index=False, encoding="utf-8")
    logger.info(f"Testes estatísticos exportados para '{out_testes_csv}'.")

    # 8. Interpretação Global via SHAP (XGBoost A e C)
    logger.info("=== GERANDO ANÁLISE SHAP PARA OS MODELOS XGBOOST (TESTE) ===")
    explainer_A = shap.TreeExplainer(modelos_treinados["xgb_A"])
    shap_vals_A = explainer_A(df_teste[COLS_A])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_A, df_teste[COLS_A], show=False, max_display=12)
    plt.title("SHAP Summary Plot - XGBoost Conjunto A (Perigo Puro)", fontsize=13, pad=15)
    plt.tight_layout()
    out_shap_A = resultados_dir / "shap_xgb_A.png"
    plt.savefig(out_shap_A, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico SHAP do XGBoost Conjunto A salvo em '{out_shap_A}'.")

    explainer_C = shap.TreeExplainer(modelos_treinados["xgb_C"])
    shap_vals_C = explainer_C(df_teste[COLS_C])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_C, df_teste[COLS_C], show=False, max_display=12)
    plt.title("SHAP Summary Plot - XGBoost Conjunto C (Híbrido C)", fontsize=13, pad=15)
    plt.tight_layout()
    out_shap_C = resultados_dir / "shap_xgb_C.png"
    plt.savefig(out_shap_C, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico SHAP do XGBoost Conjunto C salvo em '{out_shap_C}'.")

    # Identificar top 10 atributos de cada modelo XGBoost por média do valor absoluto SHAP
    shap_importancia_A = pd.Series(
        np.abs(shap_vals_A.values).mean(axis=0),
        index=COLS_A
    ).sort_values(ascending=False)

    shap_importancia_C = pd.Series(
        np.abs(shap_vals_C.values).mean(axis=0),
        index=COLS_C
    ).sort_values(ascending=False)

    # 9. Exibição dos Relatórios no Terminal
    imprimir_relatorio_final(df_resumo, df_testes_stat, melhores_params, df_validacao, shap_importancia_A, shap_importancia_C)
    logger.info("Módulo 03 concluído com sucesso!")
    return df_resumo, df_testes_stat

def imprimir_relatorio_final(
    df_resumo: pd.DataFrame,
    df_testes: pd.DataFrame,
    melhores_params: Dict[str, Any],
    df_val: pd.DataFrame,
    shap_A: pd.Series,
    shap_C: pd.Series
):
    """Exibe os resultados finais estruturados no terminal."""
    print("\n" + "=" * 115)
    print("                TABELA COMPARATIVA FINAL: BASELINES E MODELOS DE PERIGO (24 MESES DE TESTE)")
    print("=" * 115)
    header = f"{'Método':<20} | {'Cap@15':<8} | {'Cap*@15':<9} | {'CapPond@15':<10} | {'Spearman':<8} | {'Rotativ.':<8} | {'Cap@15Seca':<10} | {'MAE':<7} | {'Dev.Poisson':<11}"
    print(header)
    print("-" * 115)
    for _, r in df_resumo.iterrows():
        m_name = r["metodo"]
        c15 = f"{r['captura_15']*100:.2f}%"
        c_star = f"{r['captura_star_15']*100:.2f}%"
        c_pond = f"{r['captura_pond_15']*100:.2f}%"
        sp = f"{r['spearman']:.4f}"
        rot = f"{r['rotatividade']*100:.2f}%"
        c_seca = f"{r['captura_15_seca']*100:.2f}%"
        mae_str = f"{r['mae']:.2f}" if pd.notnull(r["mae"]) else "-"
        pdev_str = f"{r['poisson_deviance']:.2f}" if pd.notnull(r["poisson_deviance"]) else "-"
        print(f"{m_name:<20} | {c15:<8} | {c_star:<9} | {c_pond:<10} | {sp:<8} | {rot:<8} | {c_seca:<10} | {mae_str:<7} | {pdev_str:<11}")
    print("=" * 115)

    print("\n" + "=" * 105)
    print("             TESTES ESTATÍSTICOS PAREADOS (WILCOXON & BOOTSTRAP 95% NA CAPTURA@15)")
    print("=" * 105)
    header_testes = f"{'Modelo':<10} | {'Referência':<20} | {'Dif. Média':<12} | {'IC 95% Bootstrap':<25} | {'p-valor':<10} | {'Significância'}"
    print(header_testes)
    print("-" * 105)
    for _, r in df_testes.iterrows():
        mod = r["modelo"]
        comp = r["comparado_com"]
        dm = f"{r['diferenca_media']*100:+.2f} p.p."
        ic_str = f"[{r['ic_95_inf']*100:+.2f}%, {r['ic_95_sup']*100:+.2f}%]"
        pval = f"{r['p_valor']:.4f}"
        sig = "p < 0.05 (*)" if r['p_valor'] < 0.05 else "não sig."
        print(f"{mod:<10} | {comp:<20} | {dm:<12} | {ic_str:<25} | {pval:<10} | {sig}")
    print("=" * 105)

    print("\n" + "=" * 90)
    print("             HIPERPARÂMETROS SELECIONADOS NA VALIDAÇÃO TEMPORAL")
    print("=" * 90)
    for k, v in melhores_params.items():
        sub_v = df_val[(df_val["modelo"] == k.split("_")[0]) & (df_val["conjunto_features"] == k.split("_")[1])]
        best_row = sub_v.sort_values("cap15_media", ascending=False).iloc[0]
        nota_val = best_row["cap15_media"] * 100
        nota_22 = best_row["cap15_dobra_2022"] * 100
        nota_23 = best_row["cap15_dobra_2023"] * 100
        print(f"• {k.upper():<8}: Nota Validação = {nota_val:.2f}% (2022: {nota_22:.2f}%, 2023: {nota_23:.2f}%)")
        print(f"  Hiperparâmetros: {v}\n")
    print("=" * 90)

    print("\n" + "=" * 60)
    print("       TOP 10 ATRIBUTOS MAIS IMPORTANTES (SHAP)")
    print("=" * 60)
    print("--- XGBOOST CONJUNTO A (PERIGO PURO) ---")
    for i, (feat, val) in enumerate(shap_A.head(10).items(), 1):
        print(f"  {i:2d}. {feat:<25} (SHAP médio: {val:.4f})")
    print("\n--- XGBOOST CONJUNTO C (HÍBRIDO C) ---")
    for i, (feat, val) in enumerate(shap_C.head(10).items(), 1):
        print(f"  {i:2d}. {feat:<25} (SHAP médio: {val:.4f})")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    executar_modulo_perigo()
