"""
Módulo 15 — Validação e Teste Único da Recorrência no Nível 1 (Município) (Projeto APHIM)
Dissertação de Mestrado: Predição e Priorização Mensal de Municípios do Pará para Ação contra Queimadas.

Objetivos:
1. Construir as 5 variáveis de recorrência no nível do município:
   - "Teve foco" = focos_count > mediana do município no mês de calendário calculada com anos anteriores.
   - Histórico de 2016-2018 provido por data/processed_2015/aphim_base_municipio_mes.parquet.
   - 5 variáveis:
     1. meses_desde_ultimo_foco (120 se nunca teve foco acima da mediana desde 2016)
     2. anos_com_fogo_3a (0 a 3)
     3. frac_meses_fogo_24m
     4. focos_janela_retorno (t-12 a t-10)
     5. taxa_retorno_emp (com isolamento estrito: _d2022, _d2023, _final)
2. Comparar na Validação Temporal (dobras 2022 e 2023):
   - D0: 20 atributos atuais do modelo vencedor
   - D2: 20 atributos atuais + 5 atributos de recorrência do município (25 atributos)
   - Treino: alvos a partir de 2020-02 (como hoje)
   - Critério: maior Captura@15 média nas dobras 2022 e 2023.
3. Checagem de Sanidade:
   - D0 deve reproduzir 56,41% de validação (|diff| <= 0,10 p.p. / 0.0010).
4. Regra de Escolha do Vencedor:
   - D2 só substitui D0 se Captura@15 média for pelo menos 0,5 ponto maior (+0.0050).
5. Importância por Permutação:
   - Importância por permutação das features de D2 na validação.
6. Avaliação no Teste (executada estritamente uma única vez se D2 vencer):
   - Retreina D2 no treino completo (2020-02 a 2023-12) e prevê o teste (2024-01 a 2025-12).
   - Comparação pareada contra vencedor atual (features_v2_predicoes_teste.parquet), climatologia e persistência.
   - Wilcoxon pareado, Bootstrap 95% (10.000 reamostragens, semente 42) e correção de Holm.
   - Se D2 não vencer: não executa o teste.

Saídas em data/processed_recorrencia/resultados/:
- recorrencia_validacao_municipio.csv
- recorrencia_permutacao_municipio.csv
- recorrencia_predicoes_teste_municipio.parquet (se D2 vencer)
- recorrencia_teste_resumo_municipio.csv (se D2 vencer)
"""

import os
import sys
import json
import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger
from src.utils.metricas_priorizacao import calcular_metricas_mes, avaliar_metodo_mensal

logger = setup_logger("processamento_15_recorrencia_municipio")

FAIXAS_NOMES = ["0", "1", "2", "3-5", "6-9", "10-12", "13-24", "25+", "nunca"]

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

def construir_atributos_recorrencia_municipio(
    df_2015_path: Path
) -> pd.DataFrame:
    """
    Constrói a matriz de focos e calcula as 5 variáveis de recorrência municipais.
    'Teve foco' = focos_count > mediana do município no mês de calendário calculada com anos anteriores.
    """
    logger.info("Carregando histórico municipal estendido (2015-2025)...")
    df_2015 = pd.read_parquet(df_2015_path, columns=["cd_mun", "mes", "ano", "mes_num", "focos_count"])
    df_2015["cd_mun"] = df_2015["cd_mun"].astype(str).str.zfill(7)
    all_muns = sorted(df_2015["cd_mun"].unique())

    meses_todos = [f"{ano:04d}-{m:02d}" for ano in range(2016, 2026) for m in range(1, 13)]
    mes_to_idx = {m: i for i, m in enumerate(meses_todos)}
    focos_dict = df_2015.set_index(["cd_mun", "mes"])["focos_count"].to_dict()

    # Mediana histórica do mês de calendário usando apenas anos anteriores (ano < y)
    logger.info("Calculando mediana histórica de focos por município e mês de calendário (anos < y)...")
    med_dict = {}
    for (c, m_num), grp in df_2015.groupby(["cd_mun", "mes_num"]):
        grp = grp.sort_values("ano")
        anos = grp["ano"].values
        counts = grp["focos_count"].values
        for y in range(2016, 2026):
            mask = anos < y
            med_dict[(c, m_num, y)] = float(np.median(counts[mask]))

    # Matrizes A_mun (binária: acima da mediana) e F_mun (contagem real)
    A_mun = np.zeros((len(all_muns), len(meses_todos)), dtype=np.int8)
    F_mun = np.zeros((len(all_muns), len(meses_todos)), dtype=np.float32)

    for c_idx, c in enumerate(all_muns):
        for m_idx, m_str in enumerate(meses_todos):
            y = int(m_str[:4])
            mn = int(m_str[5:7])
            f = focos_dict.get((c, m_str), 0)
            F_mun[c_idx, m_idx] = f
            if f > med_dict[(c, mn, y)]:
                A_mun[c_idx, m_idx] = 1

    # Variável 1: meses_desde_ultimo_foco (acima da mediana)
    last_fire = np.full(len(all_muns), -999, dtype=np.int32)
    meses_desde_mun = np.zeros_like(A_mun, dtype=np.int32)
    for m in range(len(meses_todos)):
        has_fire = (A_mun[:, m] == 1)
        last_fire[has_fire] = m
        diff = m - last_fire
        diff[last_fire == -999] = 120
        meses_desde_mun[:, m] = diff

    # Taxa de retorno empírica por recorte
    def calc_taxa_retorno_mun(idx_fim_alvo: int):
        idx_inicio_t = mes_to_idx["2016-12"]
        idx_fim_t = idx_fim_alvo - 1
        counts_total = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}
        counts_focos = {f: {m: 0 for m in range(1, 13)} for f in FAIXAS_NOMES}

        for t in range(idx_inicio_t, idx_fim_t + 1):
            alvo_idx = t + 1
            alvo_cal_m = int(meses_todos[alvo_idx][5:7])
            fx = get_faixa_idx(meses_desde_mun[:, t])
            y_alvo = A_mun[:, alvo_idx]

            for f in FAIXAS_NOMES:
                mask_f = (fx == f)
                n = int(mask_f.sum())
                if n > 0:
                    k = int(y_alvo[mask_f].sum())
                    counts_total[f][alvo_cal_m] += n
                    counts_focos[f][alvo_cal_m] += k

        rates = {f: {} for f in FAIXAS_NOMES}
        for f in FAIXAS_NOMES:
            for m in range(1, 13):
                n = counts_total[f][m]
                k = counts_focos[f][m]
                rates[f][m] = (k + 1) / (n + 2)
        return rates

    rates_mun_d2022 = calc_taxa_retorno_mun(mes_to_idx["2021-12"])
    rates_mun_d2023 = calc_taxa_retorno_mun(mes_to_idx["2022-12"])
    rates_mun_final = calc_taxa_retorno_mun(mes_to_idx["2023-12"])

    # Montar base de atributos para mes_emissao de 2019-01 a 2025-11
    meses_emissao = [f"{ano:04d}-{m:02d}" for ano in range(2019, 2026) for m in range(1, 13)][:-1]
    records = []

    for t_str in meses_emissao:
        t = mes_to_idx[t_str]
        alvo_cal_m = int(meses_todos[t + 1][5:7])
        fx = get_faixa_idx(meses_desde_mun[:, t])

        # anos_com_fogo_3a
        w1 = (A_mun[:, t-11 : t+1].sum(axis=1) > 0).astype(int)
        w2 = (A_mun[:, t-23 : t-11].sum(axis=1) > 0).astype(int)
        w3 = (A_mun[:, t-35 : t-23].sum(axis=1) > 0).astype(int)
        anos_3a = w1 + w2 + w3

        # frac_meses_fogo_24m
        frac_24m = A_mun[:, t-23 : t+1].sum(axis=1) / 24.0

        # focos_janela_retorno (t-12 a t-10)
        focos_ret = F_mun[:, t-12 : t-9].sum(axis=1)

        for c_idx, c in enumerate(all_muns):
            f_name = fx[c_idx]
            records.append({
                "cd_mun": c,
                "mes_emissao": t_str,
                "meses_desde_ultimo_foco": int(meses_desde_mun[c_idx, t]),
                "anos_com_fogo_3a": int(anos_3a[c_idx]),
                "frac_meses_fogo_24m": float(frac_24m[c_idx]),
                "focos_janela_retorno": float(focos_ret[c_idx]),
                "taxa_retorno_emp_d2022": float(rates_mun_d2022[f_name][alvo_cal_m]),
                "taxa_retorno_emp_d2023": float(rates_mun_d2023[f_name][alvo_cal_m]),
                "taxa_retorno_emp_final": float(rates_mun_final[f_name][alvo_cal_m]),
            })

    df_rec_mun = pd.DataFrame(records)
    logger.info(f"Atributos de recorrência municipais gerados: {len(df_rec_mun):,} linhas.")
    return df_rec_mun

def buscar_hiperparametros_municipio(
    config_nome: str,
    feats: List[str],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    grade: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float, float, float]:
    """Busca os melhores hiperparâmetros nas 24 combinações para o nível municipal."""
    logger.info(f"Iniciando busca em grade ({len(grade)} combinações) para {config_nome}...")
    melhor_params = None
    melhor_media = -1.0
    melhor_c22 = -1.0
    melhor_c23 = -1.0

    for p in grade:
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

def calcular_permutacao_municipio(
    feats: List[str],
    params: Dict[str, Any],
    df_tr_22: pd.DataFrame,
    df_val_22: pd.DataFrame,
    df_tr_23: pd.DataFrame,
    df_val_23: pd.DataFrame,
    score_baseline: float
) -> pd.DataFrame:
    """Calcula a importância por permutação na validação para D2."""
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

def avaliar_teste_d2(
    feats_d2: List[str],
    params: Dict[str, Any],
    df_mod: pd.DataFrame,
    processed_dir: Path,
    out_dir: Path
):
    """
    Treina o modelo D2 em todo o conjunto de treino (2020-02 a 2023-12)
    e prevê o conjunto de teste (2024-01 a 2025-12). Executado estritamente se D2 vencer.
    """
    logger.info("=== RETREINANDO MODELO D2 NO TREINO COMPLETO E PREVENDO NO TESTE ===")
    df_treino = df_mod[(df_mod["conjunto"] == "treino") & (df_mod["mes_alvo"] >= "2020-02")].copy()
    df_teste = df_mod[df_mod["conjunto"] == "teste"].copy()

    df_treino["taxa_retorno_emp"] = df_treino["taxa_retorno_emp_final"]
    df_teste["taxa_retorno_emp"] = df_teste["taxa_retorno_emp_final"]

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
    m_final.fit(df_treino[feats_d2], df_treino["y"])
    preds_teste = np.maximum(0.0, m_final.predict(df_teste[feats_d2]))

    df_pred_out = df_teste[["cd_mun", "mes_emissao", "mes_alvo", "y"]].copy()
    df_pred_out["pred_vencedor"] = preds_teste

    out_parquet = out_dir / "resultados" / "recorrencia_predicoes_teste_municipio.parquet"
    df_pred_out.to_parquet(out_parquet, index=False)
    logger.info(f"Predições de teste salvas em '{out_parquet}'.")

    # Avaliação mensal do D2
    df_m_d2 = avaliar_metodo_mensal(
        df_periodo=df_pred_out,
        col_score="pred_vencedor",
        nome_metodo="D2_xgb_recorrencia",
        df_ancora=None,
        col_target="y"
    )

    # Carregar predições do vencedor atual (features_v2_predicoes_teste.parquet)
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

    # Baselines
    base_path = processed_dir / "resultados" / "baselines_mensal.csv"
    df_base = pd.read_csv(base_path)
    df_clim = df_base[df_base["baseline"] == "climatologia_focos"].sort_values("mes_alvo")
    df_pers = df_base[df_base["baseline"] == "persistencia"].sort_values("mes_alvo")

    caps_d2 = df_m_d2.sort_values("mes_alvo")["captura_15"].values
    caps_venc = df_m_venc.sort_values("mes_alvo")["captura_15"].values
    caps_clim = df_clim["captura_15"].values
    caps_pers = df_pers["captura_15"].values

    assert len(caps_d2) == 24, "Esperado exatamente 24 meses de teste."

    def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
        rng = np.random.RandomState(seed)
        diffs = a - b
        mean_diff = float(np.mean(diffs))
        boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        stat, pval = wilcoxon(a, b)
        venceu = int(np.sum(diffs > 0))
        return mean_diff, ci_lower, ci_upper, float(pval), venceu

    diff_venc, ci_l_v, ci_u_v, p_venc, v_venc = bootstrap_diff(caps_d2, caps_venc)
    diff_clim, ci_l_c, ci_u_c, p_clim, v_clim = bootstrap_diff(caps_d2, caps_clim)
    diff_pers, ci_l_p, ci_u_p, p_pers, v_pers = bootstrap_diff(caps_d2, caps_pers)

    pvals_raw = np.array([p_venc, p_clim, p_pers])
    pvals_holm = holm_bonferroni(pvals_raw)

    resumo_teste = pd.DataFrame([
        {
            "comparador": "vencedor_v2",
            "cap15_d2": float(np.mean(caps_d2)),
            "cap15_comparador": float(np.mean(caps_venc)),
            "diff_media": diff_venc,
            "ic95_inferior": ci_l_v,
            "ic95_superior": ci_u_v,
            "p_wilcoxon": p_venc,
            "p_holm": pvals_holm[0],
            "meses_venceu": v_venc
        },
        {
            "comparador": "climatologia",
            "cap15_d2": float(np.mean(caps_d2)),
            "cap15_comparador": float(np.mean(caps_clim)),
            "diff_media": diff_clim,
            "ic95_inferior": ci_l_c,
            "ic95_superior": ci_u_c,
            "p_wilcoxon": p_clim,
            "p_holm": pvals_holm[1],
            "meses_venceu": v_clim
        },
        {
            "comparador": "persistencia",
            "cap15_d2": float(np.mean(caps_d2)),
            "cap15_comparador": float(np.mean(caps_pers)),
            "diff_media": diff_pers,
            "ic95_inferior": ci_l_p,
            "ic95_superior": ci_u_p,
            "p_wilcoxon": p_pers,
            "p_holm": pvals_holm[2],
            "meses_venceu": v_pers
        }
    ])

    out_csv = out_dir / "resultados" / "recorrencia_teste_resumo_municipio.csv"
    resumo_teste.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Resumo do benchmark de teste salvo em '{out_csv}'.")

    print("\n" + "=" * 95)
    print("           BENCHMARK DE TESTE DO MUNICÍPIO — MODELO D2 (2024–2025)")
    print("=" * 95)
    print(resumo_teste.to_string(index=False))
    print("=" * 95 + "\n")

def main():
    t_start = time.time()
    logger.info("=== INICIANDO SCRIPT 15: VALIDAÇÃO E TESTE DA RECORRÊNCIA NO MUNICÍPIO ===")

    processed_dir = Path("data/processed")
    processed_2015_dir = Path("data/processed_2015")
    processed_deter_dir = Path("data/processed_deter")
    out_dir = Path("data/processed_recorrencia")
    resultados_dir = out_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar base de modelagem municipal atual
    mod_path = processed_deter_dir / "aphim_modelagem_v2_deter.parquet"
    if not mod_path.exists():
        raise FileNotFoundError(f"Arquivo '{mod_path}' não encontrado.")
    df_mod = pd.read_parquet(mod_path)
    df_mod["cd_mun"] = df_mod["cd_mun"].astype(str).str.zfill(7)

    # 2. Construir atributos de recorrência municipais
    base_2015_path = processed_2015_dir / "aphim_base_municipio_mes.parquet"
    if not base_2015_path.exists():
        raise FileNotFoundError(f"Arquivo '{base_2015_path}' não encontrado.")
    df_rec_mun = construir_atributos_recorrencia_municipio(base_2015_path)

    # Mesclar com a base de modelagem
    df_mod = df_mod.merge(df_rec_mun, on=["cd_mun", "mes_emissao"], how="left")

    # 3. Definir atributos de D0 e D2
    perm_path = processed_dir / "resultados" / "features_v2_permutacao.csv"
    if not perm_path.exists():
        raise FileNotFoundError(f"Arquivo '{perm_path}' não encontrado.")
    df_perm_ref = pd.read_csv(perm_path)
    feats_d0_set = set(df_perm_ref["atributo"].tolist())

    ordem_canonica = [
        "focos_t", "focos_t1", "focos_t2", "focos_media_hist_mes",
        "anom_prec_mm", "anom_t2m_media_c", "anom_umid_solo", "prec_3m", "anom_prec_3m",
        "prec_6m", "anom_prec_6m", "meses_secos_seguidos", "umid_solo_t1", "anom_umid_solo_3m",
        "razao_fogo_t", "variacao_anual_t",
        "focos_vizinhos_t", "razao_fogo_vizinhos_t",
        "mes_sin", "mes_cos"
    ]
    feats_d0 = [f for f in ordem_canonica if f in feats_d0_set]
    assert len(feats_d0) == 20, f"Esperado 20 atributos em D0, obtido {len(feats_d0)}"

    feats_rec_mun = ["meses_desde_ultimo_foco", "anos_com_fogo_3a", "frac_meses_fogo_24m", "focos_janela_retorno"]
    feats_d2 = feats_d0 + feats_rec_mun + ["taxa_retorno_emp"]
    assert len(feats_d2) == 25, f"Esperado 25 atributos em D2, obtido {len(feats_d2)}"

    # 4. Partições de Validação
    df_treino = df_mod[df_mod["conjunto"] == "treino"].copy()
    df_tr_22 = df_treino[(df_treino["mes_alvo"] >= "2020-02") & (df_treino["mes_alvo"] <= "2021-12")].copy()
    df_val_22 = df_treino[df_treino["mes_alvo"].str.startswith("2022-")].copy()
    df_tr_23 = df_treino[(df_treino["mes_alvo"] >= "2020-02") & (df_treino["mes_alvo"] <= "2022-12")].copy()
    df_val_23 = df_treino[df_treino["mes_alvo"].str.startswith("2023-")].copy()

    # Taxas empíricas específicas de cada dobra
    df_tr_22["taxa_retorno_emp"] = df_tr_22["taxa_retorno_emp_d2022"]
    df_val_22["taxa_retorno_emp"] = df_val_22["taxa_retorno_emp_d2022"]
    df_tr_23["taxa_retorno_emp"] = df_tr_23["taxa_retorno_emp_d2023"]
    df_val_23["taxa_retorno_emp"] = df_val_23["taxa_retorno_emp_d2023"]

    # 5. Avaliação e Checagem de Sanidade de D0 contra features_v2_validacao.csv (56,41%)
    val_ref_path = processed_dir / "resultados" / "features_v2_validacao.csv"
    df_val_ref = pd.read_csv(val_ref_path)
    score_esperado_d0 = float(df_val_ref[(df_val_ref["modelo"] == "xgb") & (df_val_ref["conjunto"] == "A_v2_sel")]["cap15_media"].iloc[0])

    hiper_path = processed_dir / "resultados" / "perigo_hiperparametros.json"
    with open(hiper_path, "r", encoding="utf-8") as f:
        p_d0 = json.load(f)["xgb_A"]

    logger.info("Avaliando D0 (20 atributos atuais) na validação...")
    m22_d0 = xgb.XGBRegressor(**p_d0, objective="count:poisson", random_state=42, n_jobs=-1)
    m22_d0.fit(df_tr_22[feats_d0], df_tr_22["y"])
    c22_d0 = avaliar_predicoes_val(df_val_22, np.maximum(0.0, m22_d0.predict(df_val_22[feats_d0])))

    m23_d0 = xgb.XGBRegressor(**p_d0, objective="count:poisson", random_state=42, n_jobs=-1)
    m23_d0.fit(df_tr_23[feats_d0], df_tr_23["y"])
    c23_d0 = avaliar_predicoes_val(df_val_23, np.maximum(0.0, m23_d0.predict(df_val_23[feats_d0])))

    med_d0 = (c22_d0 + c23_d0) / 2.0
    diff_sanidade_d0 = abs(med_d0 - score_esperado_d0)

    logger.info(
        f"Checagem de Sanidade D0: Reproduzido = {med_d0*100:.4f}% | "
        f"Esperado = {score_esperado_d0*100:.4f}% | Diferença = {diff_sanidade_d0*100:.4f} p.p."
    )
    if diff_sanidade_d0 > 0.0010: # > 0,10 ponto
        msg = f"FALHA CRÍTICA NA CHECAGEM DE SANIDADE DE D0: Diferença {diff_sanidade_d0*100:.4f} p.p. excede 0,10 p.p."
        logger.error(msg)
        raise ValueError(msg)
    logger.info("CHECAGEM DE SANIDADE DE D0 APROVADA: D0 reproduziu 56,41% perfeitamente!")

    # 6. Busca de Hiperparâmetros para D2 (25 atributos)
    grade_xgb = [
        {"n_estimators": ne, "learning_rate": lr, "max_depth": md, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": mcw}
        for ne, lr, md, mcw in itertools.product([300, 800], [0.03, 0.1], [3, 5, 7], [1, 10])
    ]

    p_d2, c22_d2, c23_d2, med_d2 = buscar_hiperparametros_municipio(
        "D2 (20 ATRIBUTOS + 5 RECORRENCIA MUN)", feats_d2,
        df_tr_22, df_val_22, df_tr_23, df_val_23, grade_xgb
    )

    # 7. Salvar Tabela de Validação
    df_val_mun = pd.DataFrame([
        {
            "configuracao": "D0",
            "n_atributos": len(feats_d0),
            "hiperparametros": json.dumps(p_d0),
            "cap15_dobra_2022": c22_d0,
            "cap15_dobra_2023": c23_d0,
            "cap15_media": med_d0
        },
        {
            "configuracao": "D2",
            "n_atributos": len(feats_d2),
            "hiperparametros": json.dumps(p_d2),
            "cap15_dobra_2022": c22_d2,
            "cap15_dobra_2023": c23_d2,
            "cap15_media": med_d2
        }
    ])
    out_val_csv = resultados_dir / "recorrencia_validacao_municipio.csv"
    df_val_mun.to_csv(out_val_csv, index=False, encoding="utf-8")
    logger.info(f"Tabela recorrencia_validacao_municipio.csv salva em '{out_val_csv}'.")

    # 8. Importância por Permutação de D2
    logger.info("Calculando importância por permutação na validação para D2...")
    df_perm_d2 = calcular_permutacao_municipio(
        feats_d2, p_d2,
        df_tr_22, df_val_22, df_tr_23, df_val_23,
        med_d2
    )
    out_perm_csv = resultados_dir / "recorrencia_permutacao_municipio.csv"
    df_perm_d2.to_csv(out_perm_csv, index=False, encoding="utf-8")
    logger.info(f"Importância por permutação do município salva em '{out_perm_csv}'.")

    print("\n" + "=" * 80)
    print("      TABELA DE VALIDAÇÃO TEMPORAL - NÍVEL 1 (MUNICÍPIO)")
    print("=" * 80)
    print(df_val_mun.to_string(index=False))

    print("\nIMPORTÂNCIA POR PERMUTAÇÃO D2 NO MUNICÍPIO (TOP 15):")
    print(df_perm_d2.head(15).to_string(index=False))

    # 9. Regra de Decisão do Vencedor (+0,5 p.p.)
    diff_d2_d0 = med_d2 - med_d0
    logger.info(f"\nDiferença D2 vs D0: {diff_d2_d0*100:+.3f} p.p. (limiar de vitória: +0.500 p.p.)")
    d2_venceu = diff_d2_d0 >= 0.0050

    if d2_venceu:
        logger.info(f"-> VENCEDOR: D2 VENCEU! Superou D0 em {diff_d2_d0*100:+.3f} p.p. (>= +0.500 p.p.).")
        print(f"\nDECISÃO: D2 VENCEU! (+{diff_d2_d0*100:.3f} p.p. >= +0.500 p.p.) -> Executando avaliação no teste.")
        avaliar_teste_d2(feats_d2, p_d2, df_mod, processed_dir, out_dir)
    else:
        logger.info(f"-> VENCEDOR: D0 CONTINUA! D2 não atingiu o ganho mínimo de +0.500 p.p. (ganho = {diff_d2_d0*100:+.3f} p.p.).")
        print(f"\nDECISÃO: D0 CONTINUA COMO VENCEDOR! (D2 obteve {diff_d2_d0*100:+.3f} p.p., abaixo do limiar de +0.500 p.p.).")
        print("Regra de teste respeitada: TESTE NÃO EXECUTADO para D2.")

    logger.info(f"Script 15 concluído em {time.time()-t_start:.1f} segundos!")

if __name__ == "__main__":
    main()
