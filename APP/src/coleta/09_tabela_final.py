"""
Módulo 09 — Montagem da Tabela Final Integrada (Projeto APHIM)
Dissertação de Mestrado: Predição e Classificação de Risco de Queimadas no Pará.

Consolida a grade analítica completa de 144 municípios x N meses conforme configuração.

Regras de Integração:
1. Malha e Identificadores: 144 municípios do Pará (IBGE 2024 / SIRGAS 2000).
2. Focos de Calor: NASA FIRMS (VIIRS S-NPP) por (cd_mun, mes).
3. Clima e Extremos: ERA5-Land (t0) por (cd_mun, mes) + prec_extremo.
4. Risco de Fogo INPE: Média e máximo zonal (t0) por (cd_mun, mes).
5. Uso do Solo MapBiomas: Defasagem t-1 (ano_mapbiomas = ano - 1).
6. Desmatamento PRODES: Defasagem t-1 (ano_prodes = ano - 1; ciclo encerra em 31/jul do ano AAAA-1).
7. Vulnerabilidade Social: Renda domiciliar per capita mediana (Censo 2022 SIDRA) por cd_mun.

Saídas:
- aphim_base_municipio_mes.parquet (no diretório processed da configuração)
- aphim_base_municipio_mes.csv
"""

import os
import sys
import yaml
from pathlib import Path
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_09_tabela_final")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def montar_tabela_final(processed_dir: Path) -> pd.DataFrame:
    """
    Executa a junção relacional determinística e com defasagens de todas as etapas de coleta.
    """
    logger.info("Carregando bases processadas de todas as etapas...")
    
    # 1. Malha oficial
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
    gdf_mun = gpd.read_file(gpkg_path)[["cd_mun", "nm_mun", "area_km2"]]
    gdf_mun["cd_mun"] = gdf_mun["cd_mun"].astype(str).str.zfill(7)
    
    # 2. Focos FIRMS
    focos_path = processed_dir / "focos_municipio_mes.parquet"
    if not focos_path.exists():
        raise FileNotFoundError(f"Base de focos '{focos_path}' não encontrada. Execute 02_firms.py primeiro.")
    df_focos = pd.read_parquet(focos_path)
    df_focos["cd_mun"] = df_focos["cd_mun"].astype(str).str.zfill(7)
    df_focos["mes"] = df_focos["mes"].astype(str)
    
    # 3. Clima ERA5-Land
    clima_path = processed_dir / "clima_municipio_mes.parquet"
    if not clima_path.exists():
        raise FileNotFoundError(f"Base de clima '{clima_path}' não encontrada. Execute 04_era5.py primeiro.")
    df_clima = pd.read_parquet(clima_path).drop(columns=["mes_calendario"], errors="ignore")
    df_clima["cd_mun"] = df_clima["cd_mun"].astype(str).str.zfill(7)
    df_clima["mes"] = df_clima["mes"].astype(str)
    
    # 4. INPE Risco de Fogo
    rf_path = processed_dir / "risco_fogo_inpe_municipio_mes.parquet"
    if not rf_path.exists():
        raise FileNotFoundError(f"Base de risco de fogo '{rf_path}' não encontrada. Execute 08_risco_fogo.py primeiro.")
    df_rf = pd.read_parquet(rf_path)
    df_rf["cd_mun"] = df_rf["cd_mun"].astype(str).str.zfill(7)
    df_rf["mes"] = df_rf["mes"].astype(str)
    
    # 5. MapBiomas (anual 2018 a 2024)
    mb_path = processed_dir / "mapbiomas_municipio_ano.parquet"
    if not mb_path.exists():
        raise FileNotFoundError(f"Base do MapBiomas '{mb_path}' não encontrada. Execute 05_mapbiomas.py primeiro.")
    df_mb = pd.read_parquet(mb_path)
    df_mb["cd_mun"] = df_mb["cd_mun"].astype(str).str.zfill(7)
    df_mb["ano"] = df_mb["ano"].astype(int)
    
    # 6. PRODES (anual 2018 a 2025)
    prodes_path = processed_dir / "prodes_municipio_ano.parquet"
    if not prodes_path.exists():
        raise FileNotFoundError(f"Base do PRODES '{prodes_path}' não encontrada. Execute 06_prodes.py primeiro.")
    df_prodes = pd.read_parquet(prodes_path).rename(columns={
        "desmat_km2": "desmat_km2_prodes",
        "desmat_pct_area": "desmat_pct_area_prodes"
    })
    df_prodes["cd_mun"] = df_prodes["cd_mun"].astype(str).str.zfill(7)
    df_prodes["ano"] = df_prodes["ano"].astype(int)
    
    # 7. Renda Censo 2022
    renda_path = processed_dir / "renda_municipio.parquet"
    if not renda_path.exists():
        raise FileNotFoundError(f"Base de renda '{renda_path}' não encontrada. Execute 07_renda.py primeiro.")
    df_renda = pd.read_parquet(renda_path)
    df_renda["cd_mun"] = df_renda["cd_mun"].astype(str).str.zfill(7)

    logger.info("Construindo esqueleto temporal e calculando defasagens...")
    # Base inicial da grade completa
    df_base = df_focos.copy()
    df_base["ano"] = df_base["mes"].apply(lambda x: int(x.split("-")[0]))
    df_base["mes_num"] = df_base["mes"].apply(lambda x: int(x.split("-")[1]))
    
    # Defasagens t-1
    df_base["ano_mapbiomas"] = df_base["ano"] - 1
    df_base["ano_prodes"] = df_base["ano"] - 1
    
    # Junções espaciais e derivadas de área
    logger.info("Executando junções de atributos espaciais, climáticos e socioeconômicos...")
    df_base = df_base.merge(gdf_mun, on="cd_mun", how="left")
    df_base["focos_por_km2"] = (df_base["focos_count"] / df_base["area_km2"]).round(6)
    
    # Junções climáticas e perigo meteorológico (t0)
    df_base = df_base.merge(df_clima, on=["cd_mun", "mes"], how="left")
    df_base = df_base.merge(df_rf, on=["cd_mun", "mes"], how="left")
    
    # Junção MapBiomas com defasagem t-1
    df_base = df_base.merge(
        df_mb,
        left_on=["cd_mun", "ano_mapbiomas"],
        right_on=["cd_mun", "ano"],
        how="left"
    ).drop(columns=["ano_y"]).rename(columns={"ano_x": "ano"})
    
    # Junção PRODES com defasagem t-1
    df_base = df_base.merge(
        df_prodes,
        left_on=["cd_mun", "ano_prodes"],
        right_on=["cd_mun", "ano"],
        how="left"
    ).drop(columns=["ano_y"]).rename(columns={"ano_x": "ano"})
    
    # Junção Renda Municipal
    df_base = df_base.merge(df_renda, on="cd_mun", how="left")
    
    # Ordenação canônica das colunas
    colunas_ordenadas = [
        # Identificação Espacial e Temporal
        "cd_mun", "nm_mun", "mes", "ano", "mes_num", "area_km2",
        # Focos de Calor FIRMS
        "focos_count", "focos_por_km2",
        # Variáveis Climáticas ERA5-Land (t0)
        "t2m_media_c", "t2m_max_c", "t2m_min_c", "td2m_c",
        "prec_mm", "prec_extremo", "lst_c", "umid_solo", "vento_ms",
        "ur_pct", "dpv_kpa",
        # Risco de Fogo INPE (t0)
        "rf_inpe_media", "rf_inpe_max",
        # Uso e Cobertura do Solo MapBiomas (t-1)
        "ano_mapbiomas", "area_km2_mapbiomas",
        "pct_floresta", "pct_savana", "pct_campo_natural",
        "pct_pastagem", "pct_agricultura", "pct_mosaico",
        "pct_silvicultura", "pct_urbano", "pct_mineracao",
        "pct_nao_vegetado_outros", "pct_agua", "pct_outros",
        "pct_veg_nativa", "pct_antropico", "pct_solo_exposto",
        # Desmatamento Anual PRODES (t-1)
        "ano_prodes", "desmat_km2_prodes", "desmat_pct_area_prodes",
        # Vulnerabilidade Social
        "renda_pc", "fonte_renda", "ano_ref_renda"
    ]
    
    df_final = df_base[colunas_ordenadas].sort_values(["mes", "cd_mun"]).reset_index(drop=True)
    return df_final

def validar_tabela_final(df: pd.DataFrame, processed_dir: Path, config: dict):
    """
    Validações estritas da etapa 09 (falhar com erro se não passarem):
    1. Linhas esperadas (144 municípios x N meses do período);
    2. Sem duplicatas de (cd_mun, mes);
    3. Teste automático de defasagem: ano_mapbiomas == ano - 1 e ano_prodes == ano - 1;
    4. Soma total de focos idêntica à base de focos da etapa 02;
    5. Relatório de nulos por coluna (apenas risco de fogo pode ter nulos).
    """
    logger.info("=== INICIANDO VALIDAÇÕES OBRIGATÓRIAS DA TABELA FINAL (ETAPA 09) ===")
    
    # 1. Contagem de linhas
    data_inicio = config["periodo"]["data_inicio"]
    data_fim = config["periodo"]["data_fim"]
    todos_meses = pd.date_range(start=data_inicio, end=data_fim, freq="MS")
    n_meses_esperados = len(todos_meses)
    esperado_linhas = 144 * n_meses_esperados
    
    n_linhas = len(df)
    if n_linhas != esperado_linhas:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Esperado {esperado_linhas} linhas, obtido {n_linhas}.")
    logger.info(f"Validação 1 aprovada: exatamente {n_linhas} linhas (144 municípios x {n_meses_esperados} meses).")
    
    # 2. Duplicatas
    dups = df.duplicated(subset=["cd_mun", "mes"]).sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: {dups} duplicatas de (cd_mun, mes) encontradas.")
    logger.info("Validação 2 aprovada: zero duplicatas de (cd_mun, mes).")
    
    # 3. Teste automático de defasagens t-1
    check_mb = (df["ano_mapbiomas"] == df["ano"] - 1).all()
    check_prodes = (df["ano_prodes"] == df["ano"] - 1).all()
    if not check_mb or not check_prodes:
        raise ValueError("FALHA NA VALIDAÇÃO: Inconsistência na defasagem temporal t-1 do MapBiomas ou PRODES!")
    logger.info("Validação 3 aprovada: 100% das linhas possuem defasagem t-1 estrita (ano_mapbiomas == ano-1 e ano_prodes == ano-1).")
    
    # 4. Soma total de focos FIRMS
    soma_focos = int(df["focos_count"].sum())
    focos_path = processed_dir / "focos_municipio_mes.parquet"
    if focos_path.exists():
        df_focos_check = pd.read_parquet(focos_path)
        esperado_focos = int(df_focos_check["focos_count"].sum())
        if soma_focos != esperado_focos:
            raise ValueError(f"FALHA NA VALIDAÇÃO: Soma de focos divergente! Esperado {esperado_focos}, obtido {soma_focos}.")
    logger.info(f"Validação 4 aprovada: soma total de focos idêntica à etapa 02 ({soma_focos:,} focos).")
    
    # Verificação adicional se contém o período histórico 2019-2025
    sub_19_25 = df[(df["mes"] >= "2019-01") & (df["mes"] <= "2025-12")]
    if len(sub_19_25) == 12096:
        soma_19_25 = int(sub_19_25["focos_count"].sum())
        logger.info(f"Subtotal 2019–2025: {soma_19_25:,} focos (esperado: 1.695.702).")
        if soma_19_25 != 1695702:
            raise ValueError(f"FALHA NA VALIDAÇÃO: Subtotal 2019-2025 divergente! Obtido {soma_19_25}, esperado 1.695.702.")
    
    # 5. Relatório de nulos
    nulos = df.isnull().sum()
    logger.info("\n--- RELATÓRIO DE NULOS POR COLUNA NA TABELA FINAL ---")
    colunas_com_nulo = []
    for col, count in nulos.items():
        if count > 0:
            logger.warning(f" - {col:<25}: {count} nulos ({count/len(df)*100:.2f}%)")
            colunas_com_nulo.append(col)
        else:
            logger.info(f" - {col:<25}: 0 nulos")
            
    # Somente colunas de risco de fogo podem ter nulos
    colunas_proibidas_nulos = [c for c in df.columns if c not in ["rf_inpe_media", "rf_inpe_max"]]
    nulos_proibidos = df[colunas_proibidas_nulos].isnull().sum().sum()
    if nulos_proibidos > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Encontrados {nulos_proibidos} nulos em colunas que não admitem ausência!")
    logger.info("Validação 5 aprovada: integridade nula 100% perfeita.")

def executar_montagem_tabela_final():
    logger.info("Iniciando Módulo 09: Montagem da Tabela Final Integrada APHIM...")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    
    out_parquet = processed_dir / "aphim_base_municipio_mes.parquet"
    out_csv = processed_dir / "aphim_base_municipio_mes.csv"
    
    df_final = montar_tabela_final(processed_dir)
    validar_tabela_final(df_final, processed_dir, config)
    
    logger.info(f"Exportando base final para Parquet: '{out_parquet}'...")
    df_final.to_parquet(out_parquet, index=False)
    
    logger.info(f"Exportando cópia em CSV: '{out_csv}'...")
    df_final.to_csv(out_csv, index=False, encoding="utf-8")
    
    logger.info(f"Tabela final gerada com sucesso! Dimensões: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas.")
    return df_final

if __name__ == "__main__":
    executar_montagem_tabela_final()
