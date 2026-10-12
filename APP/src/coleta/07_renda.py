"""
Módulo 07 — Renda Municipal (Vulnerabilidade Social)
Projeto APHIM: Predição e Classificação de Risco de Queimadas no Pará.

Fonte Oficial:
IBGE — Censo Demográfico 2022 (Resultados da Amostra)
Tabela SIDRA 10295: "Moradores em domicílios particulares permanentes ocupados,
exclusive os cuja condição no domicílio era pensionista, empregado(a) doméstico(a)
ou parente do(a) empregado(a) doméstico(a), valor do rendimento domiciliar mensal
per capita, médio e mediano, por sexo, cor ou raça e grupos de idade"

Variável Utilizada:
- 13534: "Valor do rendimento nominal mediano mensal domiciliar per capita"
Classificações fixadas no Total:
- Sexo: Total (c2 = 6794)
- Cor ou raça: Total (c86 = 95251)
- Grupo de idade: Total (c58 = 95253)

Endpoint SIDRA:
https://apisidra.ibge.gov.br/values/t/10295/n6/in%20n3%2015/v/13534/p/2022/c2/6794/c86/95251/c58/95253
Data de Acesso: 2026-10-11
"""

import os
import sys
import json
import yaml
import requests
from pathlib import Path
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_07_renda")

SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/10295/n6/in%20n3%2015/v/13534/p/2022/c2/6794/c86/95251/c58/95253"
FONTE_RENDA = "IBGE_Censo_2022_SIDRA_10295_v13534"
ANO_REF_RENDA = 2022

def carregar_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def obter_renda_censo_2022(raw_dir: Path) -> pd.DataFrame:
    """
    Obtém o rendimento mediano per capita do Censo 2022 via API SIDRA/IBGE.
    Possui cache bruto em JSON para garantia de idempotência e reprodutibilidade.
    """
    raw_file = raw_dir / "renda_censo_2022_sidra_10295.json"
    
    if raw_file.exists():
        logger.info(f"Carregando dados brutos de renda do cache local: '{raw_file}'...")
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.info(f"Consultando API SIDRA/IBGE (Tabela 10295, V13534) em: {SIDRA_URL}...")
        r = requests.get(SIDRA_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Dados brutos do SIDRA salvos em '{raw_file}'.")

    # A primeira linha da resposta do SIDRA é o cabeçalho descritivo
    df_raw = pd.DataFrame(data[1:])
    df_res = pd.DataFrame()
    df_res["cd_mun"] = df_raw["D1C"].astype(str).str.zfill(7)
    df_res["nm_mun_ibge"] = df_raw["D1N"].astype(str)
    
    # Tratamento de valores numéricos
    # Substitui caracteres de ausência do IBGE ('-', '...', '..') por NaN
    valores_limpos = df_raw["V"].replace({"-": None, "...": None, "..": None})
    df_res["renda_pc"] = pd.to_numeric(valores_limpos, errors="coerce")
    df_res["fonte_renda"] = FONTE_RENDA
    df_res["ano_ref_renda"] = ANO_REF_RENDA
    
    return df_res

def validar_renda(df: pd.DataFrame, gdf_mun: gpd.GeoDataFrame) -> dict:
    """
    Validações estritas da etapa 07:
    1. Exatamente 144 municípios do Pará;
    2. Sem valores nulos ou duplicatas;
    3. Mínimo, mediana e máximo;
    4. Top 5 municípios mais ricos e mais pobres (esperado: Belém e mineração no topo; Marajó na base).
    """
    logger.info("=== INICIANDO VALIDAÇÕES OBRIGATÓRIAS DE RENDA (ETAPA 07) ===")
    
    # 1. Quantidade de municípios
    n_linhas = len(df)
    if n_linhas != 144:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Esperado 144 municípios, obtido {n_linhas}.")
        
    dups = df["cd_mun"].duplicated().sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: {dups} duplicatas de 'cd_mun' encontradas.")
        
    n_nulos = df["renda_pc"].isnull().sum()
    if n_nulos > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: {n_nulos} valores nulos de renda_pc encontrados.")
        
    # Verificar cobertura de todos os municípios da malha oficial
    set_malha = set(gdf_mun["cd_mun"].astype(str).str.zfill(7))
    set_renda = set(df["cd_mun"])
    if set_malha != set_renda:
        diff = set_malha.symmetric_difference(set_renda)
        raise ValueError(f"FALHA NA VALIDAÇÃO: Códigos municipais incompatíveis com a malha: {diff}")
        
    logger.info("Validação 1 aprovada: 144 municípios do Pará cobertos, sem nulos e sem duplicatas.")
    
    # Estatísticas descritivas
    v_min = float(df["renda_pc"].min())
    v_med = float(df["renda_pc"].median())
    v_max = float(df["renda_pc"].max())
    v_media = float(df["renda_pc"].mean())
    
    logger.info(f"\n--- ESTATÍSTICAS DE RENDIMENTO DOMICILIAR MEDIANO PER CAPITA (PA) ---")
    logger.info(f" - Mínimo  : R$ {v_min:.2f}")
    logger.info(f" - Mediana : R$ {v_med:.2f}")
    logger.info(f" - Média   : R$ {v_media:.2f}")
    logger.info(f" - Máximo  : R$ {v_max:.2f}")
    
    # Mesclar com nome dos municípios da malha
    df_merged = df.merge(gdf_mun[["cd_mun", "nm_mun"]], on="cd_mun")
    
    top_pobres = df_merged.sort_values("renda_pc", ascending=True).head(5)
    top_ricos = df_merged.sort_values("renda_pc", ascending=False).head(5)
    
    logger.info("\nTop 5 Municípios com Menor Rendimento Per Capita (Mais Vulneráveis):")
    for _, r in top_pobres.iterrows():
        logger.info(f" - {r['nm_mun']} ({r['cd_mun']}): R$ {r['renda_pc']:.2f}")
        
    logger.info("\nTop 5 Municípios com Maior Rendimento Per Capita:")
    for _, r in top_ricos.iterrows():
        logger.info(f" - {r['nm_mun']} ({r['cd_mun']}): R$ {r['renda_pc']:.2f}")
        
    return {
        "min": v_min,
        "mediana": v_med,
        "max": v_max,
        "top_pobres": top_pobres,
        "top_ricos": top_ricos
    }

def executar_coleta_renda():
    logger.info("Iniciando Módulo 07: Renda Municipal (Censo 2022 SIDRA)...")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir = Path(config["caminhos"]["raw"]) / "ibge"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = processed_dir / "renda_municipio.parquet"
    
    # Carregar malha municipal oficial
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
    gdf_mun = gpd.read_file(gpkg_path)
    
    # Idempotência total
    if out_file.exists():
        logger.info(f"Arquivo final '{out_file}' já existe. Verificando integridade...")
        df_existente = pd.read_parquet(out_file)
        if len(df_existente) == 144 and df_existente["renda_pc"].isnull().sum() == 0:
            logger.info("Tabela de renda existente validada (144 linhas). Executando validações...")
            res_val = validar_renda(df_existente, gdf_mun)
            return df_existente, res_val
            
    df_renda_raw = obter_renda_censo_2022(raw_dir)
    
    # Formatação estrita das colunas de saída
    colunas_finais = ["cd_mun", "renda_pc", "fonte_renda", "ano_ref_renda"]
    df_final = df_renda_raw[colunas_finais].sort_values("cd_mun").reset_index(drop=True)
    
    res_val = validar_renda(df_final, gdf_mun)
    
    df_final.to_parquet(out_file, index=False)
    logger.info(f"Tabela de renda salva com sucesso em '{out_file}' ({len(df_final)} linhas).")
    
    return df_final, res_val

if __name__ == "__main__":
    executar_coleta_renda()
