import os
import sys
import yaml
import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para importações relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_01_ibge")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def validar_municipios(gdf: gpd.GeoDataFrame) -> bool:
    qtd_mun = len(gdf)
    area_total = gdf["area_km2"].sum()
    
    logger.info(f"Validação: {qtd_mun} municípios encontrados.")
    logger.info(f"Validação: Área total calculada = {area_total:.2f} km².")
    
    # Validação 1: 144 municípios
    if qtd_mun != 144:
        logger.error(f"FALHA NA VALIDAÇÃO: Esperado 144 municípios, obtidos {qtd_mun}.")
        return False
        
    # Validação 2: Soma das áreas ≈ 1.245.870 km² (tolerância ±1%)
    area_esperada = 1245870.0
    tol_min = area_esperada * 0.99
    tol_max = area_esperada * 1.01
    
    if not (tol_min <= area_total <= tol_max):
        logger.error(f"FALHA NA VALIDAÇÃO: Área total {area_total:.2f} km² fora do limite [{tol_min:.2f}, {tol_max:.2f}].")
        return False
        
    logger.info("VALIDAÇÃO CONCLUÍDA COM SUCESSO: Malha municipal validada!")
    return True

def baixar_malha_ibge():
    config = carregar_config()
    uf = config.get("uf", 15)
    epsg_proj = config.get("projecao_area", 5880)
    out_dir = Path(config["caminhos"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filepath = out_dir / "municipios_pa.gpkg"
    
    # Idempotência
    if out_filepath.exists():
        logger.info(f"Arquivo de saída já existe em '{out_filepath}'. Validando...")
        gdf_existente = gpd.read_file(out_filepath)
        if validar_municipios(gdf_existente):
            logger.info("Execução pulada: arquivo existente é válido.")
            return gdf_existente

    logger.info("Baixando nomes dos municípios via API de Localidades do IBGE...")
    url_nomes = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    res_nomes = requests.get(url_nomes, timeout=30)
    res_nomes.raise_for_status()
    nomes_data = res_nomes.json()
    
    # Mapeamento cd_mun (str, 7 dígitos) -> nm_mun
    mapa_nomes = {str(item["id"]): item["nome"] for item in nomes_data}
    logger.info(f"{len(mapa_nomes)} nomes de municípios carregados.")

    logger.info("Baixando malha geoespacial via API de Malhas do IBGE (GeoJSON)...")
    url_malha = f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}?intrarregiao=municipio&formato=application/vnd.geo+json"
    res_malha = requests.get(url_malha, timeout=60)
    res_malha.raise_for_status()
    
    gdf = gpd.read_file(res_malha.text)
    logger.info(f"Malha geoespacial baixada: {len(gdf)} feições.")
    
    # Ajuste e tratamento de colunas
    if "codarea" in gdf.columns:
        gdf["cd_mun"] = gdf["codarea"].astype(str)
    elif "id" in gdf.columns:
        gdf["cd_mun"] = gdf["id"].astype(str)
    else:
        raise KeyError(f"Coluna de identificador não encontrada em: {gdf.columns}")
        
    gdf["nm_mun"] = gdf["cd_mun"].map(mapa_nomes)
    
    # Cálculo da área em km² em projeção de área igual
    logger.info(f"Calculando áreas em km² utilizando EPSG:{epsg_proj}...")
    gdf_projected = gdf.to_crs(epsg=epsg_proj)
    gdf["area_km2"] = (gdf_projected.geometry.area / 1e6).round(4)
    
    # Seleção e ordenação de colunas
    gdf = gdf[["cd_mun", "nm_mun", "area_km2", "geometry"]]
    
    # Validação
    if not validar_municipios(gdf):
        raise ValueError("A malha baixada não passou nos critérios de validação.")
        
    # Salvar GeoPackage
    logger.info(f"Salvando GeoPackage em '{out_filepath}'...")
    gdf.to_file(out_filepath, driver="GPKG")
    logger.info("Script 01 (IBGE) concluído com sucesso!")
    return gdf

if __name__ == "__main__":
    baixar_malha_ibge()
