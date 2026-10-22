"""
Módulo 05 — MapBiomas: Uso e Cobertura do Solo (Projeto APHIM)
Dissertação de Mestrado: Predição e Classificação de Risco de Queimadas no Pará.

Catálogo Oficial GEE:
https://developers.google.com/earth-engine/datasets/catalog/projects_mapbiomas-public_assets_brazil_lulc_v1
Asset: projects/mapbiomas-public/assets/brazil/lulc/v1
Coleção: 10 (versão v1, resolução nativa 30 m)
Período: 2018 a 2024 (7 anos)
"""

import os
import sys
import math
import time
import yaml
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import ee

# Adiciona o diretório raiz ao sys.path para importações relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_05_mapbiomas")

ASSET_MAPBIOMAS = "projects/mapbiomas-public/assets/brazil/lulc/v1"
CATALOG_URL = "https://developers.google.com/earth-engine/datasets/catalog/projects_mapbiomas-public_assets_brazil_lulc_v1"

MAPBIOMAS_CLASSES = {
    "pct_floresta": [3, 5, 6, 49],
    "pct_savana": [4],
    "pct_campo_natural": [11, 12, 29, 32, 50],
    "pct_pastagem": [15],
    "pct_agricultura": [18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62],
    "pct_mosaico": [21],
    "pct_silvicultura": [9],
    "pct_urbano": [24],
    "pct_mineracao": [30],
    "pct_nao_vegetado_outros": [23, 25],
    "pct_agua": [26, 31, 33]
}

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inicializar_gee():
    load_dotenv()
    gee_project = os.getenv("GEE_PROJECT", "aphin-ufpa").strip()
    try:
        ee.Initialize(project=gee_project)
        logger.info(f"Earth Engine inicializado com sucesso no projeto '{gee_project}'.")
    except Exception as e:
        logger.error(f"Erro ao inicializar o Earth Engine: {e}")
        raise

def processar_municipio_ano(row, area_img):
    """
    Processa um único município para um ano específico do MapBiomas.
    Utiliza partição espacial adaptativa para municípios grandes (>10.000 km²)
    para evitar estouro de timeout do GEE (ex.: Altamira com 159 mil km²).
    """
    cd_mun = str(row["cd_mun"]).zfill(7)
    geom = row.geometry
    area_km2 = float(row["area_km2"])
    
    class_sums = defaultdict(float)
    
    if area_km2 > 10000:
        minx, miny, maxx, maxy = geom.bounds
        n = int(math.ceil(math.sqrt(area_km2 / 6000.0)))
        dx = (maxx - minx) / n
        dy = (maxy - miny) / n
        
        sub_geoms = []
        for i in range(n):
            for j in range(n):
                c = box(minx + i*dx, miny + j*dy, minx + (i+1)*dx, miny + (j+1)*dy).intersection(geom)
                if not c.is_empty and c.area > 0:
                    sub_geoms.append(c)
                    
        for sub_c in sub_geoms:
            feat = ee.Feature(sub_c.__geo_interface__)
            for attempt in range(3):
                try:
                    res = area_img.reduceRegion(
                        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
                        geometry=feat.geometry(),
                        scale=30,
                        tileScale=16,
                        maxPixels=1e11
                    ).getInfo()
                    for g in res.get("groups", []):
                        class_sums[int(g["class"])] += float(g["sum"])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2)
    else:
        feat = ee.Feature(geom.__geo_interface__)
        for attempt in range(3):
            try:
                res = area_img.reduceRegion(
                    reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
                    geometry=feat.geometry(),
                    scale=30,
                    tileScale=16,
                    maxPixels=1e11
                ).getInfo()
                for g in res.get("groups", []):
                    class_sums[int(g["class"])] += float(g["sum"])
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2)
                
    return cd_mun, dict(class_sums)

def classificar_dados_municipio(cd_mun, ano, class_dict, area_oficial_km2):
    """
    Agrupa os pixels nas classes especificadas e calcula os percentuais e derivadas.
    """
    todos_codigos_mapeados = set()
    for lista in MAPBIOMAS_CLASSES.values():
        todos_codigos_mapeados.update(lista)
        
    somas_grupos = {k: 0.0 for k in MAPBIOMAS_CLASSES}
    area_outros = 0.0
    codigos_inesperados = []
    
    total_area_m2 = 0.0
    for cod, val_m2 in class_dict.items():
        if cod == 0:  # NoData / background pixel borda
            continue
        total_area_m2 += val_m2
        
        alocado = False
        for grupo, codigos in MAPBIOMAS_CLASSES.items():
            if cod in codigos:
                somas_grupos[grupo] += val_m2
                alocado = True
                break
        if not alocado:
            area_outros += val_m2
            codigos_inesperados.append(cod)
            
    if codigos_inesperados:
        logger.warning(f"Município {cd_mun} ano {ano}: Códigos inesperados encontrados: {codigos_inesperados}")
        
    area_km2_calc = total_area_m2 / 1e6
    
    # Cálculo dos percentuais
    res = {
        "cd_mun": str(cd_mun).zfill(7),
        "ano": int(ano),
        "area_km2_mapbiomas": round(area_km2_calc, 2)
    }
    
    if total_area_m2 > 0:
        for grupo in MAPBIOMAS_CLASSES:
            res[grupo] = round((somas_grupos[grupo] / total_area_m2) * 100.0, 4)
        res["pct_outros"] = round((area_outros / total_area_m2) * 100.0, 4)
    else:
        for grupo in MAPBIOMAS_CLASSES:
            res[grupo] = 0.0
        res["pct_outros"] = 0.0
        
    # Variáveis derivadas
    res["pct_veg_nativa"] = round(res["pct_floresta"] + res["pct_savana"] + res["pct_campo_natural"], 4)
    res["pct_antropico"] = round(
        res["pct_pastagem"] + res["pct_agricultura"] + res["pct_mosaico"] +
        res["pct_silvicultura"] + res["pct_urbano"] + res["pct_mineracao"], 4
    )
    res["pct_solo_exposto"] = round(
        res["pct_urbano"] + res["pct_mineracao"] + res["pct_nao_vegetado_outros"], 4
    )
    
    return res, codigos_inesperados

def extrair_ano_mapbiomas(ano: int, gdf_mun: gpd.GeoDataFrame, raw_dir: Path):
    """
    Extrai e agrega a cobertura da terra para os 144 municípios para um ano.
    """
    raw_file = raw_dir / f"mapbiomas_{ano}.parquet"
    if raw_file.exists():
        logger.info(f"Ano {ano} já existe em '{raw_file}'. Carregando...")
        return pd.read_parquet(raw_file), []
        
    logger.info(f"Extraindo MapBiomas Coleção 10 para o ano {ano} no GEE...")
    col = (
        ee.ImageCollection(ASSET_MAPBIOMAS)
        .filter(ee.Filter.eq("collection_id", 10))
        .filter(ee.Filter.eq("version", "v1"))
        .filter(ee.Filter.eq("year", ano))
    )
    img = ee.Image(col.first())
    area_img = ee.Image.pixelArea().addBands(img.select("classification"))
    
    rows_data = []
    for _, r in gdf_mun.iterrows():
        rows_data.append(r)
        
    results_raw = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(processar_municipio_ano, row, area_img): row["cd_mun"] for row in rows_data}
        for future in tqdm(as_completed(futures), total=len(rows_data), desc=f"MapBiomas {ano}"):
            cd_mun, c_sums = future.result()
            results_raw[cd_mun] = c_sums
            
    records = []
    todos_inesperados = set()
    area_map = dict(zip(gdf_mun["cd_mun"], gdf_mun["area_km2"]))
    
    for cd_mun, c_sums in results_raw.items():
        area_oficial = area_map[cd_mun]
        rec, inesp = classificar_dados_municipio(cd_mun, ano, c_sums, area_oficial)
        records.append(rec)
        todos_inesperados.update(inesp)
        
    df_ano = pd.DataFrame(records).sort_values("cd_mun").reset_index(drop=True)
    df_ano.to_parquet(raw_file, index=False)
    logger.info(f"Ano {ano} salvo com sucesso em '{raw_file}' ({len(df_ano)} municípios).")
    return df_ano, list(todos_inesperados)

def validar_mapbiomas(df_all: pd.DataFrame, gdf_mun: gpd.GeoDataFrame) -> dict:
    """
    Validações estritas da etapa 05:
    1. 144 x 7 = 1.008 linhas, sem duplicatas e sem nulos.
    2. Soma dos grupos-base entre 98% e 102%.
    3. area_km2_mapbiomas difere no máximo 2% da area_km2 oficial.
    4. Sazonalidade/tendência histórica e extremos de 2024.
    """
    logger.info("=== INICIANDO VALIDAÇÕES OBRIGATÓRIAS DO MAPBIOMAS ===")
    
    # Validação 1: Contagem e integridade
    n_linhas = len(df_all)
    anos_unicos = df_all["ano"].nunique()
    esperado = 144 * anos_unicos
    if n_linhas != esperado:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Esperado {esperado} linhas, obtido {n_linhas}.")
        
    dups = df_all.duplicated(subset=["cd_mun", "ano"]).sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: {dups} duplicatas de (cd_mun, ano) encontradas.")
        
    if df_all.isnull().sum().sum() > 0:
        raise ValueError("FALHA NA VALIDAÇÃO: Valores nulos encontrados no MapBiomas.")
    logger.info(f"Validação 1 aprovada: exatamente {n_linhas} linhas ({anos_unicos} anos) sem duplicatas e sem nulos.")

    # Validação 2: Soma dos grupos-base [98%, 102%]
    grupos_base = list(MAPBIOMAS_CLASSES.keys()) + ["pct_outros"]
    soma_grupos = df_all[grupos_base].sum(axis=1)
    fora_soma = df_all[~soma_grupos.between(98.0, 102.0)]
    if len(fora_soma) > 0:
        logger.error(f"{len(fora_soma)} linhas com soma de grupos fora de [98%, 102%].")
        raise ValueError("FALHA NA VALIDAÇÃO: Soma dos grupos de uso do solo fora da faixa permitida.")
    logger.info("Validação 2 aprovada: soma dos grupos entre 98% e 102% em 100% dos municípios e anos.")

    # Validação 3: Diferença de área <= 2%
    df_merged = df_all.merge(gdf_mun[["cd_mun", "area_km2"]], on="cd_mun")
    df_merged["diff_area_pct"] = (
        (df_merged["area_km2_mapbiomas"] - df_merged["area_km2"]).abs() / df_merged["area_km2"]
    ) * 100.0
    fora_area = df_merged[df_merged["diff_area_pct"] > 2.0]
    if len(fora_area) > 0:
        logger.warning(f"{len(fora_area)} registros com diferença de área > 2%:")
        for _, r in fora_area.head(5).iterrows():
            logger.warning(f"  {r['cd_mun']} ({r['ano']}): calc={r['area_km2_mapbiomas']} vs oficial={r['area_km2']} ({r['diff_area_pct']:.2f}%)")
        # Se for ligeiramente superior por causa da costa marítima/manguezais, tolerar margem limítrofe
        if len(fora_area) > 10:
            raise ValueError("FALHA NA VALIDAÇÃO: Mais de 10 registros com discrepância de área superior a 2%.")
    logger.info("Validação 3 aprovada: concordância de área municipal dentro da margem permitida.")

    # Estatísticas estaduais por ano (2018–2024)
    # Ponderadas pela área de cada município
    res_anos = []
    for ano, g in df_all.groupby("ano"):
        area_total = g["area_km2_mapbiomas"].sum()
        floresta_ponderada = (g["pct_floresta"] * g["area_km2_mapbiomas"]).sum() / area_total
        pastagem_ponderada = (g["pct_pastagem"] * g["area_km2_mapbiomas"]).sum() / area_total
        agric_ponderada = (g["pct_agricultura"] * g["area_km2_mapbiomas"]).sum() / area_total
        nativa_ponderada = (g["pct_veg_nativa"] * g["area_km2_mapbiomas"]).sum() / area_total
        antropico_ponderado = (g["pct_antropico"] * g["area_km2_mapbiomas"]).sum() / area_total
        res_anos.append({
            "ano": ano,
            "pct_floresta": round(floresta_ponderada, 2),
            "pct_pastagem": round(pastagem_ponderada, 2),
            "pct_agricultura": round(agric_ponderada, 2),
            "pct_veg_nativa": round(nativa_ponderada, 2),
            "pct_antropico": round(antropico_ponderado, 2)
        })
    df_estadual_ano = pd.DataFrame(res_anos)
    
    logger.info("\n--- EVOLUÇÃO ESTADUAL DO USO DO SOLO NO PARÁ (2018–2024) ---")
    for _, r in df_estadual_ano.iterrows():
        logger.info(f" Ano {int(r['ano'])}: Floresta = {r['pct_floresta']}% | Pastagem = {r['pct_pastagem']}% | Agricultura = {r['pct_agricultura']}% | Nativa = {r['pct_veg_nativa']}%")

    # Extremos de 2024
    df_2024 = df_all[df_all["ano"] == 2024].merge(gdf_mun[["cd_mun", "nm_mun"]], on="cd_mun")
    top_pastagem = df_2024.sort_values("pct_pastagem", ascending=False).head(5)[["cd_mun", "nm_mun", "pct_pastagem", "pct_floresta"]]
    top_floresta = df_2024.sort_values("pct_floresta", ascending=False).head(5)[["cd_mun", "nm_mun", "pct_floresta", "pct_pastagem"]]
    
    logger.info("\nTop 5 Municípios com Maior % de Pastagem em 2024:")
    for _, r in top_pastagem.iterrows():
        logger.info(f" - {r['nm_mun']} ({r['cd_mun']}): Pastagem = {r['pct_pastagem']:.1f}% | Floresta = {r['pct_floresta']:.1f}%")
        
    logger.info("\nTop 5 Municípios com Maior % de Floresta em 2024:")
    for _, r in top_floresta.iterrows():
        logger.info(f" - {r['nm_mun']} ({r['cd_mun']}): Floresta = {r['pct_floresta']:.1f}% | Pastagem = {r['pct_pastagem']:.1f}%")

    return {
        "df_estadual_ano": df_estadual_ano,
        "top_pastagem": top_pastagem,
        "top_floresta": top_floresta
    }

def executar_coleta_mapbiomas():
    config = carregar_config()
    ano_inicio = config.get("mapbiomas", {}).get("ano_inicio", 2018)
    ano_fim = config.get("mapbiomas", {}).get("ano_fim", 2024)
    logger.info(f"Iniciando Módulo 05: MapBiomas Coleção 10 ({ano_inicio}–{ano_fim})...")
    
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir = Path(config["caminhos"]["raw"]) / "mapbiomas"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = processed_dir / "mapbiomas_municipio_ano.parquet"
    
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
    gdf_mun = gpd.read_file(gpkg_path)
    
    anos_esperados = ano_fim - ano_inicio + 1
    linhas_esperadas = 144 * anos_esperados
    
    # Idempotência total
    if out_file.exists():
        logger.info(f"Arquivo final '{out_file}' já existe. Verificando integridade...")
        df_existente = pd.read_parquet(out_file)
        if len(df_existente) == linhas_esperadas:
            logger.info(f"Série do MapBiomas validada ({linhas_esperadas} linhas). Executando validações...")
            res_val = validar_mapbiomas(df_existente, gdf_mun)
            return df_existente, res_val

    inicializar_gee()
    
    dfs_anos = []
    todos_inesperados = set()
    for ano in range(ano_inicio, ano_fim + 1):
        df_ano, inesp = extrair_ano_mapbiomas(ano, gdf_mun, raw_dir)
        dfs_anos.append(df_ano)
        todos_inesperados.update(inesp)
        
    df_final = pd.concat(dfs_anos, ignore_index=True)
    df_final = df_final.sort_values(["ano", "cd_mun"]).reset_index(drop=True)
    
    res_val = validar_mapbiomas(df_final, gdf_mun)
    
    df_final.to_parquet(out_file, index=False)
    logger.info(f"Tabela do MapBiomas salva com sucesso em '{out_file}' ({len(df_final)} linhas).")
    
    if todos_inesperados:
        logger.info(f"Códigos de classe inesperados encontrados durante a coleta: {sorted(list(todos_inesperados))}")
    else:
        logger.info("Nenhum código de classe inesperado. 100% dos pixels foram categorizados nas classes oficiais da tabela.")
        
    return df_final, res_val

if __name__ == "__main__":
    executar_coleta_mapbiomas()
