import os
import sys
import io
import yaml
import zipfile
import unicodedata
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_03_auditoria")

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn').upper().strip()

def baixar_inpe_ano(ano, raw_dir):
    out_zip = raw_dir / f"focos_br_ref_{ano}.zip"
    url = f"https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/Brasil_sat_ref/focos_br_ref_{ano}.zip"
    
    if not out_zip.exists():
        logger.info(f"Baixando INPE BDQueimadas satélites de referência para {ano}...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_zip, "wb") as f:
            f.write(r.content)
            
    with zipfile.ZipFile(out_zip) as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"Nenhum arquivo CSV encontrado em {out_zip}")
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f, usecols=lambda c: c in ["data_pas", "estado", "municipio", "lat", "lon"])
            return df

def executar_auditoria():
    config = carregar_config()
    raw_dir = Path(config["caminhos"]["raw"]) / "inpe"
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    ano_inicio = config["periodo"]["ano_inicio"]
    ano_fim = config["periodo"]["ano_fim"]

    # Carregar malha dos municípios do Pará para junção espacial de alta precisão
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        logger.error(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
        sys.exit(1)
    gdf_mun = gpd.read_file(gpkg_path)
    gdf_mun_4326 = gdf_mun.to_crs(epsg=4326)
    
    logger.info(f"Coletando e filtrando focos INPE BDQueimadas (satélite de referência) {ano_inicio}-{ano_fim}...")
    dfs_inpe = []
    
    for ano in range(ano_inicio, ano_fim + 1):
        df_ano = baixar_inpe_ano(ano, raw_dir)
        
        # Filtro de alta precisão: Junção Espacial estrita com os 144 municípios do Pará
        gdf_focos = gpd.GeoDataFrame(
            df_ano,
            geometry=gpd.points_from_xy(df_ano["lon"], df_ano["lat"]),
            crs="EPSG:4326"
        )
        gdf_joined = gpd.sjoin(gdf_focos, gdf_mun_4326[["cd_mun", "nm_mun", "geometry"]], how="inner", predicate="within")
        dfs_inpe.append(gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore"))
        
    df_inpe_total = pd.concat(dfs_inpe, ignore_index=True)
    logger.info(f"Total de focos INPE estritamente no Pará (Junção Espacial): {len(df_inpe_total):,}")
    
    df_inpe_total["mes"] = df_inpe_total["data_pas"].astype(str).str.slice(0, 7)
    serie_inpe = df_inpe_total.groupby("mes").size().reset_index(name="focos_inpe")
    
    # Carregar série mensal do FIRMS
    firms_path = processed_dir / "focos_municipio_mes.parquet"
    if not firms_path.exists():
        logger.error(f"Arquivo FIRMS '{firms_path}' não encontrado. Execute 02_firms.py primeiro.")
        sys.exit(1)
        
    df_firms = pd.read_parquet(firms_path)
    serie_firms = df_firms.groupby("mes")["focos_count"].sum().reset_index().rename(columns={"focos_count": "focos_firms"})
    
    # Merge das Séries Mensais
    df_comp = pd.merge(serie_inpe, serie_firms, on="mes", how="outer").fillna(0)
    df_comp = df_comp.sort_values("mes").reset_index(drop=True)
    df_comp["focos_inpe"] = df_comp["focos_inpe"].astype(int)
    df_comp["focos_firms"] = df_comp["focos_firms"].astype(int)
    df_comp["ano"] = df_comp["mes"].str.slice(0, 4)
    df_comp["razao_firms_inpe"] = (df_comp["focos_firms"] / df_comp["focos_inpe"].replace(0, 1)).round(4)
    
    out_csv = processed_dir / "auditoria_firms_vs_inpe.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Tabela de auditoria mensal atualizada salva em '{out_csv}'")
    
    # Estatísticas
    pico_inpe_idx = df_comp["focos_inpe"].idxmax()
    pico_firms_idx = df_comp["focos_firms"].idxmax()
    
    pico_inpe_mes = df_comp.loc[pico_inpe_idx, "mes"]
    pico_inpe_val = df_comp.loc[pico_inpe_idx, "focos_inpe"]
    
    pico_firms_mes = df_comp.loc[pico_firms_idx, "mes"]
    pico_firms_val = df_comp.loc[pico_firms_idx, "focos_firms"]
    
    corr_pearson = df_comp["focos_firms"].corr(df_comp["focos_inpe"], method="pearson")
    corr_spearman = df_comp["focos_firms"].rank().corr(df_comp["focos_inpe"].rank())
    
    totais_anuais = df_comp.groupby("ano")[["focos_firms", "focos_inpe"]].sum().reset_index()
    totais_anuais["razao_firms_inpe"] = (totais_anuais["focos_firms"] / totais_anuais["focos_inpe"].replace(0, 1)).round(4)
    
    print("\n" + "=" * 65)
    print("  RELATÓRIO DE AUDITORIA CORRIGIDO: NASA FIRMS vs INPE BDQUEIMADAS")
    print("=" * 65)
    print(f"Período analisado: {df_comp['mes'].min()} a {df_comp['mes'].max()} ({len(df_comp)} meses)")
    print(f"Total de focos INPE (Sat. Referência Pará): {df_comp['focos_inpe'].sum():,}")
    print(f"Total de focos FIRMS (VIIRS S-NPP Pará):    {df_comp['focos_firms'].sum():,}")
    print("-" * 65)
    print(f"Correlação de Pearson  (mensal): {corr_pearson:.4f}")
    print(f"Correlação de Spearman (mensal): {corr_spearman:.4f}")
    print("-" * 65)
    print(f"Mês de pico INPE:  {pico_inpe_mes} ({pico_inpe_val:,} focos)")
    print(f"Mês de pico FIRMS: {pico_firms_mes} ({pico_firms_val:,} focos)")
    print("-" * 65)
    print("TOTAIS ANUAIS E RAZÃO CORRIGIDOS (FIRMS / INPE):")
    print(totais_anuais.to_string(index=False))
    print("=" * 65 + "\n")
    
    return df_comp, totais_anuais

if __name__ == "__main__":
    executar_auditoria()
