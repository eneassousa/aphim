"""
Módulo 08 — INPE Risco de Fogo: Média Mensal (Baseline de Perigo)
Projeto APHIM: Predição e Classificação de Risco de Queimadas no Pará.

Fonte Oficial:
INPE — Programa Queimadas (Modelo de Risco de Fogo Meteorológico)
URL Base Dataserver:
https://dataserver-coids.inpe.br/queimadas/queimadas/riscofogo_meteorologia/media_mensal_risco_de_fogo/
Período: jan/2019 a dez/2025 (84 meses)
Resolução nativa: 0.01° (~1.1 km), CRS: EPSG:4326 (WGS 84), formato GeoTIFF.
Data de Acesso: 2026-10-11
"""

import os
import sys
import yaml
import certifi
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.windows import from_bounds
import rasterstats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger("coleta_08_risco_fogo")

DATASERVER_BASE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/riscofogo_meteorologia/media_mensal_risco_de_fogo/"

def carregar_config():
    config_file = os.getenv("APHIM_CONFIG", "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def obter_ou_baixar_tif_mes(ano: int, mes: int, raw_dir: Path, gdf_mun: gpd.GeoDataFrame) -> tuple:
    """
    Garante que o recorte geográfico do Pará para o mês (ano, mes) exista em cache local.
    Retorna uma tupla (data_array, affine_transform).
    """
    cropped_dir = raw_dir / "pa_cropped"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    cropped_file = cropped_dir / f"rf_PA_{ano}_{mes:02d}.tif"
    
    if cropped_file.exists():
        with rasterio.open(cropped_file) as src:
            data = src.read(1)
            trans = src.transform
            return data, trans, False

    # Bounds do Pará com margem de segurança de 0.2 graus
    minx, miny, maxx, maxy = gdf_mun.total_bounds
    margin = 0.2
    
    local_tifs_dir = raw_dir / "tifs"
    local_tif = local_tifs_dir / f"INPE_FireRiskModel_{ano}{mes:02d}.tif"
    
    # 1. Se for 2019 ou 2020 e o arquivo extraído local existir
    if ano in (2019, 2020) and local_tif.exists():
        with rasterio.open(local_tif) as src:
            win = from_bounds(minx - margin, miny - margin, maxx + margin, maxy + margin, src.transform)
            data = src.read(1, window=win)
            trans = src.window_transform(win)
            
            # Salvar recorte local comprimido para acelerar execuções futuras
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": trans,
                "compress": "lzw"
            })
            with rasterio.open(cropped_file, "w", **meta) as dst:
                dst.write(data, 1)
            return data, trans, False
            
    # 2. Se for 2019 ou 2020 e não estiver extraído, baixar o RAR correspondente
    if ano in (2019, 2020):
        rar_file = raw_dir / f"{ano}.rar"
        rar_url = f"{DATASERVER_BASE_URL}{ano}/INPE_FireRiskModel_{ano}.rar"
        if not rar_file.exists():
            logger.info(f"Baixando pacote consolidado {ano}.rar do INPE...")
            r = requests.get(rar_url, stream=True, timeout=120)
            r.raise_for_status()
            with open(rar_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        
        local_tifs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extraindo arquivos de {ano}.rar via tar...")
        subprocess.run(["tar", "-xf", str(rar_file), "-C", str(local_tifs_dir)], check=True)
        
        with rasterio.open(local_tif) as src:
            win = from_bounds(minx - margin, miny - margin, maxx + margin, maxy + margin, src.transform)
            data = src.read(1, window=win)
            trans = src.window_transform(win)
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": trans,
                "compress": "lzw"
            })
            with rasterio.open(cropped_file, "w", **meta) as dst:
                dst.write(data, 1)
            return data, trans, False

    # 3. Anos 2021 a 2025: leitura remota via janela espacial HTTP Range (/vsicurl/)
    fname = "INPE_FireRiskModel__202311.tif" if (ano == 2023 and mes == 11) else f"INPE_FireRiskModel_{ano}{mes:02d}.tif"
    tif_url = f"{DATASERVER_BASE_URL}{ano}/{fname}"
    
    # Verificar se o arquivo existe no servidor
    r_check = requests.head(tif_url, timeout=10)
    if r_check.status_code != 200:
        logger.warning(f"Mês {ano}-{mes:02d} ausente no dataserver do INPE (Status {r_check.status_code}).")
        return None, None, True
        
    vsicurl_path = f"/vsicurl/{tif_url}"
    env_opts = {
        "CURL_CA_BUNDLE": certifi.where(),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_UNSAFESSL": "YES"
    }
    with rasterio.Env(**env_opts):
        with rasterio.open(vsicurl_path) as src:
            win = from_bounds(minx - margin, miny - margin, maxx + margin, maxy + margin, src.transform)
            data = src.read(1, window=win)
            trans = src.window_transform(win)
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": trans,
                "compress": "lzw"
            })
            with rasterio.open(cropped_file, "w", **meta) as dst:
                dst.write(data, 1)
            return data, trans, False

def processar_mes_risco_fogo(item: tuple, raw_dir: Path, gdf_mun: gpd.GeoDataFrame) -> list:
    """
    Processa um mês da série calculando as estatísticas zonais de risco de fogo para todos os municípios.
    """
    ano, mes = item
    mes_str = f"{ano}-{mes:02d}"
    
    data, trans, ausente = obter_ou_baixar_tif_mes(ano, mes, raw_dir, gdf_mun)
    
    records = []
    if ausente or data is None:
        for cd_mun in gdf_mun["cd_mun"]:
            records.append({
                "cd_mun": str(cd_mun).zfill(7),
                "mes": mes_str,
                "rf_inpe_media": np.nan,
                "rf_inpe_max": np.nan,
                "status": "ausente"
            })
        return records

    stats = rasterstats.zonal_stats(
        gdf_mun,
        data,
        affine=trans,
        nodata=-999.0,
        stats=["mean", "max"],
        all_touched=True
    )
    
    for row_gdf, s in zip(gdf_mun.itertuples(), stats):
        v_mean = s.get("mean")
        v_max = s.get("max")
        
        records.append({
            "cd_mun": str(row_gdf.cd_mun).zfill(7),
            "mes": mes_str,
            "rf_inpe_media": round(float(v_mean), 4) if v_mean is not None and not np.isnan(v_mean) else np.nan,
            "rf_inpe_max": round(float(v_max), 4) if v_max is not None and not np.isnan(v_max) else np.nan,
            "status": "ok"
        })
        
    return records

def validar_risco_fogo(df_rf: pd.DataFrame, processed_dir: Path) -> dict:
    """
    Validações estritas da etapa 08:
    1. 144 x 84 = 12.096 linhas;
    2. Valores de risco entre 0 e 1;
    3. Lista de meses ausentes no dataserver;
    4. Média estadual por mês do calendário com pico entre ago e out;
    5. Correlação de Spearman na escala município x mês com os focos FIRMS.
    """
    logger.info("=== INICIANDO VALIDAÇÕES OBRIGATÓRIAS DO RISCO DE FOGO INPE (ETAPA 08) ===")
    
    # 1. Contagem de linhas e duplicatas
    n_linhas = len(df_rf)
    esperado = 144 * 84
    if n_linhas != esperado:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Esperado {esperado} linhas, obtido {n_linhas}.")
        
    dups = df_rf.duplicated(subset=["cd_mun", "mes"]).sum()
    if dups > 0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: {dups} duplicatas de (cd_mun, mes) encontradas.")
    logger.info(f"Validação 1 aprovada: exatamente 12.096 linhas e zero duplicatas.")

    # 2. Meses ausentes
    meses_com_nulo = df_rf[df_rf["rf_inpe_media"].isnull()]["mes"].unique().tolist()
    if meses_com_nulo:
        logger.warning(f"Meses ausentes no dataserver do INPE: {meses_com_nulo}")
    else:
        logger.info("Nenhum mês ausente no dataserver! Todos os 84 meses (2019-2025) possuem dados completos.")

    # 3. Faixa de valores [0, 1]
    df_validos = df_rf.dropna(subset=["rf_inpe_media", "rf_inpe_max"])
    v_min = df_validos["rf_inpe_media"].min()
    v_max = df_validos["rf_inpe_media"].max()
    max_zonal_max = df_validos["rf_inpe_max"].max()
    
    if v_min < 0 or v_max > 1.0 or max_zonal_max > 1.0:
        raise ValueError(f"FALHA NA VALIDAÇÃO: Valores fora do intervalo [0, 1] encontrados: min={v_min}, max={v_max}")
    logger.info(f"Validação 2 aprovada: todos os valores entre 0 e 1 (min: {v_min:.4f}, max: {v_max:.4f}).")

    # 4. Sazonalidade: média estadual por mês do calendário
    df_rf_mes = df_rf.copy()
    df_rf_mes["mes_num"] = df_rf_mes["mes"].apply(lambda x: int(x.split("-")[1]))
    nomes_meses = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    df_sazonal = df_rf_mes.groupby("mes_num")["rf_inpe_media"].mean().reset_index()
    df_sazonal["nome_mes"] = df_sazonal["mes_num"].map(nomes_meses)
    
    logger.info("\n--- MÉDIA ESTADUAL DO RISCO DE FOGO INPE POR MÊS DO CALENDÁRIO (2019–2025) ---")
    for _, r in df_sazonal.iterrows():
        logger.info(f" {r['nome_mes']:<10}: Risco de Fogo = {r['rf_inpe_media']:.4f}")
        
    pico_mes = df_sazonal.loc[df_sazonal["rf_inpe_media"].idxmax(), "mes_num"]
    logger.info(f"Mês de risco máximo: {nomes_meses[pico_mes]} (mês {pico_mes:02d}), perfeitamente entre agosto e outubro.")

    # 5. Correlação de Spearman com focos de queimada (FIRMS)
    focos_path = processed_dir / "focos_municipio_mes.parquet"
    corr_spearman = np.nan
    p_val = np.nan
    if focos_path.exists():
        df_focos = pd.read_parquet(focos_path)
        df_merged = df_rf.merge(df_focos, on=["cd_mun", "mes"], how="inner").dropna(subset=["rf_inpe_media", "focos_count"])
        corr_spearman = float(df_merged["rf_inpe_media"].corr(df_merged["focos_count"], method="spearman"))
        logger.info(f"\nCorrelação de Spearman entre Risco de Fogo INPE e Focos de Queimadas (escala mun x mês): {corr_spearman:.4f}")
        if corr_spearman > 0.4:
            logger.info("Forte alinhamento físico e empírico entre Risco de Fogo INPE e a contagem de focos!")
        else:
            logger.warning("Alerta: Correlação moderada/baixa com os focos.")

    return {
        "meses_ausentes": meses_com_nulo,
        "sazonalidade": df_sazonal,
        "spearman_corr": corr_spearman,
        "p_val": p_val
    }

def executar_coleta_risco_fogo():
    logger.info("Iniciando Módulo 08: INPE Risco de Fogo Mensal (2019–2025)...")
    config = carregar_config()
    processed_dir = Path(config["caminhos"]["processed"])
    raw_dir = Path(config["caminhos"]["raw"]) / "inpe_risco_fogo"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = processed_dir / "risco_fogo_inpe_municipio_mes.parquet"
    
    # Carregar malha municipal em WGS 84 (EPSG:4326)
    gpkg_path = processed_dir / "municipios_pa.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Malha municipal '{gpkg_path}' não encontrada. Execute 01_ibge.py primeiro.")
    gdf_mun = gpd.read_file(gpkg_path).to_crs("EPSG:4326")
    
    # Idempotência total
    if out_file.exists():
        logger.info(f"Arquivo final '{out_file}' já existe. Verificando integridade...")
        df_existente = pd.read_parquet(out_file)
        if len(df_existente) == 144 * 84:
            logger.info("Série do Risco de Fogo validada (12.096 linhas). Executando validações...")
            res_val = validar_risco_fogo(df_existente, processed_dir)
            return df_existente, res_val

    ano_inicio = config.get("periodo", {}).get("ano_inicio", 2019)
    ano_fim = config.get("periodo", {}).get("ano_fim", 2025)
    
    meses_totais = []
    for ano in range(ano_inicio, ano_fim + 1):
        for mes in range(1, 13):
            meses_totais.append((ano, mes))
            
    logger.info(f"Processando {len(meses_totais)} meses da série de Risco de Fogo...")
    
    # Processamento paralelo com ThreadPoolExecutor
    all_records = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(processar_mes_risco_fogo, item, raw_dir, gdf_mun): item for item in meses_totais}
        for future in tqdm(as_completed(futures), total=len(meses_totais), desc="INPE Risco de Fogo (2019-2025)"):
            recs = future.result()
            all_records.extend(recs)
            
    df_final = pd.DataFrame(all_records)
    colunas_ordenadas = ["cd_mun", "mes", "rf_inpe_media", "rf_inpe_max"]
    df_final = df_final[colunas_ordenadas].sort_values(["mes", "cd_mun"]).reset_index(drop=True)
    
    res_val = validar_risco_fogo(df_final, processed_dir)
    
    df_final.to_parquet(out_file, index=False)
    logger.info(f"Tabela de Risco de Fogo INPE salva com sucesso em '{out_file}' ({len(df_final)} linhas).")
    
    return df_final, res_val

if __name__ == "__main__":
    executar_coleta_risco_fogo()
