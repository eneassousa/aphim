Você vai implementar, em Python, o pipeline automático de coleta e montagem da base de dados do projeto **APHIM** (Arquitetura Preditiva Híbrida com Inteligência computacional e Modelagem multicritério). É a dissertação de mestrado em Ciência da Computação (UFPA) sobre predição e classificação de risco de focos de queimadas nos municípios do Pará. Trabalhe no repositório local `aphim`.

## Objetivo desta etapa
Gerar uma única tabela painel **município × mês**: 144 municípios do Pará × 84 meses (jan/2019 a dez/2025) = 12.096 linhas. Ela deve reunir os focos de queimadas, o clima, o uso do solo, o desmatamento, a renda e o Risco de Fogo do INPE. Ainda **não** treine modelos nem defina a variável alvo; apenas colete, limpe, agregue e valide.

## Regras gerais
- Python 3.11+. Bibliotecas: pandas, geopandas, shapely, requests, pyarrow, earthengine-api, rasterio, rasterstats, python-dotenv, tqdm.
- Estrutura: `config.yaml` (período, UF, caminhos, satélite, filtros), `.env.example` (FIRMS_MAP_KEY=, GEE_PROJECT=), `requirements.txt`, `src/coleta/` (um script por fonte: 01_ibge.py, 02_firms.py, ...), `src/processamento/` (agregações e montagem da tabela final), `data/raw/`, `data/interim/`, `data/processed/`, `logs/`.
- Credenciais **somente** por variável de ambiente (`.env`, que fica no `.gitignore`). Nunca escreva chaves no código.
- Cada script deve ser **idempotente e retomável**: se o arquivo de saída de uma etapa ou janela já existe, pule. Salve os brutos em `data/raw/<fonte>/`.
- Identificador municipal: código IBGE de **7 dígitos** (`cd_mun`, texto). Datas mensais no formato `AAAA-MM`.
- Variáveis **anuais** (MapBiomas, PRODES) entram na tabela mensal com **defasagem de 1 ano** (o mês de 2024-09 recebe o valor de 2023), para evitar vazamento de informação. Por isso, colete essas fontes a partir de **2018**.
- Registre em log o que foi baixado, quantas linhas entraram e quantas foram descartadas em cada filtro.
- Não invente identificadores de tabelas ou de assets. Se não tiver certeza do ID de uma tabela SIDRA ou de um asset do Earth Engine, consulte a documentação oficial e registre no código a fonte usada.

## Fontes e scripts (nesta ordem)

### 01 — IBGE: malha municipal do Pará (UF 15)
- Malha via API de malhas do IBGE (v3, estado 15, intrarregião município, GeoJSON) e nomes via API de localidades.
- Calcule a área em km² numa projeção de área igual (por exemplo, EPSG:5880).
- Saída: `data/processed/municipios_pa.gpkg` com `cd_mun`, `nm_mun`, `area_km2`, `geometry`.
- Validação: exatamente **144** municípios; soma das áreas ≈ 1.245.870 km² (tolerância de ±1%).

### 02 — NASA FIRMS: focos de calor (alvo do modelo)
- Fonte: **VIIRS S-NPP apenas** (`VIIRS_SNPP_SP`; use `VIIRS_SNPP_NRT` só para datas ainda não disponíveis no SP). **Não** misture NOAA-20/NOAA-21, para manter a série homogênea.
- API area: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{W,S,E,N}/{DAY_RANGE}/{AAAA-MM-DD}`. O **DAY_RANGE máximo é 5 dias**, e o limite é de 5000 transações a cada 10 minutos. Percorra **todo** o período 2019-01-01 a 2025-12-31 em janelas consecutivas de 5 dias, sem lacunas, com controle de taxa e novas tentativas. Use o bbox do Pará com margem.
- **Atenção:** a versão anterior desta coleta obteve apenas 9.765 focos válidos em 7 anos, enquanto o INPE registra cerca de 200 mil focos no Pará só de 2020 a 2024 (com outro satélite). A causa provável é a coleta não ter coberto o período inteiro. Garanta a cobertura completa e prove isso na validação.
- Filtros: (1) junção espacial com a malha, mantendo só os pontos dentro do Pará; (2) `type == 0` (vegetação presumida), se a coluna existir; (3) descartar `confidence == 'l'`.
- Saídas: `data/interim/firms_focos_pa.parquet` (pontos filtrados com `cd_mun`) e `data/processed/focos_municipio_mes.parquet` (contagem por `cd_mun` × mês, com **zeros explícitos** nos meses sem foco).
- Validação: nenhuma janela de 5 dias faltando; tabela de totais por ano e por mês; registro de quantos pontos cada filtro removeu.

### 03 — Auditoria com INPE BDQueimadas (satélite de referência)
- Baixe os focos anuais do satélite de referência (dataserver do INPE, `queimadas/focos/csv/anual/`) para 2019–2025 e filtre o estado do Pará.
- Agregue por mês e compare com a série mensal do FIRMS: correlação de Pearson e de Spearman, meses de pico e razão FIRMS/INPE por ano. Os valores absolutos **não** precisam coincidir (são sensores diferentes); a **tendência** precisa ser coerente.
- Saída: `data/processed/auditoria_firms_vs_inpe.csv` e um resumo impresso.

### 04 — ERA5-Land: clima (via Google Earth Engine)
- Use o Earth Engine (`GEE_PROJECT` no `.env`). Coleção mensal agregada do ERA5-Land; para cada município e mês, calcule a média espacial (reduceRegions) de: temperatura do ar a 2 m (°C), média e máxima mensal; precipitação total mensal (mm); umidade relativa (%), derivada da temperatura e do ponto de orvalho (fórmula de Magnus); temperatura da superfície (skin temperature, °C), como LST; umidade do solo da camada 1 (volumétrica); velocidade do vento a 10 m (a partir de u e v).
- Saída: `data/processed/clima_municipio_mes.parquet`.
- Opcional, num script separado: FMA (Fórmula de Monte Alegre) diária a partir do ERA5-Land horário (umidade relativa às 13h locais = 16 UTC), agregada por mês. Deixe isso para o final.

### 05 — MapBiomas: uso e cobertura do solo (anual, 2018–2025)
- Pelo Earth Engine, use a coleção pública mais recente do MapBiomas Brasil (confirme o asset na documentação do MapBiomas).
- Por município e ano, calcule a % da área em: formação florestal, pastagem, agricultura e área não vegetada. Documente no código quais códigos de classe entram em cada grupo.
- Saída: `data/processed/mapbiomas_municipio_ano.parquet`.

### 06 — PRODES: desmatamento anual (2018–2025)
- Fonte: PRODES Amazônia Legal (TerraBrasilis/INPE), incremento anual de desmatamento **por município**, em km².
- Saída: `data/processed/prodes_municipio_ano.parquet` com `cd_mun`, `ano`, `desmat_km2` e `desmat_pct_area`.

### 07 — Censo 2022: renda domiciliar per capita
- Fonte: IBGE/SIDRA, Censo Demográfico 2022, renda domiciliar per capita (média ou mediana) por município. Confirme o código da tabela no SIDRA.
- Valor único por município, usado depois só no módulo multicritério.
- Saída: `data/processed/renda_municipio.parquet`.

### 08 — INPE Risco de Fogo: média mensal (referência de comparação)
- Fonte: dataserver do INPE, `riscofogo_meteorologia/media_mensal_risco_de_fogo/`, de 2019 a 2025.
- Média zonal por município e mês (rasterstats).
- Saída: `data/processed/risco_fogo_inpe_municipio_mes.parquet`.

### 09 — Montagem da tabela final
- Grade completa de 144 municípios × 84 meses. Junte focos, clima e risco INPE por (`cd_mun`, `mes`), as variáveis anuais com defasagem t-1 e a renda por município.
- Inclua também `ano`, `mes_num`, `area_km2` e `focos_por_km2`.
- Saída: `data/processed/aphim_base_municipio_mes.parquet` e uma cópia em CSV.
- Validações finais (falhar com erro se não passarem): 12.096 linhas; sem duplicatas de (`cd_mun`, `mes`); relatório de valores ausentes por coluna; checagem de que nenhuma variável anual usa o próprio ano.

## Entregáveis desta etapa
1. Os scripts acima, executáveis em sequência (por exemplo, `python -m src.coleta.02_firms`), mais um `run_all.py`.
2. Um `README.md` explicando como configurar o `.env`, autenticar o Earth Engine e rodar tudo.
3. Um relatório curto (`data/processed/relatorio_coleta.md`) com os totais por fonte, os descartes de cada filtro, o resultado da auditoria FIRMS × INPE e os valores ausentes.

Comece pelo script 01, depois o 02 com a auditoria 03, e me mostre os totais anuais de focos antes de seguir para as demais fontes.