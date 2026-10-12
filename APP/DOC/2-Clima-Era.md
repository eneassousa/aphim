Continue no repositório local aphim (projeto APHIM, dissertação sobre focos de queimadas nos 144 municípios do Pará). As etapas 01 (malha IBGE), 02 (focos NASA FIRMS VIIRS S-NPP) e 03 (auditoria com o INPE) já estão prontas e validadas. Não altere essas etapas.

Agora implemente a etapa 04: clima ERA5-Land via Google Earth Engine. Siga as mesmas regras gerais do projeto:

Scripts idempotentes e retomáveis.
Credenciais só pelo .env (GEE_PROJECT).
cd_mun como texto de 7 dígitos e mês no formato AAAA-MM.
Log de tudo o que for baixado e descartado.
Papel destas variáveis no projeto

O clima é o insumo do perigo, ou seja, do modelo de ML que estima a chance de atividade de fogo acima do padrão no mês seguinte. Por decisão metodológica, o clima não entra no multicritério de vulnerabilidade. Nesta etapa, apenas colete, converta unidades e valide. Ainda não crie defasagens nem atributos de modelagem.

Fonte
Asset do Earth Engine: ECMWF/ERA5_LAND/MONTHLY_AGGR (resolução nativa de ~11 km). Confira no catálogo oficial antes de usar e registre a URL no código: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR
Bandas e conversões:
Banda	Saída	Conversão
temperature_2m	t2m_media_c	K − 273,15
temperature_2m_max	t2m_max_c	K − 273,15 (máxima do mês)
temperature_2m_min	t2m_min_c	K − 273,15
dewpoint_temperature_2m	td2m_c	K − 273,15
total_precipitation_sum	prec_mm	m × 1000; valores negativos viram 0 e são registrados no log
skin_temperature	lst_c	K − 273,15
volumetric_soil_water_layer_1	umid_solo	fração volumétrica, sem conversão
u_component_of_wind_10m, v_component_of_wind_10m	vento_ms	√(u² + v²), calculado sobre as médias mensais (documente essa limitação)
Variáveis derivadas (calcule em pandas, depois da extração):
ur_pct: umidade relativa pela fórmula de Magnus, a partir de t2m_media_c e td2m_c, limitada a [0, 100].
dpv_kpa: déficit de pressão de vapor, igual a es(T) − es(Td), com es pela mesma fórmula de Magnus.
Extração
Use reduceRegions com ee.Reducer.mean() sobre a malha data/processed/municipios_pa.gpkg (envie-a ao GEE como FeatureCollection ou como asset).
Atenção aos municípios pequenos: com pixel de ~11 km, um município pode não conter nenhum centro de pixel e voltar nulo. Use scale de 1000 m (reamostragem por vizinho mais próximo) para que todos recebam valor. Valide que nenhum município fica nulo.
Processe mês a mês ou ano a ano, para não estourar os limites do GEE. Salve cada pedaço em data/raw/era5/ e pule os que já existirem.
Períodos
Série principal: jan/2018 a dez/2025. O ano de 2018 é necessário para as defasagens que serão criadas depois. Saída: data/processed/clima_municipio_mes.parquet.
Climatologia de referência 1991–2020 (normal climatológica da OMM): média por município e por mês do calendário (1 a 12) de prec_mm, t2m_media_c, ur_pct, umid_solo e lst_c. Saída: data/processed/clima_normal_1991_2020.parquet com 144 × 12 = 1.728 linhas. Ela servirá depois para calcular anomalias (por exemplo, anomalia de chuva). As anomalias não são calculadas agora.
Validações (falhar com erro se não passarem)
Série principal com exatamente 144 × 96 = 13.824 linhas, sem duplicatas de (cd_mun, mes) e sem nulos.
Faixas plausíveis para o Pará:
t2m_media_c entre 18 e 35;
prec_mm entre 0 e 1000;
ur_pct entre 30 e 100;
umid_solo entre 0 e 0,6.
Liste os casos fora dessas faixas.
Sazonalidade coerente: a média estadual de prec_mm deve ser maior em fevereiro, março e abril e menor em julho, agosto e setembro. Imprima a média estadual por mês do calendário.
Imprima a correlação mensal, em 2019–2025, entre a média estadual de prec_mm e o total de focos do Pará (data/processed/focos_municipio_mes.parquet). Ela é esperada negativa.
Opcional (script separado 04b_fma.py, só depois de o 04 validar)

Calcule a FMA (Fórmula de Monte Alegre, Soares 1972) diária a partir de ECMWF/ERA5_LAND/HOURLY, usando a umidade relativa às 13h locais (16 UTC) e a chuva diária para a regra de abatimento. Agregue por município e mês como média e como máximo mensal. Registre no código a fonte da fórmula e dos limiares de chuva. Não rode isso antes de eu confirmar.

Entregáveis
src/coleta/04_era5.py, mais a entrada correspondente em run_all.py.
A seção de clima no data/processed/relatorio_coleta.md, com as linhas, os nulos, os valores fora de faixa, a sazonalidade e a correlação chuva × focos.
A atualização do README.md com o passo de autenticação do Earth Engine (earthengine authenticate).

Ao terminar, mostre:

a tabela da média estadual por mês do calendário (chuva, temperatura, umidade relativa);
a correlação chuva × focos;
quantos valores ficaram fora de faixa.