import React from 'react';
import { 
  CloudSun, 
  Trees, 
  Flame,
  Globe2,
  FileCheck
} from 'lucide-react';

const PIPELINE_STEPS = [
  {
    num: '01',
    title: 'Malha Municipal & Áreas IBGE',
    source: 'IBGE API v3 / Localidades',
    status: 'DONE',
    output: 'municipios_pa.gpkg',
    icon: <Globe2 size={20} color="var(--primary-teal)" />,
    desc: 'Extração dos 144 municípios do Pará (UF 15), cálculo de área com projeção Polyconic EPSG:5880 (1.247.753 km², tolerância ±1%).'
  },
  {
    num: '02',
    title: 'Focos de Calor NASA FIRMS',
    source: 'NASA FIRMS API (VIIRS S-NPP)',
    status: 'DONE',
    output: 'firms_focos_pa.parquet',
    icon: <Flame size={20} color="var(--primary-teal)" />,
    desc: 'Download contínuo de 512 janelas de 5 dias (2019-2025). Funil de qualidade consolidou 1.695.702 focos com zeros explícitos em 12.096 linhas.'
  },
  {
    num: '03',
    title: 'Auditoria Cruzada INPE BDQueimadas',
    source: 'INPE Dataserver (Satélite de Referência)',
    status: 'DONE',
    output: 'auditoria_firms_vs_inpe.csv',
    icon: <FileCheck size={20} color="var(--primary-blue)" />,
    desc: 'Auditoria com 249.210 focos do INPE. Correlação linear Pearson de 0,9822, Spearman de 0,9898 e sincronia perfeita do pico em 09/2024.'
  },
  {
    num: '04',
    title: 'Clima ERA5-Land Reanálise',
    source: 'Google Earth Engine (ECMWF)',
    status: 'DONE',
    output: 'clima_municipio_mes.parquet',
    icon: <CloudSun size={20} color="var(--primary-teal)" />,
    desc: 'Extração mensal via reduceRegions: 13.824 linhas (2018-2025), normal 1991-2020 (1.728 linhas). Correlação chuva x focos de -0,5934 e sazonalidade validada.'
  },
  {
    num: '05',
    title: 'Uso e Cobertura do Solo MapBiomas',
    source: 'MapBiomas Brasil (GEE Coleção 10)',
    status: 'DONE',
    output: 'mapbiomas_municipio_ano.parquet',
    icon: <Trees size={20} color="var(--primary-teal)" />,
    desc: 'Uso da terra em 30m: 1.008 linhas (2018–2024). Floresta estadual caiu de 77,0% para 74,3% e pastagem subiu de 14,8% para 17,4%.'
  },
  {
    num: '06',
    title: 'Desmatamento Anual PRODES',
    source: 'INPE TerraBrasilis (WFS Anual)',
    status: 'DONE',
    output: 'prodes_municipio_ano.parquet',
    icon: <FileCheck size={20} color="var(--primary-teal)" />,
    desc: 'Interseção espacial em EPSG:5880: 1.152 linhas (2018–2025). Altamira e São Félix lideram com mais de 3.000 km² acumulados.'
  },
  {
    num: '07',
    title: 'Censo Demográfico (Renda per capita)',
    source: 'IBGE SIDRA (Censo 2022 Tabela 10295)',
    status: 'DONE',
    output: 'renda_municipio.parquet',
    icon: <Globe2 size={20} color="var(--primary-teal)" />,
    desc: 'Rendimento domiciliar mediano per capita (144 municípios). Extremos de R$ 233,33 (Bagre) a R$ 1.000,00 (Novo Progresso).'
  },
  {
    num: '08',
    title: 'INPE Risco de Fogo Mensal',
    source: 'INPE Programa Queimadas (Dataserver)',
    status: 'DONE',
    output: 'risco_fogo_inpe_municipio_mes.parquet',
    icon: <Flame size={20} color="var(--primary-teal)" />,
    desc: 'Médias e máximos zonais em 0,01° (12.096 linhas). Pico em setembro (0,545) e correlação Spearman de +0,6803 com os focos.'
  },
  {
    num: '09',
    title: 'Tabela Final Integrada APHIM',
    source: 'Consolidação com defasagens t-1',
    status: 'DONE',
    output: 'aphim_base_municipio_mes.parquet (.csv)',
    icon: <FileCheck size={20} color="var(--primary-teal)" />,
    desc: 'Grade 144 municípios × 84 meses = 12.096 linhas, 44 colunas e zero nulos. 1.695.702 focos consolidados com integridade total.'
  }
];

const PipelineRoadmap = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="ref-card" style={{ padding: 22 }}>
        <h2 style={{ fontSize: '1.25rem', marginBottom: 6 }}>
          Pipeline Metodológico APHIM (Arquitetura Preditiva Híbrida)
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Fluxo de engenharia de dados e modelagem multicritério para predição e classificação de risco de queimadas.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        {PIPELINE_STEPS.map((step) => {
          const isDone = step.status === 'DONE';
          const isNext = step.status === 'NEXT';

          return (
            <div key={step.num} className="ref-card" style={{ padding: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ 
                    width: 32, 
                    height: 32, 
                    borderRadius: 6, 
                    background: isDone ? 'var(--primary-teal-light)' : isNext ? 'var(--warning-amber-bg)' : 'var(--btn-gray)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {step.icon}
                  </div>
                  <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Etapa {step.num}</span>
                </div>

                <span className={`pill-badge ${isDone ? 'success' : isNext ? 'warning' : 'info'}`}>
                  {isDone ? 'Validação Concluída' : isNext ? 'Em Andamento' : 'Planejado'}
                </span>
              </div>

              <h3 style={{ fontSize: '1rem', marginBottom: 6 }}>{step.title}</h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 12, minHeight: 48 }}>
                {step.desc}
              </p>

              <div style={{ paddingTop: 10, borderTop: '1px solid var(--border-subtle)', fontSize: '0.78rem' }}>
                <div style={{ color: 'var(--text-light)', marginBottom: 2 }}>Fonte: {step.source}</div>
                <div className="mono" style={{ color: 'var(--primary-teal)', fontWeight: 600 }}>{step.output}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PipelineRoadmap;
