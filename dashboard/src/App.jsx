import React, { useState } from 'react';
import { ParentSize } from '@visx/responsive';
import Header from './components/Header';
import MetricCards from './components/MetricCards';
import ConversionGaugeCard from './components/ConversionGaugeCard';
import TimeSeriesChart from './components/charts/TimeSeriesChart';
import SensorDistributionCard from './components/SensorDistributionCard';
import TerritorialDistributionCard from './components/TerritorialDistributionCard';
import TopMunicipiosRiskCard from './components/TopMunicipiosRiskCard';
import AnnualBarChart from './components/charts/AnnualBarChart';
import SeasonalityChart from './components/charts/SeasonalityChart';
import FunnelChart from './components/charts/FunnelChart';
import MunicipiosTable from './components/MunicipiosTable';
import AuditTable from './components/AuditTable';
import PipelineRoadmap from './components/PipelineRoadmap';
import MulticriterioBaseView from './components/MulticriterioBaseView';

import data from './data/aphimData.json';
import { 
  BarChart3, 
  MapPin, 
  Satellite, 
  AlertCircle,
  FileCheck
} from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [periodFilter, setPeriodFilter] = useState('ALL');

  const [timeSeriesMode, setTimeSeriesMode] = useState('lines'); // 'lines' ou 'lines-only'
  const [showAnnualLabels, setShowAnnualLabels] = useState(true);

  const {
    auditoria_resumo,
    funil,
    totais_anuais,
    serie_mensal,
    sazonalidade,
    ranking_municipios
  } = data;

  // Filtragem dinâmica do período temporal para a série mensal
  const filteredSerieMensal = React.useMemo(() => {
    if (!serie_mensal) return [];
    if (periodFilter === '1M') return serie_mensal.slice(-3);
    if (periodFilter === '6M') return serie_mensal.slice(-6);
    if (periodFilter === '1Y') return serie_mensal.slice(-12);
    return serie_mensal;
  }, [serie_mensal, periodFilter]);

  return (
    <div className="app-container">
      {/* Top Header com Search, Utilidades e Navegação */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Linha Superior: 4 Cards de Métricas (Idêntico à Imagem de Referência) */}
      <MetricCards summary={auditoria_resumo} onSelectTab={setActiveTab} />

      {/* ABA 1: VISÃO GERAL (Estrutura Fiel à Imagem de Referência) */}
      {activeTab === 'overview' && (
        <>
          {/* Linha Central (3 Colunas: Conversions, Performance, Session By Browser) */}
          <div className="middle-grid-3">
            {/* Coluna 1: Conversions Style (Validação Cruzada & Gauge) */}
            <ConversionGaugeCard onViewDetails={() => setActiveTab('etapa3')} />

            {/* Coluna 2: Performance Style (Série Temporal com Suporte a Linhas e Barra + Linha) */}
            <div className="ref-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div className="ref-card-header" style={{ flexWrap: 'wrap', gap: 8 }}>
                <span className="ref-card-title">Performance da Série Temporal</span>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                  {/* Seletor de Tipo: Linhas & Área vs Linhas Puras */}
                  <div className="period-pill-group">
                    <button
                      className={`period-pill-btn ${timeSeriesMode === 'lines' ? 'active' : ''}`}
                      onClick={() => setTimeSeriesMode('lines')}
                      title="Exibir ambas as séries como Linhas Suaves com Área Gradiente"
                    >
                      Linhas & Área
                    </button>
                    <button
                      className={`period-pill-btn ${timeSeriesMode === 'lines-only' ? 'active' : ''}`}
                      onClick={() => setTimeSeriesMode('lines-only')}
                      title="Exibir ambas as séries como Linhas Puras"
                    >
                      Linhas Puras
                    </button>
                  </div>

                  {/* Seletor de Período Temporal */}
                  <div className="period-pill-group">
                    {['ALL', '1M', '6M', '1Y'].map((p) => (
                      <button
                        key={p}
                        className={`period-pill-btn ${periodFilter === p ? 'active' : ''}`}
                        onClick={() => setPeriodFilter(p)}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="ref-card-body" style={{ paddingTop: 0, paddingBottom: 14 }}>
                {/* Banner de Aviso no Padrão da Imagem de Referência */}
                <div className="alert-banner-box">
                  <AlertCircle size={16} />
                  <span>
                    Aviso Metodológico: Base de dados auditada com 512 janelas contínuas sem hiatos temporais (2019–2025).
                  </span>
                </div>

                {/* Gráfico Visx Performance */}
                <div style={{ height: 260, width: '100%' }}>
                  <ParentSize>
                    {({ width, height }) => (
                      <TimeSeriesChart
                        data={filteredSerieMensal}
                        chartType={timeSeriesMode}
                        width={width}
                        height={height}
                      />
                    )}
                  </ParentSize>
                </div>

                {/* Legenda Limpa da Imagem */}
                <div className="chart-legend-row">
                  <div>
                    <span className="legend-dot" style={{ background: 'var(--primary-teal)' }} />
                    <strong>NASA FIRMS (VIIRS 375m)</strong>
                    <span style={{ fontSize: '0.72rem', color: '#64748b', marginLeft: 4 }}>
                      (Linha Verde Esmeralda)
                    </span>
                  </div>
                  <div>
                    <span className="legend-dot" style={{ background: 'var(--primary-blue)' }} />
                    <strong>INPE BDQueimadas (1km)</strong>
                    <span style={{ fontSize: '0.72rem', color: '#64748b', marginLeft: 4 }}>
                      (Linha Azul de Referência)
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Coluna 3: Session By Browser Style (Distribuição por Sensor) */}
            <SensorDistributionCard onViewAll={() => setActiveTab('etapa2')} />
          </div>

          {/* Linha Inferior (2 Colunas: Sessions by Country & Top Pages) */}
          <div className="bottom-grid-2">
            {/* Coluna 1: Sessions By Country Style (Municípios Críticos & Mapa) */}
            <TerritorialDistributionCard onViewData={() => setActiveTab('municipios')} />

            {/* Coluna 2: Top Pages Style (Tabela de Municípios & Badges de Risco) */}
            <TopMunicipiosRiskCard ranking={ranking_municipios} onViewAll={() => setActiveTab('municipios')} />
          </div>

          {/* Análises Complementares: Totais Anuais & Sazonalidade */}
          <div className="bottom-grid-2" style={{ marginTop: 24 }}>
            <div className="ref-card">
              <div className="ref-card-header">
                <span className="ref-card-title">
                  <BarChart3 size={18} color="var(--primary-teal)" />
                  Comparativo Anual Consolidado
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button
                    className={`period-pill-btn ${showAnnualLabels ? 'active' : ''}`}
                    onClick={() => setShowAnnualLabels(!showAnnualLabels)}
                    style={{
                      border: '1px solid #e2e8f0',
                      background: showAnnualLabels ? 'var(--primary-teal-light)' : 'transparent',
                      color: showAnnualLabels ? 'var(--primary-teal)' : '#64748b',
                      fontSize: '0.72rem'
                    }}
                    title="Alternar visibilidade dos rótulos de dados nos topos das colunas"
                  >
                    🏷️ Rótulos: {showAnnualLabels ? 'Ativos' : 'Ocultos'}
                  </button>
                  <span className="pill-badge info">2019 a 2025</span>
                </div>
              </div>
              <div className="ref-card-body" style={{ minHeight: 330, paddingTop: 6 }}>
                <div style={{ width: '100%', height: 320, position: 'relative' }}>
                  <ParentSize debounceTime={10}>
                    {({ width, height }) => (
                      <AnnualBarChart
                        data={totais_anuais}
                        showLabels={showAnnualLabels}
                        width={width || 500}
                        height={height || 320}
                      />
                    )}
                  </ParentSize>
                </div>
              </div>
            </div>

            <div className="ref-card">
              <div className="ref-card-header">
                <span className="ref-card-title">
                  <Satellite size={18} color="var(--primary-blue)" />
                  Ciclo Sazonal Médio de Queimadas (Jan a Dez)
                </span>
                <span className="pill-badge warning">🔥 Alta Concentração no 2º Semestre</span>
              </div>
              <div className="ref-card-body" style={{ minHeight: 380, paddingTop: 6 }}>
                <div style={{ width: '100%', height: 380, position: 'relative' }}>
                  <ParentSize debounceTime={10}>
                    {({ width, height }) => (
                      <SeasonalityChart
                        data={sazonalidade}
                        width={width || 500}
                        height={height || 380}
                      />
                    )}
                  </ParentSize>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ABA 2: ETAPA 01 - IBGE */}
      {activeTab === 'etapa1' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="ref-card" style={{ padding: 22 }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <MapPin size={20} color="var(--primary-teal)" />
              Etapa 01 — Malha Municipal e Áreas do Pará (IBGE UF 15)
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginBottom: 18 }}>
              Definição das fronteiras territoriais e cálculo de área de cada um dos 144 municípios do Pará em projeção Polyconic EPSG:5880.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>MUNICÍPIOS OFICIAIS</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 2 }}>144</div>
                <span className="pill-badge success" style={{ marginTop: 4 }}>100% validados via API IBGE</span>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>ÁREA TOTAL CALCULADA</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 2 }}>1.247.754 km²</div>
                <span className="pill-badge success" style={{ marginTop: 4 }}>Diferença de apenas +0,15% da oficial</span>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>SISTEMA DE COORDENADAS</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>EPSG:5880</div>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 4 }}>SIRGAS 2000 / Polyconic</div>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>ARTEFATO GERADO</div>
                <div className="mono" style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-teal)', marginTop: 6 }}>
                  municipios_pa.gpkg
                </div>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: 4 }}>Formato OGC GeoPackage</div>
              </div>
            </div>
          </div>

          <MunicipiosTable data={ranking_municipios} />
        </div>
      )}

      {/* ABA 3: ETAPA 02 - NASA FIRMS */}
      {activeTab === 'etapa2' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="ref-card" style={{ padding: 22 }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Satellite size={20} color="var(--primary-teal)" />
              Etapa 02 — Coleta NASA FIRMS (VIIRS S-NPP) & Funil Metodológico
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginBottom: 20 }}>
              Coleta de 512 janelas contínuas de 5 dias com resolução espacial nativa de 375m (2019 a 2025).
            </p>

            <div style={{ height: 260, marginBottom: 20 }}>
              <ParentSize>
                {({ width, height }) => (
                  <FunnelChart
                    data={funil}
                    width={width}
                    height={height}
                  />
                )}
              </ParentSize>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
              {funil.map((step) => (
                <div key={step.etapa} style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontWeight: 700, fontSize: '0.92rem', marginBottom: 4 }}>{step.etapa}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10 }}>{step.descricao}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 8, borderTop: '1px solid var(--border-subtle)' }}>
                    <span className="mono" style={{ fontWeight: 700, color: 'var(--primary-teal)' }}>
                      {step.focos.toLocaleString('pt-BR')} focos
                    </span>
                    <span className="pill-badge info">{step.pct_retida}% retido</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ABA 4: ETAPA 03 - AUDITORIA INPE */}
      {activeTab === 'etapa3' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="ref-card" style={{ padding: 22 }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <FileCheck size={20} color="var(--primary-blue)" />
              Etapa 03 — Auditoria Cruzada INPE BDQueimadas vs NASA FIRMS
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', marginBottom: 18 }}>
              Validação comparativa mensal atestando que os picos e sazonalidades do sensor VIIRS (375m) correlacionam-se estritamente com o satélite oficial de referência do INPE (1km).
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>CORRELAÇÃO LINEAR (PEARSON)</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-teal)', marginTop: 2 }}>r = 0,9822</div>
                <span className="pill-badge success" style={{ marginTop: 4 }}>Correlação linear quase perfeita</span>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>CORRELAÇÃO POSTOS (SPEARMAN)</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>ρ = 0,9898</div>
                <span className="pill-badge success" style={{ marginTop: 4 }}>Invariância estrita de ordenação</span>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>PICO MÁXIMO HISTÓRICO</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--danger-red-text)', marginTop: 2 }}>09/2024</div>
                <span className="pill-badge danger" style={{ marginTop: 4 }}>Sincronia total entre sensores</span>
              </div>
              <div style={{ background: 'var(--bg-surface-alt)', padding: 14, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>RAZÃO DE SENSIBILIDADE</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 2 }}>6,80x</div>
                <span className="pill-badge info" style={{ marginTop: 4 }}>Mais sensível a fogo de sub-bosque</span>
              </div>
            </div>
          </div>

          <AuditTable serieMensal={serie_mensal} totaisAnuais={totais_anuais} />
        </div>
      )}

      {/* ABA: BASE INTEGRADA MULTICRITÉRIO (44 VARIÁVEIS - ETAPAS 04 A 09) */}
      {activeTab === 'multicriterio' && (
        <MulticriterioBaseView />
      )}

      {/* ABA 5: 144 MUNICÍPIOS */}
      {activeTab === 'municipios' && (
        <MunicipiosTable data={ranking_municipios} />
      )}

      {/* ABA 6: PIPELINE APHIM */}
      {activeTab === 'pipeline' && (
        <PipelineRoadmap />
      )}
    </div>
  );
}

export default App;
