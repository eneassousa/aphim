import React, { useState } from 'react';
import { 
  CloudRain, 
  Flame, 
  Trees, 
  DollarSign, 
  Database, 
  Search, 
  CheckCircle2, 
  AlertTriangle, 
  FileText,
  TrendingUp,
  BarChart2,
  Table,
  Layers,
  Info
} from 'lucide-react';
import data from '../data/aphimData.json';

const MulticriterioBaseView = () => {
  const [subTab, setSubTab] = useState('tabela_final');
  const [dictSearch, setDictSearch] = useState('');
  const [dictFilterCategory, setDictFilterCategory] = useState('ALL');

  const {
    clima_resumo,
    mapbiomas_resumo,
    prodes_resumo,
    renda_resumo,
    risco_fogo_resumo,
    tabela_final_resumo,
    clima_sazonalidade = [],
    mapbiomas_evolucao = [],
    prodes_evolucao = [],
    renda_ranking_144 = [],
    dicionario_dados = [],
    amostra_tabela_final = []
  } = data;

  // Filtragem do dicionário de dados
  const filteredDict = dicionario_dados.filter((item) => {
    const matchesSearch = item.coluna.toLowerCase().includes(dictSearch.toLowerCase()) ||
                          item.desc.toLowerCase().includes(dictSearch.toLowerCase()) ||
                          item.fonte.toLowerCase().includes(dictSearch.toLowerCase());
    if (dictFilterCategory === 'ALL') return matchesSearch;
    if (dictFilterCategory === 'CLIMA') return matchesSearch && (item.coluna.startsWith('t2m') || item.coluna.startsWith('prec') || item.coluna.includes('vento') || item.coluna.includes('ur') || item.coluna.includes('dpv') || item.coluna.includes('solo') || item.coluna.includes('lst') || item.coluna.includes('td2m'));
    if (dictFilterCategory === 'FOGO') return matchesSearch && (item.coluna.includes('focos') || item.coluna.startsWith('rf_inpe'));
    if (dictFilterCategory === 'SOLO') return matchesSearch && (item.coluna.includes('pct_') || item.coluna.includes('mapbiomas'));
    if (dictFilterCategory === 'PRODES') return matchesSearch && item.coluna.includes('prodes');
    if (dictFilterCategory === 'RENDA') return matchesSearch && item.coluna.includes('renda');
    return matchesSearch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header da Seção Multicritério */}
      <div className="ref-card" style={{ padding: 22 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Database size={22} color="var(--primary-teal)" />
              Base Consolidada Multicritério APHIM (Etapas 04 a 09)
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
              Painel analítico das variáveis biofísicas, de perigo meteorológico, uso do solo, desmatamento e vulnerabilidade socioeconômica.
            </p>
          </div>

          {/* Sub-navegação interna */}
          <div className="period-pill-group" style={{ flexWrap: 'wrap' }}>
            <button 
              className={`period-pill-btn ${subTab === 'tabela_final' ? 'active' : ''}`}
              onClick={() => setSubTab('tabela_final')}
            >
              <Table size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              Base Integrada (44 Vars)
            </button>
            <button 
              className={`period-pill-btn ${subTab === 'clima_risco' ? 'active' : ''}`}
              onClick={() => setSubTab('clima_risco')}
            >
              <CloudRain size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              04 & 08. Clima e Risco
            </button>
            <button 
              className={`period-pill-btn ${subTab === 'solo_prodes' ? 'active' : ''}`}
              onClick={() => setSubTab('solo_prodes')}
            >
              <Trees size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              05 & 06. MapBiomas & PRODES
            </button>
            <button 
              className={`period-pill-btn ${subTab === 'renda' ? 'active' : ''}`}
              onClick={() => setSubTab('renda')}
            >
              <DollarSign size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              07. Renda Censo 2022
            </button>
          </div>
        </div>
      </div>

      {/* SUB-ABA 1: BASE INTEGRADA & DICIONÁRIO DE DADOS */}
      {subTab === 'tabela_final' && (
        <>
          {/* Cards de Resumo Geral da Base Final */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>GRADE TOTAL</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 2 }}>12.096 linhas</div>
              <span className="pill-badge success" style={{ marginTop: 4 }}>144 municípios × 84 meses</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>DIMENSÃO DE ATRIBUTOS</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-teal)', marginTop: 2 }}>44 colunas</div>
              <span className="pill-badge info" style={{ marginTop: 4 }}>Biofísica + Antrópica + Social</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>INTEGRIDADE NULA</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>0 nulos (100%)</div>
              <span className="pill-badge success" style={{ marginTop: 4 }}>Zero falhas em 44 variáveis</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>DEFASAGEM TEMPORAL</div>
              <div style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--warning-amber)', marginTop: 4 }}>t − 1 estrita</div>
              <span className="pill-badge warning" style={{ marginTop: 4 }}>MapBiomas & PRODES t-1</span>
            </div>
          </div>

          {/* Dicionário de Dados Interativo */}
          <div className="ref-card" style={{ padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <FileText size={18} color="var(--primary-teal)" />
                  Dicionário Oficial de Dados da Base APHIM
                </h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Total de {filteredDict.length} variáveis listadas
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                {/* Filtro por Categoria */}
                <div className="period-pill-group">
                  {['ALL', 'CLIMA', 'FOGO', 'SOLO', 'PRODES', 'RENDA'].map((cat) => (
                    <button
                      key={cat}
                      className={`period-pill-btn ${dictFilterCategory === cat ? 'active' : ''}`}
                      onClick={() => setDictFilterCategory(cat)}
                      style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                    >
                      {cat}
                    </button>
                  ))}
                </div>

                {/* Busca rápida */}
                <div className="search-input-wrapper" style={{ maxWidth: 220, height: 32 }}>
                  <Search size={14} color="var(--text-light)" />
                  <input
                    type="text"
                    placeholder="Filtrar coluna..."
                    value={dictSearch}
                    onChange={(e) => setDictSearch(e.target.value)}
                    style={{ fontSize: '0.8rem' }}
                  />
                </div>
              </div>
            </div>

            <div style={{ overflowX: 'auto', maxHeight: 380 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '2px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Coluna</th>
                    <th style={{ padding: '8px 10px' }}>Descrição</th>
                    <th style={{ padding: '8px 10px' }}>Unidade</th>
                    <th style={{ padding: '8px 10px' }}>Fonte</th>
                    <th style={{ padding: '8px 10px' }}>Escala</th>
                    <th style={{ padding: '8px 10px' }}>Defasagem</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDict.map((v, i) => (
                    <tr key={v.coluna} style={{ borderBottom: '1px solid var(--border-subtle)', background: i % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.015)' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600, fontFamily: 'monospace', color: 'var(--primary-teal)' }}>
                        {v.coluna}
                      </td>
                      <td style={{ padding: '8px 10px' }}>{v.desc}</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-muted)' }}>{v.unidade}</td>
                      <td style={{ padding: '8px 10px' }}>{v.fonte}</td>
                      <td style={{ padding: '8px 10px' }}>{v.temp}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <span className={`pill-badge ${v.defasagem === 'Nenhuma' ? 'info' : v.defasagem === 't0' ? 'success' : 'warning'}`}>
                          {v.defasagem}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Amostra da Base Consolidada */}
          <div className="ref-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 12 }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Table size={18} color="var(--primary-blue)" />
                Amostra Real da Tabela Final (`aphim_base_municipio_mes.parquet`)
              </h3>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Exibição de 20 municípios representativos durante o pico de queimadas (Agosto de 2025)
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '2px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>IBGE</th>
                    <th style={{ padding: '6px 8px' }}>Município</th>
                    <th style={{ padding: '6px 8px' }}>Mês</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Focos</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Chuva (mm)</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Temp (°C)</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Risco INPE</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Floresta (t-1)</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Pasto (t-1)</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Desmat. km² (t-1)</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Renda PC</th>
                  </tr>
                </thead>
                <tbody>
                  {amostra_tabela_final.map((r) => (
                    <tr key={r.cd_mun} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{r.cd_mun}</td>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.nm_mun}</td>
                      <td style={{ padding: '6px 8px' }}>{r.mes}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 700, color: 'var(--danger-red-text)' }}>
                        {r.focos_count.toLocaleString('pt-BR')}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{r.prec_mm.toFixed(1)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{r.t2m_media_c.toFixed(1)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600, color: 'var(--warning-amber)' }}>
                        {r.rf_inpe_media ? r.rf_inpe_media.toFixed(3) : '-'}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{r.pct_floresta.toFixed(1)}%</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{r.pct_pastagem.toFixed(1)}%</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{r.desmat_km2_prodes ? r.desmat_km2_prodes.toFixed(1) : '0.0'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600, color: 'var(--primary-teal)' }}>
                        R$ {r.renda_pc ? r.renda_pc.toFixed(2) : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* SUB-ABA 2: CLIMA & RISCO DE FOGO */}
      {subTab === 'clima_risco' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>CORRELAÇÃO CHUVA × FOCOS</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>r = -0,5934</div>
              <span className="pill-badge info" style={{ marginTop: 4 }}>Pearson: forte inibição por chuva</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>CORRELAÇÃO RISCO INPE × FOCOS</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--primary-teal)', marginTop: 2 }}>ρ = +0,6803</div>
              <span className="pill-badge success" style={{ marginTop: 4 }}>Spearman: excelente aderência física</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>MESES DE PICO DE PERIGO</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--danger-red-text)', marginTop: 2 }}>Ago a Out</div>
              <span className="pill-badge danger" style={{ marginTop: 4 }}>Pico máximo: Setembro (0,5450)</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>EVENTOS DE CHUVA EXTREMA</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 2 }}>5 casos</div>
              <span className="pill-badge warning" style={{ marginTop: 4 }}>prec_extremo (Mar/2022 &gt; 1000mm)</span>
            </div>
          </div>

          <div className="ref-card" style={{ padding: 20 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 12 }}>
              Sazonalidade Mensal Clima & Risco Meteorológico (Médias do Pará)
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '2px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Mês</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Chuva (`prec_mm`)</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Temp 2m (`t2m_media_c`)</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Umidade (`ur_pct`)</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>DPV (`dpv_kpa`)</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Risco Fogo INPE</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Total de Focos</th>
                  </tr>
                </thead>
                <tbody>
                  {clima_sazonalidade.map((m) => (
                    <tr key={m.mes_num} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 700 }}>{m.mes_nome} ({m.mes_num.toString().padStart(2, '0')})</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>{m.prec_mm.toFixed(1)} mm</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right' }}>{m.t2m_media_c.toFixed(1)} °C</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right' }}>{m.ur_pct.toFixed(1)}%</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right' }}>{m.dpv_kpa.toFixed(2)} kPa</td>
                      <td style={{ 
                        padding: '8px 10px', 
                        textAlign: 'right', 
                        fontWeight: 700,
                        color: m.rf_inpe_media > 0.4 ? 'var(--danger-red-text)' : m.rf_inpe_media > 0.1 ? 'var(--warning-amber)' : 'var(--primary-teal)' 
                      }}>
                        {m.rf_inpe_media.toFixed(4)}
                      </td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                        {m.focos_count.toLocaleString('pt-BR')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* SUB-ABA 3: USO DO SOLO & PRODES */}
      {subTab === 'solo_prodes' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
            <div className="ref-card" style={{ padding: 18 }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>MAPBIOMAS COLEÇÃO 10 (2018–2024)</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-teal)', marginBottom: 8 }}>
                Florestas: 77,0% → 74,3% | Pastagens: 14,8% → 17,4%
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                1.008 linhas municipais-anuais em resolução 30m. Classe 32 (Apicum costeiro) incorporada com sucesso a campos naturais, resultando em <strong>0% na categoria outros</strong>.
              </p>
            </div>

            <div className="ref-card" style={{ padding: 18 }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>INPE PRODES AMAZÔNIA (2018–2025)</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--danger-red-text)', marginBottom: 8 }}>
                28.416,7 km² desmatados acumulados
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                1.152 linhas anuais integradas com defasagem $t-1$ (ano PRODES encerrado em 31/jul do ano anterior à detecção de queimadas).
              </p>
            </div>
          </div>

          <div className="bottom-grid-2">
            <div className="ref-card" style={{ padding: 18 }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12 }}>
                Evolução Estadual MapBiomas (Médias dos Municípios)
              </h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '1px solid var(--border-subtle)' }}>
                      <th style={{ padding: '6px 8px' }}>Ano</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Floresta</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Pastagem</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Agricultura</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>% Veg. Nativa</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapbiomas_evolucao.map((y) => (
                      <tr key={y.ano} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 700 }}>{y.ano}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>{y.pct_floresta.toFixed(1)}%</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>{y.pct_pastagem.toFixed(1)}%</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>{y.pct_agricultura.toFixed(2)}%</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600, color: 'var(--primary-teal)' }}>
                          {y.pct_veg_nativa.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="ref-card" style={{ padding: 18 }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: 12 }}>
                Desmatamento Anual Consolidado PRODES (Pará)
              </h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '1px solid var(--border-subtle)' }}>
                      <th style={{ padding: '6px 8px' }}>Ano PRODES</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>Desmatamento (km²)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>Taxa Média Mun. (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prodes_evolucao.map((p) => (
                      <tr key={p.ano} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 700 }}>{p.ano}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600, color: 'var(--danger-red-text)' }}>
                          {p.desmat_km2.toLocaleString('pt-BR')} km²
                        </td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>{p.desmat_pct_area.toFixed(3)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-ABA 4: RENDA MUNICIPAL CENSO 2022 */}
      {subTab === 'renda' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>FONTE OFICIAL</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-heading)', marginTop: 4 }}>Censo 2022 SIDRA</div>
              <span className="pill-badge info" style={{ marginTop: 4 }}>Tabela 10295 V13534</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>MÍNIMO (MAIS VULNERÁVEL)</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--danger-red-text)', marginTop: 2 }}>R$ 233,33</div>
              <span className="pill-badge danger" style={{ marginTop: 4 }}>Bagre (Marajó)</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>MEDIANA DO PARÁ</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary-teal)', marginTop: 2 }}>R$ 509,08</div>
              <span className="pill-badge success" style={{ marginTop: 4 }}>144 municípios</span>
            </div>
            <div style={{ background: 'var(--bg-surface-alt)', padding: 16, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>MÁXIMO REGISTRADO</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>R$ 1.000,00</div>
              <span className="pill-badge info" style={{ marginTop: 4 }}>Novo Progresso (BR-163)</span>
            </div>
          </div>

          <div className="ref-card" style={{ padding: 20 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 12 }}>
              Ranking Completo de Rendimento Domiciliar per Capita (144 Municípios)
            </h3>
            <div style={{ overflowX: 'auto', maxHeight: 420 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-surface-alt)', borderBottom: '2px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Posição</th>
                    <th style={{ padding: '8px 10px' }}>Código IBGE</th>
                    <th style={{ padding: '8px 10px' }}>Município</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Rendimento Mediano per Capita</th>
                    <th style={{ padding: '8px 10px' }}>Vulnerabilidade Relativa</th>
                  </tr>
                </thead>
                <tbody>
                  {renda_ranking_144.map((r, idx) => (
                    <tr key={r.cd_mun} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{idx + 1}º</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.cd_mun}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.nm_mun}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 700, color: r.renda_pc < 350 ? 'var(--danger-red-text)' : 'var(--primary-teal)' }}>
                        R$ {r.renda_pc.toFixed(2)}
                      </td>
                      <td style={{ padding: '8px 10px' }}>
                        <span className={`pill-badge ${r.renda_pc < 350 ? 'danger' : r.renda_pc < 600 ? 'warning' : 'success'}`}>
                          {r.renda_pc < 350 ? 'Muito Alta' : r.renda_pc < 600 ? 'Média' : 'Baixa'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MulticriterioBaseView;
