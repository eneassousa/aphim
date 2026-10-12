import React from 'react';
import { 
  Flame, 
  Satellite, 
  Layers, 
  MapPin, 
  ArrowUpRight, 
  ArrowDownRight 
} from 'lucide-react';

const MetricCards = ({ summary, onSelectTab }) => {
  if (!summary) return null;

  return (
    <div className="kpi-row-4">
      {/* 1. Page View Style -> Focos NASA FIRMS */}
      <div className="ref-card kpi-ref-card">
        <div className="kpi-ref-top">
          <div className="kpi-icon-container">
            <Flame size={22} />
          </div>
          <div className="kpi-ref-labels">
            <div className="kpi-ref-title">Focos NASA FIRMS</div>
            <div className="kpi-ref-value">
              {summary.total_firms.toLocaleString('pt-BR')}
            </div>
          </div>
        </div>
        <div className="kpi-ref-bottom">
          <div className="kpi-trend positive">
            <ArrowUpRight size={14} />
            <span>+8.1%</span>
            <span style={{ color: 'var(--text-light)', fontWeight: 400 }}>vs Histórico</span>
          </div>
          <button 
            className="kpi-view-more-btn" 
            onClick={() => onSelectTab && onSelectTab('etapa2')}
          >
            View More
          </button>
        </div>
      </div>

      {/* 2. Clicks Style -> Focos INPE Referência */}
      <div className="ref-card kpi-ref-card">
        <div className="kpi-ref-top">
          <div className="kpi-icon-container">
            <Satellite size={22} />
          </div>
          <div className="kpi-ref-labels">
            <div className="kpi-ref-title">Focos INPE (Ref)</div>
            <div className="kpi-ref-value">
              {summary.total_inpe.toLocaleString('pt-BR')}
            </div>
          </div>
        </div>
        <div className="kpi-ref-bottom">
          <div className="kpi-trend positive">
            <ArrowUpRight size={14} />
            <span>+2.3%</span>
            <span style={{ color: 'var(--text-light)', fontWeight: 400 }}>Last Month</span>
          </div>
          <button 
            className="kpi-view-more-btn" 
            onClick={() => onSelectTab && onSelectTab('etapa3')}
          >
            View More
          </button>
        </div>
      </div>

      {/* 3. Conversions Style -> Correlação Pearson (r) */}
      <div className="ref-card kpi-ref-card">
        <div className="kpi-ref-top">
          <div className="kpi-icon-container">
            <Layers size={22} />
          </div>
          <div className="kpi-ref-labels">
            <div className="kpi-ref-title">Correlação Pearson (r)</div>
            <div className="kpi-ref-value">
              {summary.pearson_mensal.toString().replace('.', ',')}
            </div>
          </div>
        </div>
        <div className="kpi-ref-bottom">
          <div className="kpi-trend negative">
            <ArrowDownRight size={14} />
            <span>-0.3%</span>
            <span style={{ color: 'var(--text-light)', fontWeight: 400 }}>Desvio Relativo</span>
          </div>
          <button 
            className="kpi-view-more-btn" 
            onClick={() => onSelectTab && onSelectTab('etapa3')}
          >
            View More
          </button>
        </div>
      </div>

      {/* 4. New Users Style -> Municípios Monitorados */}
      <div className="ref-card kpi-ref-card">
        <div className="kpi-ref-top">
          <div className="kpi-icon-container">
            <MapPin size={22} />
          </div>
          <div className="kpi-ref-labels">
            <div className="kpi-ref-title">Municípios Pará</div>
            <div className="kpi-ref-value">
              {summary.municipios_total}
            </div>
          </div>
        </div>
        <div className="kpi-ref-bottom">
          <div className="kpi-trend positive">
            <ArrowUpRight size={14} />
            <span>100%</span>
            <span style={{ color: 'var(--text-light)', fontWeight: 400 }}>Cobertura IBGE</span>
          </div>
          <button 
            className="kpi-view-more-btn" 
            onClick={() => onSelectTab && onSelectTab('municipios')}
          >
            View More
          </button>
        </div>
      </div>
    </div>
  );
};

export default MetricCards;
