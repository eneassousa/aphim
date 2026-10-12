import React from 'react';

const TOP_N = 6;

const formatInt = (n) => Number(n).toLocaleString('pt-BR');
const formatDec = (n, digits) =>
  Number(n).toLocaleString('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits });

// Participação no total de focos do Pará (2019–2025). Não é um índice de risco:
// o risco só existirá depois dos módulos de perigo (ML) e vulnerabilidade (AHP/ELECTRE).
const shareBadge = (share) => {
  if (share >= 10) return 'danger';
  if (share >= 3) return 'warning';
  return 'success';
};

const TopMunicipiosRiskCard = ({ ranking = [], onViewAll }) => {
  const totalFocos = ranking.reduce((acc, m) => acc + m.focos_total, 0);
  const top = [...ranking]
    .sort((a, b) => b.focos_total - a.focos_total)
    .slice(0, TOP_N);

  return (
    <div className="ref-card">
      <div className="ref-card-header">
        <span className="ref-card-title">Top Municípios & Focos Acumulados</span>
        <button
          className="pagination-btn"
          style={{ padding: '4px 10px', fontSize: '0.78rem' }}
          onClick={onViewAll}
        >
          View All
        </button>
      </div>

      <div className="ref-card-body" style={{ padding: 0 }}>
        <table className="data-table-clean">
          <thead>
            <tr>
              <th>Município / IBGE</th>
              <th style={{ textAlign: 'right' }}>Focos FIRMS</th>
              <th style={{ textAlign: 'center' }}>Densidade / km²</th>
              <th style={{ textAlign: 'right' }}>% dos Focos do PA</th>
            </tr>
          </thead>
          <tbody>
            {top.map((mun) => {
              const share = totalFocos > 0 ? (mun.focos_total / totalFocos) * 100 : 0;
              return (
                <tr key={mun.cd_mun}>
                  <td style={{ fontWeight: 600, fontSize: '0.84rem' }}>
                    {mun.nm_mun} ({mun.cd_mun})
                  </td>
                  <td style={{ textAlign: 'right', fontWeight: 600, fontSize: '0.84rem' }}>
                    {formatInt(mun.focos_total)}
                  </td>
                  <td style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                    {formatDec(mun.focos_por_km2, 2)} focos/km²
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <span className={`pill-badge ${shareBadge(share)}`}>
                      {formatDec(share, 1)}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TopMunicipiosRiskCard;
