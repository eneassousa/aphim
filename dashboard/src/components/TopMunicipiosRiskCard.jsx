import React from 'react';

const TOP_MUNICIPIOS = [
  { path: 'São Félix do Xingu (1507300)', focos: '234.812', taxa: '2.78 focos/km²', rate: '84.2%', badgeType: 'danger' },
  { path: 'Altamira (1500602)', focos: '182.490', taxa: '1.14 focos/km²', rate: '71.5%', badgeType: 'danger' },
  { path: 'Novo Progresso (1505031)', focos: '129.504', taxa: '3.39 focos/km²', rate: '64.2%', badgeType: 'danger' },
  { path: 'Marabá (1504208)', focos: '88.210', taxa: '5.83 focos/km²', rate: '42.5%', badgeType: 'warning' },
  { path: 'Portel (1505809)', focos: '54.120', taxa: '2.14 focos/km²', rate: '22.8%', badgeType: 'warning' },
  { path: 'Paragominas (1505502)', focos: '31.405', taxa: '1.62 focos/km²', rate: '8.4%', badgeType: 'success' },
];

const TopMunicipiosRiskCard = ({ onViewAll }) => {
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
              <th style={{ textAlign: 'right' }}>Índice de Risco</th>
            </tr>
          </thead>
          <tbody>
            {TOP_MUNICIPIOS.map((mun) => (
              <tr key={mun.path}>
                <td style={{ fontWeight: 600, fontSize: '0.84rem' }}>
                  {mun.path}
                </td>
                <td style={{ textAlign: 'right', fontWeight: 600, fontSize: '0.84rem' }}>
                  {mun.focos}
                </td>
                <td style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                  {mun.taxa}
                </td>
                <td style={{ textAlign: 'right' }}>
                  <span className={`pill-badge ${mun.badgeType}`}>
                    {mun.rate}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TopMunicipiosRiskCard;
