import React, { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Award } from 'lucide-react';

const AuditTable = ({ serieMensal = [], totaisAnuais = [] }) => {
  const [selectedYear, setSelectedYear] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 12;

  const years = useMemo(() => {
    const list = Array.from(new Set(serieMensal.map((d) => String(d.ano))));
    return ['ALL', ...list];
  }, [serieMensal]);

  const filteredData = useMemo(() => {
    if (selectedYear === 'ALL') return serieMensal;
    return serieMensal.filter((d) => String(d.ano) === selectedYear);
  }, [serieMensal, selectedYear]);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredData.slice(start, start + itemsPerPage);
  }, [filteredData, currentPage]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Tabela de Totais Anuais */}
      <div className="ref-card" style={{ padding: 0 }}>
        <div className="ref-card-header" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
          <div className="ref-card-title">
            <Award size={18} color="var(--primary-teal)" />
            <span>Totais Anuais Comparativos (Série Consolidada 2019–2025)</span>
          </div>
          <span className="pill-badge success">
            Total FIRMS: 1.695.702 | INPE: 249.210
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="data-table-clean">
            <thead>
              <tr>
                <th>Ano</th>
                <th style={{ textAlign: 'right' }}>Focos FIRMS (VIIRS)</th>
                <th style={{ textAlign: 'right' }}>Focos INPE (Referência)</th>
                <th style={{ textAlign: 'right' }}>Razão FIRMS / INPE</th>
                <th style={{ textAlign: 'right' }}>% do Total Histórico</th>
                <th>Status Metodológico</th>
              </tr>
            </thead>
            <tbody>
              {totaisAnuais.map((item) => {
                const totalFirms = 1695702;
                const pct = ((item.focos_firms / totalFirms) * 100).toFixed(1);
                const isPeakYear = item.ano === 2024;

                return (
                  <tr key={item.ano} style={isPeakYear ? { background: '#fef2f2' } : undefined}>
                    <td className="mono" style={{ fontWeight: 700, color: isPeakYear ? '#dc2626' : 'var(--text-heading)' }}>
                      {item.ano} {isPeakYear && '(Pico Máximo)'}
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 600, fontFamily: 'JetBrains Mono', color: 'var(--primary-teal)' }}>
                      {item.focos_firms.toLocaleString('pt-BR')}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono', color: 'var(--primary-blue)' }}>
                      {item.focos_inpe.toLocaleString('pt-BR')}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono' }}>
                      {item.razao.toFixed(2)}x
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono' }}>
                      {pct}%
                    </td>
                    <td>
                      <span className={`pill-badge ${isPeakYear ? 'danger' : 'info'}`}>
                        {isPeakYear ? 'Ano de Seca Histórica' : 'Padrão Consistente'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Auditoria Mensal Completa */}
      <div className="ref-card" style={{ padding: 0 }}>
        <div className="ref-card-header" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
          <div className="ref-card-title">
            <span>Série Temporal Mensal Auditada (84 Meses)</span>
            <span className="pill-badge info">Correlação Pearson r = 0,9822</span>
          </div>

          <div className="period-pill-group">
            {years.map((yr) => (
              <button
                key={yr}
                className={`period-pill-btn ${selectedYear === yr ? 'active' : ''}`}
                onClick={() => {
                  setSelectedYear(yr);
                  setCurrentPage(1);
                }}
              >
                {yr}
              </button>
            ))}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="data-table-clean">
            <thead>
              <tr>
                <th>Mês / Ano</th>
                <th style={{ textAlign: 'right' }}>Focos FIRMS</th>
                <th style={{ textAlign: 'right' }}>Focos INPE</th>
                <th style={{ textAlign: 'right' }}>Razão Escala</th>
                <th style={{ textAlign: 'center' }}>Sincronia de Pico</th>
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((row) => {
                const isPeak = row.mes === '2024-09';
                return (
                  <tr key={row.mes} style={isPeak ? { background: '#fef2f2' } : undefined}>
                    <td className="mono" style={{ fontWeight: 600 }}>{row.mes}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono', fontWeight: 600, color: 'var(--primary-teal)' }}>
                      {row.focos_firms.toLocaleString('pt-BR')}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono', color: 'var(--primary-blue)' }}>
                      {row.focos_inpe.toLocaleString('pt-BR')}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono' }}>
                      {row.razao_firms_inpe ? `${row.razao_firms_inpe.toFixed(2)}x` : '-'}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {isPeak ? (
                        <span className="pill-badge danger">Ápice da Série</span>
                      ) : (
                        <span className="pill-badge success">Alinhado</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="table-pagination-bar">
          <div>
            Exibindo página {currentPage} de {totalPages} ({filteredData.length} meses)
          </div>
          <div className="pagination-controls">
            <button
              className="pagination-btn"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft size={14} /> Anterior
            </button>
            <button
              className="pagination-btn"
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            >
              Próxima <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuditTable;
