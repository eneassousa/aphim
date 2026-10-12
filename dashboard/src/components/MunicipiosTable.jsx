import React, { useState, useMemo } from 'react';
import { Search, ArrowUpDown, ChevronLeft, ChevronRight, MapPin } from 'lucide-react';

const MunicipiosTable = ({ data = [] }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState('focos_total');
  const [sortDirection, setSortDirection] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 12;

  const filteredData = useMemo(() => {
    return data.filter((mun) => {
      const term = searchTerm.toLowerCase();
      return (
        mun.nm_mun.toLowerCase().includes(term) ||
        mun.cd_mun.includes(term)
      );
    });
  }, [data, searchTerm]);

  const sortedData = useMemo(() => {
    return [...filteredData].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (typeof valA === 'string') {
        return sortDirection === 'asc'
          ? valA.localeCompare(valB)
          : valB.localeCompare(valA);
      }

      return sortDirection === 'asc' ? valA - valB : valB - valA;
    });
  }, [filteredData, sortField, sortDirection]);

  const totalPages = Math.ceil(sortedData.length / itemsPerPage) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return sortedData.slice(start, start + itemsPerPage);
  }, [sortedData, currentPage]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1);
  };

  const maxFocos = useMemo(() => Math.max(...data.map((d) => d.focos_total), 1), [data]);

  return (
    <div className="ref-card" style={{ padding: 0 }}>
      <div className="ref-card-header" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
        <div className="ref-card-title">
          <MapPin size={18} color="var(--primary-teal)" />
          <span>Malha Municipal do Pará (144 Municípios IBGE)</span>
          <span className="pill-badge info">
            {filteredData.length} municípios
          </span>
        </div>

        <div className="search-input-wrapper" style={{ width: 250, padding: '6px 12px' }}>
          <Search size={15} color="var(--text-light)" />
          <input
            type="text"
            placeholder="Buscar município ou código..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
          />
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table-clean">
          <thead>
            <tr>
              <th style={{ width: 60 }}>Rank</th>
              <th onClick={() => handleSort('cd_mun')} style={{ cursor: 'pointer' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  Cód. IBGE
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th onClick={() => handleSort('nm_mun')} style={{ cursor: 'pointer' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  Município
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th onClick={() => handleSort('area_km2')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                  Área (km²)
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th onClick={() => handleSort('focos_total')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                  Focos FIRMS
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th style={{ width: 140 }}>Distribuição</th>
              <th onClick={() => handleSort('focos_por_km2')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
                  Focos / km²
                  <ArrowUpDown size={12} />
                </div>
              </th>
              <th style={{ textAlign: 'center' }}>Nível de Risco</th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((mun, idx) => {
              const rank = (currentPage - 1) * itemsPerPage + idx + 1;
              const barWidth = `${Math.min(100, (mun.focos_total / maxFocos) * 100)}%`;
              const isHigh = mun.focos_por_km2 > 1.5;
              const isMedium = mun.focos_por_km2 > 0.5 && !isHigh;

              return (
                <tr key={mun.cd_mun}>
                  <td style={{ color: 'var(--text-light)', fontWeight: 600 }}>#{rank}</td>
                  <td className="mono" style={{ color: 'var(--text-muted)' }}>{mun.cd_mun}</td>
                  <td style={{ fontWeight: 600 }}>{mun.nm_mun}</td>
                  <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono' }}>
                    {mun.area_km2.toLocaleString('pt-BR', { maximumFractionDigits: 1 })}
                  </td>
                  <td style={{ textAlign: 'right', fontWeight: 700, fontFamily: 'JetBrains Mono', color: 'var(--primary-teal)' }}>
                    {mun.focos_total.toLocaleString('pt-BR')}
                  </td>
                  <td>
                    <div className="progress-track" style={{ height: 5 }}>
                      <div className="progress-fill" style={{ width: barWidth, background: 'var(--primary-teal)' }} />
                    </div>
                  </td>
                  <td style={{ textAlign: 'right', fontFamily: 'JetBrains Mono' }}>
                    {mun.focos_por_km2.toFixed(3)}
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className={`pill-badge ${isHigh ? 'danger' : isMedium ? 'warning' : 'success'}`}>
                      {isHigh ? 'Crítico' : isMedium ? 'Moderado' : 'Controlado'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="table-pagination-bar">
        <div>
          Exibindo {Math.min(filteredData.length, (currentPage - 1) * itemsPerPage + 1)} a {Math.min(filteredData.length, currentPage * itemsPerPage)} de {filteredData.length} municípios
        </div>
        <div className="pagination-controls">
          <button
            className="pagination-btn"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft size={14} /> Anterior
          </button>
          <span>Página {currentPage} de {totalPages}</span>
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
  );
};

export default MunicipiosTable;
