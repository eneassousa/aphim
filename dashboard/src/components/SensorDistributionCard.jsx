import React from 'react';

const SENSOR_DATA = [
  { name: 'VIIRS S-NPP (375m)', pct: '62.5%', count: '1.06M' },
  { name: 'NOAA-20 VIIRS', pct: '18.5%', count: '313K' },
  { name: 'INPE Ref. (1km)', pct: '12.3%', count: '249K' },
  { name: 'Aqua/Terra MODIS', pct: '4.8%', count: '81K' },
  { name: 'Outros Sensores', pct: '1.9%', count: '32K' },
  { name: 'Satélites Geoest.', pct: '1.05%', count: '17K' },
];

const SensorDistributionCard = ({ onViewAll }) => {
  return (
    <div className="ref-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div className="ref-card-header">
        <span className="ref-card-title">Distribuição por Sensor</span>
      </div>

      <div className="ref-card-body" style={{ paddingTop: 0 }}>
        <table className="compact-list-table">
          <tbody>
            {SENSOR_DATA.map((item) => (
              <tr key={item.name}>
                <td style={{ fontWeight: 500, fontSize: '0.84rem' }}>{item.name}</td>
                <td className="text-muted" style={{ fontSize: '0.82rem' }}>{item.pct}</td>
                <td className="text-right" style={{ fontSize: '0.82rem' }}>{item.count}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <button 
          className="view-details-btn" 
          style={{ marginTop: 16 }}
          onClick={onViewAll}
        >
          View All
        </button>
      </div>
    </div>
  );
};

export default SensorDistributionCard;
