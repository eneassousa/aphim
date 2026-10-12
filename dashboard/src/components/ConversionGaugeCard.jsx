import React from 'react';

const ConversionGaugeCard = ({ onViewDetails }) => {
  // Parâmetros do arco semicircular com traços (gauge da imagem de referência)
  const totalTicks = 32;
  const activeTicks = 22; // ~68% preenchido como na imagem
  const radius = 64;
  const centerX = 95;
  const centerY = 85;

  const ticks = Array.from({ length: totalTicks }).map((_, i) => {
    // Ângulo de -180 graus (esquerda) até 0 graus (direita)
    const angleDeg = -180 + (i / (totalTicks - 1)) * 180;
    const angleRad = (angleDeg * Math.PI) / 180;
    
    const rInner = radius - 8;
    const rOuter = radius + 6;

    const x1 = centerX + rInner * Math.cos(angleRad);
    const y1 = centerY + rInner * Math.sin(angleRad);
    const x2 = centerX + rOuter * Math.cos(angleRad);
    const y2 = centerY + rOuter * Math.sin(angleRad);

    const isActive = i <= activeTicks;
    return { x1, y1, x2, y2, isActive, id: i };
  });

  return (
    <div className="ref-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div className="ref-card-header">
        <span className="ref-card-title">Validação & Aderência</span>
      </div>

      <div className="ref-card-body" style={{ paddingTop: 6 }}>
        <div className="gauge-metric-container">
          <svg className="gauge-svg" viewBox="0 0 190 105">
            {ticks.map((t) => (
              <line
                key={t.id}
                x1={t.x1}
                y1={t.y1}
                x2={t.x2}
                y2={t.y2}
                stroke={t.isActive ? 'var(--primary-teal)' : '#e2e8f0'}
                strokeWidth={3}
                strokeLinecap="round"
              />
            ))}
          </svg>
          <div className="gauge-center-val">98.2%</div>
          <div className="gauge-center-label">Consistência Estatística (r)</div>
        </div>

        <div className="gauge-comparison-row">
          <div className="gauge-comp-col">
            <span className="gauge-comp-label">NASA FIRMS</span>
            <span className="gauge-comp-val">1.695k</span>
          </div>
          <div className="gauge-comp-col">
            <span className="gauge-comp-label">INPE Ref.</span>
            <span className="gauge-comp-val">249.2k</span>
          </div>
        </div>

        <button 
          className="view-details-btn" 
          onClick={onViewDetails}
        >
          View Details
        </button>
      </div>
    </div>
  );
};

export default ConversionGaugeCard;
