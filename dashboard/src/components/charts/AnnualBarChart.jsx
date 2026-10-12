import React, { useMemo, useState } from 'react';
import { Group } from '@visx/group';
import { Bar, LinePath, AreaClosed } from '@visx/shape';
import { scaleBand, scaleLinear, scalePoint } from '@visx/scale';
import { AxisBottom, AxisLeft, AxisRight } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { LinearGradient } from '@visx/gradient';
import { curveMonotoneX } from '@visx/curve';
import { useTooltip, TooltipWithBounds } from '@visx/tooltip';

const formatCompact = (val) => {
  if (val >= 1000) {
    return `${(val / 1000).toFixed(0)}k`;
  }
  return String(val);
};

const AnnualBarChart = ({
  data = [],
  width = 500,
  height = 320,
  showLabels = true,
  defaultMode = 'lines', // 'lines' (padrão solicitado) ou 'bars'
  margin = { top: 32, right: 45, bottom: 42, left: 52 }
}) => {
  const [chartMode, setChartMode] = useState(defaultMode);

  const {
    showTooltip,
    hideTooltip,
    tooltipData,
    tooltipLeft = 0,
    tooltipTop = 0,
  } = useTooltip();

  // Dimensões seguras para evitar tela em branco
  const safeWidth = Math.max(width || 450, 280);
  const safeHeight = Math.max(height || 320, 260);

  const innerWidth = Math.max(0, safeWidth - margin.left - margin.right);
  const innerHeight = Math.max(0, safeHeight - margin.top - margin.bottom);

  // Escalas para modo LINHAS
  const xPointScale = useMemo(() => {
    return scalePoint({
      range: [0, innerWidth],
      domain: data.map((d) => String(d.ano)),
      padding: 0.2,
    });
  }, [innerWidth, data]);

  // Escalas para modo BARRAS
  const x0BandScale = useMemo(() => {
    return scaleBand({
      range: [0, innerWidth],
      domain: data.map((d) => String(d.ano)),
      padding: 0.28,
    });
  }, [innerWidth, data]);

  const x1BandScale = useMemo(() => {
    return scaleBand({
      range: [0, x0BandScale.bandwidth()],
      domain: ['firms', 'inpe'],
      padding: 0.12,
    });
  }, [x0BandScale]);

  const maxFirms = useMemo(() => {
    if (!data.length) return 100;
    return Math.max(...data.map((d) => d.focos_firms), 100);
  }, [data]);

  const maxInpe = useMemo(() => {
    if (!data.length) return 50;
    return Math.max(...data.map((d) => d.focos_inpe), 50);
  }, [data]);

  // Escala Y para FIRMS (Verde Esmeralda)
  const yFirmsScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxFirms * 1.25], // 25% de margem para rótulos
      nice: true,
    });
  }, [innerHeight, maxFirms]);

  // Escala Y para INPE (Azul) - no modo linhas tem eixo próprio à direita
  const yInpeScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxInpe * 1.25],
      nice: true,
    });
  }, [innerHeight, maxInpe]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Botões de alternância rápida: Linhas (solicitado) vs Barras */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
        padding: '0 4px'
      }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: '0.74rem', color: '#64748b', fontWeight: 600 }}>Formato:</span>
          <button
            onClick={() => setChartMode('lines')}
            style={{
              padding: '2px 8px',
              fontSize: '0.72rem',
              fontWeight: 600,
              borderRadius: 4,
              border: '1px solid',
              borderColor: chartMode === 'lines' ? '#00a884' : '#e2e8f0',
              background: chartMode === 'lines' ? '#e6f7f2' : '#ffffff',
              color: chartMode === 'lines' ? '#007a60' : '#64748b',
              cursor: 'pointer'
            }}
          >
            📈 Linhas Contínuas
          </button>
          <button
            onClick={() => setChartMode('bars')}
            style={{
              padding: '2px 8px',
              fontSize: '0.72rem',
              fontWeight: 600,
              borderRadius: 4,
              border: '1px solid',
              borderColor: chartMode === 'bars' ? '#00a884' : '#e2e8f0',
              background: chartMode === 'bars' ? '#e6f7f2' : '#ffffff',
              color: chartMode === 'bars' ? '#007a60' : '#64748b',
              cursor: 'pointer'
            }}
          >
            📊 Barras
          </button>
        </div>

        <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
          Total acumulado: <strong style={{ color: '#0f172a' }}>1.695.702 focos</strong>
        </div>
      </div>

      <svg width={safeWidth} height={safeHeight - 35}>
        <defs>
          <LinearGradient
            id="annual-firms-grad"
            from="#00a884"
            fromOpacity={0.25}
            to="#00a884"
            toOpacity={0.02}
          />
          <LinearGradient
            id="annual-inpe-grad"
            from="#3b82f6"
            fromOpacity={0.16}
            to="#3b82f6"
            toOpacity={0.01}
          />
        </defs>

        <Group left={margin.left} top={margin.top}>
          <GridRows
            scale={yFirmsScale}
            width={innerWidth}
            stroke="#edf2f7"
            strokeDasharray="3 3"
          />

          {/* MODO 1: LINHAS (SOLICITADO PELO USUÁRIO) */}
          {chartMode === 'lines' ? (
            <>
              {/* Área Sombreada FIRMS */}
              <AreaClosed
                data={data}
                x={(d) => xPointScale(String(d.ano))}
                y={(d) => yFirmsScale(d.focos_firms)}
                yScale={yFirmsScale}
                fill="url(#annual-firms-grad)"
                curve={curveMonotoneX}
              />

              {/* Área Sombreada INPE */}
              <AreaClosed
                data={data}
                x={(d) => xPointScale(String(d.ano))}
                y={(d) => yInpeScale(d.focos_inpe)}
                yScale={yInpeScale}
                fill="url(#annual-inpe-grad)"
                curve={curveMonotoneX}
              />

              {/* Linha INPE BDQueimadas */}
              <LinePath
                data={data}
                x={(d) => xPointScale(String(d.ano))}
                y={(d) => yInpeScale(d.focos_inpe)}
                stroke="#3b82f6"
                strokeWidth={2.5}
                strokeDasharray="4 3"
                curve={curveMonotoneX}
              />

              {/* Linha NASA FIRMS */}
              <LinePath
                data={data}
                x={(d) => xPointScale(String(d.ano))}
                y={(d) => yFirmsScale(d.focos_firms)}
                stroke="#00a884"
                strokeWidth={3}
                curve={curveMonotoneX}
              />

              {/* Pontos e Rótulos */}
              {data.map((d) => {
                const cx = xPointScale(String(d.ano));
                const firmsY = yFirmsScale(d.focos_firms);
                const inpeY = yInpeScale(d.focos_inpe);

                return (
                  <Group key={`pts-${d.ano}`}>
                    {/* Marcador INPE */}
                    <circle
                      cx={cx}
                      cy={inpeY}
                      r={4.5}
                      fill="#ffffff"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      style={{ cursor: 'pointer' }}
                      onMouseMove={() => {
                        showTooltip({
                          tooltipData: { ano: d.ano, fonte: 'INPE BDQueimadas (1km)', valor: d.focos_inpe, razao: d.razao, cor: '#3b82f6' },
                          tooltipLeft: cx + margin.left,
                          tooltipTop: inpeY + margin.top,
                        });
                      }}
                      onMouseLeave={hideTooltip}
                    />

                    {/* Marcador FIRMS */}
                    <circle
                      cx={cx}
                      cy={firmsY}
                      r={5.5}
                      fill="#00a884"
                      stroke="#ffffff"
                      strokeWidth={2}
                      style={{ cursor: 'pointer' }}
                      onMouseMove={() => {
                        showTooltip({
                          tooltipData: { ano: d.ano, fonte: 'NASA FIRMS (VIIRS 375m)', valor: d.focos_firms, razao: d.razao, cor: '#00a884' },
                          tooltipLeft: cx + margin.left,
                          tooltipTop: firmsY + margin.top,
                        });
                      }}
                      onMouseLeave={hideTooltip}
                    />

                    {/* Rótulo FIRMS */}
                    {showLabels && (
                      <g>
                        <rect
                          x={cx - 18}
                          y={firmsY - 19}
                          width={36}
                          height={14}
                          rx={3}
                          fill="#ffffff"
                          stroke="#00a884"
                          strokeWidth={0.8}
                          opacity={0.95}
                        />
                        <text
                          x={cx}
                          y={firmsY - 8}
                          textAnchor="middle"
                          fill="#007a60"
                          fontSize={9.5}
                          fontWeight={700}
                          fontFamily="JetBrains Mono"
                        >
                          {formatCompact(d.focos_firms)}
                        </text>
                      </g>
                    )}

                    {/* Rótulo INPE */}
                    {showLabels && (
                      <g>
                        <rect
                          x={cx - 16}
                          y={inpeY + 7}
                          width={32}
                          height={13}
                          rx={3}
                          fill="#ffffff"
                          stroke="#3b82f6"
                          strokeWidth={0.8}
                          opacity={0.95}
                        />
                        <text
                          x={cx}
                          y={inpeY + 17}
                          textAnchor="middle"
                          fill="#1d4ed8"
                          fontSize={9}
                          fontWeight={700}
                          fontFamily="JetBrains Mono"
                        >
                          {formatCompact(d.focos_inpe)}
                        </text>
                      </g>
                    )}
                  </Group>
                );
              })}

              {/* Eixo Y Direito para INPE no modo Linhas */}
              <AxisRight
                left={innerWidth}
                scale={yInpeScale}
                numTicks={5}
                tickFormat={(val) => `${(val / 1000).toFixed(0)}k`}
                stroke="transparent"
                tickStroke="transparent"
                tickLabelProps={() => ({
                  fill: '#3b82f6',
                  fontSize: 10,
                  fontWeight: 600,
                  fontFamily: 'JetBrains Mono',
                  textAnchor: 'start',
                  dx: '0.33em',
                  dy: '0.33em',
                })}
              />
            </>
          ) : (
            /* MODO 2: BARRAS */
            data.map((d) => {
              const anoStr = String(d.ano);
              const groupX = x0BandScale(anoStr);
              const firmsX = groupX + x1BandScale('firms');
              const inpeX = groupX + x1BandScale('inpe');
              const barW = x1BandScale.bandwidth();

              const firmsY = yFirmsScale(d.focos_firms);
              const inpeY = yFirmsScale(d.focos_inpe);

              const firmsH = Math.max(0, innerHeight - firmsY);
              const inpeH = Math.max(0, innerHeight - inpeY);

              return (
                <Group key={`bar-group-${anoStr}`}>
                  <Bar
                    x={firmsX}
                    y={firmsY}
                    width={barW}
                    height={firmsH}
                    fill="#00a884"
                    rx={3.5}
                    ry={3.5}
                    opacity={0.92}
                    onMouseMove={() => {
                      showTooltip({
                        tooltipData: { ano: d.ano, fonte: 'NASA FIRMS (VIIRS 375m)', valor: d.focos_firms, razao: d.razao, cor: '#00a884' },
                        tooltipLeft: firmsX + margin.left + barW / 2,
                        tooltipTop: firmsY + margin.top,
                      });
                    }}
                    onMouseLeave={hideTooltip}
                  />

                  {showLabels && (
                    <text
                      x={firmsX + barW / 2}
                      y={firmsY - 6}
                      textAnchor="middle"
                      fill="#007a60"
                      fontSize={9.5}
                      fontWeight={700}
                      fontFamily="JetBrains Mono"
                    >
                      {formatCompact(d.focos_firms)}
                    </text>
                  )}

                  <Bar
                    x={inpeX}
                    y={inpeY}
                    width={barW}
                    height={inpeH}
                    fill="#3b82f6"
                    rx={3.5}
                    ry={3.5}
                    opacity={0.92}
                    onMouseMove={() => {
                      showTooltip({
                        tooltipData: { ano: d.ano, fonte: 'INPE BDQueimadas (1km)', valor: d.focos_inpe, razao: d.razao, cor: '#3b82f6' },
                        tooltipLeft: inpeX + margin.left + barW / 2,
                        tooltipTop: inpeY + margin.top,
                      });
                    }}
                    onMouseLeave={hideTooltip}
                  />

                  {showLabels && (
                    <text
                      x={inpeX + barW / 2}
                      y={inpeY - 6}
                      textAnchor="middle"
                      fill="#2563eb"
                      fontSize={9.5}
                      fontWeight={700}
                      fontFamily="JetBrains Mono"
                    >
                      {formatCompact(d.focos_inpe)}
                    </text>
                  )}
                </Group>
              );
            })
          )}

          {/* Eixo X: Anos */}
          <AxisBottom
            top={innerHeight}
            scale={chartMode === 'lines' ? xPointScale : x0BandScale}
            stroke="#e2e8f0"
            tickStroke="#e2e8f0"
            tickLabelProps={() => ({
              fill: '#475569',
              fontSize: 11,
              fontWeight: 600,
              fontFamily: 'Plus Jakarta Sans',
              textAnchor: 'middle',
              dy: '0.4em',
            })}
          />

          {/* Eixo Y Esquerdo: NASA FIRMS */}
          <AxisLeft
            scale={yFirmsScale}
            numTicks={5}
            tickFormat={(val) => `${(val / 1000).toFixed(0)}k`}
            stroke="transparent"
            tickStroke="transparent"
            tickLabelProps={() => ({
              fill: '#00a884',
              fontSize: 10.5,
              fontWeight: 600,
              fontFamily: 'JetBrains Mono',
              textAnchor: 'end',
              dx: '-0.33em',
              dy: '0.33em',
            })}
          />
        </Group>
      </svg>

      {/* Legenda dos Dados */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 20,
        marginTop: 4,
        fontSize: '0.76rem',
        color: '#64748b'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 4, borderRadius: 2, background: '#00a884', display: 'inline-block' }} />
          <strong>NASA FIRMS (375m)</strong>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 4, borderRadius: 2, background: '#3b82f6', display: 'inline-block' }} />
          <strong>INPE BDQueimadas (1km)</strong>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#94a3b8', fontSize: '0.72rem' }}>
          <span>(Pico em 2024: 414.486 focos)</span>
        </div>
      </div>

      {/* Tooltip Dinâmico */}
      {tooltipData && (
        <TooltipWithBounds
          top={tooltipTop}
          left={tooltipLeft}
          className="visx-tooltip-custom"
          style={{
            zIndex: 100,
            boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
            padding: '10px 14px',
            backgroundColor: '#ffffff'
          }}
        >
          <div style={{ fontWeight: 800, color: '#1e293b', marginBottom: 4 }}>
            Ano de {tooltipData.ano}
          </div>
          <div style={{ color: tooltipData.cor, fontWeight: 700, fontSize: '0.84rem' }}>
            {tooltipData.fonte}: {tooltipData.valor.toLocaleString('pt-BR')} focos
          </div>
          <div style={{ fontSize: '0.76rem', color: '#64748b', marginTop: 4, paddingTop: 4, borderTop: '1px dashed #e2e8f0' }}>
            Razão Anual FIRMS/INPE: <strong style={{ color: '#0f172a' }}>{tooltipData.razao}x</strong>
          </div>
        </TooltipWithBounds>
      )}
    </div>
  );
};

export default AnnualBarChart;
