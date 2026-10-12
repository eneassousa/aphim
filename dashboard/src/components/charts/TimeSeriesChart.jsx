import React, { useMemo, useCallback } from 'react';
import { Group } from '@visx/group';
import { AreaClosed, LinePath, Bar, Line } from '@visx/shape';
import { scaleTime, scaleLinear } from '@visx/scale';
import { AxisBottom, AxisLeft, AxisRight } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { LinearGradient } from '@visx/gradient';
import { useTooltip, TooltipWithBounds } from '@visx/tooltip';
import { localPoint } from '@visx/event';
import { curveMonotoneX } from '@visx/curve';

const getDate = (d) => new Date(`${d.mes}-01T00:00:00`);
const getFirms = (d) => d.focos_firms;
const getInpe = (d) => d.focos_inpe;

const TimeSeriesChart = ({
  data = [],
  width,
  height,
  chartType = 'lines', // 'lines' ou 'lines-only'
  margin = { top: 20, right: 45, bottom: 35, left: 45 }
}) => {
  const {
    showTooltip,
    hideTooltip,
    tooltipData,
    tooltipLeft = 0,
    tooltipTop = 0,
  } = useTooltip();

  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const displayData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data;
  }, [data]);

  const dateScale = useMemo(() => {
    if (!displayData.length) return null;
    return scaleTime({
      range: [0, innerWidth],
      domain: [getDate(displayData[0]), getDate(displayData[displayData.length - 1])],
    });
  }, [innerWidth, displayData]);

  const maxFirms = useMemo(() => {
    if (!displayData.length) return 100;
    return Math.max(...displayData.map(getFirms), 100);
  }, [displayData]);

  const maxInpe = useMemo(() => {
    if (!displayData.length) return 50;
    return Math.max(...displayData.map(getInpe), 50);
  }, [displayData]);

  // Escala para FIRMS (Eixo Esquerdo - Verde)
  const firmsScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxFirms * 1.12],
      nice: true,
    });
  }, [innerHeight, maxFirms]);

  // Escala para INPE (Eixo Direito - Azul)
  const inpeScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxInpe * 1.12],
      nice: true,
    });
  }, [innerHeight, maxInpe]);

  const handleTooltip = useCallback(
    (event) => {
      const { x } = localPoint(event) || { x: 0 };
      const adjustedX = x - margin.left;
      if (!dateScale || !displayData.length) return;
      const x0 = dateScale.invert(adjustedX);
      
      let closest = displayData[0];
      let minDiff = Infinity;
      displayData.forEach((d) => {
        const diff = Math.abs(getDate(d) - x0);
        if (diff < minDiff) {
          minDiff = diff;
          closest = d;
        }
      });

      if (closest) {
        showTooltip({
          tooltipData: closest,
          tooltipLeft: dateScale(getDate(closest)) + margin.left,
          tooltipTop: firmsScale(getFirms(closest)) + margin.top,
        });
      }
    },
    [dateScale, displayData, firmsScale, margin, showTooltip]
  );

  if (width < 100 || height < 100 || !dateScale) return null;

  const showArea = chartType !== 'lines-only';

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height}>
        <defs>
          {/* Gradiente Verde Esmeralda (FIRMS) */}
          <LinearGradient
            id="teal-line-gradient"
            from="#00a884"
            fromOpacity={0.28}
            to="#00a884"
            toOpacity={0.01}
          />
          {/* Gradiente Azul Cobalto (INPE) */}
          <LinearGradient
            id="blue-line-gradient"
            from="#3b82f6"
            fromOpacity={0.22}
            to="#3b82f6"
            toOpacity={0.01}
          />
        </defs>

        <Group left={margin.left} top={margin.top}>
          {/* Linhas de Grade Sutis */}
          <GridRows
            scale={firmsScale}
            width={innerWidth}
            stroke="#edf2f7"
            strokeDasharray="3 3"
          />

          {/* Áreas com Gradiente Translúcido */}
          {showArea && (
            <>
              {/* Área FIRMS */}
              <AreaClosed
                data={displayData}
                x={(d) => dateScale(getDate(d))}
                y={(d) => firmsScale(getFirms(d))}
                yScale={firmsScale}
                strokeWidth={0}
                fill="url(#teal-line-gradient)"
                curve={curveMonotoneX}
              />
              {/* Área INPE */}
              <AreaClosed
                data={displayData}
                x={(d) => dateScale(getDate(d))}
                y={(d) => inpeScale(getInpe(d))}
                yScale={inpeScale}
                strokeWidth={0}
                fill="url(#blue-line-gradient)"
                curve={curveMonotoneX}
              />
            </>
          )}

          {/* LINHA 1: NASA FIRMS (VIIRS 375m) - Verde Esmeralda */}
          <LinePath
            data={displayData}
            x={(d) => dateScale(getDate(d))}
            y={(d) => firmsScale(getFirms(d))}
            stroke="#00a884"
            strokeWidth={2.8}
            curve={curveMonotoneX}
          />

          {/* LINHA 2: INPE BDQueimadas (Satélite Referência 1km) - Azul Cobalto */}
          <LinePath
            data={displayData}
            x={(d) => dateScale(getDate(d))}
            y={(d) => inpeScale(getInpe(d))}
            stroke="#3b82f6"
            strokeWidth={2.4}
            strokeDasharray={chartType === 'lines-only' ? '5 3' : undefined}
            curve={curveMonotoneX}
          />

          {/* Indicadores de Marcadores / Pontos em Série Curta ou Filtro Selecionado */}
          {displayData.length <= 24 &&
            displayData.map((d) => (
              <g key={`point-${d.mes}`}>
                <circle
                  cx={dateScale(getDate(d))}
                  cy={firmsScale(getFirms(d))}
                  r={3.5}
                  fill="#ffffff"
                  stroke="#00a884"
                  strokeWidth={2}
                />
                <circle
                  cx={dateScale(getDate(d))}
                  cy={inpeScale(getInpe(d))}
                  r={3}
                  fill="#ffffff"
                  stroke="#3b82f6"
                  strokeWidth={2}
                />
              </g>
            ))}

          {/* Linha Guia Vertical (Crosshair) no Mês Hovered */}
          {tooltipData && (
            <g>
              <Line
                from={{ x: dateScale(getDate(tooltipData)), y: 0 }}
                to={{ x: dateScale(getDate(tooltipData)), y: innerHeight }}
                stroke="#94a3b8"
                strokeWidth={1.2}
                strokeDasharray="4 4"
                pointerEvents="none"
              />
              {/* Ponto Destaque FIRMS */}
              <circle
                cx={dateScale(getDate(tooltipData))}
                cy={firmsScale(getFirms(tooltipData))}
                r={6}
                fill="#00a884"
                stroke="#ffffff"
                strokeWidth={2.5}
                pointerEvents="none"
              />
              {/* Ponto Destaque INPE */}
              <circle
                cx={dateScale(getDate(tooltipData))}
                cy={inpeScale(getInpe(tooltipData))}
                r={5.5}
                fill="#3b82f6"
                stroke="#ffffff"
                strokeWidth={2.5}
                pointerEvents="none"
              />
            </g>
          )}

          {/* Eixo X com formatação de Mês e Ano */}
          <AxisBottom
            top={innerHeight}
            scale={dateScale}
            numTicks={width > 600 ? 9 : 5}
            tickFormat={(val) => {
              const d = new Date(val);
              const m = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][d.getMonth()];
              const yr = String(d.getFullYear()).slice(2);
              return displayData.length > 24 ? `${m}/${yr}` : `${m}`;
            }}
            stroke="#e2e8f0"
            tickStroke="#e2e8f0"
            tickLabelProps={() => ({
              fill: '#64748b',
              fontSize: 10.5,
              fontFamily: 'Plus Jakarta Sans',
              textAnchor: 'middle',
              dy: '0.4em',
            })}
          />

          {/* Eixo Y Esquerdo (FIRMS - Escala Verde Esmeralda) */}
          <AxisLeft
            scale={firmsScale}
            numTicks={5}
            tickFormat={(val) => {
              return val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val;
            }}
            stroke="transparent"
            tickStroke="transparent"
            tickLabelProps={() => ({
              fill: '#00a884',
              fontSize: 10.5,
              fontWeight: 600,
              fontFamily: 'JetBrains Mono',
              textAnchor: 'end',
              dx: '-0.4em',
              dy: '0.33em',
            })}
          />

          {/* Eixo Y Direito (INPE - Escala Azul) */}
          <AxisRight
            left={innerWidth}
            scale={inpeScale}
            numTicks={5}
            tickFormat={(val) => {
              return val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val;
            }}
            stroke="transparent"
            tickStroke="transparent"
            tickLabelProps={() => ({
              fill: '#3b82f6',
              fontSize: 10.5,
              fontWeight: 600,
              fontFamily: 'JetBrains Mono',
              textAnchor: 'start',
              dx: '0.4em',
              dy: '0.33em',
            })}
          />

          {/* Hover Capture Bar Transparente */}
          <Bar
            x={0}
            y={0}
            width={innerWidth}
            height={innerHeight}
            fill="transparent"
            onTouchStart={handleTooltip}
            onTouchMove={handleTooltip}
            onMouseMove={handleTooltip}
            onMouseLeave={hideTooltip}
          />
        </Group>
      </svg>

      {/* Tooltip Limpo e Comparativo com Crosshair */}
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
          <div style={{ fontWeight: 700, marginBottom: 6, color: '#1e293b', fontSize: '0.86rem' }}>
            Mês: {tooltipData.mes}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, color: '#00a884', fontWeight: 600, fontSize: '0.84rem' }}>
            <span>● NASA FIRMS (VIIRS 375m):</span>
            <span style={{ fontFamily: 'JetBrains Mono' }}>{tooltipData.focos_firms.toLocaleString('pt-BR')} focos</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 14, color: '#3b82f6', fontWeight: 600, fontSize: '0.84rem', marginTop: 2 }}>
            <span>● INPE BDQueimadas (1km):</span>
            <span style={{ fontFamily: 'JetBrains Mono' }}>{tooltipData.focos_inpe.toLocaleString('pt-BR')} focos</span>
          </div>
          <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px dashed #e2e8f0', fontSize: '0.76rem', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
            <span>Razão de Sensibilidade:</span>
            <strong style={{ color: '#0f172a' }}>{(tooltipData.focos_firms / Math.max(1, tooltipData.focos_inpe)).toFixed(2)}x</strong>
          </div>
        </TooltipWithBounds>
      )}
    </div>
  );
};

export default TimeSeriesChart;
