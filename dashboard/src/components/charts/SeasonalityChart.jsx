import React, { useMemo } from 'react';
import { Group } from '@visx/group';
import { AreaClosed, LinePath } from '@visx/shape';
import { scaleLinear } from '@visx/scale';
import { AxisBottom, AxisLeft, AxisRight } from '@visx/axis';
import { GridRows } from '@visx/grid';
import { LinearGradient } from '@visx/gradient';
import { useTooltip, TooltipWithBounds } from '@visx/tooltip';
import { curveCardinal } from '@visx/curve';
import { Flame, AlertTriangle, ShieldCheck, TrendingUp } from 'lucide-react';

const MESES_NOMES = [
  'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
  'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'
];

const SeasonalityChart = ({
  data = [],
  width,
  height = 320,
  margin = { top: 32, right: 40, bottom: 40, left: 45 }
}) => {
  const {
    showTooltip,
    hideTooltip,
    tooltipData,
    tooltipLeft = 0,
    tooltipTop = 0,
  } = useTooltip();

  // Dimensões seguras para evitar tela em branco caso o container demore a calcular
  const safeWidth = Math.max(width || 450, 280);
  const safeHeight = Math.max(height || 340, 260);

  const innerWidth = Math.max(0, safeWidth - margin.left - margin.right);
  const innerHeight = Math.max(0, safeHeight - margin.top - margin.bottom);

  const xScale = useMemo(() => {
    return scaleLinear({
      range: [0, innerWidth],
      domain: [1, 12],
    });
  }, [innerWidth]);

  const maxFirms = useMemo(() => {
    if (!data.length) return 100;
    return Math.max(...data.map((d) => d.focos_firms_medio), 100);
  }, [data]);

  const maxInpe = useMemo(() => {
    if (!data.length) return 50;
    return Math.max(...data.map((d) => d.focos_inpe_medio || 0), 50);
  }, [data]);

  const yScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxFirms * 1.25], // Folga para rótulos acima do pico
      nice: true,
    });
  }, [innerHeight, maxFirms]);

  const yInpeScale = useMemo(() => {
    return scaleLinear({
      range: [innerHeight, 0],
      domain: [0, maxInpe * 1.25],
      nice: true,
    });
  }, [innerHeight, maxInpe]);

  // Total anual de focos médios para calcular % de concentração
  const totalFirmsAnual = useMemo(() => {
    return data.reduce((acc, curr) => acc + (curr.focos_firms_medio || 0), 0);
  }, [data]);

  // Delimitação da Janela Crítica de Alta (Agosto=8 a Novembro=11, com transição de Julho a Dez)
  const peakStartX = xScale(7.5); // Meio de Julho para início de Agosto
  const peakEndX = xScale(11.5);   // Fim de Novembro
  const peakWidth = Math.max(0, peakEndX - peakStartX);

  return (
    <div style={{ position: 'relative' }}>
      <svg width={safeWidth} height={safeHeight}>
        <defs>
          <LinearGradient
            id="season-teal-grad"
            from="#00a884"
            fromOpacity={0.28}
            to="#00a884"
            toOpacity={0.02}
          />
          <LinearGradient
            id="season-blue-grad"
            from="#3b82f6"
            fromOpacity={0.16}
            to="#3b82f6"
            toOpacity={0.0}
          />
          <LinearGradient
            id="peak-alert-zone"
            from="#f59e0b"
            fromOpacity={0.14}
            to="#ef4444"
            toOpacity={0.04}
          />
        </defs>

        <Group left={margin.left} top={margin.top}>
          {/* FAIXA DE DESTAQUE: JANELA CRÍTICA DE ALTA (AGOSTO A NOVEMBRO) */}
          <rect
            x={peakStartX}
            y={0}
            width={peakWidth}
            height={innerHeight}
            fill="url(#peak-alert-zone)"
            rx={6}
          />
          <line
            x1={peakStartX}
            y1={0}
            x2={peakStartX}
            y2={innerHeight}
            stroke="#f59e0b"
            strokeDasharray="4 3"
            strokeWidth={1.2}
            opacity={0.65}
          />
          <line
            x1={peakEndX}
            y1={0}
            x2={peakEndX}
            y2={innerHeight}
            stroke="#f59e0b"
            strokeDasharray="4 3"
            strokeWidth={1.2}
            opacity={0.65}
          />

          {/* Tag de Alerta no Topo da Faixa Crítica */}
          <rect
            x={peakStartX + 6}
            y={-22}
            width={Math.min(180, peakWidth - 12)}
            height={18}
            rx={4}
            fill="#fffbeb"
            stroke="#fde68a"
            strokeWidth={1}
          />
          <text
            x={peakStartX + 12}
            y={-10}
            fill="#b45309"
            fontSize={9.5}
            fontWeight={700}
            fontFamily="Plus Jakarta Sans"
          >
            🔥 Janela Crítica (Ago–Nov: 88,4%)
          </text>

          {/* Linhas de Grade Sutis */}
          <GridRows
            scale={yScale}
            width={innerWidth}
            stroke="#edf2f7"
            strokeDasharray="3 3"
          />

          {/* Curva INPE BDQueimadas (Azul de Referência) */}
          <AreaClosed
            data={data}
            x={(d) => xScale(d.mes_num)}
            y={(d) => yInpeScale(d.focos_inpe_medio || 0)}
            yScale={yInpeScale}
            strokeWidth={0}
            fill="url(#season-blue-grad)"
            curve={curveCardinal}
          />
          <LinePath
            data={data}
            x={(d) => xScale(d.mes_num)}
            y={(d) => yInpeScale(d.focos_inpe_medio || 0)}
            stroke="#3b82f6"
            strokeWidth={1.8}
            strokeDasharray="5 3"
            curve={curveCardinal}
          />

          {/* Curva FIRMS (Verde Esmeralda Principal) */}
          <AreaClosed
            data={data}
            x={(d) => xScale(d.mes_num)}
            y={(d) => yScale(d.focos_firms_medio)}
            yScale={yScale}
            strokeWidth={0}
            fill="url(#season-teal-grad)"
            curve={curveCardinal}
          />
          <LinePath
            data={data}
            x={(d) => xScale(d.mes_num)}
            y={(d) => yScale(d.focos_firms_medio)}
            stroke="#00a884"
            strokeWidth={2.8}
            curve={curveCardinal}
          />

          {/* Eixo X: Meses */}
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            numTicks={12}
            tickFormat={(val) => MESES_NOMES[val - 1]}
            stroke="#e2e8f0"
            tickStroke="#e2e8f0"
            tickLabelProps={(val) => {
              const isHigh = val >= 8 && val <= 11;
              return {
                fill: isHigh ? '#b45309' : '#64748b',
                fontWeight: isHigh ? 700 : 500,
                fontSize: 10.5,
                fontFamily: 'Plus Jakarta Sans',
                textAnchor: 'middle',
                dy: '0.4em',
              };
            }}
          />

          {/* Eixo Y Esquerdo: NASA FIRMS */}
          <AxisLeft
            scale={yScale}
            numTicks={5}
            tickFormat={(val) => `${(val / 1000).toFixed(0)}k`}
            stroke="transparent"
            tickStroke="transparent"
            tickLabelProps={() => ({
              fill: '#00a884',
              fontSize: 10,
              fontWeight: 600,
              fontFamily: 'JetBrains Mono',
              textAnchor: 'end',
              dx: '-0.33em',
              dy: '0.33em',
            })}
          />

          {/* Eixo Y Direito: INPE */}
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

          {/* PONTOS & RÓTULOS DE DADOS NO PERÍODO DE ALTA */}
          {data.map((d) => {
            const isHigh = d.mes_num >= 8 && d.mes_num <= 11;
            const isPeak = d.mes_num === 9; // Setembro é o pico máximo
            const cx = xScale(d.mes_num);
            const cy = yScale(d.focos_firms_medio);

            return (
              <g key={`season-pt-${d.mes_num}`}>
                {/* Ponto na curva INPE */}
                <circle
                  cx={cx}
                  cy={yInpeScale(d.focos_inpe_medio || 0)}
                  r={isHigh ? 3 : 2}
                  fill="#ffffff"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                />

                {/* Ponto na curva FIRMS */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isPeak ? 5.5 : isHigh ? 4.5 : 3.5}
                  fill={isPeak ? '#ef4444' : isHigh ? '#f59e0b' : '#ffffff'}
                  stroke={isPeak ? '#ffffff' : '#00a884'}
                  strokeWidth={2}
                  style={{ cursor: 'pointer' }}
                  onMouseMove={() => {
                    showTooltip({
                      tooltipData: d,
                      tooltipLeft: cx + margin.left,
                      tooltipTop: cy + margin.top,
                    });
                  }}
                  onMouseLeave={hideTooltip}
                />

                {/* RÓTULOS DE DADOS NO PERÍODO DE ALTA (Ago, Set, Out, Nov) */}
                {isHigh && (
                  <g>
                    {/* Badge destacado no Pico (Setembro) */}
                    {isPeak ? (
                      <g transform={`translate(${cx}, ${cy - 12})`}>
                        <rect
                          x={-34}
                          y={-18}
                          width={68}
                          height={17}
                          rx={4}
                          fill="#ef4444"
                          filter="drop-shadow(0 2px 4px rgba(239, 68, 68, 0.25))"
                        />
                        <text
                          x={0}
                          y={-6}
                          textAnchor="middle"
                          fill="#ffffff"
                          fontSize={9.5}
                          fontWeight={800}
                          fontFamily="JetBrains Mono"
                        >
                          ★ 70.8k
                        </text>
                      </g>
                    ) : (
                      /* Rótulos para Agosto, Outubro e Novembro */
                      <g transform={`translate(${cx}, ${cy - 8})`}>
                        <rect
                          x={-22}
                          y={-14}
                          width={44}
                          height={14}
                          rx={3}
                          fill="#ffffff"
                          stroke="#e2e8f0"
                          strokeWidth={1}
                        />
                        <text
                          x={0}
                          y={-3.5}
                          textAnchor="middle"
                          fill="#1e293b"
                          fontSize={9}
                          fontWeight={700}
                          fontFamily="JetBrains Mono"
                        >
                          {(d.focos_firms_medio / 1000).toFixed(1)}k
                        </text>
                      </g>
                    )}
                  </g>
                )}
              </g>
            );
          })}
        </Group>
      </svg>

      {/* MINI PAINEL DE INFORMAÇÕES DA ALTA TEMPORADA */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 10,
        marginTop: 12,
        padding: '10px 14px',
        background: '#f8fafc',
        borderRadius: 8,
        border: '1px solid #edf2f7'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: '#fee2e2', color: '#ef4444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Flame size={15} />
          </div>
          <div>
            <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600 }}>PICO MÁXIMO</div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#1e293b' }}>
              Setembro <span style={{ color: '#00a884', fontFamily: 'JetBrains Mono' }}>(70.8k focos)</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: '#fef3c7', color: '#d97706', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <AlertTriangle size={15} />
          </div>
          <div>
            <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600 }}>JANELA CRÍTICA (AGO–NOV)</div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#b45309' }}>
              88,4% <span style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 500 }}>do fogo anual</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: '#e0f2fe', color: '#0284c7', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShieldCheck size={15} />
          </div>
          <div>
            <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600 }}>PERÍODO CHUVOSO (JAN–JUN)</div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0369a1' }}>
              Apenas 2,7% <span style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 500 }}>do total</span>
            </div>
          </div>
        </div>
      </div>

      {/* Legenda Informativa */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 16,
        marginTop: 10,
        fontSize: '0.75rem',
        color: '#64748b'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#00a884', display: 'inline-block' }} />
          <span>NASA FIRMS (375m) Média</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#3b82f6', display: 'inline-block' }} />
          <span>INPE BDQueimadas (1km) Média</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f59e0b', opacity: 0.5, display: 'inline-block' }} />
          <span>Faixa de Alta Severidade</span>
        </div>
      </div>

      {/* Tooltip Aprimorado com Informações Sazonais */}
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
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginBottom: 6 }}>
            <span style={{ fontWeight: 800, color: '#1e293b', fontSize: '0.88rem' }}>
              {MESES_NOMES[tooltipData.mes_num - 1]}
            </span>
            <span style={{
              fontSize: '0.68rem',
              fontWeight: 700,
              padding: '2px 6px',
              borderRadius: 4,
              backgroundColor: tooltipData.mes_num >= 8 && tooltipData.mes_num <= 11 ? '#fef2f2' : '#f0fdf4',
              color: tooltipData.mes_num >= 8 && tooltipData.mes_num <= 11 ? '#dc2626' : '#16a34a'
            }}>
              {tooltipData.mes_num === 9 ? '🔥 PICO HISTÓRICO' : tooltipData.mes_num >= 8 && tooltipData.mes_num <= 11 ? '⚠️ ALTA SEVERIDADE' : '🌧️ PERÍODO ÚMIDO'}
            </span>
          </div>

          <div style={{ color: '#00a884', fontWeight: 600, fontSize: '0.82rem', display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>Média FIRMS:</span>
            <span style={{ fontFamily: 'JetBrains Mono' }}>
              {Math.round(tooltipData.focos_firms_medio).toLocaleString('pt-BR')} focos
            </span>
          </div>

          <div style={{ color: '#3b82f6', fontWeight: 600, fontSize: '0.82rem', display: 'flex', justifyContent: 'space-between', gap: 12, marginTop: 2 }}>
            <span>Média INPE:</span>
            <span style={{ fontFamily: 'JetBrains Mono' }}>
              {Math.round(tooltipData.focos_inpe_medio || 0).toLocaleString('pt-BR')} focos
            </span>
          </div>

          <div style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: '1px dashed #e2e8f0',
            fontSize: '0.74rem',
            color: '#64748b',
            display: 'flex',
            justifyContent: 'space-between'
          }}>
            <span>Representatividade Anual:</span>
            <strong style={{ color: '#0f172a' }}>
              {((tooltipData.focos_firms_medio / totalFirmsAnual) * 100).toFixed(1)}% do total
            </strong>
          </div>
        </TooltipWithBounds>
      )}
    </div>
  );
};

export default SeasonalityChart;
