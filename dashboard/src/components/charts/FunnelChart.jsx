import React, { useMemo } from 'react';
import { Group } from '@visx/group';
import { Bar } from '@visx/shape';
import { scaleBand, scaleLinear } from '@visx/scale';
import { AxisLeft } from '@visx/axis';

const FunnelChart = ({
  data = [],
  width,
  height = 240,
  margin = { top: 15, right: 90, bottom: 20, left: 190 }
}) => {
  const innerWidth = Math.max(0, width - margin.left - margin.right);
  const innerHeight = Math.max(0, height - margin.top - margin.bottom);

  const yScale = useMemo(() => {
    return scaleBand({
      range: [0, innerHeight],
      domain: data.map((d) => d.etapa),
      padding: 0.35,
    });
  }, [innerHeight, data]);

  const maxVal = useMemo(() => Math.max(...data.map((d) => d.focos), 100), [data]);

  const xScale = useMemo(() => {
    return scaleLinear({
      range: [0, innerWidth],
      domain: [0, maxVal * 1.05],
      nice: true,
    });
  }, [innerWidth, maxVal]);

  if (width < 100 || height < 100) return null;

  return (
    <svg width={width} height={height}>
      <Group left={margin.left} top={margin.top}>
        {/* Track de fundo */}
        {data.map((d) => {
          const y = yScale(d.etapa);
          const barHeight = yScale.bandwidth();
          return (
            <Bar
              key={`bg-${d.etapa}`}
              x={0}
              y={y}
              width={innerWidth}
              height={barHeight}
              fill="#f1f5f9"
              rx={4}
            />
          );
        })}

        {/* Barras de dados limpas */}
        {data.map((d, i) => {
          const y = yScale(d.etapa);
          const barWidth = xScale(d.focos);
          const barHeight = yScale.bandwidth();
          const color = i === 0 ? '#64748b' : '#00a884';

          return (
            <Group key={`bar-${d.etapa}`}>
              <Bar
                x={0}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={color}
                rx={4}
              />
              <text
                x={barWidth + 8}
                y={y + barHeight / 2}
                dy="0.35em"
                fontSize={11}
                fontWeight={600}
                fill="#334155"
                fontFamily="JetBrains Mono"
              >
                {d.focos.toLocaleString('pt-BR')} ({d.pct_retida}%)
              </text>
            </Group>
          );
        })}

        <AxisLeft
          scale={yScale}
          stroke="transparent"
          tickStroke="transparent"
          tickLabelProps={() => ({
            fill: '#475569',
            fontSize: 11,
            fontWeight: 500,
            fontFamily: 'Plus Jakarta Sans',
            textAnchor: 'end',
            dx: '-0.5em',
            dy: '0.33em',
          })}
        />
      </Group>
    </svg>
  );
};

export default FunnelChart;
