import React, { useEffect, useRef, useState, useMemo } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { MapPin, Flame, Eye, RotateCcw, Layers } from 'lucide-react';
import paraGeoData from '../data/para_municipios.json';

const CRITICAL_MUNICIPIOS = [
  { cd_mun: '1507300', name: 'São Félix do Xingu', color: '#1e3a8a', val: '202.031', pct: 93, density: '2.398 focos/km²', coords: [-7.2294, -52.2523] },
  { cd_mun: '1500602', name: 'Altamira', color: '#00a884', val: '217.269', pct: 100, density: '1.362 focos/km²', coords: [-6.4811, -53.8853] },
  { cd_mun: '1505031', name: 'Novo Progresso', color: '#f59e0b', val: '116.234', pct: 53, density: '3.045 focos/km²', coords: [-8.0605, -55.6122] },
  { cd_mun: '1505809', name: 'Portel', color: '#10b981', val: '74.594', pct: 34, density: '2.934 focos/km²', coords: [-2.5656, -50.9556] },
  { cd_mun: '1504208', name: 'Marabá', color: '#06b6d4', val: '44.576', pct: 20, density: '2.947 focos/km²', coords: [-5.63, -50.0163] }
];

const CAPITAL_BELEM = {
  name: 'Belém (Capital)',
  coords: [-1.2407, -48.4599],
  color: '#475569'
};

const TerritorialDistributionCard = ({ onViewData }) => {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geojsonLayerRef = useRef(null);
  const markersLayerRef = useRef(null);

  const [mapMode, setMapMode] = useState('critical'); // 'critical' ou 'density'
  const [selectedMun, setSelectedMun] = useState(null);

  // Função para obter cor de densidade no modo coroplético
  const getColorByDensity = (density) => {
    if (density >= 2.5) return '#dc2626'; // Vermelho intenso
    if (density >= 1.5) return '#ea580c'; // Laranja forte
    if (density >= 0.8) return '#f59e0b'; // Âmbar
    if (density >= 0.3) return '#0284c7'; // Azul céu
    return '#94a3b8';                     // Cinza suave
  };

  // Inicialização do Mapa Leaflet com OpenStreetMap
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Criar mapa Leaflet
    const map = L.map(mapContainerRef.current, {
      center: [-4.2, -52.8],
      zoom: 5,
      zoomControl: false,
      attributionControl: false
    });

    // Camada Oficial de Tiles do OpenStreetMap
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    // Adiciona controle de zoom no canto superior direito
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Camada de Marcadores
    const markersGroup = L.layerGroup().addTo(map);
    markersLayerRef.current = markersGroup;

    mapInstanceRef.current = map;

    // Garantir renderização imediata sem bordas cinzas em tela cheia
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 150);

    const handleResize = () => {
      map.invalidateSize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Atualizar polígonos GeoJSON dos 144 municípios e marcadores quando mapMode mudar
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Remover camada GeoJSON anterior se existir
    if (geojsonLayerRef.current) {
      map.removeLayer(geojsonLayerRef.current);
    }
    if (markersLayerRef.current) {
      markersLayerRef.current.clearLayers();
    }

    // Estilo dos municípios no Leaflet
    const styleFeature = (feature) => {
      const p = feature.properties;
      const isCritical = CRITICAL_MUNICIPIOS.find((c) => c.cd_mun === p.cd_mun);

      if (mapMode === 'density') {
        const fillColor = getColorByDensity(p.focos_por_km2);
        return {
          fillColor: fillColor,
          weight: 1,
          opacity: 0.8,
          color: '#ffffff',
          fillOpacity: 0.55
        };
      } else {
        return {
          fillColor: isCritical ? isCritical.color : '#64748b',
          weight: isCritical ? 2 : 0.8,
          opacity: isCritical ? 0.9 : 0.4,
          color: isCritical ? isCritical.color : '#475569',
          fillOpacity: isCritical ? 0.35 : 0.08
        };
      }
    };

    // Criar camada GeoJSON dos 144 municípios
    const geoLayer = L.geoJSON(paraGeoData, {
      style: styleFeature,
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        
        // Popup enriquecido do município
        const popupContent = `
          <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; line-height: 1.4; padding: 4px;">
            <div style="font-weight: 800; font-size: 13px; color: #0f172a; margin-bottom: 4px;">${p.nm_mun}</div>
            <div style="color: #00a884; font-weight: 700;">🔥 ${p.focos_total ? p.focos_total.toLocaleString('pt-BR') : 0} focos</div>
            <div style="color: #475569; font-size: 11px; margin-top: 2px;">Densidade: <strong>${p.focos_por_km2 || 0} focos/km²</strong></div>
            <div style="color: #94a3b8; font-size: 10px;">Área: ${p.area_km2?.toLocaleString('pt-BR')} km²</div>
          </div>
        `;
        layer.bindPopup(popupContent, { closeButton: false });

        // Efeitos de Hover
        layer.on({
          mouseover: (e) => {
            const l = e.target;
            l.setStyle({
              weight: 3,
              color: '#0f172a',
              fillOpacity: 0.65
            });
            l.bringToFront();
          },
          mouseout: (e) => {
            geoLayer.resetStyle(e.target);
          },
          click: () => {
            setSelectedMun(p.nm_mun);
          }
        });
      }
    }).addTo(map);

    geojsonLayerRef.current = geoLayer;

    // Adicionar marcadores com badges customizados nos Polos Críticos
    CRITICAL_MUNICIPIOS.forEach((crit) => {
      const icon = L.divIcon({
        className: 'custom-osm-pin',
        html: `
          <div style="
            display: flex;
            align-items: center;
            gap: 4px;
            background: #ffffff;
            border: 2px solid ${crit.color};
            border-radius: 14px;
            padding: 2px 7px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.22);
            font-size: 10.5px;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
            cursor: pointer;
          ">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: ${crit.color}; display: inline-block;"></span>
            <span>${crit.name}</span>
          </div>
        `,
        iconSize: [120, 24],
        iconAnchor: [60, 12]
      });

      const marker = L.marker(crit.coords, { icon });
      marker.bindPopup(`
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; padding: 4px;">
          <strong style="color: #0f172a; font-size: 13px;">${crit.name}</strong>
          <div style="color: ${crit.color}; font-weight: 700; margin-top: 2px;">Total: ${crit.val} focos</div>
          <div style="color: #64748b; font-size: 11px;">${crit.density}</div>
        </div>
      `);
      markersLayerRef.current.addLayer(marker);
    });

    // Marcador de Belém (Capital)
    const belemIcon = L.divIcon({
      className: 'custom-osm-pin-belem',
      html: `
        <div style="
          background: #334155;
          color: #ffffff;
          border-radius: 10px;
          padding: 2px 6px;
          font-size: 9.5px;
          font-weight: 700;
          box-shadow: 0 2px 6px rgba(0,0,0,0.2);
          white-space: nowrap;
        ">
          ★ Belém
        </div>
      `,
      iconSize: [60, 20],
      iconAnchor: [30, 10]
    });
    const belemMarker = L.marker(CAPITAL_BELEM.coords, { icon: belemIcon });
    markersLayerRef.current.addLayer(belemMarker);

    // Ajustar enquadramento do mapa para o Pará
    map.fitBounds(geoLayer.getBounds(), { padding: [10, 10] });

  }, [mapMode]);

  // Função para voar até um município quando clicado na lista lateral
  const flyToMunicipio = (mun) => {
    const map = mapInstanceRef.current;
    if (!map) return;
    setSelectedMun(mun.name);
    map.flyTo(mun.coords, 7.5, { duration: 1.2 });
  };

  // Resetar zoom para todo o Pará
  const resetMapZoom = () => {
    const map = mapInstanceRef.current;
    const geoLayer = geojsonLayerRef.current;
    if (map && geoLayer) {
      setSelectedMun(null);
      map.fitBounds(geoLayer.getBounds(), { padding: [10, 10] });
    }
  };

  return (
    <div className="ref-card">
      <div className="ref-card-header" style={{ flexWrap: 'wrap', gap: 8 }}>
        <span className="ref-card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <MapPin size={18} color="var(--primary-teal)" />
          Distribuição Espacial por Municípios Críticos (OpenStreetMap)
        </span>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Seletor de Modo: Polos vs Densidade */}
          <div className="period-pill-group">
            <button
              className={`period-pill-btn ${mapMode === 'critical' ? 'active' : ''}`}
              onClick={() => setMapMode('critical')}
              title="Exibir polos críticos com pins e nomes"
              style={{ fontSize: '0.72rem' }}
            >
              Polos Críticos
            </button>
            <button
              className={`period-pill-btn ${mapMode === 'density' ? 'active' : ''}`}
              onClick={() => setMapMode('density')}
              title="Exibir calor por densidade (focos/km²)"
              style={{ fontSize: '0.72rem' }}
            >
              Densidade (focos/km²)
            </button>
          </div>

          {/* Botão Resetar Visão do Pará */}
          <button 
            className="period-pill-btn"
            style={{ padding: '3px 8px', fontSize: '0.72rem' }}
            onClick={resetMapZoom}
            title="Enquadrar todo o Estado do Pará"
          >
            <RotateCcw size={12} style={{ display: 'inline', marginRight: 3 }} />
            <span>Ajustar UF</span>
          </button>

          {/* Botão Ver Tabela Completa */}
          <button 
            className="pagination-btn"
            style={{ padding: '3px 8px', fontSize: '0.74rem' }}
            onClick={onViewData}
            title="Ir para a tabela analítica dos 144 municípios"
          >
            <Eye size={13} />
            <span>Dados</span>
          </button>
        </div>
      </div>

      <div className="ref-card-body">
        <div className="map-split-view">
          {/* Lado Esquerdo: MAPA REAL OPENSTREETMAP VIA LEAFLET */}
          <div 
            style={{ 
              position: 'relative', 
              width: '100%', 
              height: '310px', 
              borderRadius: 8, 
              border: '1px solid #cbd5e1', 
              overflow: 'hidden',
              boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.06)'
            }}
          >
            {/* Contêiner Leaflet */}
            <div 
              ref={mapContainerRef} 
              style={{ width: '100%', height: '100%', zIndex: 1 }} 
            />

            {/* Badge discreto de OpenStreetMap no rodapé do mapa */}
            <div style={{
              position: 'absolute',
              bottom: 4,
              left: 6,
              background: 'rgba(255, 255, 255, 0.85)',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: '9px',
              color: '#475569',
              zIndex: 500,
              backdropFilter: 'blur(3px)',
              border: '1px solid #e2e8f0'
            }}>
              🗺️ OpenStreetMap • Pará (144 mun.)
            </div>
          </div>

          {/* Lado Direito: LISTA DOS MUNICÍPIOS CRÍTICOS COM INTERAÇÃO AO CLIQUE */}
          <div className="region-progress-list">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#64748b' }}>
                POLOS DE PRESSÃO (Clique para localizar)
              </span>
              <span style={{ fontSize: '0.70rem', color: '#94a3b8' }}>Focos (2019–2025)</span>
            </div>

            {CRITICAL_MUNICIPIOS.map((reg) => {
              const isSelected = selectedMun === reg.name;

              return (
                <div 
                  key={reg.name} 
                  className="region-progress-item"
                  style={{
                    padding: '5px 8px',
                    borderRadius: 6,
                    background: isSelected ? '#f1f5f9' : 'transparent',
                    border: isSelected ? '1px solid #cbd5e1' : '1px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  onClick={() => flyToMunicipio(reg)}
                  title={`Clique para dar zoom em ${reg.name} no OpenStreetMap`}
                >
                  <div className="region-item-header">
                    <span className="region-name" style={{ fontWeight: isSelected ? 800 : 600 }}>
                      <span 
                        style={{ 
                          width: 8, 
                          height: 8, 
                          borderRadius: '50%', 
                          background: reg.color, 
                          display: 'inline-block' 
                        }} 
                      />
                      {reg.name}
                    </span>
                    <span className="region-val" style={{ fontFamily: 'JetBrains Mono', fontWeight: 700 }}>
                      {reg.val}
                    </span>
                  </div>

                  <div className="progress-track" style={{ height: 5, marginTop: 4 }}>
                    <div 
                      className="progress-fill" 
                      style={{ width: `${reg.pct}%`, background: reg.color }} 
                    />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
                    <span style={{ fontSize: '0.67rem', color: '#0284c7', fontWeight: 600 }}>
                      🔍 Ver no mapa
                    </span>
                    <span style={{ fontSize: '0.67rem', color: '#64748b' }}>
                      {reg.density}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Rodapé explicativo do Pará */}
            <div style={{ 
              marginTop: 6, 
              paddingTop: 6, 
              borderTop: '1px dashed #e2e8f0', 
              fontSize: '0.71rem', 
              color: '#64748b',
              display: 'flex',
              alignItems: 'center',
              gap: 4
            }}>
              <Flame size={12} color="#f59e0b" />
              <span>Estes 5 municípios concentram mais de <strong>40%</strong> de todas as queimadas do Pará.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TerritorialDistributionCard;
