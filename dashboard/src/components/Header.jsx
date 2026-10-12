import React, { useState } from 'react';
import { 
  Search, 
  Moon, 
  Sun, 
  Maximize2, 
  Bell, 
  Settings, 
  TrendingUp, 
  MapPin, 
  Satellite, 
  FileCheck, 
  Table2, 
  Layers,
  GraduationCap,
  Database
} from 'lucide-react';

const Header = ({ activeTab, setActiveTab, onSearch }) => {
  const [darkMode, setDarkMode] = useState(false);
  const [searchInput, setSearchInput] = useState('');

  const toggleDarkMode = () => {
    const next = !darkMode;
    setDarkMode(next);
    if (next) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  return (
    <div style={{ marginBottom: 24 }}>
      {/* Barra de Busca e Utilidades do Topo (Referência Exata da Imagem) */}
      <div className="top-navbar">
        <div className="search-input-wrapper">
          <Search size={16} color="var(--text-light)" />
          <input 
            type="text" 
            placeholder="Search..." 
            value={searchInput}
            onChange={(e) => {
              setSearchInput(e.target.value);
              if (onSearch) onSearch(e.target.value);
            }}
          />
        </div>

        <div className="top-navbar-actions">
          <button 
            className="action-icon-btn" 
            title={darkMode ? "Modo Claro" : "Modo Escuro"}
            onClick={toggleDarkMode}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          <button 
            className="action-icon-btn" 
            title="Tela Cheia"
            onClick={toggleFullscreen}
          >
            <Maximize2 size={17} />
          </button>

          <button className="action-icon-btn" title="Notificações Científicas">
            <Bell size={18} />
            <span className="badge-counter">5</span>
          </button>

          <button className="action-icon-btn" title="Configurações">
            <Settings size={18} />
          </button>

          <div 
            className="user-profile-avatar" 
            title="Pesquisador PPGCC / UFPA - Enéas Sousa"
          >
            PA
          </div>
        </div>
      </div>

      {/* Título Acadêmico & Abas de Navegação */}
      <div className="academic-header-bar">
        <div className="academic-info">
          <h1>
            <GraduationCap size={24} color="var(--primary-teal)" />
            APHIM Platform — Dashboard Científico de Focos de Queimadas
          </h1>
          <p>
            Dissertação de Mestrado PPGCC / UFPA | Pará (144 Municípios | 2019–2025 | 1.695.702 detecções validadas)
          </p>
        </div>

        <nav className="header-tabs" aria-label="Navegação da Plataforma">
          <button
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <TrendingUp size={15} />
            Visão Geral
          </button>
          <button
            className={`tab-button ${activeTab === 'etapa1' ? 'active' : ''}`}
            onClick={() => setActiveTab('etapa1')}
          >
            <MapPin size={15} />
            01. IBGE & Malha
          </button>
          <button
            className={`tab-button ${activeTab === 'etapa2' ? 'active' : ''}`}
            onClick={() => setActiveTab('etapa2')}
          >
            <Satellite size={15} />
            02. FIRMS (Funil)
          </button>
          <button
            className={`tab-button ${activeTab === 'etapa3' ? 'active' : ''}`}
            onClick={() => setActiveTab('etapa3')}
          >
            <FileCheck size={15} />
            03. Auditoria INPE
          </button>
          <button
            className={`tab-button ${activeTab === 'multicriterio' ? 'active' : ''}`}
            onClick={() => setActiveTab('multicriterio')}
          >
            <Database size={15} />
            Base Integrada (44 Vars)
          </button>
          <button
            className={`tab-button ${activeTab === 'municipios' ? 'active' : ''}`}
            onClick={() => setActiveTab('municipios')}
          >
            <Table2 size={15} />
            144 Municípios
          </button>
          <button
            className={`tab-button ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            <Layers size={15} />
            Pipeline APHIM
          </button>
        </nav>
      </div>
    </div>
  );
};

export default Header;
