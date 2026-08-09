import React from 'react';

export default function Navbar({ activeTab, setActiveTab, theme, toggleTheme }) {
  return (
    <header className="navbar">
      <div className="nav-inner">
        <div className="nav-brand" onClick={() => setActiveTab('dashboard')}>
          <div className="brand-icon">
            <i className="fa-solid fa-heart-pulse"></i>
          </div>
          <div className="brand-text">
            Kidney<span>AI</span>
          </div>
        </div>

        <nav className="nav-items">
          <button
            className={`nav-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <i className="fa-solid fa-cloud-arrow-up"></i> Image Upload
          </button>
          
          <button
            className={`nav-btn ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            <i className="fa-solid fa-x-ray"></i> Image Results
          </button>

          <button
            className={`nav-btn ${activeTab === 'clinical' ? 'active' : ''}`}
            onClick={() => setActiveTab('clinical')}
          >
            <i className="fa-solid fa-vial"></i> Clinical Predictor
          </button>

          <button
            className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <i className="fa-solid fa-chart-line"></i> Dashboard
          </button>
        </nav>

        <button className="theme-toggle-btn" onClick={toggleTheme} title="Toggle Theme">
          <i className={`fa-solid ${theme === 'dark' ? 'fa-sun' : 'fa-moon'}`}></i>
        </button>
      </div>
    </header>
  );
}
