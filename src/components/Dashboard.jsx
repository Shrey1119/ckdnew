import React from 'react';

export default function Dashboard({ onGoUpload }) {
  return (
    <div>
      {/* Hero Overview */}
      <div className="hero-banner">
        <div className="hero-content">
          <div className="badge-pill">
            <span className="pulse-dot"></span> ResNet-50 & KNN Multimodal Pipeline
          </div>
          <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>KidneyAI Clinical Portal</h1>
          <p style={{ color: 'var(--text-sub)', fontSize: '0.95rem', maxWidth: '580px' }}>
            Advanced deep learning diagnostic system for CT Kidney imaging (Normal, Cyst, Tumor, Stone) and patient blood biomarkers.
          </p>
          <button className="btn-primary" onClick={onGoUpload} style={{ marginTop: '1.2rem' }}>
            <i className="fa-solid fa-cloud-arrow-up"></i> Upload CT Scan Image
          </button>
        </div>

        <div className="hero-stats">
          <div className="stat-card">
            <div className="stat-value">99.67%</div>
            <div className="stat-label">Accuracy</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">100.0%</div>
            <div className="stat-label">Precision</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">1,500</div>
            <div className="stat-label">Dataset Scans</div>
          </div>
        </div>
      </div>

      {/* Audit Metrics */}
      <div className="card">
        <div className="card-title">
          <i className="fa-solid fa-chart-pie icon-accent"></i> Phase 2 Model Audit Metrics
        </div>

        <div className="metrics-grid" style={{ marginTop: '1rem' }}>
          <div className="metric-card">
            <span className="m-title">Model Accuracy</span>
            <span className="m-val" style={{ color: 'var(--emerald)' }}>99.67%</span>
            <span className="m-sub">Validated on test set</span>
          </div>
          <div className="metric-card">
            <span className="m-title">Precision</span>
            <span className="m-val" style={{ color: 'var(--emerald)' }}>100.0%</span>
            <span className="m-sub">Zero false positive rate</span>
          </div>
          <div className="metric-card">
            <span className="m-title">Recall</span>
            <span className="m-val" style={{ color: 'var(--cyan)' }}>98.82%</span>
            <span className="m-sub">Sensitivity score</span>
          </div>
          <div className="metric-card">
            <span className="m-title">Specificity</span>
            <span className="m-val" style={{ color: 'var(--emerald)' }}>100.0%</span>
            <span className="m-sub">Healthy class accuracy</span>
          </div>
        </div>
      </div>
    </div>
  );
}
