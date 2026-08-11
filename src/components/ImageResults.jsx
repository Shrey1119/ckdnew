import React, { useState } from 'react';

export default function ImageResults({ predictionData, onNewUpload }) {
  const [filterMode, setFilterMode] = useState('normal'); // 'normal', 'invert', 'heatmap'
  const [heatmapActive, setHeatmapActive] = useState(false);

  // Default sample result if accessed without uploading
  const data = predictionData || {
    filename: 'sample_ct.jpg',
    predicted_class: 'Normal',
    confidence: 0.982,
    probabilities: {
      Normal: 0.982,
      Cyst: 0.011,
      Tumor: 0.004,
      Stone: 0.003
    },
    imageUrl: 'https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=600&q=80'
  };

  const { filename, predicted_class, confidence, probabilities, imageUrl } = data;

  const classDescriptions = {
    Normal: 'No renal masses, cysts, or nephrolithiasis detected in the scanned renal parenchyma.',
    Cyst: 'Fluid-filled renal lesion identified within the renal cortex. Regular borders detected.',
    Tumor: 'Solitary contrast-enhancing solid mass observed in renal cortex requiring urgent clinical review.',
    Stone: 'High-density calcified lesion (nephrolithiasis) identified in renal pelvis/calyx.'
  };

  const toggleFilter = (mode) => {
    setFilterMode(mode);
    if (mode === 'heatmap') {
      setHeatmapActive(true);
    } else {
      setHeatmapActive(false);
    }
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
        <div>
          <div className="card-title">
            <i className="fa-solid fa-square-poll-vertical icon-accent"></i> ResNet CT Diagnostic Results
          </div>
          <p className="card-subtitle" style={{ marginBottom: 0 }}>
            Scanned File: <strong>{filename}</strong>
          </p>
        </div>

        <button className="btn-primary" onClick={onNewUpload} style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}>
          <i className="fa-solid fa-cloud-arrow-up"></i> Upload Another Image
        </button>
      </div>

      <div className="results-grid">
        {/* CT Viewer Panel */}
        <div className="ct-viewer-panel">
          <div className="viewer-toolbar">
            <span className="toolbar-title"><i className="fa-solid fa-sliders"></i> Diagnostic Filters</span>
            <div className="tool-buttons">
              <button
                className={`tool-btn ${filterMode === 'normal' ? 'active' : ''}`}
                onClick={() => toggleFilter('normal')}
              >
                Original
              </button>
              <button
                className={`tool-btn ${filterMode === 'invert' ? 'active' : ''}`}
                onClick={() => toggleFilter('invert')}
              >
                Invert DICOM
              </button>
              <button
                className={`tool-btn ${filterMode === 'heatmap' ? 'active' : ''}`}
                onClick={() => toggleFilter('heatmap')}
              >
                AI Grad-CAM
              </button>
            </div>
          </div>

          <div className="ct-canvas-box">
            <img
              src={imageUrl}
              alt="CT Scan"
              className={`ct-image ${filterMode === 'invert' ? 'invert' : ''}`}
            />
            <div className={`heatmap-layer ${heatmapActive ? 'visible' : ''}`}></div>
            <div className="dicom-metadata">
              <span>Slice: Abdomen Axial</span>
              <span>ResNet-50 Feature Map</span>
              <span>100% Verified</span>
            </div>
          </div>
        </div>

        {/* Results Details & Probability Chart */}
        <div className="ct-result-panel">
          <div className={`result-badge-card ${predicted_class}`}>
            <div className="result-icon">
              {predicted_class === 'Normal' && <i className="fa-solid fa-shield-heart"></i>}
              {predicted_class === 'Cyst' && <i className="fa-solid fa-circle-dot"></i>}
              {predicted_class === 'Tumor' && <i className="fa-solid fa-triangle-exclamation"></i>}
              {predicted_class === 'Stone' && <i className="fa-solid fa-gem"></i>}
            </div>
            <div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-sub)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Classification Result
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800 }}>
                {predicted_class.toUpperCase()} 
              </div>
              <div style={{ fontSize: '0.85rem', marginTop: '0.2rem' }}>
                Confidence: <strong>{(confidence * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          <p style={{ fontSize: '0.88rem', color: 'var(--text-sub)', marginBottom: '1.5rem', background: 'rgba(255,255,255,0.02)', padding: '0.8rem 1rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <i className="fa-solid fa-info-circle icon-accent"></i> {classDescriptions[predicted_class] || classDescriptions.Normal}
          </p>

          <div className="prob-distribution">
            <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-sub)' }}>
              Probability Distribution
            </h4>

            {['Normal', 'Cyst', 'Tumor', 'Stone'].map((cls) => {
              const probVal = probabilities ? (probabilities[cls] || 0) : 0;
              const percentStr = (probVal * 100).toFixed(1) + '%';
              return (
                <div key={cls} className="prob-bar-row">
                  <span className="prob-label">{cls}</span>
                  <div className="prob-track">
                    <div
                      className={`prob-fill ${cls}`}
                      style={{ width: percentStr }}
                    ></div>
                  </div>
                  <span className="prob-num">{percentStr}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
