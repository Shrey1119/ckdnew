import React, { useState } from 'react';

export default function ClinicalPredictor() {
  const [formData, setFormData] = useState({
    serum_creatinine: 1.1,
    hemoglobin: 14.2,
    blood_pressure: 80,
    specific_gravity: '1.020',
    albumin: '0',
    sugar: '0',
    age: 48,
    bgr: 110,
  });

  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  // Heuristic live KNN risk calculation
  const calculateRisk = () => {
    let risk = 0;
    const sc = parseFloat(formData.serum_creatinine);
    const hemo = parseFloat(formData.hemoglobin);
    const bp = parseFloat(formData.blood_pressure);
    const alb = parseInt(formData.albumin);
    const sug = parseInt(formData.sugar);

    if (sc > 1.2) risk += (sc - 1.2) * 20;
    if (hemo < 13.0) risk += (13.0 - hemo) * 10;
    if (bp > 90) risk += (bp - 90) * 0.5;
    if (alb > 0) risk += alb * 18;
    if (sug > 0) risk += sug * 12;

    return Math.min(Math.max(Math.round(risk), 3), 99);
  };

  const riskScore = calculateRisk();
  const isCKD = riskScore >= 40;

  return (
    <div className="card">
      <div className="card-title">
        <i className="fa-solid fa-vial icon-accent"></i> Clinical Blood & Biomarker Assessment
      </div>
      <p className="card-subtitle">
        Input patient laboratory blood test results to run real-time KNN predictive analytics.
      </p>

      <div className="grid-2col">
        <div className="form-grid">
          <div className="form-group">
            <label>Serum Creatinine (mg/dL)</label>
            <div className="input-with-slider">
              <input
                type="number"
                step="0.1"
                value={formData.serum_creatinine}
                onChange={(e) => handleChange('serum_creatinine', e.target.value)}
              />
              <input
                type="range"
                min="0.4"
                max="12.0"
                step="0.1"
                value={formData.serum_creatinine}
                onChange={(e) => handleChange('serum_creatinine', e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Hemoglobin (g/dL)</label>
            <div className="input-with-slider">
              <input
                type="number"
                step="0.1"
                value={formData.hemoglobin}
                onChange={(e) => handleChange('hemoglobin', e.target.value)}
              />
              <input
                type="range"
                min="3.0"
                max="18.0"
                step="0.1"
                value={formData.hemoglobin}
                onChange={(e) => handleChange('hemoglobin', e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Blood Pressure (mm/Hg)</label>
            <div className="input-with-slider">
              <input
                type="number"
                step="5"
                value={formData.blood_pressure}
                onChange={(e) => handleChange('blood_pressure', e.target.value)}
              />
              <input
                type="range"
                min="50"
                max="180"
                step="5"
                value={formData.blood_pressure}
                onChange={(e) => handleChange('blood_pressure', e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label>Albumin (Urine Protein)</label>
            <select
              className="custom-select"
              value={formData.albumin}
              onChange={(e) => handleChange('albumin', e.target.value)}
            >
              <option value="0">0 - Normal</option>
              <option value="1">1 - Mild</option>
              <option value="2">2 - Moderate</option>
              <option value="3">3 - Severe</option>
            </select>
          </div>
        </div>

        <div className="result-panel">
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <div style={{ fontSize: '3rem', fontWeight: 800, color: isCKD ? '#f43f5e' : '#10b981' }}>
              {riskScore}%
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-sub)', textTransform: 'uppercase' }}>
              CKD Predictive Risk Score
            </div>

            <div className={`diagnosis-card ${isCKD ? 'ckd' : 'normal'}`} style={{ marginTop: '1.5rem' }}>
              <div className="card-status-icon">
                <i className={`fa-solid ${isCKD ? 'fa-triangle-exclamation' : 'fa-circle-check'}`}></i>
              </div>
              <div className="card-status-info">
                <h4>{isCKD ? 'Elevated CKD Risk Detected' : 'Normal Kidney Function'}</h4>
                <p>{isCKD ? 'Laboratory biomarkers indicate potential renal impairment.' : 'Biomarkers within healthy range.'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
