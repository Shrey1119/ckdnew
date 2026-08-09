import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ImageUpload from './components/ImageUpload';
import ImageResults from './components/ImageResults';
import ClinicalPredictor from './components/ClinicalPredictor';
import Dashboard from './components/Dashboard';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [predictionData, setPredictionData] = useState(null);
  const [theme, setTheme] = useState('dark');

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    document.body.className = newTheme;
  };

  const handleUploadSuccess = (result) => {
    setPredictionData(result);
    setActiveTab('results');
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      <main className="page-content">
        {activeTab === 'upload' && (
          <ImageUpload onUploadSuccess={handleUploadSuccess} />
        )}

        {activeTab === 'results' && (
          <ImageResults
            predictionData={predictionData}
            onNewUpload={() => setActiveTab('upload')}
          />
        )}

        {activeTab === 'clinical' && <ClinicalPredictor />}

        {activeTab === 'dashboard' && (
          <Dashboard onGoUpload={() => setActiveTab('upload')} />
        )}
      </main>
    </div>
  );
}
