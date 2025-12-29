import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // --- Auth State ---
  const [user, setUser] = useState(null); // 'null' means not logged in
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
  const [usernameInput, setUsernameInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [authError, setAuthError] = useState('');

  // --- Main App State ---
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // --- AUTH HANDLERS ---

  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');
    const endpoint = authMode === 'login' ? '/login' : '/register';
    
    try {
      const response = await axios.post(`http://localhost:8000${endpoint}`, {
        username: usernameInput,
        password: passwordInput
      });

      if (authMode === 'register') {
        alert("Registration successful! Please log in.");
        setAuthMode('login');
      } else {
        // Login successful
        setUser(response.data.username);
      }
    } catch (err) {
      setAuthError(err.response?.data?.detail || "Authentication failed");
    }
  };

  const handleLogout = () => {
    setUser(null);
    setResult(null);
    setSelectedFile(null);
    setUsernameInput('');
    setPasswordInput('');
  };

  // --- FILE HANDLERS ---

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setError('');
    setResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/process-manual', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to process. Check backend console.');
    } finally {
      setLoading(false);
    }
  };

  // --- RENDER ---

  if (!user) {
    // LOGIN SCREEN
    return (
      <div className="auth-container">
        <div className="auth-box">
          <h2>{authMode === 'login' ? 'Login' : 'Register'}</h2>
          <form onSubmit={handleAuth}>
            <input 
              type="text" 
              placeholder="Username" 
              value={usernameInput}
              onChange={(e) => setUsernameInput(e.target.value)}
              required 
            />
            <input 
              type="password" 
              placeholder="Password" 
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              required 
            />
            <button type="submit" className="big-upload-button auth-btn">
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </form>
          
          {authError && <p className="error-message">{authError}</p>}
          
          <p className="toggle-auth" onClick={() => {
              setAuthMode(authMode === 'login' ? 'register' : 'login');
              setAuthError('');
          }}>
            {authMode === 'login' ? "New here? Register" : "Already have an account? Login"}
          </p>
        </div>
      </div>
    );
  }

  // MAIN APP SCREEN
  return (
    <div className="app-container">
      <header>
        <div className="logo">
          <div className="logo-icon"></div>
          IKEA Text Extractor
        </div>
        <div className="top-nav-controls">
            <span>Welcome, <b>{user}</b></span>
            <button onClick={handleLogout} className="action-button small">Logout</button>
        </div>
      </header>

      <main className="hero-section">
        <div className="converter-container">
          <div className="main-icon">📄</div>
          <h1>Manual Converter</h1>
          <p class="subtitle">Upload PDF manuals to extract plain text.</p>

          <div className="upload-area">
            <input type="file" id="file-upload" accept=".pdf" onChange={handleFileChange} style={{display: 'none'}} />
            <label htmlFor="file-upload" className="big-upload-button">
              {selectedFile ? selectedFile.name : "Choose PDF File"}
            </label>
            
            <button 
                onClick={handleUpload} 
                className="action-button"
                disabled={loading || !selectedFile}
            >
                {loading ? "Analyzing..." : "Start Conversion"}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}
        </div>

        {result && (
          <div className="results-container">
            <h2>Instructions</h2>
            <div className="results-text">
                <pre>{JSON.stringify(result.instructions, null, 2)}</pre>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;