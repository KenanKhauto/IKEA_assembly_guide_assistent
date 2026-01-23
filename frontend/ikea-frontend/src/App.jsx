import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "http://localhost:8000";

export default function App() {
  // Data
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [renderedSteps, setRenderedSteps] = useState([]);

  // UI state
  const [loading, setLoading] = useState(false);
  const [resultTitle, setResultTitle] = useState("");
  const [cacheKey, setCacheKey] = useState("");
  const [assemblySteps, setAssemblySteps] = useState([]);  
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const [currentStep, setCurrentStep] = useState(0);

  // Load products on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_URL}/products`);
        if (!res.ok) throw new Error("Failed to fetch products");
        const data = await res.json();
        setProducts(data);
      } catch (e) {
        console.error(e);
        setError("Could not load products. Is the backend running?");
      }
    })();
  }, []);

  const hasResults = useMemo(() => Boolean(resultTitle), [resultTitle]);

  async function fetchManualText(productName) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(
        `${API_URL}/manual-text?product_name=${encodeURIComponent(productName)}`
      );
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || "Failed to fetch instructions");
      }

      setResultTitle(`Instructions for: ${data.product_name}`);
      setRenderedSteps(data.output_text_list || []);
      setAssemblySteps(data.assembly_instructions || []);
      setCurrentStep(0);
    } catch (e) {
      console.error(e);
      setError(e.message || "Failed to fetch instructions.");
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(file) {
    if (!file) return;
    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/process-manual`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || "Processing failed");
      }

      setResultTitle(`Instructions for: ${data.filename} ${data.cached ? "(cached)" : ""}`);
      setCacheKey(data.cache_key || "");
      setAssemblySteps(data.assembly_instructions || []);
      setRenderedSteps(data.output_text_list || []);
      setCurrentStep(0);
    } catch (e) {
      console.error(e);
      setError(e.message || "Failed to process file.");
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function onProductChange(e) {
    const name = e.target.value;
    setSelectedProduct(name);
    if (name) fetchManualText(name);
  }

  function onFileChange(e) {
    const file = e.target.files?.[0];
    handleUpload(file);
  }

  return (
    <>
      <header>
        <div className="logo">
          <div className="logo-icon" />
          IKEA Text Extractor
        </div>

        <div className="top-nav-controls">
          <select
            className="product-select"
            value={selectedProduct}
            onChange={onProductChange}
          >
            <option value="" disabled>
              Select a Product...
            </option>
            {products.map((p) => (
              <option key={`${p.category}-${p.product_name}`} value={p.product_name}>
                {p.category}: {p.product_name}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className="hero-section">
        <div className="converter-container">
          <div className="main-icon">📄</div>
          <h1>IKEA Manual to Text Converter</h1>
          <p className="subtitle">Select a product from the top menu OR upload a new PDF.</p>

          <input
            ref={fileInputRef}
            type="file"
            id="file-upload"
            accept=".pdf"
            onChange={onFileChange}
            style={{ display: "none" }}
          />

          <label
            htmlFor="file-upload"
            className="big-upload-button"
            style={{ opacity: loading ? 0.7 : 1 }}
          >
            Upload New File
          </label>

          {loading && <div className="loading">Processing... please wait (this may take a minute)</div>}
          {error && <div className="loading" style={{ color: "#e74c3c" }}>{error}</div>}
        </div>

        {hasResults && (
          <div className="results-area">
            <div className="results-header">{resultTitle}</div>

            <div className="instruction-text">
                {renderedSteps.length ? (
                  <>
                    <div className="steps-nav">
                      {renderedSteps.map((_, idx) => (
                        <button
                          key={assemblySteps[idx]?.step_id ?? idx}
                          className={`steps-nav-btn ${idx === currentStep ? "active" : ""}`}
                          onClick={() => setCurrentStep(idx)}
                          type="button"
                        >
                          {idx + 1}
                        </button>
                      ))}
                    </div>

                    <div className="steps-nav-actions">
                      <button
                        type="button"
                        className="steps-action-btn"
                        onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
                        disabled={currentStep === 0}
                      >
                        ← Prev
                      </button>

                      <div className="steps-counter">
                        Step {currentStep + 1} / {renderedSteps.length}
                      </div>

                      <button
                        type="button"
                        className="steps-action-btn"
                        onClick={() => setCurrentStep((s) => Math.min(renderedSteps.length - 1, s + 1))}
                        disabled={currentStep === renderedSteps.length - 1}
                      >
                        Next →
                      </button>
                    </div>

                    {/* Render ONLY the selected step */}
                    {(() => {
                      const step = assemblySteps[currentStep];
                      const stepText = renderedSteps[currentStep];
                      const imgUrl =
                        cacheKey && step?.step_id
                          ? `${API_URL}/static/crops/step_crops/${cacheKey}/step_${step.step_id}.png`
                          : "";

                      return (
                        <div className="step-card">
                          

                          {imgUrl && (
                            <img
                              src={imgUrl}
                              alt={`Step ${currentStep + 1}`}
                              className="step-image"
                              onError={(e) => (e.currentTarget.style.display = "none")}
                            />
                          )}

                          <div className="step-markdown">
                            <ReactMarkdown>{stepText}</ReactMarkdown>
                          </div>
                        </div>
                      );
                    })()}
                  </>
                ) : (
                  "No instructions generated yet."
                )}

            </div>
          </div>
        )}
      </main>

      <footer>
        <p>© 2025 IKEA Text Extractor Project.</p>
      </footer>
    </>
  );
}
