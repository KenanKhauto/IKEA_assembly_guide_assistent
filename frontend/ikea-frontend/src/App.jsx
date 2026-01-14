import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

export default function App() {
  // Data
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("");

  // UI state
  const [loading, setLoading] = useState(false);
  const [resultTitle, setResultTitle] = useState("");
  const [cacheKey, setCacheKey] = useState("");
  const [assemblySteps, setAssemblySteps] = useState([]);  
  const [error, setError] = useState("");
  const [plainText, setPlainText] = useState("");
  const fileInputRef = useRef(null);

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
      setPlainText(data.instructions || "No text instructions generated yet.");
      setAssemblySteps([]); // clear structured steps
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
      setPlainText(""); // clear old text view
    } catch (e) {
      console.error(e);
      setError(e.message || "Failed to process file.");
    } finally {
      setLoading(false);
      // reset file input so same file can be re-uploaded
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
                            {plainText ? (
                  <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{plainText}</pre>
                ) : assemblySteps.length ? (
                  <ol style={{ margin: 0, paddingLeft: "1.2rem" }}>
                    {assemblySteps.map((s, idx) => (
                      <li key={s.step_id ?? idx} style={{ marginBottom: "1rem" }}>
                        <div style={{ fontWeight: 700 }}>{s.action_summary}</div>
                        {(() => {
                            const imgUrl =
                              cacheKey && s.step_id
                                ? `${API_URL}/static/crops/step_crops/${cacheKey}/step_${s.step_id}.png`
                                : "";
                            return imgUrl ? (
                              <img
                                src={imgUrl}
                                alt={`Step ${idx + 1}`}
                                style={{
                                  width: "100%",
                                  maxWidth: "900px",
                                  borderRadius: "10px",
                                  margin: "0.75rem 0",
                                  border: "1px solid #ddd",
                                }}
                                onError={(e) => (e.currentTarget.style.display = "none")}
                              />
                            ) : null;
                          })()}
                        {s.objects?.length ? (
                          <div><b>Objects:</b> {s.objects.join(", ")}</div>
                        ) : null}

                        {s.fasteners?.length ? (
                          <div><b>Fasteners:</b> {s.fasteners.join(", ")}</div>
                        ) : null}

                        {s.quantities ? (
                          <div>
                            <b>Quantities:</b>{" "}
                            {Object.entries(s.quantities)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(", ")}
                          </div>
                        ) : null}

                        {s.warnings?.length ? (
                          <div style={{ color: "#c0392b" }}>
                            <b>Warnings:</b> {s.warnings.join(" | ")}
                          </div>
                        ) : null}

                        {s.confidence != null ? (
                          <div style={{ opacity: 0.8 }}>
                            <b>Confidence:</b> {String(s.confidence)}
                          </div>
                        ) : null}
                      </li>
                    ))}
                  </ol>
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
