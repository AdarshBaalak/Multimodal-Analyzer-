
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";
export default function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  function sentimentToNumber(sent) {
    if (!sent) return 0;
    const s = sent.toLowerCase();
    if (s.includes("pos")) return 1;
    if (s.includes("neg")) return -1;
    return 0;
  }

  function drawTrend() {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const w = c.width;
    const h = c.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#04131b";
    ctx.fillRect(0, 0, w, h);

    if (history.length === 0) return;
    const items = history.slice(0, 20).reverse();

    ctx.beginPath();
    for (let i = 0; i < items.length; i++) {
      const v = sentimentToNumber(items[i].result.text_sentiment);
      const x = (i / (items.length - 1 || 1)) * (w - 20) + 10;
      const y = h / 2 - v * (h / 2 - 10);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      ctx.fillStyle = v > 0 ? "#16a34a" : v < 0 ? "#dc2626" : "#facc15";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  useEffect(() => { if (history.length) drawTrend(); }, [history]);

  async function handleAnalyze() {
    if (!text && !image) {
      alert("Please enter text or upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("text", text);
    if (image) formData.append("image", image);

    setLoading(true);
    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/analyze",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(res.data);

      const entry = {
        text,
        image: image ? URL.createObjectURL(image) : null,
        result: res.data,
        timestamp: new Date().toLocaleTimeString(),
      };
      setHistory((prev) => [entry, ...prev]);
    } catch (err) {
      console.error("Error analyzing:", err);
      alert("Error analyzing. Check backend deployment!");
    } finally {
      setLoading(false);
    }
  }

  return (
   <div className="AppContainer">
    <div style={{ fontFamily: "Inter, sans-serif", background: "#0f172a", color: "white", minHeight: "100vh", padding: "40px", display: "flex", flexDirection: "column", alignItems: "center"}}>
      <h1 style={{ fontSize: "2rem", marginBottom: "1rem" }}>Multimodal Sentiment Analyzer</h1>
      <div style={{ background: "#1e293b", padding: "20px", borderRadius: "12px", width: "100%", maxWidth: "600px" }}>
        <textarea placeholder="Enter your text..." value={text} onChange={(e) => setText(e.target.value)}
          style={{ width: "100%", padding: "10px", borderRadius: "8px", background: "#0f172a", color: "white", border: "1px solid #334155", marginBottom: "10px", resize: "none", height: "80px" }} />
        <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} style={{ marginBottom: "10px" }} />
        <button onClick={handleAnalyze} disabled={loading} style={{ background: "#2563eb", color: "white", border: "none", padding: "10px 20px", borderRadius: "8px", cursor: "pointer", width: "100%", fontWeight: "600" }}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {result && <div style={{ marginTop: "20px", background: "#1e293b", padding: "20px", borderRadius: "12px", width: "100%", maxWidth: "600px" }}>
        <h2 style={{ color: "#60a5fa" }}>Result</h2>
        <p><strong>Text Sentiment:</strong> {result.text_sentiment}</p>
        <p><strong>Text Summary:</strong> {result.text_summary}</p>
        <p><strong>Topic:</strong> {result.topic}</p>
        <p><strong>Image Classification:</strong> {result.image_labels?.join(", ")}</p>
        <p><strong>OCR Text:</strong> {result.ocr_text}</p>
        <p><strong>Toxicity Score:</strong> {result.toxicity_score}</p>
        <p><strong>Automated Response:</strong> {result.automated_response}</p>
      </div>}

      <div style={{ marginTop: "30px", background: "#1e293b", padding: "20px", borderRadius: "12px", width: "100%", maxWidth: "600px" }}>
        <h2 style={{ color: "#60a5fa" }}>Sentiment Trend</h2>
        <canvas ref={canvasRef} width={320} height={100} style={{ background: "#0f172a", borderRadius: "8px", marginTop: "10px", width: "100%" }} />

        {history.length > 0 && <div style={{ marginTop: "20px", maxHeight: "200px", overflowY: "auto", background: "#0f172a", borderRadius: "8px", padding: "10px" }}>
          <h3 style={{ color: "#60a5fa" }}>History</h3>
          {history.map((item, idx) => <div key={idx} style={{ borderBottom: "1px solid #334155", padding: "8px 0" }}>
            <p style={{ margin: 0, color: "#94a3b8" }}><strong>Time:</strong> {item.timestamp}</p>
            <p style={{ margin: 0 }}><strong>Sentiment:</strong> <span style={{ color: item.result.text_sentiment === "POSITIVE" ? "#16a34a" : item.result.text_sentiment === "NEGATIVE" ? "#dc2626" : "#facc15" }}>{item.result.text_sentiment}</span></p>
            <p style={{ margin: 0 }}><strong>Text:</strong> {item.text || "—"}</p>
            {item.image && <img src={item.image} alt="Uploaded" width={60} height={40} style={{ borderRadius: "4px", marginTop: "5px", objectFit: "cover" }} />}
          </div>)}
        </div>}
      </div>
    </div>
    </div>
  );
}
