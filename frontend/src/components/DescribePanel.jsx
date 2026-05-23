import React, { useState } from "react";
import ClarifyModal from "./ClarifyModal.jsx";

export default function DescribePanel({
  description,
  onDescriptionChange,
  onGenerate,
  isGenerating,
  error,
  xml,
}) {
  const [showClarify, setShowClarify] = useState(false);
  const [clarification, setClarification] = useState(null);

  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      onGenerate(clarification?.clarified_description || description);
    }
  };

  const openClarify = () => setShowClarify(true);
  const closeClarify = () => setShowClarify(false);
  const saveClarify = (vals) => {
    setClarification(vals);
    setShowClarify(false);
  };

  return (
    <div className="describe-panel">
      <h1>
        <span>for</span>ge
      </h1>

      <textarea
        placeholder={"Describe a mechanical system...\n\ne.g. A steel ball falling onto a trampoline and bouncing.\n\nCtrl+Enter to generate."}
        value={description}
        onChange={(e) => onDescriptionChange(e.target.value)}
        onKeyDown={handleKey}
      />

      <div style={{display:"flex",gap:8}}>
        <button
          className="generate-btn"
          onClick={() => onGenerate(clarification?.clarified_description || description)}
          disabled={isGenerating || !description.trim()}
        >
          {isGenerating ? "Generating..." : "Generate"}
        </button>
        <button className="generate-btn" onClick={openClarify} style={{background:"#1f2a33"}}>Clarify</button>
      </div>

      {clarification?.assumptions?.length > 0 && (
        <div className="clarify-summary">
          <strong>Assumptions</strong>
          {clarification.assumptions.map((item) => (
            <span key={item.key}>{item.label}: {item.value}</span>
          ))}
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {xml && (
        <details className="xml-accordion">
          <summary>View MJCF XML</summary>
          <pre>{xml}</pre>
        </details>
      )}

      <ClarifyModal
        isOpen={showClarify}
        description={description}
        onClose={closeClarify}
        onSave={saveClarify}
        initial={clarification}
      />
    </div>
  );
}
