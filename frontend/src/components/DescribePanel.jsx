import React from "react";

export default function DescribePanel({
  description,
  onDescriptionChange,
  onGenerate,
  isGenerating,
  error,
  xml,
}) {
  const handleKey = (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      onGenerate();
    }
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

      <button
        className="generate-btn"
        onClick={onGenerate}
        disabled={isGenerating || !description.trim()}
      >
        {isGenerating ? "Generating..." : "Generate"}
      </button>

      {error && <div className="error-box">{error}</div>}

      {xml && (
        <details className="xml-accordion">
          <summary>View MJCF XML</summary>
          <pre>{xml}</pre>
        </details>
      )}
    </div>
  );
}
