import React from "react";

export default function ParametersPanel({ parameters, duration, onDurationChange }) {
  const entries = Object.entries(parameters);

  return (
    <div className="params-panel">
      <h2>Parameters</h2>

      {entries.length === 0 ? (
        <p className="param-empty">Generate a simulation to see parameters.</p>
      ) : (
        entries.map(([key, param]) => (
          <div className="param-item" key={key}>
            <div className="param-header">
              <span className="param-name">{key.replace(/_/g, " ")}</span>
              <span className="param-value">
                {param.value}
                <span className="param-unit">{param.unit}</span>
              </span>
            </div>
            {param.description && (
              <p className="param-desc">{param.description}</p>
            )}
            <input
              className="param-slider"
              type="range"
              min={param.min}
              max={param.max}
              step={(param.max - param.min) / 100}
              defaultValue={param.value}
              readOnly
            />
          </div>
        ))
      )}

      <div className="sim-duration">
        <label>
          Simulation Duration
          <span className="sim-duration-value"> {duration}s</span>
        </label>
        <input
          className="param-slider"
          type="range"
          min={2}
          max={30}
          step={1}
          value={duration}
          onChange={(e) => onDurationChange(Number(e.target.value))}
        />
      </div>
    </div>
  );
}
