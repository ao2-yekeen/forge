import React, { useEffect, useState } from "react";

const emptyClarification = {
  assumptions: [],
  questions: [],
  clarified_description: "",
};

export default function ClarifyModal({ isOpen, description, onClose, onSave, initial = null }) {
  const [clarification, setClarification] = useState(initial || emptyClarification);
  const [answers, setAnswers] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    fetch("http://localhost:8000/clarify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return;
        setClarification(data);
        const nextAnswers = {};
        for (const item of data.assumptions || []) {
          nextAnswers[item.key] = item.value;
        }
        for (const item of data.questions || []) {
          nextAnswers[item.key] = item.default || "";
        }
        setAnswers(nextAnswers);
      })
      .catch(() => {
        if (!cancelled) setError("Could not reach the backend clarification service.");
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen, description]);

  if (!isOpen) return null;

  const updateAnswer = (key, value) => {
    setAnswers((current) => ({ ...current, [key]: value }));
  };

  const save = () => {
    fetch("http://localhost:8000/clarify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description, answers }),
    })
      .then((res) => res.json())
      .then(onSave)
      .catch(() => {
        const assumptions = (clarification.assumptions || []).map((item) => ({
          ...item,
          value: answers[item.key] || item.value,
        }));
        const questionText = (clarification.questions || [])
          .filter((item) => answers[item.key])
          .map((item) => `- ${item.label}: ${answers[item.key]}`)
          .join("\n");
        onSave({
          assumptions,
          questions: clarification.questions || [],
          clarified_description: `${description}\n\nGeneration assumptions:\n${assumptions.map((item) => `- ${item.label}: ${item.value}`).join("\n")}${questionText ? `\n\nClarification answers:\n${questionText}` : ""}`,
        });
      });
  };

  return (
    <div className="clarify-backdrop">
      <div className="clarify-modal">
        <div className="clarify-header">
          <h3>Clarify assumptions</h3>
          <button className="clarify-close" onClick={onClose}>x</button>
        </div>

        {isLoading && <div className="clarify-note">Building useful defaults...</div>}
        {error && <div className="error-box">{error}</div>}

        {!isLoading && (
          <>
            <div className="clarify-section">
              <h4>Assumptions</h4>
              {(clarification.assumptions || []).map((item) => (
                <label key={item.key} className="clarify-field">
                  <span>{item.label}</span>
                  <textarea
                    value={answers[item.key] || ""}
                    onChange={(e) => updateAnswer(item.key, e.target.value)}
                    rows={2}
                  />
                </label>
              ))}
            </div>

            {(clarification.questions || []).length > 0 && (
              <div className="clarify-section">
                <h4>Questions</h4>
                {clarification.questions.map((item) => (
                  <label key={item.key} className="clarify-field">
                    <span>{item.prompt}</span>
                    <input
                      value={answers[item.key] || ""}
                      onChange={(e) => updateAnswer(item.key, e.target.value)}
                    />
                  </label>
                ))}
              </div>
            )}

            {(clarification.questions || []).length === 0 && (
              <div className="clarify-note">No extra questions needed. These defaults should be enough.</div>
            )}
          </>
        )}

        <div className="clarify-actions">
          <button className="sim-btn" onClick={onClose}>Cancel</button>
          <button className="sim-btn primary" onClick={save} disabled={isLoading}>Use these</button>
        </div>
      </div>
    </div>
  );
}
