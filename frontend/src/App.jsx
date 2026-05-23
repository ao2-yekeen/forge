import React, { useState, useCallback } from "react";
import "./App.css";
import DescribePanel from "./components/DescribePanel.jsx";
import Viewport from "./components/Viewport.jsx";
import ParametersPanel from "./components/ParametersPanel.jsx";

export default function App() {
  const [description, setDescription] = useState("");
  const [xml, setXml] = useState(null);
  const [parameters, setParameters] = useState({});
  const [actuatorSchedule, setActuatorSchedule] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [simState, setSimState] = useState("idle"); // idle | loading | playing | paused | done
  const [simTime, setSimTime] = useState(0);
  const [duration, setDuration] = useState(10);
  const [simError, setSimError] = useState(null);

  const handleGenerate = useCallback(async (desc) => {
    if (!desc.trim()) return;
    setIsGenerating(true);
    setError(null);
    setXml(null);
    setParameters({});
    setActuatorSchedule([]);
    setSimState("idle");
    setSimTime(0);
    setSimError(null);
    try {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: desc }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setXml(data.xml);
        setParameters(data.parameters || {});
        setActuatorSchedule(data.actuator_schedule || []);
      }
    } catch (e) {
      setError("Could not reach backend. Is it running on port 8000?");
    } finally {
      setIsGenerating(false);
    }
  }, []);

  return (
    <div id="app-root">
      <DescribePanel
        description={description}
        onDescriptionChange={setDescription}
        onGenerate={handleGenerate}
        isGenerating={isGenerating}
        error={error}
        xml={xml}
      />
      <Viewport
        xml={xml}
        actuatorSchedule={actuatorSchedule}
        duration={duration}
        simState={simState}
        onSimStateChange={setSimState}
        onSimTime={setSimTime}
        simTime={simTime}
        simError={simError}
        onSimError={setSimError}
      />
      <ParametersPanel
        parameters={parameters}
        duration={duration}
        onDurationChange={setDuration}
      />
    </div>
  );
}
