import React, { useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

const BACKEND_WS = "ws://localhost:8000/ws/simulate";

function makeGeometry(type, size) {
  switch (type) {
    case "box":
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    case "sphere":
      return new THREE.SphereGeometry(size[0], 16, 16);
    case "cylinder":
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 16);
    case "capsule":
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
    case "plane":
      return new THREE.PlaneGeometry(10, 10);
    case "ellipsoid":
      return new THREE.SphereGeometry(size[0], 16, 16);
    default:
      return new THREE.BoxGeometry(0.1, 0.1, 0.1);
  }
}

// Cylinder/capsule need +90° X rotation: Three.js axis is +Y, MuJoCo default is +Z.
// Plane does NOT need this — its Three.js normal (+Z) already maps to MuJoCo +Z,
// and the rootGroup -90° X correctly converts it to Three.js +Y (up/floor).
const NEEDS_BASE_ROT = new Set(["cylinder", "capsule"]);

export default function Viewport({
  xml,
  actuatorSchedule,
  duration,
  simState,
  onSimStateChange,
  onSimTime,
  simTime,
  simError,
  onSimError,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const rafRef = useRef(null);
  const wsRef = useRef(null);
  const framesRef = useRef([]);
  const frameIndexRef = useRef(0);
  const bodyGroupsRef = useRef([]);
  const lastRealTimeRef = useRef(null);
  const simTimeRef = useRef(0);
  const simStateRef = useRef("idle");
  const wsCompleteRef = useRef(false); // true once WS sends "done"
  const orbitRef = useRef({ theta: Math.PI / 4, phi: Math.PI / 3, radius: 5, target: new THREE.Vector3(0, 0, 0) });
  const dragRef = useRef(null);
  const autoFramedRef = useRef(false);

  // Keep ref in sync with prop for use inside rAF
  simStateRef.current = simState;

  // ---- Three.js setup ----
  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0B0F1A);
    scene.fog = new THREE.Fog(0x0B0F1A, 20, 50);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 200);
    cameraRef.current = camera;
    updateCameraFromOrbit();

    const ambient = new THREE.AmbientLight(0x404060, 1.5);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 6, 4);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const grid = new THREE.GridHelper(10, 20, 0x1e2535, 0x1e2535);
    scene.add(grid);

    function animate() {
      rafRef.current = requestAnimationFrame(animate);
      tickSimulation();
      renderer.render(scene, camera);
    }
    animate();

    const ro = new ResizeObserver(() => {
      const w2 = mount.clientWidth;
      const h2 = mount.clientHeight;
      renderer.setSize(w2, h2);
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
    });
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  function updateCameraFromOrbit() {
    const { theta, phi, radius, target } = orbitRef.current;
    const cam = cameraRef.current;
    if (!cam) return;
    cam.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta),
    );
    cam.lookAt(target);
  }

  // ---- Orbit controls ----
  useEffect(() => {
    const mount = mountRef.current;

    function onMouseDown(e) {
      dragRef.current = { x: e.clientX, y: e.clientY, button: e.button };
    }
    function onMouseMove(e) {
      if (!dragRef.current) return;
      const dx = e.clientX - dragRef.current.x;
      const dy = e.clientY - dragRef.current.y;
      dragRef.current.x = e.clientX;
      dragRef.current.y = e.clientY;
      const orbit = orbitRef.current;
      if (dragRef.current.button === 0) {
        orbit.theta -= dx * 0.01;
        orbit.phi = Math.max(0.1, Math.min(Math.PI - 0.1, orbit.phi + dy * 0.01));
      } else if (dragRef.current.button === 2) {
        const cam = cameraRef.current;
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        cam.getWorldDirection(new THREE.Vector3());
        right.setFromMatrixColumn(cam.matrixWorld, 0).normalize();
        up.setFromMatrixColumn(cam.matrixWorld, 1).normalize();
        orbit.target.addScaledVector(right, -dx * 0.005 * orbit.radius * 0.1);
        orbit.target.addScaledVector(up, dy * 0.005 * orbit.radius * 0.1);
      }
      updateCameraFromOrbit();
    }
    function onMouseUp() { dragRef.current = null; }
    function onWheel(e) {
      orbitRef.current.radius = Math.max(0.5, Math.min(30, orbitRef.current.radius + e.deltaY * 0.01));
      updateCameraFromOrbit();
    }
    function onCtxMenu(e) { e.preventDefault(); }

    mount.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    mount.addEventListener("wheel", onWheel, { passive: true });
    mount.addEventListener("contextmenu", onCtxMenu);
    return () => {
      mount.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      mount.removeEventListener("wheel", onWheel);
      mount.removeEventListener("contextmenu", onCtxMenu);
    };
  }, []);

  // ---- Clear scene bodies ----
  function clearBodies() {
    const scene = sceneRef.current;
    bodyGroupsRef.current.forEach((g) => scene.remove(g));
    if (bodyGroupsRef.current.rootGroup) {
      scene.remove(bodyGroupsRef.current.rootGroup);
    }
    bodyGroupsRef.current = [];
    framesRef.current = [];
    frameIndexRef.current = 0;
    wsCompleteRef.current = false;
    lastRealTimeRef.current = null;
    simTimeRef.current = 0;
    autoFramedRef.current = false;
    onSimTime(0);
  }

  // ---- Build Three.js scene from init message ----
  function buildScene(geoms, bodyNames) {
    clearBodies();
    const scene = sceneRef.current;

    // rootGroup rotates Z-up (MuJoCo) → Y-up (Three.js)
    const rootGroup = new THREE.Group();
    rootGroup.rotation.x = -Math.PI / 2;
    scene.add(rootGroup);

    const bodyGroups = bodyNames.map((name) => {
      const g = new THREE.Group();
      g.name = name;
      rootGroup.add(g);
      return g;
    });
    bodyGroupsRef.current = bodyGroups;
    bodyGroupsRef.current.rootGroup = rootGroup;

    geoms.forEach((geom) => {
      const bodyGroup = bodyGroups[geom.body_id];
      if (!bodyGroup) return;

      const geo = makeGeometry(geom.type, geom.size);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
        opacity: geom.rgba[3],
        transparent: geom.rgba[3] < 1,
        metalness: 0.6,
        roughness: 0.4,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      mesh.position.set(...geom.pos);

      // MuJoCo [w,x,y,z] → Three.js Quaternion(x,y,z,w)
      const [qw, qx, qy, qz] = geom.quat;
      const geomQuat = new THREE.Quaternion(qx, qy, qz, qw);

      if (NEEDS_BASE_ROT.has(geom.type)) {
        const baseQuat = new THREE.Quaternion();
        baseQuat.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
        mesh.quaternion.copy(geomQuat).multiply(baseQuat);
      } else {
        mesh.quaternion.copy(geomQuat);
      }

      bodyGroup.add(mesh);
    });
  }

  // ---- Apply a simulation frame to the scene ----
  function applyFrame(frame) {
    const bodyGroups = bodyGroupsRef.current;
    frame.bodies.forEach((b, i) => {
      const g = bodyGroups[i];
      if (!g) return;
      g.position.set(...b.pos);
      const [qw, qx, qy, qz] = b.quat;
      g.quaternion.set(qx, qy, qz, qw);
    });
    simTimeRef.current = frame.time;
    onSimTime(frame.time);

    // Auto-frame camera on first frame (skip worldbody at index 0)
    if (!autoFramedRef.current && frame.bodies.length > 1) {
      autoFramedRef.current = true;
      const bodies = frame.bodies.slice(1);
      const cx = bodies.reduce((s, b) => s + b.pos[0], 0) / bodies.length;
      const cy = bodies.reduce((s, b) => s + b.pos[1], 0) / bodies.length;
      const cz = bodies.reduce((s, b) => s + b.pos[2], 0) / bodies.length;
      const spread = Math.max(
        ...bodies.map((b) => Math.sqrt((b.pos[0]-cx)**2 + (b.pos[1]-cy)**2 + (b.pos[2]-cz)**2)),
        0.5
      );
      // MuJoCo (x,y,z) Z-up → Three.js (x, z, -y) Y-up (rootGroup rotation.x = -PI/2)
      orbitRef.current.target.set(cx, cz, -cy);
      orbitRef.current.radius = Math.max(spread * 3.5, 2.0);
      updateCameraFromOrbit();
    }
  }

  // ---- Simulation playback tick (called from rAF) ----
  function tickSimulation() {
    if (simStateRef.current !== "playing") return;
    const frames = framesRef.current;
    if (frames.length === 0) return;

    const now = performance.now();
    if (lastRealTimeRef.current === null) {
      lastRealTimeRef.current = now;
    }
    const elapsed = (now - lastRealTimeRef.current) / 1000;
    lastRealTimeRef.current = now;

    simTimeRef.current += elapsed;
    while (
      frameIndexRef.current < frames.length - 1 &&
      frames[frameIndexRef.current + 1].time <= simTimeRef.current
    ) {
      frameIndexRef.current++;
    }
    applyFrame(frames[frameIndexRef.current]);

    // Only mark done when WS streaming is complete and we've played all frames
    if (wsCompleteRef.current && frameIndexRef.current >= frames.length - 1) {
      onSimStateChange("done");
    }
  }

  // ---- WebSocket simulation stream ----
  // autoPlay=false: build scene + show frame 0 as static preview, stay "paused"
  // autoPlay=true: start animating immediately
  const startSimulation = useCallback((autoPlay = false) => {
    if (!xml) return;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearBodies();
    onSimError(null);
    onSimStateChange("loading");

    const ws = new WebSocket(BACKEND_WS);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ xml, duration, actuator_schedule: actuatorSchedule }));
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === "init") {
        buildScene(msg.geoms, msg.body_names);
      } else if (msg.type === "frame") {
        framesRef.current.push(msg);
        // Show initial pose as soon as the first frame arrives
        if (framesRef.current.length === 1) {
          applyFrame(framesRef.current[0]);
          if (!autoPlay) {
            onSimStateChange("paused");
          } else {
            lastRealTimeRef.current = null;
            onSimStateChange("playing");
          }
        }
      } else if (msg.type === "done") {
        wsCompleteRef.current = true;
        // If still loading (no frames came), go idle
        if (simStateRef.current === "loading") {
          onSimStateChange("idle");
        }
      } else if (msg.type === "error") {
        onSimError("Simulation error: " + msg.message);
        onSimStateChange("idle");
      }
    };

    ws.onerror = () => {
      onSimError("Could not connect to simulation backend (ws://localhost:8000).");
      onSimStateChange("idle");
    };
  }, [xml, duration, actuatorSchedule]);

  // Ref so useEffect can call startSimulation without it being a dep
  const startSimulationRef = useRef(startSimulation);
  startSimulationRef.current = startSimulation;

  // Auto-load scene whenever xml changes (after Generate)
  useEffect(() => {
    if (xml) {
      startSimulationRef.current(false);
    }
  }, [xml]);

  const handlePlay = useCallback(() => {
    if (simState === "paused") {
      lastRealTimeRef.current = null;
      onSimStateChange("playing");
    } else if (simState === "done") {
      // Rewind and replay from buffered frames
      frameIndexRef.current = 0;
      simTimeRef.current = 0;
      lastRealTimeRef.current = null;
      onSimTime(0);
      onSimStateChange("playing");
    } else if (simState === "idle") {
      startSimulation(true);
    }
    // "loading" → wait for scene to build; "playing" → already running
  }, [simState, startSimulation]);

  const handlePause = useCallback(() => {
    if (simState === "playing") onSimStateChange("paused");
  }, [simState]);

  const handleRestart = useCallback(() => {
    if (framesRef.current.length > 0) {
      // Rewind to initial pose without re-simulating
      frameIndexRef.current = 0;
      simTimeRef.current = 0;
      lastRealTimeRef.current = null;
      onSimTime(0);
      applyFrame(framesRef.current[0]);
      onSimStateChange("paused");
    } else {
      startSimulation(false);
    }
  }, [startSimulation]);

  const hasXml = !!xml;
  const isLoading = simState === "loading";

  return (
    <div className="viewport" ref={mountRef}>
      {!hasXml && (
        <div className="viewport-overlay">
          <div className="viewport-placeholder">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
              <rect x="8" y="8" width="48" height="48" rx="6" stroke="#00E5FF" strokeWidth="2"/>
              <path d="M20 44 L32 20 L44 44" stroke="#00E5FF" strokeWidth="2" strokeLinejoin="round"/>
              <circle cx="32" cy="36" r="3" fill="#00E5FF"/>
            </svg>
            <p>Describe a system and click Generate</p>
          </div>
        </div>
      )}

      {simError && (
        <div className="viewport-overlay" style={{pointerEvents:"none"}}>
          <div style={{background:"rgba(255,77,79,0.12)",border:"1px solid #ff4d4f",borderRadius:8,color:"#ff4d4f",fontSize:13,padding:"12px 18px",maxWidth:440,textAlign:"center",lineHeight:1.6}}>
            {simError}
          </div>
        </div>
      )}

      {hasXml && (
        <>
          <div className="time-display">
            {isLoading ? "Loading..." : `${simTime.toFixed(2)}s`}
          </div>
          <div className="sim-controls">
            {simState === "playing" ? (
              <button className="sim-btn" onClick={handlePause}>Pause</button>
            ) : (
              <button
                className="sim-btn primary"
                onClick={handlePlay}
                disabled={isLoading}
              >
                {simState === "done" ? "Replay" : simState === "paused" ? "Play" : "Play"}
              </button>
            )}
            <button className="sim-btn" onClick={handleRestart} disabled={isLoading}>
              Restart
            </button>
          </div>
        </>
      )}
    </div>
  );
}
