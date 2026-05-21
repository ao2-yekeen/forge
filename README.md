# Forge

Describe a mechanical system in plain English — Forge generates a MuJoCo physics simulation and renders it in 3D in your browser.

![Three-panel layout: describe → simulate → visualize]

## How it works

1. Type a description (e.g. *"A hinged door swinging open under gravity"*)
2. Click **Generate** — an LLM produces valid MJCF XML
3. Click **Play** — the backend runs the physics and streams frames to the frontend
4. Watch the simulation in the 3D viewport with orbit controls

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + Three.js (Vite) |
| Backend | FastAPI + MuJoCo Python |
| LLM | Ollama (`llama3.2`) |

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) running locally

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
ollama pull llama3.2
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

- **Left-drag** — orbit camera
- **Right-drag** — pan
- **Scroll** — zoom
- **Play / Pause / Restart** — simulation controls
- **Simulation Duration** slider — set how long the simulation runs

## Example prompts

- A solid steel cube falling under gravity
- A pendulum with a heavy ball on a rod
- A hinged door swinging open from gravity
- A multi-link robotic arm with three joints
- A trebuchet arm swinging
- A Newton's cradle with five balls
