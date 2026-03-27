/**
 * TrafficLightPanel – full dashboard panel for traffic light optimization.
 *
 * Shows: traffic light visualization, fuzzy/RL/GA controls, simulation results.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { trafficLightApi } from '../../services/api';
import type { TLState, TLDecision, RLStatus, GAStatus, SimResult } from '../../services/api';

// ── Traffic Light Visual ────────────────────────────────────────────────────

function TrafficLightVisual({ phases, activePhase }: { phases: TLState['phases']; activePhase: number }) {
  const labels = ['N-S', 'E-W'];
  return (
    <div className="flex gap-4 justify-center">
      {phases.map((p, i) => (
        <div key={i} className="flex flex-col items-center gap-1">
          <span className="text-[10px] font-bold text-slate-500">{labels[i]}</span>
          <div className="w-12 bg-slate-800 rounded-xl p-1.5 flex flex-col gap-1.5 shadow-lg">
            {(['red', 'yellow', 'green'] as const).map((color) => (
              <div
                key={color}
                className={`w-9 h-9 rounded-full border-2 transition-all duration-500 ${
                  p.color === color
                    ? color === 'red'
                      ? 'bg-red-500 border-red-400 shadow-[0_0_12px_rgba(239,68,68,0.7)]'
                      : color === 'yellow'
                      ? 'bg-yellow-400 border-yellow-300 shadow-[0_0_12px_rgba(250,204,21,0.7)]'
                      : 'bg-green-500 border-green-400 shadow-[0_0_12px_rgba(34,197,94,0.7)]'
                    : 'bg-slate-700 border-slate-600'
                }`}
              />
            ))}
          </div>
          <div className="text-center mt-1">
            <div className="text-xs font-bold text-slate-700">{p.green_time.toFixed(0)}s</div>
            <div className="text-[10px] text-slate-400">{p.queue_length} xe</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Mode Selector ────────────────────────────────────────────────────────────

function ModeSelector({ mode, onChange }: { mode: string; onChange: (m: string) => void }) {
  const modes = [
    { value: 'manual', label: 'Thu cong', desc: 'Co dinh 30s' },
    { value: 'fuzzy', label: 'Fuzzy', desc: 'Logic mo' },
    { value: 'rl', label: 'RL', desc: 'Q-Learning' },
    { value: 'auto', label: 'Auto', desc: 'RL + Fuzzy' },
  ];
  return (
    <div className="grid grid-cols-4 gap-1">
      {modes.map((m) => (
        <button
          key={m.value}
          onClick={() => onChange(m.value)}
          className={`py-2 px-1 rounded-lg text-center transition-all ${
            mode === m.value
              ? 'bg-accent text-white shadow'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
          }`}
        >
          <div className="text-xs font-bold">{m.label}</div>
          <div className="text-[9px] opacity-70">{m.desc}</div>
        </button>
      ))}
    </div>
  );
}

// ── RL Training Panel ────────────────────────────────────────────────────────

function RLPanel({ status, onTrain }: { status: RLStatus | null; onTrain: () => void }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-600">Q-Learning Agent</span>
        <button
          onClick={onTrain}
          className="px-3 py-1 text-[11px] font-bold rounded-lg bg-purple-100 text-purple-700 hover:bg-purple-200 transition-all"
        >
          Train (200 ep)
        </button>
      </div>
      {status && (
        <div className="grid grid-cols-2 gap-1 text-[10px]">
          <Stat label="Episodes" value={status.total_episodes} />
          <Stat label="Steps" value={status.total_steps.toLocaleString()} />
          <Stat label="Epsilon" value={status.epsilon.toFixed(3)} />
          <Stat label="Avg Reward" value={status.avg_reward.toFixed(1)} />
          <Stat label="States" value={status.states_visited} />
          <Stat label="Max Q" value={status.max_q.toFixed(2)} />
        </div>
      )}
    </div>
  );
}

// ── GA Panel ─────────────────────────────────────────────────────────────────

function GAPanel({
  status,
  onStart,
  onStop,
}: {
  status: GAStatus | null;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-600">Genetic Algorithm</span>
        <div className="flex gap-1">
          <button
            onClick={onStart}
            disabled={status?.running}
            className="px-3 py-1 text-[11px] font-bold rounded-lg bg-green-100 text-green-700 hover:bg-green-200 disabled:opacity-40 transition-all"
          >
            Start GA
          </button>
          <button
            onClick={onStop}
            disabled={!status?.running}
            className="px-3 py-1 text-[11px] font-bold rounded-lg bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-40 transition-all"
          >
            Stop
          </button>
        </div>
      </div>
      {status && (
        <>
          <div className="grid grid-cols-2 gap-1 text-[10px]">
            <Stat label="Generation" value={`${status.generation}/${status.total_generations}`} />
            <Stat label="Best Fitness" value={status.best_fitness.toFixed(2)} />
            <Stat label="Avg Fitness" value={status.avg_fitness.toFixed(2)} />
            <Stat
              label="Status"
              value={status.running ? 'Dang chay...' : 'San sang'}
              color={status.running ? 'text-green-600' : 'text-slate-500'}
            />
          </div>
          {/* Mini fitness chart */}
          {status.history.length > 1 && (
            <FitnessChart history={status.history} />
          )}
        </>
      )}
    </div>
  );
}

// ── Mini Fitness Chart ───────────────────────────────────────────────────────

function FitnessChart({ history }: { history: GAStatus['history'] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d')!;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w;
    canvas.height = h;

    ctx.clearRect(0, 0, w, h);

    const vals = history.map((h) => h.best_fitness);
    const avgVals = history.map((h) => h.avg_fitness);
    const minV = Math.min(...vals, ...avgVals);
    const maxV = Math.max(...vals, ...avgVals);
    const range = maxV - minV || 1;

    const toX = (i: number) => (i / (history.length - 1)) * w;
    const toY = (v: number) => h - ((v - minV) / range) * (h - 4) - 2;

    // Avg fitness (gray)
    ctx.beginPath();
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    avgVals.forEach((v, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
    });
    ctx.stroke();

    // Best fitness (blue)
    ctx.beginPath();
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    vals.forEach((v, i) => {
      i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
    });
    ctx.stroke();
  }, [history]);

  return (
    <div className="mt-1">
      <div className="flex justify-between text-[9px] text-slate-400 mb-0.5">
        <span>Fitness History</span>
        <span>
          <span className="text-blue-500">best</span> / <span className="text-slate-400">avg</span>
        </span>
      </div>
      <canvas ref={canvasRef} className="w-full h-16 rounded bg-slate-50 border border-slate-200" />
    </div>
  );
}

// ── Simulation Results ───────────────────────────────────────────────────────

function SimResults({ results }: { results: SimResult[] }) {
  if (!results.length) return null;
  return (
    <div className="space-y-1">
      <span className="text-xs font-semibold text-slate-600">So sanh ket qua</span>
      <div className="overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500">
              <th className="text-left py-1">Mode</th>
              <th className="text-right py-1">Reward</th>
              <th className="text-right py-1">Wait (s)</th>
              <th className="text-right py-1">Cleared</th>
              <th className="text-right py-1">Queue</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.mode} className="border-b border-slate-100">
                <td className="py-1 font-semibold">{r.mode}</td>
                <td className="text-right py-1">{r.avg_reward.toFixed(1)}</td>
                <td className="text-right py-1">{r.avg_wait.toFixed(1)}</td>
                <td className="text-right py-1">{r.total_cleared}</td>
                <td className="text-right py-1">{r.avg_queue.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Stat Helper ──────────────────────────────────────────────────────────────

function Stat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="bg-slate-50 rounded px-2 py-1">
      <div className="text-[9px] text-slate-400">{label}</div>
      <div className={`font-bold ${color || 'text-slate-700'}`}>{value}</div>
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────

export function TrafficLightPanel() {
  const [state, setState] = useState<TLState | null>(null);
  const [decision, setDecision] = useState<TLDecision | null>(null);
  const [rlStatus, setRLStatus] = useState<RLStatus | null>(null);
  const [gaStatus, setGAStatus] = useState<GAStatus | null>(null);
  const [simResults, setSimResults] = useState<SimResult[]>([]);
  const [training, setTraining] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  // Poll state
  useEffect(() => {
    const poll = async () => {
      try {
        const [s, rl, ga] = await Promise.all([
          trafficLightApi.getState(),
          trafficLightApi.getRLStatus(),
          trafficLightApi.getGAStatus(),
        ]);
        setState(s);
        setRLStatus(rl);
        setGAStatus(ga);
      } catch { /* ignore */ }
    };
    poll();
    pollRef.current = setInterval(poll, 2000);
    return () => clearInterval(pollRef.current);
  }, []);

  const handleModeChange = useCallback(async (mode: string) => {
    await trafficLightApi.setMode(mode);
    setState((s) => s ? { ...s, mode } : s);
  }, []);

  const handleDecide = useCallback(async () => {
    const d = await trafficLightApi.decide();
    setDecision(d);
    // Refresh state
    const s = await trafficLightApi.getState();
    setState(s);
  }, []);

  const handleAdvance = useCallback(async () => {
    await trafficLightApi.advance();
    const s = await trafficLightApi.getState();
    setState(s);
  }, []);

  const handleTrainRL = useCallback(async () => {
    setTraining(true);
    try {
      const res = await trafficLightApi.trainRL(200, 100);
      setRLStatus(res.stats);
    } catch { /* ignore */ }
    setTraining(false);
  }, []);

  const handleStartGA = useCallback(async () => {
    await trafficLightApi.startGA({
      population_size: 50,
      generations: 30,
      eval_episodes: 5,
      eval_cycles: 50,
    });
  }, []);

  const handleStopGA = useCallback(async () => {
    await trafficLightApi.stopGA();
  }, []);

  const handleSimulate = useCallback(async () => {
    setSimulating(true);
    try {
      const results = await trafficLightApi.simulate({
        episodes: 10,
        cycles_per_episode: 100,
      });
      setSimResults(results);
    } catch { /* ignore */ }
    setSimulating(false);
  }, []);

  return (
    <div className="flex flex-col gap-3 p-3 bg-white rounded-xl border border-slate-200 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-800">Traffic Light Optimization</h3>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-50 text-accent font-semibold">
          {state?.mode?.toUpperCase() || 'MANUAL'}
        </span>
      </div>

      {/* Traffic Light Visual */}
      {state && (
        <TrafficLightVisual phases={state.phases} activePhase={state.active_phase} />
      )}

      {/* Controls */}
      <div className="flex gap-2">
        <button
          onClick={handleDecide}
          className="flex-1 py-2 text-xs font-bold rounded-lg bg-accent text-white hover:bg-blue-700 transition-all"
        >
          Tinh Green Time
        </button>
        <button
          onClick={handleAdvance}
          className="flex-1 py-2 text-xs font-semibold rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-100 transition-all"
        >
          Chuyen Phase
        </button>
      </div>

      {/* Decision Result */}
      {decision && (
        <div className="bg-blue-50 rounded-lg px-3 py-2 text-xs">
          <span className="font-bold text-accent">{decision.green_time}s</span>
          <span className="text-slate-500"> ({decision.method}) | Queue: {decision.queue} | Wait: {decision.wait}s</span>
        </div>
      )}

      {/* Mode Selector */}
      <ModeSelector mode={state?.mode || 'manual'} onChange={handleModeChange} />

      {/* Cycle info */}
      {state && (
        <div className="text-[10px] text-slate-400 text-center">
          Cycle #{state.cycle_count} | Phase {state.active_phase}
        </div>
      )}

      <hr className="border-slate-100" />

      {/* RL */}
      <RLPanel
        status={rlStatus}
        onTrain={handleTrainRL}
      />
      {training && (
        <div className="text-[10px] text-purple-600 animate-pulse font-semibold text-center">
          Dang train RL agent (200 episodes)...
        </div>
      )}

      <hr className="border-slate-100" />

      {/* GA */}
      <GAPanel
        status={gaStatus}
        onStart={handleStartGA}
        onStop={handleStopGA}
      />

      <hr className="border-slate-100" />

      {/* Simulation */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-600">Mo phong & So sanh</span>
          <button
            onClick={handleSimulate}
            disabled={simulating}
            className="px-3 py-1 text-[11px] font-bold rounded-lg bg-amber-100 text-amber-700 hover:bg-amber-200 disabled:opacity-40 transition-all"
          >
            {simulating ? 'Dang chay...' : 'Chay Mo Phong'}
          </button>
        </div>
        <SimResults results={simResults} />
      </div>
    </div>
  );
}
