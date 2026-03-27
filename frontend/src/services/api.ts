/**
 * API Service – all REST calls to FastAPI backend.
 * Base URL proxied via Vite dev server to http://localhost:8000
 */

import type {
  ModelInfo,
  RoiRequest,
  Settings,
  StreamStartRequest,
  SuccessResponse,
  VehicleStats,
  FramePayload,
} from '../types/detection';

const BASE = '/api/v1';

async function apiFetch<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Models ────────────────────────────────────────────────────────────────────

export const modelApi = {
  list: () => apiFetch<ModelInfo[]>('/models'),

  upload: async (file: File): Promise<SuccessResponse> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/models/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  load: (name: string) =>
    apiFetch<SuccessResponse>('/models/load', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  delete: (name: string) =>
    apiFetch<SuccessResponse>(`/models/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
};

// ── Stream ────────────────────────────────────────────────────────────────────

export interface StreamStatus {
  active: boolean;
  fps: number;
  frame_count: number;
  error: string | null;
}

export interface DeviceInfo {
  cuda_available: boolean;
  device_name: string | null;
}

export const streamApi = {
  start: (body: StreamStartRequest) =>
    apiFetch<SuccessResponse>('/stream/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  stop: () =>
    apiFetch<SuccessResponse>('/stream/stop', { method: 'POST' }),

  getStatus: () => apiFetch<StreamStatus>('/stream/status'),

  getFrame: () => apiFetch<FramePayload>('/stream/frame'),

  getDevice: () => apiFetch<DeviceInfo>('/stream/device'),
};

// ── Detection / ROI / Stats ───────────────────────────────────────────────────

export const detectionApi = {
  setRoi: (body: RoiRequest) =>
    apiFetch<SuccessResponse>('/roi', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  clearRoi: () => apiFetch<SuccessResponse>('/roi', { method: 'DELETE' }),

  getStats: () => apiFetch<VehicleStats>('/stats'),

  resetStats: () => apiFetch<SuccessResponse>('/stats/reset', { method: 'POST' }),

  getTimeline: () => apiFetch<{ timeline: Array<{ t: number; v: number }> }>('/timeline'),

  getSettings: () => apiFetch<Settings>('/settings'),

  updateSettings: (body: Partial<Settings>) =>
    apiFetch<SuccessResponse>('/settings', {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
};

// ── Traffic Light ─────────────────────────────────────────────────────────────

export interface TLPhase {
  phase_id: number;
  color: 'red' | 'yellow' | 'green';
  green_time: number;
  remaining: number;
  queue_length: number;
  avg_wait: number;
}

export interface TLState {
  phases: TLPhase[];
  active_phase: number;
  cycle_count: number;
  mode: string;
  time_elapsed: number;
}

export interface TLDecision {
  green_time: number;
  method: string;
  phase: number;
  queue: number;
  wait: number;
}

export interface RLStatus {
  states_visited: number;
  total_steps: number;
  total_episodes: number;
  epsilon: number;
  avg_reward: number;
  avg_q: number;
  max_q: number;
}

export interface GAStatus {
  running: boolean;
  generation: number;
  total_generations: number;
  best_fitness: number;
  avg_fitness: number;
  history: Array<{ generation: number; best_fitness: number; avg_fitness: number; global_best: number }>;
}

export interface SimResult {
  mode: string;
  episodes: number;
  avg_reward: number;
  avg_wait: number;
  total_cleared: number;
  avg_queue: number;
}

export const trafficLightApi = {
  getState: () => apiFetch<TLState>('/traffic-light/state'),

  decide: () => apiFetch<TLDecision>('/traffic-light/decide', { method: 'POST' }),

  advance: () => apiFetch<SuccessResponse>('/traffic-light/advance', { method: 'POST' }),

  setMode: (mode: string) =>
    apiFetch<SuccessResponse>('/traffic-light/mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),

  // Fuzzy
  getFuzzyParams: () => apiFetch<Record<string, Record<string, number[]>>>('/traffic-light/fuzzy/params'),

  fuzzyDecide: (queue: number, wait: number) =>
    apiFetch<{ green_time: number; queue_input: number; wait_input: number }>(
      `/traffic-light/fuzzy/decide?queue=${queue}&wait=${wait}`
    ),

  // RL
  getRLStatus: () => apiFetch<RLStatus>('/traffic-light/rl/status'),

  trainRL: (episodes = 200, cycles = 100) =>
    apiFetch<{ success: boolean; stats: RLStatus }>(
      `/traffic-light/rl/train?episodes=${episodes}&cycles=${cycles}`,
      { method: 'POST' }
    ),

  // GA
  getGAStatus: () => apiFetch<GAStatus>('/traffic-light/ga/status'),

  startGA: (body: { population_size: number; generations: number; eval_episodes: number; eval_cycles: number }) =>
    apiFetch<SuccessResponse>('/traffic-light/ga/start', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  stopGA: () => apiFetch<SuccessResponse>('/traffic-light/ga/stop', { method: 'POST' }),

  // Simulation
  simulate: (body: {
    mode?: string; episodes?: number; cycles_per_episode?: number;
    arrival_rate_0?: number; arrival_rate_1?: number; fixed_green?: number;
  }) =>
    apiFetch<SimResult[]>('/traffic-light/simulate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};
