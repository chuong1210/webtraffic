/**
 * useDetection – central hook using REST API only (no WebSocket).
 * Provides: stats and actions; frame/detections are not streamed in real-time anymore.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { detectionApi, streamApi } from '../services/api';
import type { Detection, VehicleStats, Settings } from '../types/detection';

const DEFAULT_STATS: VehicleStats = {
  total: 0,
  count_in: 0,
  count_out: 0,
  classes: {},
  classes_in: {},
  classes_out: {},
  counting_mode: 'all',
  fps: 0,
  frame_count: 0,
  stream_active: false,
  model_loaded: false,
  model_name: '',
  roi_active: false,
  conf_threshold: 0.35,
  line_position: 0.55,
  congestion: {
    is_congested: false,
    vehicle_count: 0,
    threshold: 10,
    duration_seconds: 0,
    stable_duration: 5,
    message: '',
    level: 'normal',
  },
};

export function useDetection() {
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [detections, setDetections]     = useState<Detection[]>([]);
  const [stats, setStats]               = useState<VehicleStats>(DEFAULT_STATS);
  const [streamActive, setStreamActive] = useState(false);
  const settingsTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const [wsConnected] = useState(false);
  const frameTimerRef = useRef<ReturnType<typeof setInterval>>();

  // Load initial stats once
  useEffect(() => {
    detectionApi.getStats().then(setStats).catch(() => {});
  }, []);

  const reloadStats = useCallback(async () => {
    try {
      const s = await detectionApi.getStats();
      setStats(s);
    } catch {
      // ignore
    }
  }, []);

  // Adaptive frame poll: fire next request immediately after previous one completes.
  // This avoids request pile-up from fixed setInterval when server is slow.
  useEffect(() => {
    if (!streamActive) {
      if (frameTimerRef.current) {
        clearTimeout(frameTimerRef.current);
        frameTimerRef.current = undefined;
      }
      setCurrentFrame(null);
      setDetections([]);
      return;
    }

    let cancelled = false;
    let emptyCount = 0;

    const poll = async () => {
      if (cancelled) return;
      try {
        const payload = await streamApi.getFrame();

        if (!payload || !payload.frame) {
          emptyCount++;
          if (emptyCount > 20) {
            console.warn('[useDetection] stream appears dead, stopping poll');
            setStreamActive(false);
            return;
          }
          // No frame yet — wait 200ms before retry
          if (!cancelled) frameTimerRef.current = setTimeout(poll, 200);
          return;
        }
        emptyCount = 0;

        if (payload.stats && !payload.stats.stream_active) {
          console.warn('[useDetection] backend reports stream inactive');
          setStreamActive(false);
          return;
        }

        setCurrentFrame(payload.frame);
        setDetections(payload.detections ?? []);
        setStats(payload.stats);
      } catch (err) {
        console.error('[useDetection] frame poll error:', err);
        // Back-off on error
        if (!cancelled) frameTimerRef.current = setTimeout(poll, 500);
        return;
      }
      // Fire immediately after response — no artificial delay
      if (!cancelled) frameTimerRef.current = setTimeout(poll, 0);
    };

    poll();

    return () => {
      cancelled = true;
      if (frameTimerRef.current) {
        clearTimeout(frameTimerRef.current);
        frameTimerRef.current = undefined;
      }
    };
  }, [streamActive]);

  // ── Actions ────────────────────────────────────────────────────────────────

  const startStream = useCallback(async (url: string) => {
    await streamApi.start({ url });
    setStreamActive(true);
  }, []);

  const stopStream = useCallback(async () => {
    await streamApi.stop();
    setStreamActive(false);
  }, []);

  const setRoi = useCallback(async (points: number[][]) => {
    await detectionApi.setRoi({ points, active: true });
  }, []);

  const clearRoi = useCallback(async () => {
    await detectionApi.clearRoi();
  }, []);

  const resetCount = useCallback(async () => {
    try {
      await detectionApi.resetStats();
      setStats((s) => ({ ...s, total: 0, classes: {}, congestion: DEFAULT_STATS.congestion }));
    } catch (err) {
      console.error('[useDetection] reset failed:', err);
    }
  }, []);

  const updateSettings = useCallback(
    (patch: Partial<Settings>) => {
      clearTimeout(settingsTimerRef.current);
      settingsTimerRef.current = setTimeout(async () => {
        await detectionApi.updateSettings(patch);
      }, 400);
    },
    []
  );

  return {
    // State
    currentFrame,
    detections,
    stats,
    wsConnected,
    streamActive,
    // Actions
    startStream,
    stopStream,
    reloadStats,
    setRoi,
    clearRoi,
    resetCount,
    updateSettings,
  };
}
