/**
 * Dashboard – main page assembling all components.
 * Uses useDetection hook for all state management.
 */

import { useState, useEffect, useCallback, type ReactNode } from 'react';
import { useDetection } from '../../hooks/useDetection';
import { VideoPlayer } from '../../components/VideoPlayer';
import { RoiDrawer, RoiCanvasOverlay } from '../../components/RoiDrawer';
import type { RoiPoint } from '../../components/RoiDrawer';
import { ModelUploader } from '../../components/ModelUploader';
import { CounterPanel } from '../../components/CounterPanel';
import { CameraWall } from '../../components/CameraWall/CameraWall';
import { TrafficLightPanel } from '../../components/TrafficLight';
import { modelApi, detectionApi, streamApi } from '../../services/api';
import type { DeviceInfo } from '../../services/api';
import type { ModelInfo, Settings, Toast } from '../../types/detection';

// ── Camera presets (control room) ─────────────────────────────────────────────
const CAMERA_PRESETS = [
  { id: 'cam-1201', label: 'Camera 12 – Cổng chính',   location: 'KCN DD', url: 'rtsp://hctech:Admin@789@kcndd.cameraddns.net:554/Streaming/channels/1201' },
  { id: 'cam-601',  label: 'Camera 6 – Cổng phụ',      location: 'KCN DD', url: 'rtsp://hctech:Admin@789@kcndd.cameraddns.net:554/Streaming/channels/601' },
  { id: 'cam-401',  label: 'Camera 4 – Nội khu A',     location: 'KCN DD', url: 'rtsp://hctech:Admin@789@kcndd.cameraddns.net:554/Streaming/channels/401' },
  { id: 'cam-101',  label: 'Camera 1 – Ngã tư trung tâm', location: 'KCN DD', url: 'rtsp://hctech:Admin@789@kcndd.cameraddns.net:554/Streaming/channels/101' },
  { id: 'cam-2701', label: 'Camera 27 – Đường vòng',   location: 'KCN DD', url: 'rtsp://hctech:Admin@789@kcndd.cameraddns.net:554/Streaming/channels/2701' },
] as const;

// ── Toast helper ──────────────────────────────────────────────────────────────
let toastId = 0;

export function Dashboard() {
  const {
    currentFrame, detections, stats, wsConnected,
    startStream, stopStream, reloadStats, setRoi, clearRoi, resetCount, updateSettings,
  } = useDetection();

  const [streamUrl, setStreamUrl]     = useState('');
  const [connecting, setConnecting]   = useState(false);
  const [streamOn, setStreamOn]       = useState(false);
  const [models, setModels]           = useState<ModelInfo[]>([]);
  const [settings, setSettings]       = useState<Settings>({
    conf_threshold: 0.35,
    line_position: 0.55,
    max_fps: 30,
    tracker_type: 'bytetrack',
    counting_mode: 'all',
    congestion_threshold: 10,
    congestion_duration: 5,
  });
  const [toasts, setToasts]           = useState<Toast[]>([]);
  const [roiActive, setRoiActive]     = useState(false);
  const [roiPoints, setRoiPoints]     = useState<RoiPoint[]>([]);
  const [roiDrawing, setRoiDrawing]   = useState(false);
  const [countingEnabled, setCountingEnabled] = useState(true);
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo>({ cuda_available: false, device_name: null });

  // Merge live stats from backend with local settings for UI
  const statsForView = {
    ...stats,
    conf_threshold: settings.conf_threshold,
    line_position: settings.line_position,
  };

  const congestion = stats.congestion;

  const addToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = ++toastId;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  }, []);

  // Load models list
  const reloadModels = useCallback(() => {
    modelApi.list().then(setModels).catch(() => {});
  }, []);

  // Load settings and device (GPU/CPU)
  useEffect(() => {
    detectionApi.getSettings().then(setSettings).catch(() => {});
    reloadModels();
  }, [reloadModels]);

  useEffect(() => {
    streamApi.getDevice().then(setDeviceInfo).catch(() => {});
  }, []);

  // Stream connect
  const handleConnect = async () => {
    const url = streamUrl.trim();
    if (!url) { addToast('Nhap URL hoac duong dan video', 'error'); return; }
    if (/youtube\.com\/watch\?v=|youtu\.be\//i.test(url)) {
      const idMatch = url.match(/(?:watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{10,15})/i);
      if (!idMatch) {
        addToast('Link YouTube khong hop le. Kiem tra video ID', 'error');
        return;
      }
    }
    setConnecting(true);
    try {
      await startStream(url);
      setStreamOn(true);
      addToast('Stream dang ket noi...', 'info');
      const deadline = Date.now() + 25000;
      const t = setInterval(async () => {
        if (Date.now() > deadline) {
          clearInterval(t);
          setStreamOn(false);
          addToast('Khong the khoi dong stream.', 'error');
          return;
        }
        try {
          const status = await streamApi.getStatus();
          if (status.error) {
            clearInterval(t);
            addToast(status.error, 'error');
            setStreamOn(false);
          } else if (status.active) {
            clearInterval(t);
          }
        } catch {
          // ignore
        }
      }, 1500);
    } catch (e: any) {
      addToast(e.message || 'Khong the ket noi stream', 'error');
    } finally {
      setConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    await stopStream();
    setStreamOn(false);
    addToast('Stream da dung', 'info');
  };

  // ROI — convert canvas coords → video coords before sending to backend
  const handleApplyRoi = async (points: number[][]) => {
    // Get the video container size (canvas overlay) and original video dimensions
    const videoContainer = document.querySelector('.relative.flex-1') as HTMLElement | null;
    const videoCanvas = videoContainer?.querySelector('canvas') as HTMLCanvasElement | null;

    // We need the original video dimensions from the last decoded frame
    // imgRef in VideoPlayer holds naturalWidth/naturalHeight, but we can also
    // get it from the stats or by decoding the current frame.
    // Simplest: read the VideoPlayer canvas dimensions vs the container.
    // The RoiCanvasOverlay sits over the same container as VideoPlayer.
    // Points are in container-pixel coords. We need to map to video-pixel coords.

    if (videoCanvas && currentFrame) {
      // Decode frame to get natural dimensions
      const img = new Image();
      await new Promise<void>((resolve) => {
        img.onload = () => resolve();
        img.onerror = () => resolve();
        img.src = `data:image/jpeg;base64,${currentFrame}`;
      });

      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        const cw = videoCanvas.offsetWidth;
        const ch = videoCanvas.offsetHeight;
        const imgRatio = img.naturalWidth / img.naturalHeight;
        const canRatio = cw / ch;
        let dw: number, dh: number, dx: number, dy: number;
        if (imgRatio > canRatio) {
          dw = cw; dh = cw / imgRatio; dx = 0; dy = (ch - dh) / 2;
        } else {
          dh = ch; dw = ch * imgRatio; dx = (cw - dw) / 2; dy = 0;
        }
        const scaleX = img.naturalWidth / dw;
        const scaleY = img.naturalHeight / dh;

        const videoPoints = points.map(([x, y]) => [
          Math.round((x - dx) * scaleX),
          Math.round((y - dy) * scaleY),
        ]);
        await setRoi(videoPoints);
        setRoiActive(true);
        addToast(`ROI da ap dung (${points.length} diem)`, 'success');
        return;
      }
    }

    // Fallback: send as-is (may be wrong if canvas != video size)
    await setRoi(points);
    setRoiActive(true);
    addToast(`ROI da ap dung (${points.length} diem)`, 'success');
  };

  const handleClearRoi = async () => {
    await clearRoi();
    setRoiActive(false);
    setRoiPoints([]);
    setRoiDrawing(false);
    addToast('ROI da xoa', 'info');
  };

  // Export CSV
  const handleExport = async () => {
    const s = stats;
    const rows = [
      ['Metric', 'Value'],
      ['Total', s.total],
      ['IN (top->bottom)', s.count_in ?? 0],
      ['OUT (bottom->top)', s.count_out ?? 0],
      ['FPS', s.fps],
      ['Model', s.model_name],
      ['---', '---'],
      ['Class', 'Total', 'IN', 'OUT'],
      ...Object.entries(s.classes).map(([cls, count]) => [
        cls, count, s.classes_in?.[cls] ?? 0, s.classes_out?.[cls] ?? 0,
      ]),
    ];
    const csv = rows.map((r) => r.join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `stats_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
    a.click();
    addToast('CSV da xuat', 'success');
  };

  // Settings sliders
  const handleSettingChange = (key: keyof Settings, value: number | string) => {
    setSettings((s) => ({ ...s, [key]: value }));
    updateSettings({ [key]: value } as Partial<Settings>);
  };

  return (
    <div className="flex flex-col min-h-screen bg-bg-base font-sans">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between gap-3 px-5 min-h-14 py-2 bg-white border-b border-slate-200 shadow-sm shrink-0 z-50">
        <div className="flex items-center gap-3 shrink-0">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" className="shrink-0">
            <circle cx="16" cy="16" r="14" stroke="#2563eb" strokeWidth="1.5" strokeDasharray="4 2"/>
            <path d="M8 21L12 12L16 18L21 10L25 21Z" fill="#2563eb" opacity="0.85"/>
          </svg>
          <div className="min-w-0">
            <h1 className="text-sm font-extrabold text-slate-800 truncate">Traffic Monitor</h1>
            <p className="text-[10px] text-slate-500 truncate">YOLOv8 Vehicle Detection &amp; Counting</p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-[11px] text-slate-500 shrink-0">
          <span className="text-2xl font-bold text-accent tabular-nums">{stats.fps.toFixed(1)}</span>
          <span>FPS</span>
          <span>·</span>
          <span>Frame {stats.frame_count.toLocaleString()}</span>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-end min-w-0">
          <StatusPill label={stats.model_loaded ? stats.model_name.replace('.pt', '') : 'No Model'} active={stats.model_loaded} />
          <StatusPill label={deviceInfo.cuda_available ? 'GPU' : 'CPU'} active={deviceInfo.cuda_available} title={deviceInfo.device_name ?? undefined} />
          <StatusPill label={streamOn ? 'Live' : 'No Stream'} active={streamOn} />
          <StatusPill label={wsConnected ? 'WS Connected' : 'WS Offline'} active={wsConnected} />
        </div>
      </header>


      {/* ── Body ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <aside className="w-80 shrink-0 border-r border-slate-200 overflow-y-auto bg-white flex flex-col gap-2 p-3">

          {/* Model upload + library */}
          <SideCard title="Model (.pt)" icon="🧠">
            <ModelUploader
              models={models}
              onModelsChange={reloadModels}
              onToast={addToast}
              onReloadStats={reloadStats}
            />
          </SideCard>

          {/* Camera Wall + Stream controls */}
          <SideCard title="Camera / Stream" icon="📡">
            <div className="flex flex-col gap-3">

              {/* Camera wall grid with live thumbnails */}
              <CameraWall
                cameras={CAMERA_PRESETS}
                selectedUrl={streamUrl}
                activeUrl={streamUrl}
                onSelect={setStreamUrl}
                onConnect={(url) => { setStreamUrl(url); setTimeout(handleConnect, 50); }}
                streamOn={streamOn}
                connecting={connecting}
              />

              {/* Manual URL */}
              <div className="flex flex-col gap-1">
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Hoặc nhập URL thủ công</p>
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleConnect()}
                  placeholder="rtsp://... hoặc C:/path/video.mp4"
                  className="w-full px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800 placeholder-slate-400 outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors"
                />
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleConnect}
                  disabled={connecting || streamOn}
                  className="flex-1 py-2 text-xs font-bold rounded-lg bg-accent text-white disabled:opacity-40 hover:bg-blue-700 transition-all"
                >
                  {connecting ? 'Connecting...' : '▶ Connect'}
                </button>
                <button
                  onClick={handleDisconnect}
                  disabled={!streamOn}
                  className="flex-1 py-2 text-xs font-semibold rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-100 disabled:opacity-30 transition-all"
                >
                  ■ Stop
                </button>
              </div>
            </div>
          </SideCard>

          {/* ROI */}
          <SideCard title="ROI" icon="🎯">
            <RoiDrawer
              onApply={handleApplyRoi}
              onClear={handleClearRoi}
              active={roiActive}
              points={roiPoints}
              setPoints={setRoiPoints}
              isDrawing={roiDrawing}
              setIsDrawing={setRoiDrawing}
            />
          </SideCard>

          {/* Settings */}
          <SideCard title="Cai Dat" icon="⚙️">
            <SliderField
              label="Confidence"
              value={settings.conf_threshold}
              min={0.1} max={0.95} step={0.01}
              display={settings.conf_threshold.toFixed(2)}
              onChange={(v) => handleSettingChange('conf_threshold', v)}
            />
            <SliderField
              label="Counting Line"
              value={settings.line_position}
              min={0.1} max={0.9} step={0.01}
              display={`${Math.round(settings.line_position * 100)}%`}
              onChange={(v) => handleSettingChange('line_position', v)}
            />

            <div className="mt-2">
              <span className="text-[11px] text-slate-500 block mb-1">Tracker</span>
              <select
                value={settings.tracker_type ?? 'bytetrack'}
                onChange={(e) => {
                  const v = e.target.value;
                  setSettings((s) => ({ ...s, tracker_type: v }));
                  updateSettings({ tracker_type: v });
                  const names: Record<string, string> = { bytetrack: 'ByteTrack', botsort: 'BoT-SORT' };
                  addToast(`Da chuyen tracker sang ${names[v] ?? v}`, 'success');
                }}
                className="w-full text-xs border border-slate-300 rounded px-2 py-1.5 bg-white text-slate-700"
              >
                <option value="bytetrack">ByteTrack (recommended)</option>
                <option value="botsort">BoT-SORT</option>
              </select>
            </div>

            <div className="mt-2">
              <span className="text-[11px] text-slate-500 block mb-1">Che do dem</span>
              <select
                value={settings.counting_mode ?? 'all'}
                onChange={(e) => {
                  const v = e.target.value as 'all' | 'direction';
                  setSettings((s) => ({ ...s, counting_mode: v }));
                  updateSettings({ counting_mode: v });
                  addToast(v === 'all' ? 'Dem tat ca (ko phan biet chieu)' : 'Dem theo 2 chieu IN/OUT', 'success');
                }}
                className="w-full text-xs border border-slate-300 rounded px-2 py-1.5 bg-white text-slate-700"
              >
                <option value="all">Dem tat ca (tong hop)</option>
                <option value="direction">Dem theo chieu (IN / OUT)</option>
              </select>
            </div>

            <div className="mt-2 flex items-center justify-between">
              <span className="text-[11px] text-slate-500">Enable Counting</span>
              <button
                type="button"
                onClick={() => setCountingEnabled((v) => !v)}
                className={`relative inline-flex h-4 w-8 items-center rounded-full border transition-colors ${
                  countingEnabled ? 'bg-accent border-accent' : 'bg-slate-200 border-slate-300'
                }`}
              >
                <span
                  className={`inline-block h-3 w-3 rounded-full bg-white shadow transform transition-transform ${
                    countingEnabled ? 'translate-x-4' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </SideCard>

          {/* Congestion Settings */}
          <SideCard title="Canh Bao Ket Xe" icon="🚨">
            <SliderField
              label="Nguong phuong tien"
              value={settings.congestion_threshold ?? 10}
              min={1} max={50} step={1}
              display={`${settings.congestion_threshold ?? 10}`}
              onChange={(v) => handleSettingChange('congestion_threshold', v)}
            />
            <SliderField
              label="Thoi gian on dinh (s)"
              value={settings.congestion_duration ?? 5}
              min={1} max={60} step={1}
              display={`${settings.congestion_duration ?? 5}s`}
              onChange={(v) => handleSettingChange('congestion_duration', v)}
            />
            {/* Live congestion status */}
            {congestion && (
              <div className={`mt-2 px-3 py-2 rounded-lg text-xs font-semibold ${
                congestion.level === 'critical' ? 'bg-red-100 text-red-700 border border-red-200' :
                congestion.level === 'warning' ? 'bg-amber-100 text-amber-700 border border-amber-200' :
                'bg-green-50 text-green-700 border border-green-200'
              }`}>
                {congestion.level === 'normal'
                  ? `Binh thuong (${congestion.vehicle_count} xe)`
                  : `${congestion.level === 'critical' ? 'Nghiem trong' : 'Canh bao'}: ${congestion.vehicle_count} xe / ${congestion.duration_seconds.toFixed(0)}s`
                }
              </div>
            )}
          </SideCard>

        </aside>

        {/* ── Main ─────────────────────────────────────────────────────────── */}
        <main className="flex flex-1 min-w-0 min-h-0 overflow-hidden">

          {/* Video panel */}
          <div className="flex flex-col flex-1 p-3 gap-2 min-h-0">
            <div className="relative flex-1" style={{ minHeight: '400px' }}>
              <VideoPlayer
                frame={currentFrame}
                detections={detections}
                stats={statsForView}
                showLine={countingEnabled}
              />
              <RoiCanvasOverlay
                points={roiPoints}
                setPoints={setRoiPoints}
                isDrawing={roiDrawing}
                setIsDrawing={setRoiDrawing}
                onApply={handleApplyRoi}
                onClear={handleClearRoi}
                active={roiActive}
              />

              {/* Congestion floating pill – bồng bềnh trên video */}
              {congestion && congestion.is_congested && (
                <div className={`
                  absolute bottom-5 left-1/2 -translate-x-1/2 z-20
                  flex items-center gap-2 px-4 py-2 rounded-full
                  shadow-2xl border backdrop-blur-sm
                  text-sm font-bold pointer-events-none select-none
                  animate-bounce
                  ${congestion.level === 'critical'
                    ? 'bg-red-600/90 border-red-400 text-white'
                    : 'bg-amber-500/90 border-amber-300 text-white'}
                `}>
                  <span>{congestion.level === 'critical' ? '🚨' : '⚠️'}</span>
                  <span>
                    {congestion.level === 'critical' ? 'KẸT XE NGHIÊM TRỌNG' : 'MẬT ĐỘ CAO'}
                    {' — '}{congestion.vehicle_count} xe / {congestion.duration_seconds.toFixed(0)}s
                  </span>
                </div>
              )}
            </div>

            {/* Toolbar */}
            <div className="flex items-center gap-2 shrink-0 flex-wrap">
              <TbBadge color="text-accent" label={`${statsForView.fps.toFixed(1)} FPS`} />
              <TbBadge label={`Frame: ${statsForView.frame_count.toLocaleString()}`} />
              {(settings.counting_mode ?? 'all') === 'direction' ? (
                <>
                  <TbBadge color="text-green-600" label={`IN: ${statsForView.count_in ?? 0}`} />
                  <TbBadge color="text-orange-500" label={`OUT: ${statsForView.count_out ?? 0}`} />
                </>
              ) : (
                <TbBadge color="text-accent" label={`Total: ${statsForView.total}`} />
              )}
              <TbBadge color={roiActive ? 'text-accent' : ''} label={`ROI: ${roiActive ? 'ON' : 'OFF'}`} />
              <TbBadge
                color={congestion?.is_congested ? 'text-red-600' : 'text-green-600'}
                label={congestion?.is_congested ? `Ket xe (${congestion.vehicle_count})` : 'Luu thong'}
              />
              <div className="flex-1" />
              <TbBadge color="text-slate-600" label={`Conf: ${settings.conf_threshold.toFixed(2)}`} />
              <TbBadge color="text-slate-600" label={`Line: ${Math.round((statsForView.line_position ?? settings.line_position) * 100)}%`} />
            </div>
          </div>

          {/* Stats + Traffic Light panel (right sidebar) */}
          <aside className="w-64 shrink-0 border-l border-slate-200 overflow-y-auto p-3 bg-white flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">Thong Ke</h2>
              <span className={`w-2 h-2 rounded-full ${streamOn ? 'bg-accent animate-pulse' : 'bg-slate-300'}`} />
            </div>
            <CounterPanel
              stats={countingEnabled ? statsForView : { ...statsForView, total: 0, classes: {} }}
              onReset={resetCount}
              onExport={handleExport}
            />
            <hr className="border-slate-100" />
            <TrafficLightPanel />
          </aside>

        </main>
      </div>

      {/* ── Toast Container ───────────────────────────────────────────────── */}
      <div className="fixed top-16 right-4 z-[9999] flex flex-col gap-2 pointer-events-none">
        {toasts.map((t) => (
          <ToastItem key={t.id} {...t} />
        ))}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusPill({ label, active, title }: { label: string; active: boolean; title?: string }) {
  return (
    <div title={title} className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold border transition-all ${
      active ? 'border-accent/40 text-accent bg-blue-50' : 'border-slate-200 text-slate-500 bg-slate-50'
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-accent animate-pulse' : 'bg-slate-300'}`} />
      {label}
    </div>
  );
}

function SideCard({ title, icon, children }: { title: string; icon: string; children: ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:border-slate-300 transition-colors">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 w-full px-3 py-2.5 text-left hover:bg-slate-50 transition-colors"
      >
        <span className="text-sm">{icon}</span>
        <span className="flex-1 text-xs font-semibold text-slate-800">{title}</span>
        <span className={`text-slate-400 text-xs transition-transform ${open ? '' : '-rotate-90'}`}>▾</span>
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}

function SliderField({ label, value, min, max, step, display, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  display: string; onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-[11px] text-slate-500">{label}</span>
        <span className="text-[11px] font-bold text-accent tabular-nums">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 appearance-none bg-slate-200 rounded-full cursor-pointer accent-accent"
        style={{ accentColor: '#2563eb' }}
      />
    </div>
  );
}

function TbBadge({ label, color = 'text-slate-500' }: { label: string; color?: string }) {
  return (
    <span className={`px-3 py-1 rounded-full text-[11px] font-semibold bg-slate-100 border border-slate-200 ${color}`}>
      {label}
    </span>
  );
}

function ToastItem({ message, type }: Toast) {
  const styles = {
    success: 'border-blue-200 bg-blue-50 text-accent',
    error:   'border-red-200 bg-red-50 text-red-600',
    info:    'border-slate-200 bg-slate-50 text-slate-700',
    warning: 'border-amber-200 bg-amber-50 text-amber-700',
  };
  const icons  = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  return (
    <div className={`flex items-start gap-2.5 px-4 py-3 rounded-xl border text-xs shadow-lg pointer-events-auto animate-toastIn max-w-xs ${styles[type]}`}>
      <span className="shrink-0 mt-0.5">{icons[type]}</span>
      <span>{message}</span>
    </div>
  );
}
