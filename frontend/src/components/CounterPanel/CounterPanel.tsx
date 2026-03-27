/**
 * CounterPanel – displays vehicle counts.
 * Mode "all":       show total + class breakdown (no direction)
 * Mode "direction": show IN/OUT + class breakdown per direction
 */

import type { VehicleStats } from '../../types/detection';

const CLASS_COLORS: Record<string, string> = {
  car:        '#2563eb',
  truck:      '#475569',
  bus:        '#0ea5e9',
  motorcycle: '#64748b',
  motorbike:  '#64748b',
  bicycle:    '#0891b2',
  person:     '#6366f1',
  pedestrian: '#6366f1',
  'container truck': '#334155',
};

interface Props {
  stats: VehicleStats;
  onReset: () => void;
  onExport: () => void;
}

export function CounterPanel({ stats, onReset, onExport }: Props) {
  const mode = stats.counting_mode ?? 'all';
  const entries = Object.entries(stats.classes).sort((a, b) => b[1] - a[1]);
  const maxCount = entries[0]?.[1] || 1;

  return (
    <div className="flex flex-col gap-4">

      {/* ── Total card ── */}
      <div className="relative overflow-hidden p-4 rounded-xl border border-slate-200 bg-slate-50 text-center">
        <p className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-1">
          Tong xe da dem
        </p>
        <p className="text-5xl font-black text-accent relative tabular-nums">
          {stats.total.toString().padStart(3, '0')}
        </p>

        {/* IN / OUT — chỉ hiện khi mode = direction */}
        {mode === 'direction' && (
          <div className="flex justify-center gap-4 mt-3">
            <div className="flex items-center gap-1.5">
              <span className="text-sm text-green-600">&#x2193;</span>
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">IN</span>
              <span className="text-lg font-bold text-green-600 tabular-nums">{stats.count_in ?? 0}</span>
            </div>
            <div className="w-px h-5 bg-slate-300" />
            <div className="flex items-center gap-1.5">
              <span className="text-sm text-orange-500">&#x2191;</span>
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">OUT</span>
              <span className="text-lg font-bold text-orange-500 tabular-nums">{stats.count_out ?? 0}</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Per-class breakdown ── */}
      {entries.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          {entries.map(([cls, count]) => {
            const c = CLASS_COLORS[cls] || '#94a3b8';
            const inCount = stats.classes_in?.[cls] ?? 0;
            const outCount = stats.classes_out?.[cls] ?? 0;
            return (
              <div key={cls} className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-slate-200 bg-white hover:border-slate-300 transition-all">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: c }} />
                <span className="flex-1 text-xs font-medium capitalize text-slate-700">{cls}</span>

                {/* IN/OUT per class — chỉ hiện khi mode = direction */}
                {mode === 'direction' && (
                  <div className="flex items-center gap-1.5 text-[10px] tabular-nums">
                    <span className="text-green-600" title="IN">{inCount}</span>
                    <span className="text-slate-300">/</span>
                    <span className="text-orange-500" title="OUT">{outCount}</span>
                  </div>
                )}

                <span className="text-base font-bold tabular-nums ml-1" style={{ color: c }}>{count}</span>
              </div>
            );
          })}

          {/* Bar chart */}
          <div className="mt-2 flex flex-col gap-1.5">
            <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Distribution</p>
            {entries.map(([cls, count]) => {
              const c = CLASS_COLORS[cls] || '#94a3b8';
              const pct = Math.round((count / maxCount) * 100);
              return (
                <div key={cls} className="flex items-center gap-2 text-xs">
                  <span className="w-16 text-slate-500 capitalize truncate shrink-0">{cls}</span>
                  <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${pct}%`, background: c }}
                    />
                  </div>
                  <span className="w-6 text-right text-slate-500 shrink-0">{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-500 text-center py-4">Chua co du lieu…</p>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={onReset}
          className="flex-1 py-2 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:border-amber-500 hover:text-amber-600 transition-all"
        >
          Reset
        </button>
        <button
          onClick={onExport}
          className="flex-1 py-2 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:border-accent hover:text-accent transition-all"
        >
          CSV
        </button>
      </div>
    </div>
  );
}
