/**
 * ModelUploader – drag-and-drop upload + models library.
 */

import { useState, useCallback, type DragEvent } from 'react';
import type { ModelInfo } from '../../types/detection';
import { modelApi } from '../../services/api';

interface Props {
  models: ModelInfo[];
  onModelsChange: () => void;
  onToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  onReloadStats: () => void;
}

export function ModelUploader({ models, onModelsChange, onToast, onReloadStats }: Props) {
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const upload = useCallback(
    (file: File) => {
      if (!file.name.endsWith('.pt')) {
        onToast('Chỉ chấp nhận file .pt!', 'error');
        return;
      }

      setUploading(true);
      setProgress(0);

      // Use XHR for upload progress
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      form.append('file', file);

      xhr.open('POST', '/api/v1/models/upload');
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        setUploading(false);
        try {
          const data = JSON.parse(xhr.responseText);
          if (data.success) {
            onToast(data.message, 'success');
            onModelsChange();
            onReloadStats();
          } else {
            onToast(data.detail || 'Upload thất bại', 'error');
          }
        } catch {
          onToast('Server error', 'error');
        }
      };
      xhr.onerror = () => { setUploading(false); onToast('Network error', 'error'); };
      xhr.send(form);
    },
    [onToast, onModelsChange]
  );

  const handleDrop = useCallback(
    (e: DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) upload(file);
    },
    [upload]
  );

  return (
    <div className="flex flex-col gap-3">
      {/* Upload zone */}
      <label
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 border-dashed cursor-pointer transition-all text-center ${
          dragOver
            ? 'border-accent bg-blue-50'
            : 'border-slate-200 bg-slate-50 hover:border-accent/50 hover:bg-blue-50/50'
        }`}
      >
        <input
          type="file"
          accept=".pt"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
        <span className="text-2xl">📦</span>
        <span className="text-xs text-slate-600">
          Drag &amp; drop <strong className="text-slate-800">*.pt</strong> (YOLOv3/v8/v26) hoặc click
        </span>
      </label>

      {/* Upload progress */}
      {uploading && (
        <div>
          <div className="h-1.5 rounded-full bg-slate-200 overflow-hidden">
            <div
              className="h-full bg-accent transition-all rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-1">Uploading… {progress}%</p>
        </div>
      )}

      {/* Models library – always visible */}
      <div className="flex flex-col gap-1.5">
        <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Models</p>
        {models.length > 0 ? (
          <div className="flex items-center gap-2">
            <select
              className="flex-1 min-w-0 text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-slate-700"
              value={selectedModel ?? (models[0]?.name ?? '')}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((m) => {
                const displayName = m.name.includes('/')
                  ? m.name.split('/')[0]
                  : m.name.replace(/\.pt$/i, '');
                return (
                  <option key={m.name} value={m.name} title={m.name}>
                    {displayName}{m.active ? ' • active' : ''}
                  </option>
                );
              })}
            </select>
            <button
              type="button"
              title="Load selected model"
              onClick={async () => {
                const name = selectedModel ?? models[0]?.name;
                if (!name) return;
                try {
                  await modelApi.load(name);
                  onToast(`Loaded '${name}'`, 'success');
                  onReloadStats();
                  onModelsChange();
                } catch (e: any) {
                  onToast(e.message, 'error');
                }
              }}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-100 text-accent hover:bg-blue-200"
            >
              ▶ Load
            </button>
            <button
              type="button"
              title="Delete selected model"
              onClick={async () => {
                const name = selectedModel ?? models[0]?.name;
                if (!name) return;
                if (!confirm(`Xóa '${name}'?`)) return;
                try {
                  await modelApi.delete(name);
                  onToast(`Deleted '${name}'`, 'info');
                  onModelsChange();
                  setSelectedModel(null);
                } catch (e: any) {
                  onToast(e.message, 'error');
                }
              }}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-red-100 text-danger hover:bg-red-200"
            >
              ✕
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={async () => {
              try {
                await modelApi.load('default');
                onToast('Loading default model...', 'info');
                onReloadStats();
                onModelsChange();
              } catch (e: any) {
                onToast(e.message, 'error');
              }
            }}
            className="w-full py-2 text-xs font-semibold rounded-lg bg-blue-100 text-accent hover:bg-blue-200 transition-all"
          >
            ▶ Load Default Model
          </button>
        )}
      </div>
    </div>
  );
}
