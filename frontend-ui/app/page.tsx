'use client';

import React, { useState } from 'react';
import { Activity, Upload, FileText, AlertTriangle, Crosshair } from 'lucide-react';

const modalityOptions = [
  { value: 'brain-mri', label: 'Brain MRI' },
  { value: 'chest-xray', label: 'Chest X-ray' },
  { value: 'skin-lesion', label: 'Skin lesion' },
];

const GAUGE_R = 52;
const GAUGE_C = 2 * Math.PI * GAUGE_R;

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [doctorNotes, setDoctorNotes] = useState('');
  const [selectedModality, setSelectedModality] = useState<string>('brain-mri');
  const [revealed, setRevealed] = useState(false);

  const [predictionData, setPredictionData] = useState<{
    mainLabel: string;
    confidenceScore: number;
    distribution: { [key: string]: number };
    gradCamUrl?: string | null;
  } | null>(null);

  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setPredictionData(null);
      setRevealed(false);
    }
  };

  const executeAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setIsScanning(true);
    setRevealed(false);
    setPredictionData(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';
      const response = await fetch(`${backendUrl}/api/predict/${selectedModality}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server status: ${response.status}`);
      }

      const data = await response.json();

      let mainLabel = 'Undetermined';
      let confidenceScore = 0;
      let distribution: { [key: string]: number } = {};

      const rawProbabilities = data.probabilities || data.Probabilities || data.breakdown || null;
      if (rawProbabilities && typeof rawProbabilities === 'object') {
        Object.entries(rawProbabilities).forEach(([key, val]) => {
          const num = Number(val);
          if (!isNaN(num)) distribution[key] = num;
        });
      }

      if (data.prediction && String(data.prediction).toLowerCase() !== 'success') {
        mainLabel = data.prediction;
      }

      const rawConfidence = data.confidence ?? data.Confidence ?? null;
      if (rawConfidence !== null && !isNaN(Number(rawConfidence))) {
        confidenceScore = Number(rawConfidence);
      }

      const gradCamUrlValue = data.grad_cam_image ? `data:image/png;base64,${data.grad_cam_image}` : null;

      setTimeout(() => {
        setIsScanning(false);
        setPredictionData({
          mainLabel: String(mainLabel),
          confidenceScore: Number(confidenceScore),
          distribution,
          gradCamUrl: gradCamUrlValue,
        });
        setLoading(false);
        setTimeout(() => setRevealed(true), 60);
      }, 1600);
    } catch (error) {
      console.error('Connection error:', error);
      alert('Could not reach the backend. Check that the API is running.');
      setIsScanning(false);
      setLoading(false);
    }
  };

  const formatConfidenceString = (score: number) => {
    const normalized = score <= 1 && score >= 0 ? score * 100 : score;
    return Number(normalized.toFixed(1));
  };

  const [exporting, setExporting] = useState(false);

  const openReportWindow = (reportText: string) => {
    const w = window.open('', '_blank', 'width=820,height=1000');
    if (!w) {
      alert('Please allow pop-ups to export the report.');
      return;
    }
    const gradcam = predictionData?.gradCamUrl
      ? `<img src="${predictionData.gradCamUrl}" style="max-width:300px;border:1px solid #e4e7ec;border-radius:8px;margin-top:14px" />`
      : '';
    const safe = reportText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    w.document.write(`<!DOCTYPE html><html><head><title>OmniMed Report</title>
      <style>
        body{font-family:'Segoe UI',system-ui,sans-serif;color:#101828;max-width:720px;margin:32px auto;padding:0 24px;line-height:1.55}
        .head{display:flex;align-items:center;gap:10px;border-bottom:2px solid #7c3aed;padding-bottom:12px;margin-bottom:20px}
        .logo{width:34px;height:34px;border-radius:8px;background:#7c3aed;color:#fff;display:grid;place-items:center;font-weight:700;font-size:18px}
        h1{font-size:18px;margin:0}.sub{font-size:12px;color:#667085}
        pre{white-space:pre-wrap;font-family:inherit;font-size:14px;background:#f7f8fa;border:1px solid #e4e7ec;border-radius:8px;padding:16px}
        .foot{margin-top:24px;font-size:11px;color:#98a2b3;text-align:center;border-top:1px solid #e4e7ec;padding-top:12px}
        @media print{.noprint{display:none}}
      </style></head><body>
        <div class="head"><div class="logo">&#9671;</div>
          <div><h1>OmniMed Clinical Report</h1><div class="sub">Medical imaging analysis &middot; ${new Date().toLocaleString()}</div></div>
        </div>
        <pre>${safe}</pre>
        ${gradcam}
        <div class="foot">For demonstration and educational use. Not a medical device.</div>
        <button class="noprint" onclick="window.print()" style="margin-top:22px;padding:10px 20px;background:#7c3aed;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:14px">Save as PDF</button>
      </body></html>`);
    w.document.close();
  };

  const handlePrintExport = async () => {
    if (!predictionData || exporting) return;
    setExporting(true);
    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';
      const norm = (v: number) => (v <= 1 && v >= 0 ? v : v / 100);
      const res = await fetch(`${backendUrl}/api/report/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modality: selectedModality,
          prediction: predictionData.mainLabel,
          confidence: norm(predictionData.confidenceScore),
          probabilities: Object.fromEntries(
            Object.entries(predictionData.distribution).map(([k, v]) => [k, norm(v)])
          ),
          metadata: { Generated: new Date().toLocaleString(), Tool: 'OmniMed' },
          notes: doctorNotes || null,
        }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      openReportWindow(data.report);
    } catch (err) {
      console.error('Report generation failed:', err);
      alert('Could not generate the report. Check that the backend is running.');
    } finally {
      setExporting(false);
    }
  };

  const labelText = (raw: string) => raw.replace(/_/g, ' ');

  const pct = predictionData ? formatConfidenceString(predictionData.confidenceScore) : 0;
  const lowConfidence = predictionData != null && pct < 75;
  const gaugeOffset = revealed ? GAUGE_C * (1 - pct / 100) : GAUGE_C;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-[var(--hair)] bg-[var(--surface)]/70 backdrop-blur">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid place-items-center w-9 h-9 rounded-lg bg-violet-600 text-white shadow-[0_0_20px_rgba(139,92,246,0.45)]">
              <svg viewBox="0 0 36 36" className="w-5 h-5" fill="none" stroke="currentColor"
     strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
  <circle cx="15" cy="15" r="8" />
  <line x1="20.7" y1="20.7" x2="28.5" y2="28.5" strokeWidth="3" />
  <polyline points="8,15 11,15 12.5,11 14.8,19.5 16.5,9.5 18,15 22,15" />
</svg>
            </div>
            <div>
              <h1 className="text-[17px] font-semibold leading-tight tracking-tight">OmniMed</h1>
              <p className="text-xs text-[var(--muted)]">Medical imaging analysis</p>
            </div>
          </div>
          <div className="no-print flex items-center gap-2 text-xs text-[var(--muted)]">
            <span className="live w-1.5 h-1.5 rounded-full bg-violet-400" />
            Analysis engine ready
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT: viewport + input */}
        <div className="lg:col-span-7 space-y-6">
          <section className="bg-[var(--surface)] border border-[var(--hair)] rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold">Viewport</h2>
              <span className="mono text-[11px] text-[var(--dim)] uppercase tracking-widest">{selectedModality}</span>
            </div>

            <div className="mb-4">
              <label className="block text-xs font-medium text-[var(--muted)] mb-1.5">Imaging type</label>
              <select
                value={selectedModality}
                onChange={(e) => setSelectedModality(e.target.value)}
                className="w-full bg-[var(--surface-2)] border border-[var(--hair)] rounded-lg px-3 py-2.5 text-sm focus:border-violet-500 focus:outline-none"
              >
                {modalityOptions.map((o) => (
                  <option key={o.value} value={o.value} className="bg-[#16161f]">{o.label}</option>
                ))}
              </select>
            </div>

            {/* The hero: a radiology lightbox with HUD */}
            <label className="relative block cursor-pointer group">
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <div className="relative rounded-xl bg-[#05050a] border border-[var(--hair)] overflow-hidden min-h-[320px] grid place-items-center">
                {previewUrl ? (
                  <>
                    <img src={previewUrl} alt="Uploaded scan" className="w-full object-contain max-h-[380px] mx-auto" />

                    {/* faint grid + crosshair */}
                    <div className="pointer-events-none absolute inset-0 opacity-[0.10]"
                      style={{ backgroundImage: 'linear-gradient(var(--violet) 1px, transparent 1px), linear-gradient(90deg, var(--violet) 1px, transparent 1px)', backgroundSize: '32px 32px' }} />
                    <div className="pointer-events-none absolute left-1/2 top-0 bottom-0 w-px bg-violet-400/15" />
                    <div className="pointer-events-none absolute top-1/2 left-0 right-0 h-px bg-violet-400/15" />

                    {/* sweep beam while scanning */}
                    {isScanning && (
                      <div className="sweep pointer-events-none absolute left-0 w-full h-10"
                        style={{ background: 'linear-gradient(180deg, transparent, rgba(168,85,247,0.35), transparent)', boxShadow: '0 0 24px 4px rgba(168,85,247,0.35)' }} />
                    )}

                    {/* HUD corner readouts */}
                    <div className="mono pointer-events-none absolute inset-0 text-[10px] text-violet-300/70 tracking-wider">
                      <span className="absolute top-2.5 left-3">{selectedModality.toUpperCase()}</span>
                      <span className="absolute top-2.5 right-3">W/L 248 / 124</span>
                      <span className="absolute bottom-2.5 left-3">ZOOM 100%</span>
                      <span className="absolute bottom-2.5 right-3 flex items-center gap-1"><Crosshair className="w-3 h-3" /> {isScanning ? 'SCANNING' : 'READY'}</span>
                    </div>
                    {/* corner ticks */}
                    <span className="pointer-events-none absolute top-2 left-2 w-3.5 h-3.5 border-l border-t border-violet-400/40" />
                    <span className="pointer-events-none absolute top-2 right-2 w-3.5 h-3.5 border-r border-t border-violet-400/40" />
                    <span className="pointer-events-none absolute bottom-2 left-2 w-3.5 h-3.5 border-l border-b border-violet-400/40" />
                    <span className="pointer-events-none absolute bottom-2 right-2 w-3.5 h-3.5 border-r border-b border-violet-400/40" />
                  </>
                ) : (
                  <div className="flex flex-col items-center text-center py-20 text-[var(--dim)] group-hover:text-violet-300 transition-colors">
                    <Upload className="w-9 h-9 mb-3 stroke-[1.5]" />
                    <span className="text-sm text-[var(--muted)]">Upload a scan image</span>
                    <span className="text-xs mt-1">PNG or JPEG</span>
                  </div>
                )}
              </div>
            </label>

            {file && (
              <button
                onClick={executeAnalysis}
                disabled={loading}
                className="no-print w-full mt-4 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors shadow-[0_0_24px_rgba(139,92,246,0.35)]"
              >
                {loading ? 'Analyzing…' : 'Run analysis'}
              </button>
            )}
          </section>
        </div>

        {/* RIGHT: findings */}
        <div className="lg:col-span-5">
          <section className="bg-[var(--surface)] border border-[var(--hair)] rounded-2xl p-5">
            <h2 className="text-sm font-semibold mb-5">Findings</h2>

            {predictionData ? (
              <div className="space-y-6">
                {/* Confidence dial — the signature */}
                <div className="reveal relative grid place-items-center py-2">
                  <div className="pointer-events-none absolute w-40 h-40 rounded-full"
                    style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.22), transparent 70%)' }} />
                  <svg width="148" height="148" viewBox="0 0 148 148" className="relative">
                    <circle cx="74" cy="74" r={GAUGE_R} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="9" />
                    <circle
                      cx="74" cy="74" r={GAUGE_R} fill="none" stroke="url(#g)" strokeWidth="9" strokeLinecap="round"
                      strokeDasharray={GAUGE_C} strokeDashoffset={gaugeOffset}
                      transform="rotate(-90 74 74)"
                      style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.22,1,0.36,1)' }}
                    />
                    <defs>
                      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stopColor="#8b5cf6" />
                        <stop offset="1" stopColor="#c084fc" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute flex flex-col items-center">
                    <span className="mono text-3xl font-semibold tracking-tight">{pct}<span className="text-lg text-[var(--muted)]">%</span></span>
                    <span className="text-[11px] text-[var(--muted)] uppercase tracking-widest mt-0.5">Confidence</span>
                  </div>
                </div>

                {/* Predicted class */}
                <div className="reveal rounded-xl bg-violet-500/10 border border-violet-500/25 p-4 text-center">
                  <div className="text-xs text-[var(--muted)] mb-1">Predicted class</div>
                  <div className="text-xl font-semibold capitalize text-violet-100">{labelText(predictionData.mainLabel)}</div>
                </div>

                {lowConfidence && (
                  <div className="reveal flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-200">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <p className="text-[13px] leading-relaxed">Low confidence (&lt;75%). Recommend manual review by a clinician.</p>
                  </div>
                )}

                {/* Grad-CAM */}
                {predictionData.gradCamUrl && (
                  <div className="reveal rounded-xl overflow-hidden border border-[var(--hair)]">
                    <div className="px-3 py-2 border-b border-[var(--hair)] mono text-[11px] text-[var(--muted)] uppercase tracking-wider">Model attention · Grad-CAM</div>
                    <img src={predictionData.gradCamUrl} alt="Grad-CAM attention map" className="w-full h-auto object-contain bg-[#05050a]" />
                  </div>
                )}

                {/* Threshold */}
                <div className="no-print">
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-[var(--muted)]">Hide classes below</span>
                    <span className="mono text-violet-300">{confidenceThreshold}%</span>
                  </div>
                  <input type="range" min="0" max="100" value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full accent-violet-500" />
                </div>

                {/* Probabilities — cascade fill */}
                <div>
                  <div className="text-xs text-[var(--muted)] mb-3">Class probabilities</div>
                  <div className="space-y-3">
                    {Object.entries(predictionData.distribution)
                      .sort((a, b) => b[1] - a[1])
                      .map(([key, val], i) => {
                        const p = val <= 1 ? Math.round(val * 100) : Math.round(val);
                        if (p < confidenceThreshold) return null;
                        const isTop = labelText(key) === labelText(predictionData.mainLabel);
                        return (
                          <div key={key}>
                            <div className="flex justify-between text-[13px] mb-1">
                              <span className="capitalize">{labelText(key)}</span>
                              <span className="mono text-[var(--muted)]">{p}%</span>
                            </div>
                            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                              <div className="h-full rounded-full"
                                style={{
                                  width: revealed ? `${p}%` : '0%',
                                  background: isTop ? 'linear-gradient(90deg,#8b5cf6,#c084fc)' : 'rgba(139,92,246,0.3)',
                                  transition: 'width 0.9s cubic-bezier(0.22,1,0.36,1)',
                                  transitionDelay: `${0.15 + i * 0.12}s`,
                                }} />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>

                {/* Notes */}
                <div>
                  <label className="block text-xs text-[var(--muted)] mb-1.5">Clinical notes</label>
                  <textarea value={doctorNotes} onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Add observations or impressions…"
                    className="w-full h-24 p-3 text-[13px] bg-[var(--surface-2)] border border-[var(--hair)] rounded-lg focus:border-violet-500 focus:outline-none resize-none leading-relaxed" />
                </div>

                <div className="pt-1">
                  <button onClick={handlePrintExport} disabled={exporting}
                    className="no-print w-full flex items-center justify-center gap-2 border border-[var(--hair)] hover:bg-white/5 disabled:opacity-50 text-[var(--ink)] text-sm py-2.5 px-4 rounded-lg transition-colors">
                    <FileText className="w-4 h-4" /> {exporting ? 'Generating…' : 'Export report'}
                  </button>
                  <p className="text-[11px] text-[var(--dim)] text-center mt-3">For demonstration and educational use. Not a medical device.</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-center py-24 text-[var(--dim)] border border-dashed border-[var(--hair)] rounded-xl">
                <Activity className="w-7 h-7 mb-3 opacity-40" />
                <p className="text-sm">Run an analysis to see findings.</p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
