import React, { useState, useRef, useEffect } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  ScanLine,
  FileCheck2,
  Cpu,
  QrCode,
  Landmark,
  UserCheck,
  ArrowRight,
  Layers,
  SlidersHorizontal,
  Copy,
  Check,
  FileWarning,
  ChevronRight,
  ChevronDown,
  RotateCcw,
  Sparkles,
  ExternalLink,
  Info
} from 'lucide-react';

const BENCHMARK_SCENARIOS = [
  {
    id: 'scenario_0',
    index: '00',
    filename: 'genuine_marksheet.jpg',
    title: 'Authentic Baseline Marksheet',
    tag: 'Authentic Baseline',
    expected: 'PASS',
    applicantName: 'AARAV SHARMA',
    applicantDob: '15-08-2006',
    desc: 'Untouched physical scan, verified DSC-9 root signature, valid Central Board archive.'
  },
  {
    id: 'scenario_1',
    index: '01',
    filename: 'tampered_marksheet.jpg',
    title: 'Edited Genuine Marksheet',
    tag: 'ELA + EXIF + QR',
    expected: 'FAIL',
    applicantName: 'AARAV SHARMA',
    applicantDob: '15-08-2006',
    desc: 'ENG-104 grade modified E to B (+3px drift, Photoshop EXIF, QR payload contradiction).'
  },
  {
    id: 'scenario_2',
    index: '02',
    filename: 'fabricated_certificate.jpg',
    title: 'Fabricated Certificate (Real Univ)',
    tag: 'QR Signature + Archive',
    expected: 'FAIL',
    applicantName: 'ROHAN VERMA',
    applicantDob: '10-04-2005',
    desc: 'Genuine board header, forged FAKE-000 signature, non-existent Roll BSE-2026-999.'
  },
  {
    id: 'scenario_3',
    index: '03',
    filename: 'unrecognised_institution.jpg',
    title: 'Unrecognised Institution / Diploma Mill',
    tag: 'UGC Blacklist',
    expected: 'FAIL',
    applicantName: 'KAVYA JOSHI',
    applicantDob: '22-11-2004',
    desc: 'Issued by Global Digital Open University of India, unaccredited on UGC blacklist.'
  },
  {
    id: 'scenario_4',
    index: '04',
    filename: 'ai_generated_doc.jpg',
    title: 'Fully AI-Generated Document',
    tag: 'Gen-AI Check + No QR',
    expected: 'FAIL',
    applicantName: 'VIKRAM SINGHANIA',
    applicantDob: '05-09-2006',
    desc: 'Synthetically generated document with Midjourney/SDXL metadata and missing QR seal.'
  },
  {
    id: 'scenario_5',
    index: '05',
    filename: 'genuine_marksheet.jpg',
    title: 'Genuine Document, Wrong Person',
    tag: 'Identity Match',
    expected: 'FAIL',
    applicantName: 'PRIYA PATEL',
    applicantDob: '18-07-2005',
    desc: 'Aarav Sharma genuine marksheet submitted by applicant Priya Patel (Impersonation).'
  }
];

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [applicantName, setApplicantName] = useState('AARAV SHARMA');
  const [applicantDob, setApplicantDob] = useState('15-08-2006');
  const [selectedScenarioId, setSelectedScenarioId] = useState(null);

  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [activeTab, setActiveTab] = useState('verdict');
  const [selectedCheckId, setSelectedCheckId] = useState('step0');
  const [auditOpen, setAuditOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  // Auto-select failing check if result fails
  useEffect(() => {
    if (result?.five_check_stack) {
      const stack = result.five_check_stack;
      const failingKey = ['check1', 'check2', 'check3', 'check4', 'check5'].find(
        (k) => stack[k]?.status === 'FAIL'
      );
      if (failingKey) {
        setSelectedCheckId(failingKey);
      } else {
        setSelectedCheckId('step0');
      }
    }
  }, [result]);

  const handleResetSession = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setSelectedScenarioId(null);
    setApplicantName('AARAV SHARMA');
    setApplicantDob('15-08-2006');
    setSelectedCheckId('step0');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setError(null);
    setResult(null);
    setSelectedScenarioId(null);
    setFile(selectedFile);
    const isPdf = selectedFile.type === 'application/pdf' || selectedFile.name.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreviewUrl(objectUrl);
    } else {
      setPreviewUrl(null);
    }
    verifyUploadedFile(selectedFile, applicantName, applicantDob);
  };

  const verifyUploadedFile = async (fileToUpload, appName, appDob) => {
    setLoading(true);
    setError(null);
    setLoadingStep('Stage 1/5: Document intake & auto-alignment...');

    const t1 = setTimeout(() => {
      setLoadingStep('Stage 2/5: Scanning EXIF metadata & generative AI signatures...');
    }, 1200);

    const t2 = setTimeout(() => {
      setLoadingStep('Stage 3/5: Computing Q=85 ELA compression & baseline drift...');
    }, 3200);

    const t3 = setTimeout(() => {
      setLoadingStep('Stage 4/5: Decoding cryptographic QR & DigiLocker archive match...');
    }, 5500);

    const t4 = setTimeout(() => {
      setLoadingStep('Stage 5/5: Executing cross-document applicant identity check...');
    }, 7800);

    try {
      const formData = new FormData();
      formData.append('file', fileToUpload);
      if (appName) formData.append('applicant_name', appName);
      if (appDob) formData.append('applicant_dob', appDob);

      const response = await fetch('/api/verify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
      setActiveTab('verdict');
    } catch (err) {
      setError(err.message || 'Failed to scan document.');
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleTriggerScenario = async (scenario) => {
    setSelectedScenarioId(scenario.id);
    setApplicantName(scenario.applicantName);
    setApplicantDob(scenario.applicantDob);
    setLoading(true);
    setError(null);
    setResult(null);
    setLoadingStep(`Running 5-Check Stack on ${scenario.title}...`);

    try {
      const params = new URLSearchParams();
      if (scenario.applicantName) params.append('applicant_name', scenario.applicantName);
      if (scenario.applicantDob) params.append('applicant_dob', scenario.applicantDob);

      const response = await fetch(`/api/verify-sample/${scenario.filename}?${params.toString()}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
      setPreviewUrl(null);
      setFile({ name: scenario.filename });
      setActiveTab('verdict');
    } catch (err) {
      setError(err.message || 'Failed to verify benchmark scenario.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const handleCopyAudit = () => {
    if (!result?.audit_report) return;
    navigator.clipboard.writeText(JSON.stringify(result.audit_report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const fiveStack = result?.five_check_stack || null;
  const currentVerdict = result?.verdict || (result?.status === 'ACCEPTED' ? 'PASS' : result?.status === 'REJECTED' ? 'FAIL' : 'REFER');

  const checkConfigs = [
    {
      id: 'step0',
      stepNum: 'STEP 0',
      title: 'Intake & Extraction',
      subtitle: 'Read every field',
      icon: ScanLine,
      data: fiveStack?.step0,
    },
    {
      id: 'check1',
      stepNum: 'CHECK 1',
      title: 'File Forensics',
      subtitle: 'Metadata, saves, signature',
      icon: Cpu,
      data: fiveStack?.check1,
    },
    {
      id: 'check2',
      stepNum: 'CHECK 2',
      title: 'Pixel Analysis',
      subtitle: 'ELA, fonts, overlays',
      icon: Layers,
      data: fiveStack?.check2,
    },
    {
      id: 'check3',
      stepNum: 'CHECK 3',
      title: 'QR & Signature',
      subtitle: 'Document agrees with itself',
      icon: QrCode,
      data: fiveStack?.check3,
    },
    {
      id: 'check4',
      stepNum: 'CHECK 4',
      title: 'Issuer Source',
      subtitle: 'DigiLocker / university record',
      icon: Landmark,
      data: fiveStack?.check4,
    },
    {
      id: 'check5',
      stepNum: 'CHECK 5',
      title: 'Cross-Document Match',
      subtitle: 'Names, dates, photo',
      icon: UserCheck,
      data: fiveStack?.check5,
    },
  ];

  const selectedCheckConfig = checkConfigs.find((c) => c.id === selectedCheckId) || checkConfigs[0];

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 flex flex-col font-sans">
      {/* ------------------------------------------------------------------- */}
      {/* TOP INSTITUTIONAL NAVIGATION BAR (Sticky White Header)              */}
      {/* ------------------------------------------------------------------- */}
      <header className="bg-white/95 backdrop-blur-md border-b border-slate-200/90 sticky top-0 z-40 px-6 py-3.5 shadow-[0_1px_2px_0_rgba(15,23,42,0.03)]">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 text-white p-2 rounded-lg shadow-sm">
              <ShieldCheck className="w-5 h-5" strokeWidth={2} />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-bold text-slate-900 tracking-tight text-lg">
                  DocTamperGuard
                </span>
                <span className="bg-slate-100 text-slate-600 font-mono text-[11px] font-medium px-2.5 py-0.5 rounded-full border border-slate-200/80 tracking-wide uppercase">
                  Enterprise KYC & Credential Verification
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 bg-emerald-50 text-emerald-700 border border-emerald-200/90 px-3 py-1 rounded-full text-xs font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span>Verification Engine Active</span>
            </div>

            {result?.processing_time_ms && (
              <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-500 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-md font-mono">
                <span>Latency:</span>
                <span className="font-semibold text-slate-800">
                  {(result.processing_time_ms / 1000).toFixed(2)}s
                </span>
              </div>
            )}

            <button
              onClick={handleResetSession}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200/90 hover:bg-slate-50 text-slate-700 rounded-lg text-xs font-medium transition shadow-sm active:scale-95"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-500" strokeWidth={1.75} />
              <span>Reset Session</span>
            </button>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------------- */}
      {/* MAIN WIDESCREEN DASHBOARD (12 Columns)                              */}
      {/* ------------------------------------------------------------------- */}
      <main className="max-w-[1600px] w-full mx-auto p-6 flex-1">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* =============================================================== */}
          {/* LEFT COLUMN: Document Intake & Benchmark Suite (4 Columns)     */}
          {/* =============================================================== */}
          <div className="lg:col-span-4 space-y-5">
            {/* 1. Applicant Identity Context Card */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
              <div className="flex items-center gap-2 mb-1.5">
                <UserCheck className="w-4 h-4 text-indigo-600" strokeWidth={1.75} />
                <h2 className="text-sm font-bold text-slate-900 tracking-tight">
                  Applicant Identity Context
                </h2>
              </div>
              <p className="text-xs text-slate-500 mb-4">
                Enables Check 5 cross-referencing against submitted credentials.
              </p>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Candidate Full Name
                  </label>
                  <input
                    type="text"
                    value={applicantName}
                    onChange={(e) => setApplicantName(e.target.value)}
                    placeholder="e.g. AARAV SHARMA"
                    className="w-full px-3 py-2 text-xs bg-slate-50/80 border border-slate-200 rounded-lg text-slate-900 font-medium focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-500/15 outline-none transition"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Date of Birth
                  </label>
                  <input
                    type="text"
                    value={applicantDob}
                    onChange={(e) => setApplicantDob(e.target.value)}
                    placeholder="DD-MM-YYYY (e.g. 15-08-2006)"
                    className="w-full px-3 py-2 text-xs bg-slate-50/80 border border-slate-200 rounded-lg text-slate-900 font-medium focus:bg-white focus:border-indigo-600 focus:ring-2 focus:ring-indigo-500/15 outline-none transition"
                  />
                </div>
              </div>
            </div>

            {/* 2. Document Dropzone Card */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
              <div className="flex items-center gap-2 mb-1.5">
                <ScanLine className="w-4 h-4 text-indigo-600" strokeWidth={1.75} />
                <h2 className="text-sm font-bold text-slate-900 tracking-tight">
                  Document Ingestion
                </h2>
              </div>
              <p className="text-xs text-slate-500 mb-4">
                Upload physical scan, marksheet photo, or academic degree.
              </p>

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  if (e.dataTransfer.files?.[0]) {
                    handleFileSelect(e.dataTransfer.files[0]);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                  isDragging
                    ? 'border-indigo-600 bg-indigo-50/50'
                    : 'border-slate-300 hover:border-indigo-500 bg-slate-50/50 hover:bg-indigo-50/20'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,.pdf,image/jpeg,image/png,application/pdf"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                  }}
                />

                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <ScanLine className="w-6 h-6" strokeWidth={1.75} />
                </div>

                <div className="text-xs font-semibold text-slate-800 mb-1">
                  Click to select or drag document here
                </div>
                <div className="text-[11px] text-slate-500 mb-3">
                  PDF, JPEG, PNG • Max 10MB
                </div>

                <div className="inline-flex items-center gap-1.5 text-[10px] font-mono font-medium text-slate-500 bg-white border border-slate-200 px-2.5 py-0.5 rounded-full shadow-2xs">
                  <span>Tamper-Preserving Lossless Pipeline</span>
                </div>
              </div>
            </div>

            {/* 3. Standardized Benchmark Test Suite */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-5 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-indigo-600" strokeWidth={1.75} />
                  <h2 className="text-sm font-bold text-slate-900 tracking-tight">
                    Institutional Test Scenarios
                  </h2>
                </div>
                <span className="text-[11px] font-mono text-slate-400">1-Click Audit</span>
              </div>
              <p className="text-xs text-slate-500 mb-4">
                Ground-truth benchmark test files targeting the 5 fraud vectors.
              </p>

              <div className="space-y-2.5">
                {BENCHMARK_SCENARIOS.map((sc) => {
                  const isSelected = selectedScenarioId === sc.id;
                  const isFail = sc.expected === 'FAIL';
                  return (
                    <div
                      key={sc.id}
                      onClick={() => handleTriggerScenario(sc)}
                      className={`group border rounded-lg p-3 cursor-pointer transition-all ${
                        isSelected
                          ? 'border-indigo-600 bg-indigo-50/40 shadow-xs'
                          : 'border-slate-200 hover:border-indigo-300 bg-white hover:bg-slate-50/60'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-slate-400 group-hover:text-indigo-600">
                            {sc.index}
                          </span>
                          <span className="text-xs font-semibold text-slate-800">
                            {sc.title}
                          </span>
                        </div>
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                            isFail
                              ? 'bg-rose-50 text-rose-700 border-rose-200'
                              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          }`}
                        >
                          {sc.expected}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-500">
                        <span className="font-mono text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded border border-slate-200/80">
                          {sc.tag}
                        </span>
                        <div className="flex items-center gap-1 text-slate-400 group-hover:text-indigo-600 transition">
                          <span className="text-[10px]">Audit</span>
                          <ChevronRight className="w-3 h-3" strokeWidth={2} />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* =============================================================== */}
          {/* RIGHT COLUMN: Verification Results & Five-Check Stack (8 Cols) */}
          {/* =============================================================== */}
          <div className="lg:col-span-8 space-y-6">
            {/* Loading Banner */}
            {loading && (
              <div className="bg-white border border-indigo-200 rounded-xl p-6 shadow-sm flex items-center gap-4">
                <div className="w-9 h-9 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
                  <ScanLine className="w-5 h-5 animate-spin" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-bold text-slate-900 mb-0.5">
                    Executing Five-Check Verification Stack...
                  </div>
                  <div className="text-xs text-slate-600 font-mono">
                    {loadingStep || 'Evaluating forensic layers...'}
                  </div>
                </div>
              </div>
            )}

            {/* Error Banner */}
            {error && (
              <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 flex items-center gap-3 text-rose-800">
                <FileWarning className="w-5 h-5 text-rose-600 shrink-0" strokeWidth={2} />
                <div className="text-xs font-medium">{error}</div>
              </div>
            )}

            {/* 1. Executive Decision Banner */}
            {result && (
              <div
                className={`bg-white border rounded-xl p-6 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)] border-l-4 ${
                  currentVerdict === 'PASS'
                    ? 'border-l-emerald-500 border-slate-200/90'
                    : currentVerdict === 'REFER'
                    ? 'border-l-amber-500 border-slate-200/90'
                    : 'border-l-rose-600 border-slate-200/90'
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border tracking-wide uppercase ${
                        currentVerdict === 'PASS'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : currentVerdict === 'REFER'
                          ? 'bg-amber-50 text-amber-700 border-amber-200'
                          : 'bg-rose-50 text-rose-700 border-rose-200'
                      }`}
                    >
                      {currentVerdict === 'PASS' ? (
                        <ShieldCheck className="w-4 h-4 text-emerald-600" strokeWidth={2} />
                      ) : (
                        <ShieldAlert className="w-4 h-4 text-rose-600" strokeWidth={2} />
                      )}
                      <span>OFFICIAL VERDICT: {currentVerdict}</span>
                    </span>

                    <span className="text-xs text-slate-500">
                      Policy Status:{' '}
                      <strong className="text-slate-800 font-semibold">{result.status}</strong>
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-[11px] text-slate-400 font-mono">CONFIDENCE</div>
                      <div className="text-lg font-bold font-mono text-slate-900 leading-tight">
                        {result.confidence_score?.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Extracted Document Metadata Pills */}
                {fiveStack?.step0?.data && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mb-3.5 pt-3 border-t border-slate-100">
                    <div className="bg-slate-50 border border-slate-200/70 rounded-lg px-3 py-2">
                      <div className="text-[10px] text-slate-400 uppercase font-mono font-medium">
                        Recognised Institution
                      </div>
                      <div className="text-xs font-bold text-slate-800 truncate">
                        {fiveStack.step0.data.institution_name || 'UNKNOWN'}
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200/70 rounded-lg px-3 py-2">
                      <div className="text-[10px] text-slate-400 uppercase font-mono font-medium">
                        Candidate Full Name
                      </div>
                      <div className="text-xs font-bold text-slate-800 truncate">
                        {fiveStack.step0.data.candidate_name || 'UNKNOWN'}
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-200/70 rounded-lg px-3 py-2">
                      <div className="text-[10px] text-slate-400 uppercase font-mono font-medium">
                        Roll Number / ID
                      </div>
                      <div className="text-xs font-bold text-slate-800 font-mono truncate">
                        {fiveStack.step0.data.roll_no || 'UNKNOWN'}
                      </div>
                    </div>
                  </div>
                )}

                <p className="text-xs text-slate-600 leading-relaxed font-medium">
                  {result.summary_message}
                </p>
              </div>
            )}

            {/* 2. THE FIVE-CHECK VERIFICATION STACK (Interactive Flowchart) */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-6 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1">
                <div>
                  <h2 className="text-sm font-bold text-slate-900 tracking-tight uppercase">
                    THE FIVE-CHECK VERIFICATION STACK
                  </h2>
                  <p className="text-xs text-slate-500">
                    Each check covers a gap the others leave open.
                  </p>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Multi-Layer Forensic Defense
                </div>
              </div>

              {/* 2-Row Connected Pipeline Grid */}
              <div className="mt-5 space-y-4">
                {/* Row 1: STEP 0 -> CHECK 1 -> CHECK 2 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 relative">
                  {checkConfigs.slice(0, 3).map((chk) => {
                    const isSelected = selectedCheckId === chk.id;
                    const st = chk.data?.status || 'PENDING';
                    const isFail = st === 'FAIL';
                    const isPass = st === 'PASS';
                    const isRefer = st === 'REFER';
                    const IconComp = chk.icon;

                    return (
                      <div
                        key={chk.id}
                        onClick={() => setSelectedCheckId(chk.id)}
                        className={`relative rounded-xl p-4 border cursor-pointer transition-all ${
                          isSelected
                            ? 'border-indigo-600 ring-2 ring-indigo-500/15 bg-indigo-50/20 shadow-xs'
                            : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-[11px] font-bold text-indigo-600">
                            {chk.stepNum}
                          </span>
                          <span
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                              isPass
                                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                : isFail
                                ? 'bg-rose-50 text-rose-700 border-rose-200'
                                : isRefer
                                ? 'bg-amber-50 text-amber-700 border-amber-200'
                                : 'bg-slate-100 text-slate-500 border-slate-200'
                            }`}
                          >
                            {st}
                          </span>
                        </div>

                        <div className="flex items-center gap-2 mb-1">
                          <IconComp className="w-4 h-4 text-slate-700" strokeWidth={1.75} />
                          <div className="text-xs font-bold text-slate-900 leading-tight">
                            {chk.title}
                          </div>
                        </div>

                        <div className="text-[11px] text-slate-500 italic">
                          ({chk.subtitle})
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Row 2: CHECK 3 -> CHECK 4 -> CHECK 5 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 relative">
                  {checkConfigs.slice(3, 6).map((chk) => {
                    const isSelected = selectedCheckId === chk.id;
                    const st = chk.data?.status || 'PENDING';
                    const isFail = st === 'FAIL';
                    const isPass = st === 'PASS';
                    const isRefer = st === 'REFER';
                    const IconComp = chk.icon;

                    return (
                      <div
                        key={chk.id}
                        onClick={() => setSelectedCheckId(chk.id)}
                        className={`relative rounded-xl p-4 border cursor-pointer transition-all ${
                          isSelected
                            ? 'border-indigo-600 ring-2 ring-indigo-500/15 bg-indigo-50/20 shadow-xs'
                            : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-[11px] font-bold text-indigo-600">
                            {chk.stepNum}
                          </span>
                          <span
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                              isPass
                                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                : isFail
                                ? 'bg-rose-50 text-rose-700 border-rose-200'
                                : isRefer
                                ? 'bg-amber-50 text-amber-700 border-amber-200'
                                : 'bg-slate-100 text-slate-500 border-slate-200'
                            }`}
                          >
                            {st}
                          </span>
                        </div>

                        <div className="flex items-center gap-2 mb-1">
                          <IconComp className="w-4 h-4 text-slate-700" strokeWidth={1.75} />
                          <div className="text-xs font-bold text-slate-900 leading-tight">
                            {chk.title}
                          </div>
                        </div>

                        <div className="text-[11px] text-slate-500 italic">
                          ({chk.subtitle})
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Interactive Inspection Drawer */}
              {selectedCheckConfig && (
                <div className="mt-4 bg-slate-50/80 border border-slate-200 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-200">
                    <div className="flex items-center gap-2">
                      <selectedCheckConfig.icon className="w-4 h-4 text-indigo-600" strokeWidth={1.75} />
                      <span className="text-xs font-bold text-slate-900 uppercase">
                        {selectedCheckConfig.stepNum}: {selectedCheckConfig.title}
                      </span>
                    </div>

                    <span
                      className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full border ${
                        selectedCheckConfig.data?.status === 'PASS'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : selectedCheckConfig.data?.status === 'FAIL'
                          ? 'bg-rose-50 text-rose-700 border-rose-200'
                          : 'bg-slate-100 text-slate-600 border-slate-200'
                      }`}
                    >
                      {selectedCheckConfig.data?.status || 'PENDING'}
                    </span>
                  </div>

                  <div className="text-xs font-semibold text-slate-800 mb-2">
                    {selectedCheckConfig.data?.verdict_title || 'Awaiting Document Scan Execution'}
                  </div>

                  {selectedCheckConfig.data?.details && (
                    <ul className="space-y-1.5 text-xs text-slate-600 font-mono">
                      {selectedCheckConfig.data.details.map((d, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-slate-400">•</span>
                          <span>{d}</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  {/* If Step 0, display extracted subject grades table */}
                  {selectedCheckConfig.id === 'step0' && selectedCheckConfig.data?.data?.extracted_grades && (
                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <div className="text-[11px] font-bold text-slate-700 mb-1.5 font-mono uppercase">
                        Extracted Table Row Grades:
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                        {Object.entries(selectedCheckConfig.data.data.extracted_grades).map(([code, gr]) => (
                          <div key={code} className="bg-white border border-slate-200 rounded px-2 py-1 text-center font-mono">
                            <div className="text-[10px] text-slate-400">{code}</div>
                            <div className="text-xs font-bold text-slate-800">{gr}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* If Check 3, display QR payload */}
                  {selectedCheckConfig.id === 'check3' && selectedCheckConfig.data?.qr_payload && (
                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <div className="text-[11px] font-bold text-slate-700 mb-1 font-mono uppercase">
                        Signed QR Payload Data:
                      </div>
                      <pre className="text-[11px] bg-white border border-slate-200 rounded p-2.5 overflow-x-auto text-slate-700">
                        {JSON.stringify(selectedCheckConfig.data.qr_payload, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Three Pill Outcomes & Institutional Quote */}
              <div className="mt-5 pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wide mr-1">
                    Outcomes:
                  </span>
                  <span
                    className={`text-xs font-bold px-3 py-1 rounded-full border transition-all ${
                      currentVerdict === 'PASS'
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
                        : 'bg-emerald-50 text-emerald-700 border-emerald-200 opacity-60'
                    }`}
                  >
                    PASS
                  </span>
                  <span
                    className={`text-xs font-bold px-3 py-1 rounded-full border transition-all ${
                      currentVerdict === 'REFER'
                        ? 'bg-amber-500 text-white border-amber-500 shadow-sm'
                        : 'bg-amber-50 text-amber-700 border-amber-200 opacity-60'
                    }`}
                  >
                    REFER
                  </span>
                  <span
                    className={`text-xs font-bold px-3 py-1 rounded-full border transition-all ${
                      currentVerdict === 'FAIL'
                        ? 'bg-rose-600 text-white border-rose-600 shadow-sm'
                        : 'bg-rose-50 text-rose-700 border-rose-200 opacity-60'
                    }`}
                  >
                    FAIL
                  </span>
                </div>

                <div className="text-[11px] text-slate-500 italic max-w-md text-right sm:text-right text-center">
                  "Forensics asks whether the file was changed. The issuer check asks whether the facts are true. Consistency asks whether they belong to this person."
                </div>
              </div>
            </div>

            {/* 3. Optical Inspection Workbench (Segmented Light Tabs) */}
            <div className="bg-white border border-slate-200/90 rounded-xl p-6 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-2">
                  <FileCheck2 className="w-4 h-4 text-indigo-600" strokeWidth={1.75} />
                  <h2 className="text-sm font-bold text-slate-900 tracking-tight">
                    Optical Inspection Workbench
                  </h2>
                </div>

                {/* Segmented Light Tabs */}
                <div className="bg-slate-100 p-1 rounded-lg border border-slate-200/80 inline-flex self-start sm:self-auto">
                  <button
                    onClick={() => setActiveTab('verdict')}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                      activeTab === 'verdict'
                        ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Annotated Verdict
                  </button>
                  <button
                    onClick={() => setActiveTab('ela')}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                      activeTab === 'ela'
                        ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    ELA Heatmap
                  </button>
                  <button
                    onClick={() => setActiveTab('split')}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                      activeTab === 'split'
                        ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Side-by-Side
                  </button>
                  <button
                    onClick={() => setActiveTab('ocr')}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                      activeTab === 'ocr'
                        ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    OCR Row Grid
                  </button>
                </div>
              </div>

              {/* Studio Frame Viewer */}
              <div className="bg-slate-100/80 border border-slate-200/90 rounded-xl p-4 flex items-center justify-center min-h-[460px]">
                {result?.visuals ? (
                  <div className="w-full flex justify-center">
                    {activeTab === 'verdict' && (
                      <img
                        src={result.visuals.verdict_overlay_b64}
                        alt="Annotated Verdict"
                        className="max-h-[700px] w-auto rounded-lg shadow-sm border border-slate-200 bg-white object-contain"
                      />
                    )}
                    {activeTab === 'ela' && (
                      <img
                        src={result.visuals.ela_heatmap_b64}
                        alt="ELA Compression Map"
                        className="max-h-[700px] w-auto rounded-lg shadow-sm border border-slate-200 bg-white object-contain"
                      />
                    )}
                    {activeTab === 'ocr' && (
                      <img
                        src={result.visuals.segmentation_map_b64}
                        alt="OCR Row-Grid Map"
                        className="max-h-[700px] w-auto rounded-lg shadow-sm border border-slate-200 bg-white object-contain"
                      />
                    )}
                    {activeTab === 'split' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
                        <div className="space-y-1.5 text-center">
                          <span className="text-[11px] font-mono text-slate-500 font-medium">
                            Aligned Capture
                          </span>
                          <img
                            src={result.visuals.aligned_image_b64}
                            alt="Aligned RGB"
                            className="w-full rounded-lg shadow-sm border border-slate-200 bg-white"
                          />
                        </div>
                        <div className="space-y-1.5 text-center">
                          <span className="text-[11px] font-mono text-slate-500 font-medium">
                            ELA Compression Energy
                          </span>
                          <img
                            src={result.visuals.ela_heatmap_b64}
                            alt="ELA Residual"
                            className="w-full rounded-lg shadow-sm border border-slate-200 bg-white"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ) : previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Upload Preview"
                    className="max-h-[500px] w-auto rounded-lg shadow-sm border border-slate-200 bg-white object-contain"
                  />
                ) : (
                  <div className="text-center py-16 text-slate-400">
                    <ScanLine className="w-12 h-12 mx-auto mb-3 text-slate-300" strokeWidth={1.5} />
                    <div className="text-xs font-semibold text-slate-600 mb-1">
                      No document loaded in optical workbench
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Select an institutional benchmark on the left or upload a certificate.
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 4. Isolated Micro-Forgery Evidence Cards (When Tampering Flagged) */}
            {result?.visuals?.zoom_crops?.length > 0 && (
              <div className="bg-white border border-rose-200 rounded-xl p-6 shadow-[0_1px_3px_0_rgba(15,23,42,0.03)] bg-rose-50/15">
                <div className="flex items-center gap-2 mb-2">
                  <ShieldAlert className="w-4 h-4 text-rose-600" strokeWidth={1.75} />
                  <h2 className="text-sm font-bold text-slate-900 tracking-tight">
                    Micro-Forgery Forensic Evidence Patches
                  </h2>
                </div>
                <p className="text-xs text-slate-500 mb-4">
                  Isolated target glyphs exhibiting sub-pixel spatial baseline drift or ELA compression anomalies.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.visuals.zoom_crops.map((z, idx) => (
                    <div
                      key={idx}
                      className="bg-white border border-rose-200/90 rounded-lg p-4 shadow-2xs space-y-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs text-slate-800">
                          {z.label}
                        </span>
                        <span className="text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded">
                          CONF: {z.confidence?.toFixed(1)}%
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div className="text-center">
                          <div className="text-[10px] font-mono text-slate-400 mb-1">
                            Original RGB Crop
                          </div>
                          <img
                            src={z.patch_b64}
                            alt="RGB Patch"
                            className="w-full h-28 object-contain bg-slate-50 border border-slate-200 rounded"
                          />
                        </div>
                        <div className="text-center">
                          <div className="text-[10px] font-mono text-slate-400 mb-1">
                            ELA Heatmap Residual
                          </div>
                          <img
                            src={z.ela_crop_b64}
                            alt="ELA Patch"
                            className="w-full h-28 object-contain bg-slate-50 border border-slate-200 rounded"
                          />
                        </div>
                      </div>

                      {z.reasons?.length > 0 && (
                        <div className="text-[11px] text-slate-600 space-y-1 font-mono pt-1">
                          {z.reasons.map((r, ri) => (
                            <div key={ri} className="flex items-start gap-1.5">
                              <span className="text-rose-500 font-bold">•</span>
                              <span>{r}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 5. Audit Compliance JSON Drawer */}
            {result?.audit_report && (
              <div className="bg-white border border-slate-200/90 rounded-xl overflow-hidden shadow-[0_1px_3px_0_rgba(15,23,42,0.03)]">
                <button
                  onClick={() => setAuditOpen(!auditOpen)}
                  className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-50 transition text-left"
                >
                  <div className="flex items-center gap-2.5">
                    <SlidersHorizontal className="w-4 h-4 text-slate-600" strokeWidth={1.75} />
                    <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
                      Official JSON Compliance Record
                    </span>
                    <span className="text-[11px] font-mono text-slate-400">
                      ({Object.keys(result.audit_report).length} fields)
                    </span>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="text-xs text-indigo-600 font-medium">
                      {auditOpen ? 'Collapse' : 'Expand'}
                    </span>
                    <ChevronDown
                      className={`w-4 h-4 text-slate-400 transition-transform ${
                        auditOpen ? 'rotate-180' : ''
                      }`}
                      strokeWidth={2}
                    />
                  </div>
                </button>

                {auditOpen && (
                  <div className="border-t border-slate-200 bg-slate-50/60 p-5">
                    <div className="flex justify-end mb-2">
                      <button
                        onClick={handleCopyAudit}
                        className="inline-flex items-center gap-1.5 px-3 py-1 bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 text-xs font-medium rounded shadow-2xs transition"
                      >
                        {copied ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-600" strokeWidth={2} />
                            <span className="text-emerald-700">Copied to Clipboard</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5 text-slate-500" strokeWidth={1.75} />
                            <span>Copy Audit JSON</span>
                          </>
                        )}
                      </button>
                    </div>

                    <pre className="text-xs font-mono bg-white border border-slate-200 rounded-lg p-4 text-slate-800 overflow-x-auto max-h-96 leading-relaxed shadow-2xs">
                      {JSON.stringify(result.audit_report, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-[1600px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            DocTamperGuard Enterprise Suite — Passive Vision Forensics & Multi-Check Cryptographic Verification
          </span>
          <span className="font-mono text-[11px] text-slate-400">
            Compliant with DigiLocker, ISO/IEC 27001 & UGC Accreditation Guidelines
          </span>
        </div>
      </footer>
    </div>
  );
}
