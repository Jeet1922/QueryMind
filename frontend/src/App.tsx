import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowDownToLine,
  ArrowUp,
  BarChart3,
  Check,
  ChevronDown,
  Clock3,
  Database,
  History,
  LayoutDashboard,
  LoaderCircle,
  MessageSquareText,
  Plus,
  Search,
  Send,
  Sparkles,
  Table2,
  X,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type DataRow = Record<string, string | number | boolean | null>;
type QueryResponse = {
  request_id: string;
  question: string;
  intent: string;
  answer: string;
  insight_summary?: string | null;
  visualization: { type: string; title: string };
  supporting_data: DataRow[];
  execution_details: {
    selected_procedures: string[];
    rows_returned: number;
    execution_time_ms: number;
  };
  ragsources: string[];
};
type HistoryEntry = {
  request_id: string;
  question: string;
  intent: string;
  status: string;
};
type Turn = { question: string; result: QueryResponse };
type ModelRouteStatus = { action: string; provider: string; model: string; configured: boolean };

type ChartPoint = DataRow & { label: string; value: number };

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';
const suggestions = [
  'Which AI tools have the highest adoption?',
  'Compare AI adoption across departments',
  'Which departments have the highest adoption risk?',
  'Show the adoption forecast for ChatGPT',
];
const chartPalette = ['#ef7658', '#3a9188', '#e3b34d', '#6984b9', '#b4779a', '#8a9d64'];

function chartData(rows: DataRow[]): { points: ChartPoint[]; labelKey: string; valueKey: string } {
  if (!rows.length) return { points: [], labelKey: '', valueKey: '' };
  const keys = Object.keys(rows[0]);
  const numericKeys = keys.filter((key) => typeof rows[0][key] === 'number');
  const valueKey = numericKeys.find((key) => !/(^id$|_id$)/i.test(key) && /(adoption|usage|rate|score|count|points|anomaly|duration)/i.test(key))
    ?? numericKeys.find((key) => !/(^id$|_id$)/i.test(key))
    ?? numericKeys[0]
    ?? '';
  const labelKey = keys.find((key) => key !== valueKey && typeof rows[0][key] === 'string') ?? keys[0];
  const points = rows
    .filter((row) => typeof row[valueKey] === 'number')
    .map((row) => ({ ...row, label: String(row[labelKey] ?? ''), value: Number(row[valueKey]) }));
  return { points, labelKey, valueKey };
}

function niceLabel(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [modelRoutes, setModelRoutes] = useState<ModelRouteStatus[]>([]);
  const [databaseBackend, setDatabaseBackend] = useState('checking');
  const [activeTurn, setActiveTurn] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const current = turns[activeTurn];
  const rows = current?.result.supporting_data ?? [];
  const { points, labelKey, valueKey } = useMemo(() => chartData(rows), [rows]);
  const visualization = current?.result.visualization.type ?? 'bar';
  const isLine = visualization === 'line';
  const isDonut = visualization === 'donut';
  const isHorizontal = visualization === 'horizontal_bar';

  async function refreshHistory() {
    try {
      const response = await fetch(`${API_BASE}/api/v1/query/history`);
      if (response.ok) setHistory((await response.json()) as HistoryEntry[]);
    } catch {
      // History is supplemental; query submission reports connection errors.
    }
  }

  useEffect(() => {
    void refreshHistory();
    fetch(`${API_BASE}/api/v1/health`)
      .then((response) => response.ok ? response.json() as Promise<{ database_backend: string }> : null)
      .then((status) => setDatabaseBackend(status?.database_backend ?? 'unavailable'))
      .catch(() => setDatabaseBackend('unavailable'));
    fetch(`${API_BASE}/api/v1/models/status`)
      .then((response) => response.ok ? response.json() as Promise<ModelRouteStatus[]> : [])
      .then(setModelRoutes)
      .catch(() => setModelRoutes([]));
  }, []);

  async function ask(nextQuestion: string) {
    const cleaned = nextQuestion.trim();
    if (!cleaned || loading) return;
    setLoading(true);
    setError(null);
    setQuestion(cleaned);
    try {
      const response = await fetch(`${API_BASE}/api/v1/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: cleaned }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? 'The query could not be completed.');
      const result = payload as QueryResponse;
      setTurns((previous) => [...previous, { question: cleaned, result }]);
      setActiveTurn(turns.length);
      setQuestion('');
      setView(result.supporting_data?.length ? 'chart' : 'table');
      void refreshHistory();
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Unable to connect to QueryMind.');
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(question);
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void ask(question);
    }
  }

  function exportCsv() {
    if (!rows.length) return;
    const columns = Object.keys(rows[0]);
    const csv = [columns.join(','), ...rows.map((row) => columns.map((column) => JSON.stringify(row[column] ?? '')).join(','))].join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    link.download = 'querymind-query-results.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function startNewQuery() {
    setTurns([]);
    setActiveTurn(0);
    setQuestion('');
    setError(null);
  }

  return (
    <div className="workspace">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand-row">
          <a className="brand" href="#home" aria-label="QueryMind home">
            <span className="brand-mark"><Activity size={19} strokeWidth={2.4} /></span>
            <span>Query<span className="brand-light">Mind</span></span>
          </a>
          <button className="sidebar-close icon-button" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
        </div>
        <button className="new-query-button" onClick={startNewQuery}><Plus size={17} /> New analysis</button>
        <div className="nav-caption">WORKSPACE</div>
        <button className="nav-item nav-selected"><MessageSquareText size={17} /> Ask QueryMind</button>
        <button className="nav-item" onClick={() => current && setView('chart')}><LayoutDashboard size={17} /> Analytics canvas</button>
        <div className="history-heading"><span className="nav-caption">RECENT QUESTIONS</span><History size={14} /></div>
        <div className="history-list">
          {history.length ? history.slice(0, 12).map((item) => (
            <button className="history-item" key={item.request_id} title={item.question} onClick={() => setQuestion(item.question)}>
              <span className="history-dot" />{item.question}
            </button>
          )) : <p className="empty-history">Your recent analyses will appear here.</p>}
        </div>
        <div className="sidebar-foot"><span className="status-indicator" />Local analytics connected</div>
      </aside>

      {sidebarOpen && <button className="sidebar-backdrop" aria-label="Close navigation overlay" onClick={() => setSidebarOpen(false)} />}

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs">
            <button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><LayoutDashboard size={19} /></button>
            <span>Workspace</span><span className="breadcrumb-slash">/</span><strong>Ask QueryMind</strong>
          </div>
          <div className="topbar-right"><span className="model-status"><span className="status-indicator" /> {databaseBackend === 'postgresql' ? 'Neon PostgreSQL' : databaseBackend === 'sqlite' ? 'SQLite demo' : 'Checking database'}</span><span className="model-route-status" title={modelRoutes.map((route) => `${route.action}: ${route.configured ? route.model : 'not configured'}`).join('\n')}>{modelRoutes.filter((route) => route.configured).length}/{modelRoutes.length || 4} models</span><button className="user-avatar" title="QueryMind workspace">QM</button></div>
        </header>

        <div className="page-content">
          <section className="query-intro">
            <p className="section-kicker"><Sparkles size={14} /> DECISION INTELLIGENCE</p>
            <h1>Ask your data<br /><span>what matters.</span></h1>
            <p className="intro-copy">Explore AI adoption, team performance, and business signals in plain English.</p>
          </section>

          <section className="conversation" aria-live="polite">
            {turns.length === 0 ? (
              <div className="welcome-block">
                <div className="welcome-icon"><BarChart3 size={22} /></div>
                <div><h2>What would you like to understand?</h2><p>Ask a question or start with one of these analyses.</p></div>
                <div className="suggestion-grid">
                  {suggestions.map((suggestion, index) => (
                    <button className="suggestion" key={suggestion} onClick={() => void ask(suggestion)}>
                      <span className={`suggestion-index suggestion-${index}`}>0{index + 1}</span><span>{suggestion}</span><ArrowUp size={15} />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="conversation-turn">
                <div className="user-message"><span className="user-message-icon">You</span><p>{current.question}</p></div>
                <div className="assistant-message">
                  <div className="assistant-mark"><Sparkles size={15} /></div>
                  <div className="assistant-body">
                    <div className="answer-label">QUERYMIND <span>·</span> ENGLISH SUMMARY</div>
                    <p className="answer-text">{current.result.answer}</p>
                    {current.result.insight_summary && <p className="insight-summary"><Sparkles size={13} /> {current.result.insight_summary}</p>}
                    <div className="answer-foot"><Check size={14} /> Grounded in {current.result.execution_details.rows_returned} returned records</div>
                  </div>
                </div>

                <section className="analytics-panel">
                  <div className="analytics-header">
                    <div><span className="analytics-kicker"><BarChart3 size={14} /> ANALYTICS CANVAS</span><h2>{current.result.visualization.title}</h2></div>
                    <div className="analytics-actions">
                      <div className="view-switch" role="tablist" aria-label="Result view">
                        <button role="tab" aria-selected={view === 'chart'} className={view === 'chart' ? 'active' : ''} onClick={() => setView('chart')} title="Chart view"><BarChart3 size={16} /></button>
                        <button role="tab" aria-selected={view === 'table'} className={view === 'table' ? 'active' : ''} onClick={() => setView('table')} title="Table view"><Table2 size={16} /></button>
                      </div>
                      <button className="export-button" onClick={exportCsv} disabled={!rows.length}><ArrowDownToLine size={15} /><span>Export</span></button>
                    </div>
                  </div>

                  {view === 'chart' && points.length > 0 ? (
                    <div className="chart-area">
                      <div className="chart-legend"><span className="legend-swatch" />{niceLabel(valueKey)}</div>
                      <ResponsiveContainer width="100%" height="100%">
                        {isDonut ? (
                          <PieChart><Pie data={points} dataKey="value" nameKey="label" innerRadius={70} outerRadius={112} paddingAngle={3} stroke="none">{points.map((point, index) => <Cell key={point.label} fill={chartPalette[index % chartPalette.length]} />)}</Pie><Tooltip formatter={(value) => [Number(value).toFixed(2), niceLabel(valueKey)]} /></PieChart>
                        ) : isLine ? (
                          <LineChart data={points} margin={{ top: 14, right: 20, left: 0, bottom: 6 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9e6e0" /><XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: '#817f78', fontSize: 11 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: '#817f78', fontSize: 11 }} width={38} /><Tooltip /><Line type="monotone" dataKey="value" stroke="#3a9188" strokeWidth={3} dot={{ r: 4, fill: '#3a9188', stroke: '#fff', strokeWidth: 2 }} activeDot={{ r: 6 }} /></LineChart>
                        ) : (
                          <BarChart data={points} layout={isHorizontal ? 'vertical' : 'horizontal'} margin={{ top: 12, right: 18, left: isHorizontal ? 12 : 0, bottom: 8 }}><CartesianGrid strokeDasharray="3 5" vertical={!isHorizontal} horizontal={isHorizontal} stroke="#e9e6e0" /><XAxis type={isHorizontal ? 'number' : 'category'} dataKey={isHorizontal ? undefined : 'label'} axisLine={false} tickLine={false} tick={{ fill: '#817f78', fontSize: 11 }} /><YAxis type={isHorizontal ? 'category' : 'number'} dataKey={isHorizontal ? 'label' : undefined} axisLine={false} tickLine={false} tick={{ fill: '#817f78', fontSize: 11 }} width={isHorizontal ? 130 : 42} /><Tooltip cursor={{ fill: '#f4f0e9' }} /><Bar dataKey="value" radius={isHorizontal ? [0, 5, 5, 0] : [5, 5, 0, 0]} maxBarSize={40}>{points.map((point, index) => <Cell key={point.label} fill={chartPalette[index % chartPalette.length]} />)}</Bar></BarChart>
                        )}
                      </ResponsiveContainer>
                    </div>
                  ) : view === 'chart' ? <div className="empty-chart"><Database size={22} /><p>No numeric results are available to chart.</p></div> : null}

                  {view === 'table' && rows.length ? (
                    <div className="table-scroll"><table><thead><tr>{Object.keys(rows[0]).map((column) => <th key={column}>{niceLabel(column)}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={`${rowIndex}-${String(row[labelKey])}`}>{Object.entries(row).map(([column, value]) => <td key={column}>{typeof value === 'number' ? Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3) : String(value ?? '—')}</td>)}</tr>)}</tbody></table></div>
                  ) : null}

                  <footer className="analytics-footer">
                    <span><Database size={13} /> {current.result.execution_details.rows_returned} rows</span>
                    <span><Clock3 size={13} /> {current.result.execution_details.execution_time_ms} ms</span>
                    <button className="source-toggle" title={`Sources: ${current.result.ragsources.join(', ')}`}><Search size={13} /> Sources <ChevronDown size={13} /></button>
                  </footer>
                </section>
                {current.result.execution_details.selected_procedures.length > 0 && <p className="procedure-note">Analysis: {current.result.execution_details.selected_procedures.map(niceLabel).join(' · ')}</p>}
              </div>
            )}
            {error && <div className="error-box" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}><X size={16} /></button></div>}
          </section>

          <form className="composer" onSubmit={onSubmit}>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={onComposerKeyDown} rows={2} placeholder="Ask a question about adoption, teams, or business performance..." aria-label="Ask an analytics question" />
            <div className="composer-bottom"><span><Sparkles size={13} /> Answers in English · charts from your data</span><div><span className="key-hint">ENTER TO SEND</span><button className="send-button" type="submit" disabled={loading || !question.trim()} aria-label="Send question">{loading ? <LoaderCircle size={17} className="spin" /> : <Send size={16} />}</button></div></div>
          </form>
          <p className="disclaimer">QueryMind summarizes available records. Forecasts and risk scores are model-generated estimates.</p>
        </div>
      </main>
    </div>
  );
}
