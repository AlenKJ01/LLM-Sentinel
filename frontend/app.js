/* =============================================================================
   app.js — Adaptive LLM Orchestrator Frontend Logic
   -----------------------------------------------------------------------------
   Sections:
     1. Configuration
     2. DOM References
     3. Execute Pipeline
     4. Render Output
     5. Render Metadata
     6. Logs
     7. UI Helpers
     8. Init
   ============================================================================= */

/* ─── 1. Configuration ───────────────────────────────────────────────────── */

const API_BASE = ''; // Empty string = same origin. Change if API is on a different host.


/* ─── 2. DOM References ──────────────────────────────────────────────────── */

// Input controls
const taskInput      = document.getElementById('task-input');
const submitBtn      = document.getElementById('submit-btn');
const preferFast     = document.getElementById('prefer-fast');
const preferLocal    = document.getElementById('prefer-local');
const errorBanner    = document.getElementById('error-banner');

// Output area
const outputEmpty    = document.getElementById('output-empty');
const outputContent  = document.getElementById('output-content');

// Metadata cells
const metaModel      = document.getElementById('meta-model');
const metaProvider   = document.getElementById('meta-provider');
const metaType       = document.getElementById('meta-type');
const metaComplexity = document.getElementById('meta-complexity');
const metaLatency    = document.getElementById('meta-latency');
const metaRetries    = document.getElementById('meta-retries');
const metaConfVal    = document.getElementById('meta-confidence-value');
const confidenceBar  = document.getElementById('confidence-bar');

// Logs
const logsContainer  = document.getElementById('logs-container');
const refreshLogsBtn = document.getElementById('refresh-logs-btn');

// Footer
const footerReqId    = document.getElementById('footer-request-id');


/* ─── 3. Execute Pipeline ────────────────────────────────────────────────── */

/**
 * Sends the task to POST /execute and handles the full response cycle.
 */
submitBtn.addEventListener('click', async () => {
  const task = taskInput.value.trim();

  if (!task) {
    shakeBanner('Please enter a task.');
    return;
  }

  setLoading(true);
  clearOutput();
  clearError();

  try {
    const res = await fetch(`${API_BASE}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task,
        prefer_fast: preferFast.checked,
        prefer_local: preferLocal.checked,
        metadata: {}
      })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    renderOutput(data);
    renderMetadata(data);
    await fetchLogs();

  } catch (err) {
    showError(err.message || 'Unexpected error. Check the browser console.');
    console.error('[Orchestrator] Execute error:', err);
  } finally {
    setLoading(false);
  }
});


/* ─── 4. Render Output ───────────────────────────────────────────────────── */

/**
 * Display the model's response text in the output panel.
 * @param {Object} data - OrchestratorResult from the API.
 */
function renderOutput(data) {
  outputEmpty.style.display = 'none';
  outputContent.style.display = 'block';
  outputContent.textContent = data.output;
}

/** Hide the output and restore the placeholder. */
function clearOutput() {
  outputEmpty.style.display = 'block';
  outputContent.style.display = 'none';
  outputContent.textContent = '';
}


/* ─── 5. Render Metadata ─────────────────────────────────────────────────── */

/**
 * Populate the sidebar metadata grid from the API response.
 * @param {Object} data - OrchestratorResult from the API.
 */
function renderMetadata(data) {
  // Model + Provider
  metaModel.textContent    = data.model_used     || '—';
  metaProvider.textContent = data.model_provider || '—';
  metaProvider.className   = 'value';

  // Task classification
  metaType.textContent       = data.task_type  || '—';
  metaComplexity.textContent = data.complexity || '—';

  // Performance
  metaLatency.textContent = data.latency_ms
    ? `${data.latency_ms.toFixed(0)} ms`
    : '—';

  // Retries — colour-code: 0=green, 1-2=accent, 3+=warn
  const retries = data.retries ?? 0;
  metaRetries.textContent = retries;
  metaRetries.className = 'value ' + (
    retries === 0 ? 'success' :
    retries  <  3 ? ''        :
                    'warn'
  );

  // Validation confidence bar
  const conf = data.validation_confidence ?? 0;
  const pct  = Math.round(conf * 100);

  metaConfVal.textContent    = `${pct}%`;
  confidenceBar.style.width  = `${pct}%`;
  confidenceBar.style.background =
    conf >= 0.7 ? 'var(--success)' :
    conf >= 0.4 ? 'var(--accent)'  :
                  'var(--warn)';

  // Footer request ID
  footerReqId.textContent = data.request_id
    ? `REQ: ${data.request_id.slice(0, 8)}`
    : '';
}


/* ─── 6. Logs ────────────────────────────────────────────────────────────── */

/**
 * Fetch recent structured log entries from GET /logs and render them.
 */
async function fetchLogs() {
  try {
    const res = await fetch(`${API_BASE}/logs?limit=60`);
    if (!res.ok) return;
    const logs = await res.json();
    renderLogs(logs);
  } catch {
    // Logs are non-critical — fail silently
  }
}

/**
 * Render log entries into the sidebar logs container.
 * @param {Array} logs - Array of log entry objects from the API.
 */
function renderLogs(logs) {
  logsContainer.innerHTML = '';

  if (!logs.length) {
    logsContainer.innerHTML =
      '<div style="padding:16px;color:var(--muted);font-family:var(--mono);font-size:11px">No logs yet</div>';
    return;
  }

  logs.forEach(entry => {
    const el    = document.createElement('div');
    el.className = 'log-entry';

    const event = entry.event || 'unknown';
    const body  = buildLogBody(entry);

    el.innerHTML = `
      <span class="log-event ${event}">${event.replace(/_/g, ' ')}</span>
      <span class="log-body">${body}</span>
    `;

    logsContainer.appendChild(el);
  });
}

/**
 * Build a compact human-readable summary string from a log entry object.
 * Picks only the most relevant fields to avoid clutter.
 * @param {Object} entry - A single log entry.
 * @returns {string} HTML string.
 */
function buildLogBody(entry) {
  const parts = [];

  if (entry.model_provider) parts.push(`<strong>${entry.model_provider}</strong>`);
  if (entry.model_name)     parts.push(entry.model_name);
  if (entry.task_type)      parts.push(`type:<strong>${entry.task_type}</strong>`);
  if (entry.complexity)     parts.push(`complexity:<strong>${entry.complexity}</strong>`);
  if (entry.latency_ms)     parts.push(`${entry.latency_ms}ms`);
  if (entry.retries !== undefined) parts.push(`retries:${entry.retries}`);
  if (entry.confidence !== undefined)
    parts.push(`conf:${(entry.confidence * 100).toFixed(0)}%`);
  if (entry.strategy)  parts.push(`strategy:<strong>${entry.strategy}</strong>`);
  if (entry.error)
    parts.push(`<span style="color:var(--warn)">${entry.error}</span>`);
  if (entry.message)   parts.push(entry.message);
  if (entry.task_preview)
    parts.push(
      entry.task_preview.slice(0, 50) +
      (entry.task_preview.length > 50 ? '…' : '')
    );

  return parts.join(' · ') || JSON.stringify(entry).slice(0, 80);
}

// Manual refresh button
refreshLogsBtn.addEventListener('click', fetchLogs);


/* ─── 7. UI Helpers ──────────────────────────────────────────────────────── */

/**
 * Toggle the loading state on the submit button.
 * @param {boolean} on
 */
function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.classList.toggle('loading', on);
}

/**
 * Show an error message in the banner below the submit button.
 * @param {string} msg
 */
function showError(msg) {
  errorBanner.textContent = `Error: ${msg}`;
  errorBanner.classList.add('visible');
}

/** Hide the error banner. */
function clearError() {
  errorBanner.classList.remove('visible');
  errorBanner.textContent = '';
}

/**
 * Show error and focus the textarea (used for empty-input validation).
 * @param {string} msg
 */
function shakeBanner(msg) {
  showError(msg);
  taskInput.focus();
}


/* ─── 8. Init ────────────────────────────────────────────────────────────── */

// Load logs on page open
fetchLogs();

// Auto-refresh logs every 10 seconds
setInterval(fetchLogs, 10_000);
