// ── Word Counter ───────────────────────────────────────────────────────────────
const textarea    = document.getElementById('textInput');
const wordCounter = document.getElementById('wordCounter');

textarea.addEventListener('input', () => {
  const words = textarea.value.trim().split(/\s+/).filter(Boolean).length;
  wordCounter.textContent = `${words} word${words !== 1 ? 's' : ''}`;
  wordCounter.classList.toggle('ready', words >= 30);
});

// ── Clear ──────────────────────────────────────────────────────────────────────
function clearAll() {
  textarea.value = '';
  wordCounter.textContent = '0 words';
  wordCounter.classList.remove('ready');
  document.getElementById('errorBox').style.display   = 'none';
  document.getElementById('emptyState').style.display = 'block';
  document.getElementById('verdictCard').style.display  = 'none';
  document.getElementById('metricsRow').style.display   = 'none';
  document.getElementById('featuresCard').style.display = 'none';
}

// ── Analyze ────────────────────────────────────────────────────────────────────
async function analyze() {
  const text = textarea.value.trim();
  const btn  = document.getElementById('analyzeBtn');
  const err  = document.getElementById('errorBox');

  err.style.display = 'none';
  btn.querySelector('.btn-text').style.display   = 'none';
  btn.querySelector('.btn-loader').style.display = 'flex';
  btn.disabled = true;

  try {
    const res  = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      err.textContent    = data.error || 'Something went wrong.';
      err.style.display  = 'block';
      return;
    }

    renderResults(data);

  } catch (e) {
    err.textContent   = 'Network error — is Flask running?';
    err.style.display = 'block';
  } finally {
    btn.querySelector('.btn-text').style.display   = 'flex';
    btn.querySelector('.btn-loader').style.display = 'none';
    btn.disabled = false;
  }
}

// ── Render Results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  const isHuman = data.label === 0;

  // Hide empty state
  document.getElementById('emptyState').style.display = 'none';

  // ── Verdict Card ──
  const card = document.getElementById('verdictCard');
  card.style.display = 'block';
  card.style.animation = 'none';
  void card.offsetWidth; // reflow
  card.style.animation = '';

  // Glow
  const glow = document.getElementById('verdictGlow');
  glow.style.background = isHuman
    ? 'radial-gradient(ellipse at 50% 0%, rgba(22,163,74,0.12) 0%, transparent 70%)'
    : 'radial-gradient(ellipse at 50% 0%, rgba(220,38,38,0.12) 0%, transparent 70%)';

  // Joker emoji
  document.getElementById('verdictJoker').textContent = isHuman ? '✅' : '🤖';

  // Label
  const lbl = document.getElementById('verdictLabel');
  lbl.textContent  = isHuman ? 'Real Human' : 'AI Generated';
  lbl.style.color  = isHuman ? '#22c55e' : '#ef4444';

  document.getElementById('verdictSub').textContent = isHuman
    ? 'Writing patterns match human authorship'
    : 'Writing patterns match LLM output';

  // Confidence
  const conf = document.getElementById('verdictConf');
  conf.textContent = isHuman
    ? `${data.human_prob}%`
    : `${data.ai_prob}%`;
  conf.style.color = isHuman ? '#22c55e' : '#ef4444';

  // Probability bar — animate after paint
  requestAnimationFrame(() => {
    document.getElementById('probHuman').style.width = `${data.human_prob}%`;
    document.getElementById('probAI').style.width    = `${data.ai_prob}%`;
    document.getElementById('probHumanVal').textContent = `${data.human_prob}%`;
    document.getElementById('probAIVal').textContent    = `${data.ai_prob}%`;
  });

  // Insight tags
  renderInsights(data.features, data.benchmarks, isHuman);

  // ── Metrics Row ──
  document.getElementById('metricsRow').style.display = 'grid';
  document.getElementById('mBurstiness').textContent =
    data.features.burstiness.toFixed(1);
  document.getElementById('mContractions').textContent =
    data.features.contraction_rate.toFixed(2) + '%';
  document.getElementById('mTTR').textContent =
    data.features.type_token_ratio.toFixed(3);
  document.getElementById('mWords').textContent =
    data.word_count.toLocaleString();

  // ── Features Card ──
  renderFeatures(data.features, data.benchmarks);
}

// ── Insight Tags ───────────────────────────────────────────────────────────────
function renderInsights(features, benchmarks, isHuman) {
  const checks = [
    ['burstiness',       'high',  'Varied sentence rhythm',  'Uniform sentence rhythm'],
    ['contraction_rate', 'high',  'Natural contractions',     'Avoids contractions'],
    ['type_token_ratio', 'low',   'Organic vocabulary',       'Artificially diverse vocab'],
    ['question_rate',    'high',  'Rhetorical questions',     'No rhetorical questions'],
    ['comma_rate',       'low',   'Natural comma usage',      'Heavy list-style commas'],
  ];

  const wrap = document.getElementById('insightTags');
  wrap.innerHTML = '';

  checks.forEach(([feat, humanDir, humanTag, aiTag], i) => {
    const val  = features[feat] || 0;
    const hAvg = benchmarks[feat]?.human || 0;
    const aAvg = benchmarks[feat]?.ai    || 0;
    const mid  = (hAvg + aAvg) / 2;
    const isHumanSide = humanDir === 'high' ? val >= mid : val < mid;

    const tag = document.createElement('span');
    tag.className = `insight-tag ${isHumanSide ? 'human' : 'ai'}`;
    tag.textContent = isHumanSide ? humanTag : aiTag;
    tag.style.animationDelay = `${i * 0.06}s`;
    wrap.appendChild(tag);
  });
}

// ── Feature Bars ───────────────────────────────────────────────────────────────
function renderFeatures(features, benchmarks) {
  const card = document.getElementById('featuresCard');
  const grid = document.getElementById('featuresGrid');
  card.style.display = 'block';
  grid.innerHTML = '';

  const DISPLAY = [
    ['burstiness',        'Burstiness'],
    ['contraction_rate',  'Contraction Rate'],
    ['type_token_ratio',  'Vocab Diversity (TTR)'],
    ['avg_sentence_length','Avg Sentence Length'],
    ['comma_rate',        'Comma Rate'],
    ['punctuation_density','Punctuation Density'],
  ];

  DISPLAY.forEach(([feat, label]) => {
    const h   = benchmarks[feat]?.human || 0;
    const a   = benchmarks[feat]?.ai    || 0;
    const v   = features[feat]          || 0;
    const max = Math.max(h, a, v, 0.001);

    const row = document.createElement('div');
    row.className = 'feature-row';
    row.innerHTML = `
      <div class="feature-row-header">
        <span class="feature-name">${label}</span>
        <div class="feature-vals">
          <span><span class="dot-green">●</span> ${h.toFixed(2)}</span>
          <span><span class="dot-red">●</span>   ${a.toFixed(2)}</span>
          <span><span class="dot-purple">●</span>${v.toFixed(2)}</span>
        </div>
      </div>
      <div class="bar-group">
        <div class="bar-row">
          <span class="bar-label-left">Human</span>
          <div class="bar-track">
            <div class="bar-fill human" style="width:0%"
                 data-target="${(h/max*100).toFixed(1)}"></div>
          </div>
          <span class="bar-label-right">${h.toFixed(2)}</span>
        </div>
        <div class="bar-row">
          <span class="bar-label-left">AI Avg</span>
          <div class="bar-track">
            <div class="bar-fill ai" style="width:0%"
                 data-target="${(a/max*100).toFixed(1)}"></div>
          </div>
          <span class="bar-label-right">${a.toFixed(2)}</span>
        </div>
        <div class="bar-row">
          <span class="bar-label-left">Your text</span>
          <div class="bar-track">
            <div class="bar-fill input" style="width:0%"
                 data-target="${(v/max*100).toFixed(1)}"></div>
          </div>
          <span class="bar-label-right">${v.toFixed(2)}</span>
        </div>
      </div>
    `;
    grid.appendChild(row);
  });

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill[data-target]').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  });
}

// ── Enter key shortcut ─────────────────────────────────────────────────────────
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.metaKey) analyze();
});