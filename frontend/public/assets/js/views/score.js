import { h, svg, fmt, titleCase, tierByKey, tierBounds, tween, revealChildren, growBars, RISK_TIERS } from '../util.js';
import { api } from '../api.js';
import { store, subscribe, addSessionResult, clearSession, decisionThreshold } from '../store.js';
import { coerceTransaction, FIELDS, EXTRA_KEY, SAMPLE_INPUT } from '../schema.js';
import { notice, offlineNotice, tierPill, bar } from '../components.js';

const ARC = 'M 20 120 A 100 100 0 0 1 220 120';

function gauge(score, tone) {
  const threshold = decisionThreshold();
  const bounds = tierBounds(threshold);
  const value = svg('path', {
    d: ARC,
    class: 'gauge-value',
    pathLength: 100,
    'stroke-dasharray': '100 100',
    'stroke-dashoffset': 100,
  });
  const segments = RISK_TIERS.map((tier, i) => {
    const range = bounds[i];
    if (!range || range[0] === range[1]) return null;
    const [start, end] = range;
    return svg('path', {
      d: 'M 6 120 A 114 114 0 0 1 234 120',
      class: 'gauge-seg',
      pathLength: 100,
      stroke: `var(--risk-${tier.tone})`,
      'stroke-dasharray': `${Math.max(0, end - start - 0.6)} 100`,
      'stroke-dashoffset': -start,
    });
  }).filter(Boolean);
  const thresholdPoint = threshold != null && threshold >= 0 && threshold < 0.35 ? threshold * 100 : null;
  const breakpoints = [...new Set([0, thresholdPoint, 35, 75, 100].filter((n) => n != null))];
  const labels = breakpoints.map((n) => {
    const angle = Math.PI * (1 - n / 100);
    const x = 120 + Math.cos(angle) * 128;
    const y = 122 - Math.sin(angle) * 128;
    return svg('text', { x: x.toFixed(1), y: y.toFixed(1), class: 'gauge-label', 'text-anchor': 'middle' }, String(n));
  });

  const number = h('span', { class: 'gauge-number', 'aria-hidden': 'true' }, '0.0');
  const wrap = h(
    'div',
    { class: 'gauge-wrap', dataset: tone ? { tone } : undefined, role: 'img', 'aria-label': `Risk score ${score} of 100` },
    svg('svg', { viewBox: '-14 -14 268 140', class: 'gauge' }, ...segments, svg('path', { d: ARC, class: 'gauge-track' }), value, ...labels),
    number,
  );

  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      value.setAttribute('stroke-dashoffset', String(100 - Math.max(0, Math.min(100, score))));
      tween(0, score, 1300, (v) => (number.textContent = v.toFixed(1)));
    }),
  );
  return wrap;
}

function renderResult(result, payload) {
  const tier = tierByKey(result.risk_category);
  const modelProbabilities = Object.entries(result.base_model_probabilities || {}).filter(([, value]) => Number.isFinite(Number(value)));
  const signals = Array.isArray(result.context_signals) ? result.context_signals : [];

  return h(
    'div',
    { class: 'result', dataset: tier ? { tone: tier.tone } : undefined },
    h(
      'div',
      { class: 'result-head', 'data-reveal': true },
      gauge(result.risk_score, tier && tier.tone),
      tierPill(result.risk_category),
      h('p', { class: 'recommendation' }, result.auth_recommendation),
    ),
    h(
      'dl',
      { class: 'kv', 'data-reveal': true },
      ...[
        ['Fraud probability', fmt.pct(result.fraud_probability, 2)],
        ['Processing time', fmt.ms(result.processing_time_ms)],
        ['Transaction ID', result.transaction_id],
        ['Timestamp', result.timestamp],
      ].map(([k, v]) => h('div', null, h('dt', null, k), h('dd', null, v))),
    ),
    h(
      'section',
      { 'data-reveal': true },
      h('h3', { class: 'sub-title' }, 'Base model probabilities · 0–1'),
      modelProbabilities.length
        ? h(
            'ul',
            { class: 'bars' },
            modelProbabilities.map(([name, value]) =>
              bar({ name: titleCase(name), value: Number(value), tone: tier && tier.tone, display: fmt.dec(Number(value)) }),
            ),
          )
        : notice({
            kind: 'info',
            title: 'No base-model breakdown returned',
            body: ['The API returned the transaction-level fraud probability without per-model probabilities.'],
          }),
    ),
    h(
      'section',
      { 'data-reveal': true },
      h('h3', { class: 'sub-title' }, 'Context signals'),
      h(
        'ul',
        { class: 'factors' },
        signals.length
          ? signals.map((signal, i) => h('li', null, h('span', null, String(i + 1).padStart(2, '0')), h('span', null, signal)))
          : h('li', { class: 'muted' }, 'No context signals returned.'),
      ),
    ),
    h(
      'details',
      { class: 'disclosure', 'data-reveal': true },
      h('summary', null, 'Submitted payload'),
      h('pre', null, JSON.stringify(payload, null, 2)),
    ),
  );
}

export function initScore(root) {
  const form = root.querySelector('#score-form');
  const resultSlot = root.querySelector('[data-slot="score-result"]');
  const summary = root.querySelector('#score-summary');
  const submit = form.querySelector('[type="submit"]');
  const hour = form.querySelector('#f-hour_of_day');
  const hourOut = form.querySelector('#o-hour_of_day');
  const sessionSlot = root.querySelector('[data-slot="score-session"]');
  const clearBtn = root.querySelector('[data-action="clear-session"]');

  const syncHour = () => {
    hourOut.textContent = String(hour.value).padStart(2, '0') + ':00';
    hour.style.setProperty('--fill', `${(Number(hour.value) / 23) * 100}%`);
  };
  hour.addEventListener('input', syncHour);
  syncHour();

  function readForm() {
    const data = new FormData(form);
    const raw = {};
    for (const field of FIELDS) raw[field.key] = data.get(field.key);
    raw[EXTRA_KEY] = data.get(EXTRA_KEY);
    return raw;
  }

  function showErrors(errors) {
    for (const key of [...FIELDS.map((field) => field.key), EXTRA_KEY]) {
      const input = form.querySelector(`#f-${key}`);
      const err = form.querySelector(`#e-${key}`);
      const match = errors.find((e) => e.field === key);
      if (input) input.setAttribute('aria-invalid', match ? 'true' : 'false');
      if (err) err.textContent = match ? `${field.key} ${match.message}` : '';
    }
    summary.textContent = errors.length ? `${errors.length} field${errors.length > 1 ? 's need' : ' needs'} attention before scoring.` : '';
    if (errors.length) form.querySelector(`#f-${errors[0].field}`)?.focus();
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const { value, errors } = coerceTransaction(readForm());
    showErrors(errors);
    if (errors.length) return;

    submit.disabled = true;
    submit.textContent = 'Scoring…';
    resultSlot.setAttribute('aria-busy', 'true');
    try {
      const result = await api.score(value);
      resultSlot.replaceChildren(renderResult(result, value));
      revealChildren(resultSlot);
      growBars(resultSlot);
      addSessionResult({ source: 'score', at: new Date(), input: value, result });
    } catch (error) {
      resultSlot.replaceChildren(
        error.kind === 'unreachable'
          ? offlineNotice(error)
          : notice({ kind: 'offline', title: `Scoring failed (HTTP ${error.status || '—'})`, body: [error.message] }),
      );
      revealChildren(resultSlot);
    } finally {
      submit.disabled = false;
      submit.textContent = 'Run risk score';
      resultSlot.removeAttribute('aria-busy');
    }
  });

  root.querySelector('[data-action="load-example"]').addEventListener('click', () => {
    form.reset();
    for (const [key, value] of Object.entries(SAMPLE_INPUT)) {
      const input = form.querySelector(`#f-${key}`);
      if (!input) continue;
      if (input.type === 'checkbox') input.checked = value === '1';
      else input.value = value;
    }
    syncHour();
    showErrors([]);
  });

  form.addEventListener('reset', () => {
    requestAnimationFrame(() => {
      syncHour();
      showErrors([]);
    });
  });

  clearBtn.addEventListener('click', clearSession);

  function renderSession() {
    const entries = store.session.filter((entry) => entry.source === 'score');
    clearBtn.disabled = store.session.length === 0;
    if (!entries.length) {
      sessionSlot.replaceChildren(h('p', { class: 'muted' }, 'No transactions scored in this session yet.'));
      return;
    }
    sessionSlot.replaceChildren(
      h(
        'div',
        { class: 'table-wrap' },
        h(
          'table',
          { class: 'data-table' },
          h(
            'thead',
            null,
            h('tr', null, ['Time', 'Transaction ID', 'Amount', 'Score', 'Category', 'Probability'].map((t) => h('th', { scope: 'col' }, t))),
          ),
          h(
            'tbody',
            null,
            entries.map(({ at, input, result }) => {
              const tier = tierByKey(result.risk_category);
              return h(
                'tr',
                null,
                h('td', null, fmt.time(at)),
                h('td', null, result.transaction_id),
                h('td', { class: 'num' }, fmt.dec(input.TransactionAmt, 2)),
                h('td', { class: 'num tone-text', dataset: tier ? { tone: tier.tone } : undefined }, fmt.dec(result.risk_score, 1)),
                h('td', null, result.risk_category),
                h('td', { class: 'num' }, fmt.pct(result.fraud_probability, 2)),
              );
            }),
          ),
        ),
      ),
    );
  }

  subscribe(renderSession);
  renderSession();

  return {};
}
