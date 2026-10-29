import { h, fmt, isNum, titleCase, revealChildren, growBars } from '../util.js';
import { store, subscribe } from '../store.js';
import { notice, offlineNotice, row, bar } from '../components.js';

const PREFERRED = ['auc', 'average_precision', 'precision', 'recall', 'f1', 'specificity', 'accuracy'];
const HIGHER_IS_BETTER = new Set(['auc', 'average_precision', 'precision', 'recall', 'f1', 'specificity', 'accuracy']);

export function modelEntries(metrics) {
  if (!metrics || typeof metrics !== 'object') return [];
  return Object.entries(metrics).filter(([, value]) => value && typeof value === 'object' && !Array.isArray(value));
}

function numericKeys(entries) {
  const keys = new Set();
  for (const [, m] of entries) for (const [k, v] of Object.entries(m)) if (isNum(v)) keys.add(k);
  return [...keys].sort((a, b) => {
    const ia = PREFERRED.indexOf(a);
    const ib = PREFERRED.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
  });
}

const display = (v) => (isNum(v) ? (Math.abs(v) <= 1 ? fmt.dec(v) : fmt.dec(v, 2)) : '—');

function confusion(matrix) {
  const valid = Array.isArray(matrix) && matrix.length === 2 && matrix.every((r) => Array.isArray(r) && r.length === 2 && r.every(isNum));
  if (!valid || matrix.flat().every((n) => n === 0)) return null;
  const [[tn, fp], [fn, tp]] = matrix;
  return h(
    'div',
    { class: 'matrix', role: 'table', 'aria-label': 'Confusion matrix' },
    h('div', { class: 'head', role: 'columnheader' }, 'Actual ╲ Predicted'),
    h('div', { class: 'head', role: 'columnheader' }, 'Legit'),
    h('div', { class: 'head', role: 'columnheader' }, 'Fraud'),
    h('div', { class: 'head', role: 'rowheader' }, 'Legit'),
    h('div', { class: 'cell', role: 'cell' }, fmt.int(tn)),
    h('div', { class: 'cell', role: 'cell' }, fmt.int(fp)),
    h('div', { class: 'head', role: 'rowheader' }, 'Fraud'),
    h('div', { class: 'cell', role: 'cell' }, fmt.int(fn)),
    h('div', { class: 'cell', role: 'cell' }, fmt.int(tp)),
  );
}

export function initPerformance(root) {
  const slot = root.querySelector('[data-slot="performance-body"]');
  let signature = '';

  function render(animate) {
    const metrics = store.metrics;
    const sig = metrics ? JSON.stringify(metrics) : `state:${store.metricsCheckedAt ? 'offline' : 'checking'}`;
    if (sig === signature && !animate) return;
    signature = sig;

    if (!metrics) {
      slot.replaceChildren(
        store.metricsCheckedAt ? offlineNotice(store.metricsError) : h('p', { class: 'muted' }, 'Loading GET /metrics…'),
      );
      if (animate) revealChildren(slot);
      return;
    }

    const entries = modelEntries(metrics);
    if (!entries.length) {
      slot.replaceChildren(
        notice({
          kind: 'unavailable',
          title: 'Evaluation metrics unavailable',
          body: [
            h(
              'p',
              null,
              'GET /metrics returned an empty ',
              h('code', null, 'metrics'),
              ' object, which means ',
              h('code', null, 'outputs/results/model_metrics.json'),
              ' is not present on the server. Generate it with the project’s own training and evaluation pipeline. No placeholder values are shown here.',
            ),
          ],
        }),
      );
      if (animate) revealChildren(slot);
      return;
    }

    const keys = numericKeys(entries);
    const sorted = [...entries].sort((a, b) => (isNum(b[1].auc) ? b[1].auc : -1) - (isNum(a[1].auc) ? a[1].auc : -1));
    const withMatrix = sorted.filter(([, m]) => confusion(m.confusion_matrix));

    slot.replaceChildren(
      h(
        'section',
        null,
        h(
          'div',
          { class: 'section-head' },
          h('div', null, h('p', { class: 'eyebrow' }, 'A — Leaderboard'), h('h3', { class: 'section-title' }, 'Ranked by AUC')),
          h('span', { class: 'section-meta' }, `${entries.length} evaluated · model_metrics.json`),
        ),
        h(
          'ol',
          { class: 'rows' },
          sorted.map(([name, m], i) =>
            row({
              label: `#${String(i + 1).padStart(2, '0')}`,
              value: titleCase(name),
              small: true,
              tags: keys.filter((k) => isNum(m[k])).slice(0, 4).map((k) => `${k.toUpperCase()} ${display(m[k])}`),
            }),
          ),
        ),
      ),
      ...keys
        .filter((k) => HIGHER_IS_BETTER.has(k) && entries.every(([, m]) => !isNum(m[k]) || (m[k] >= 0 && m[k] <= 1)))
        .slice(0, 6)
        .map((key, i) =>
          h(
            'section',
            { class: 'section', 'data-reveal': true },
            h(
              'div',
              { class: 'section-head' },
              h('div', null, h('p', { class: 'eyebrow' }, `${String.fromCharCode(66 + i)} — Metric`), h('h3', { class: 'section-title' }, key.toUpperCase())),
              h('span', { class: 'section-meta' }, 'Scale 0–1'),
            ),
            h(
              'ul',
              { class: 'bars' },
              [...entries]
                .filter(([, m]) => isNum(m[key]))
                .sort((a, b) => b[1][key] - a[1][key])
                .map(([name, m]) => bar({ name: titleCase(name), value: m[key], display: fmt.dec(m[key]) })),
            ),
          ),
        ),
      h(
        'section',
        { class: 'section', 'data-reveal': true },
        h('div', { class: 'section-head' }, h('div', null, h('p', { class: 'eyebrow' }, 'Full table'), h('h3', { class: 'section-title' }, 'All returned metrics'))),
        h(
          'div',
          { class: 'table-wrap' },
          h(
            'table',
            { class: 'data-table' },
            h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Model'), keys.map((k) => h('th', { scope: 'col', class: 'num' }, k)))),
            h(
              'tbody',
              null,
              sorted.map(([name, m]) => h('tr', null, h('td', null, titleCase(name)), keys.map((k) => h('td', { class: 'num' }, display(m[k]))))),
            ),
          ),
        ),
      ),
      withMatrix.length
        ? h(
            'section',
            { class: 'section', 'data-reveal': true },
            h('div', { class: 'section-head' }, h('div', null, h('p', { class: 'eyebrow' }, 'Confusion'), h('h3', { class: 'section-title' }, 'Confusion matrices'))),
            h(
              'div',
              { class: 'stack' },
              withMatrix.map(([name, m]) => h('div', { class: 'stack-sm' }, h('h4', { class: 'sub-title' }, titleCase(name)), confusion(m.confusion_matrix))),
            ),
          )
        : null,
    );
    if (animate) {
      revealChildren(slot);
      growBars(slot);
    }
  }

  subscribe(() => render(false));
  render(false);
  return { enter: () => render(true) };
}
