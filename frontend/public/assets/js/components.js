import { h, tierByKey, RISK_TIERS, tierRangeLabel } from './util.js';
import { refreshSystem } from './store.js';

export function notice({ kind = 'info', title, body = [], actions = [] }) {
  return h(
    'div',
    { class: 'notice', dataset: { kind }, 'data-reveal': true },
    h('p', { class: 'notice-title' }, title),
    ...body.map((part) => (part instanceof Node ? part : h('p', null, part))),
    actions.length ? h('div', { class: 'notice-actions' }, ...actions) : null,
  );
}

export function offlineNotice(error) {
  return notice({
    kind: 'offline',
    title: 'FraudShield API unreachable',
    body: [
      error && error.message ? error.message : 'The backend did not respond.',
      h(
        'p',
        null,
        'This page talks to the FastAPI service on the same origin (',
        h('code', null, '/health'),
        ', ',
        h('code', null, '/model-info'),
        ', ',
        h('code', null, '/score'),
        '). A hosted preview has no FastAPI server, so no data is shown here. From Project2, run ',
        h('code', null, 'uvicorn api:app --host 127.0.0.1 --port 8000'),
        ' and open ',
        h('code', null, 'http://127.0.0.1:8000/'),
        '.',
      ),
    ],
    actions: [h('button', { class: 'btn btn--ghost btn--sm', type: 'button', onclick: () => refreshSystem() }, 'Retry connection')],
  });
}

export function tagList(items) {
  if (!items || !items.length) return null;
  return h(
    'ul',
    { class: 'tags' },
    items.map((item) => h('li', { class: 'tag' }, item)),
  );
}

export function row({ label, value, small = false, muted = false, note, tags, tone, compact = false }) {
  return h(
    'li',
    { class: `row${compact ? ' is-compact' : ''}`, 'data-reveal': true, dataset: tone ? { tone } : undefined },
    h('span', { class: 'row-label' }, label),
    h(
      'div',
      { class: 'row-main' },
      h('span', { class: `row-value${small ? ' is-small' : ''}${muted ? ' is-muted' : ''}${tone ? ' tone-text' : ''}` }, value),
      note ? h('span', { class: 'row-note' }, note) : null,
    ),
    tagList(tags) || h('span'),
  );
}

export function tierPill(key) {
  const tier = tierByKey(key);
  return h(
    'span',
    { class: 'tier-pill', dataset: tier ? { tone: tier.tone } : undefined },
    h('span', { class: 'swatch' }),
    key || 'UNKNOWN',
  );
}

export function tierRows(threshold) {
  return RISK_TIERS.map((tier, index) =>
    row({
      label: tierRangeLabel(index, threshold),
      value: tier.label,
      tone: tier.tone,
      small: true,
      compact: true,
      note: tier.action,
      tags: [tier.key],
    }),
  );
}

export function bar({ name, value, max = 1, tone, display }) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return h(
    'li',
    { class: 'bar-row', dataset: tone ? { tone } : undefined },
    h('span', { class: 'bar-name', title: name }, name),
    h(
      'span',
      { class: 'bar-track', 'aria-hidden': 'true' },
      h('span', { class: 'bar-fill', style: { '--w': `${(ratio * 100).toFixed(2)}%` } }),
    ),
    h('span', { class: 'bar-val' }, display ?? String(value)),
  );
}
