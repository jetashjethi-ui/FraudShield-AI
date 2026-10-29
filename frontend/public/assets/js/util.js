export const EASE = 'cubic-bezier(.16, 1, .3, 1)';

// Category cut-offs follow score_transaction() in api.py: RED ≥ 0.75,
// ORANGE ≥ 0.35, YELLOW ≥ decision_threshold (from /model-info), else GREEN.
export const RISK_TIERS = [
  { key: 'GREEN_APPROVE', label: 'Approve', tone: 'green', action: 'Approve under the configured decision threshold.' },
  { key: 'YELLOW_PIN_VERIFY', label: 'PIN verify', tone: 'yellow', action: 'Request additional verification.' },
  { key: 'ORANGE_BIOMETRIC', label: 'Biometric', tone: 'orange', action: 'Request step-up verification and review.' },
  { key: 'RED_BLOCK', label: 'Block', tone: 'red', action: 'Block and send for human review (demo recommendation only).' },
];

// Returns [start, end] score bounds (0–100) per tier. With no known threshold,
// GREEN/YELLOW share an unknown boundary and are returned as null.
export function tierBounds(threshold) {
  const valid = typeof threshold === 'number' && Number.isFinite(threshold) && threshold >= 0 && threshold <= 1;
  const greenEnd = valid ? Math.min(35, threshold * 100) : null;
  return [greenEnd == null ? null : [0, greenEnd], greenEnd == null ? null : [greenEnd, 35], [35, 75], [75, 100]];
}

export function tierRangeLabel(index, threshold) {
  const bounds = tierBounds(threshold)[index];
  if (!bounds) return index === 0 ? '< threshold' : 'threshold – 35';
  const [a, b] = bounds;
  const f = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
  if (a === b) return 'not used';
  return index === 3 ? `${f(a)}–${f(b)}` : `${f(a)}–<${f(b)}`;
}

export function tierByKey(key) {
  return RISK_TIERS.find((tier) => tier.key === key) || null;
}

export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key === 'style' && typeof value === 'object') {
      for (const [prop, val] of Object.entries(value)) el.style.setProperty(prop, val);
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, String(value));
  }
  appendChildren(el, children);
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null) continue;
    el.setAttribute(key, String(value));
  }
  appendChildren(el, children);
  return el;
}

function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export const isNum = (value) => typeof value === 'number' && Number.isFinite(value);

export const fmt = {
  int: (n) => (isNum(n) ? Math.round(n).toLocaleString('en-US') : '—'),
  dec: (n, digits = 4) => (isNum(n) ? n.toFixed(digits) : '—'),
  pct: (n, digits = 1) => (isNum(n) ? `${(n * 100).toFixed(digits)}%` : '—'),
  money: (n) =>
    isNum(n) ? n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }) : '—',
  time: (date) => date.toLocaleTimeString('en-US', { hour12: false }),
  ms: (n) => (isNum(n) ? `${n.toFixed(2)} ms` : '—'),
};

export function titleCase(key) {
  return String(key)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function easeOutExpo(t) {
  return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

export function tween(from, to, duration, onUpdate) {
  if (reducedMotion() || duration <= 0) {
    onUpdate(to);
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / duration, 1);
    onUpdate(from + (to - from) * easeOutExpo(t));
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function revealChildren(root, { selector = '[data-reveal]', delay = 0 } = {}) {
  if (reducedMotion() || !root) return;
  const items = Array.from(root.querySelectorAll(selector)).slice(0, 14);
  items.forEach((item, index) => {
    item.animate(
      [
        { opacity: 0, transform: 'translateY(18px)' },
        { opacity: 1, transform: 'none' },
      ],
      { duration: 900, delay: delay + index * 55, easing: EASE, fill: 'backwards' },
    );
  });
}

export function growBars(root) {
  if (reducedMotion() || !root) return;
  root.querySelectorAll('.bar-fill').forEach((bar, index) => {
    bar.style.setProperty('--delay', `${Math.min(index, 12) * 60}ms`);
    bar.classList.remove('is-entering');
    void bar.offsetWidth;
    bar.classList.add('is-entering');
  });
}

export function downloadText(filename, text, type = 'text/csv;charset=utf-8') {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = h('a', { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
