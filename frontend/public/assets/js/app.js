import { EASE, reducedMotion, revealChildren, wait } from './util.js';
import { store, subscribe, refreshSystem, setAutoRefresh, serviceState } from './store.js';
import { initOverview } from './views/overview.js';
import { initScore } from './views/score.js';
import { initBatch } from './views/batch.js';
import { initPerformance } from './views/performance.js';
import { initROI } from './views/roi.js';
import { initStatus } from './views/status.js';

// Query-parameter routing (?view=score). Route strings are only ever compared
// against this whitelist and looked up in a Map — never passed to querySelector.
const VIEWS = {
  overview: { n: '01', label: 'Overview', init: initOverview },
  score: { n: '02', label: 'Score', init: initScore },
  batch: { n: '03', label: 'Batch', init: initBatch },
  performance: { n: '04', label: 'Performance', init: initPerformance },
  roi: { n: '05', label: 'ROI', init: initROI },
  status: { n: '06', label: 'System', init: initStatus },
};

const sections = new Map();
const controllers = new Map();
const navLinks = [];
let current = null;
let transitionToken = 0;

const nav = document.getElementById('nav');
const app = document.getElementById('app');
const hero = document.getElementById('hero');
const crumb = document.querySelector('[data-slot="crumb"]');
const navStatusDot = document.querySelector('[data-slot="nav-dot"]');
const navStatusText = document.querySelector('[data-slot="nav-status"]');
const navToggle = document.getElementById('nav-toggle');

function viewFromURL() {
  const view = new URLSearchParams(window.location.search).get('view');
  return view && Object.hasOwn(VIEWS, view) ? view : null;
}

function scrollBehavior() {
  return reducedMotion() ? 'auto' : 'smooth';
}

function scrollToApp(behavior = scrollBehavior()) {
  const top = app.getBoundingClientRect().top + window.scrollY - 1;
  window.scrollTo({ top, behavior });
}

function scrollToHero() {
  window.scrollTo({ top: 0, behavior: scrollBehavior() });
  window.dispatchEvent(new CustomEvent('fs:hero-return'));
}

function setNavState(view) {
  for (const link of navLinks) {
    if (link.dataset.nav === view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  const meta = VIEWS[view];
  crumb.textContent = `${meta.n} — ${meta.label}`;
  document.title = `${meta.label} — FraudShield AI`;
}

async function transition(from, to) {
  const token = ++transitionToken;
  const outgoing = from ? sections.get(from) : null;
  const incoming = sections.get(to);
  const animate = !reducedMotion();

  if (outgoing && outgoing !== incoming) {
    if (animate) {
      outgoing.getAnimations().forEach((a) => a.cancel());
      await outgoing
        .animate(
          [
            { opacity: 1, transform: 'none' },
            { opacity: 0, transform: 'translateY(-12px)' },
          ],
          { duration: 220, easing: 'cubic-bezier(.7, 0, .84, 0)', fill: 'forwards' },
        )
        .finished.catch(() => {});
    }
    if (token !== transitionToken) return;
    outgoing.hidden = true;
    outgoing.getAnimations().forEach((a) => a.cancel());
  }

  for (const [name, section] of sections) if (name !== to) section.hidden = true;
  incoming.hidden = false;
  controllers.get(to)?.enter?.();

  if (animate) {
    incoming.getAnimations().forEach((a) => a.cancel());
    incoming.animate(
      [
        { opacity: 0, transform: 'translateY(26px) scale(.995)' },
        { opacity: 1, transform: 'none' },
      ],
      { duration: 700, easing: EASE },
    );
    revealChildren(incoming, { delay: 80 });
  }
}

function show(view, { push = false, scroll = true, focus = true } = {}) {
  if (push) {
    const url = new URL(window.location.href);
    url.searchParams.set('view', view);
    if (url.href !== window.location.href) history.pushState({ view }, '', url);
  }
  const previous = current;
  current = view;
  setNavState(view);
  if (previous !== view) transition(previous, view);
  if (scroll) scrollToApp();
  if (focus) {
    sections.get(view).querySelector('.view-title')?.focus({ preventScroll: true });
  }
  closeMenu();
}

function closeMenu() {
  nav.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
}

function renderNavStatus() {
  const state = serviceState();
  navStatusDot.dataset.state = state;
  navStatusText.textContent = {
    checking: 'Checking API',
    healthy: 'API online',
    degraded: 'API degraded',
    offline: 'API offline',
  }[state];
}

function bindNavigation() {
  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const target = event.target instanceof Element ? event.target : null;
    const navLink = target?.closest('a[data-nav]');
    if (navLink && Object.hasOwn(VIEWS, navLink.dataset.nav)) {
      event.preventDefault();
      show(navLink.dataset.nav, { push: true });
      return;
    }
    const home = target?.closest('[data-to-hero]');
    if (home) {
      event.preventDefault();
      closeMenu();
      scrollToHero();
    }
  });

  window.addEventListener('popstate', () => {
    const view = viewFromURL();
    if (view) show(view, { scroll: true, focus: false });
    else scrollToHero();
  });

  navToggle.addEventListener('click', () => {
    const open = !nav.classList.contains('is-open');
    nav.classList.toggle('is-open', open);
    navToggle.setAttribute('aria-expanded', String(open));
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && nav.classList.contains('is-open')) {
      closeMenu();
      navToggle.focus();
    }
  });

  const initButton = document.getElementById('initialize');
  initButton.addEventListener('click', async () => {
    window.dispatchEvent(new CustomEvent('fs:initialize'));
    document.body.classList.add('is-initializing');
    await wait(reducedMotion() ? 0 : 560);
    show('overview', { push: true });
    setTimeout(() => document.body.classList.remove('is-initializing'), 1200);
  });

  new IntersectionObserver(
    ([entry]) => nav.classList.toggle('is-solid', entry.intersectionRatio < 0.85),
    { threshold: [0.85] },
  ).observe(hero);
}

function init() {
  for (const section of document.querySelectorAll('section[data-view]')) {
    const name = section.dataset.view;
    if (Object.hasOwn(VIEWS, name)) sections.set(name, section);
  }
  for (const link of document.querySelectorAll('a[data-nav]')) navLinks.push(link);

  for (const [name, meta] of Object.entries(VIEWS)) {
    const section = sections.get(name);
    if (section) controllers.set(name, meta.init(section) || {});
  }

  bindNavigation();
  subscribe(renderNavStatus);
  renderNavStatus();

  const initial = viewFromURL();
  show(initial || 'overview', { scroll: false, focus: false });
  if (initial) requestAnimationFrame(() => scrollToApp('auto'));

  refreshSystem();
  setAutoRefresh(store.autoRefresh);
}

init();
