import { h, fmt, revealChildren } from '../util.js';
import { api } from '../api.js';
import { notice, offlineNotice, row } from '../components.js';

// Percentages are entered in human-friendly percent units and converted to
// fractions for the FastAPI ROIRequest.
const ASSUMPTIONS = [
  { key: 'annual_transactions', label: 'Annual transactions', value: '1000000', unit: 'transactions / year', min: 1, max: 1_000_000_000_000, integer: true },
  { key: 'fraud_rate', label: 'Fraud rate', value: '3.5', unit: '% · enter a value greater than 0 and less than 100', min: 0, max: 100, percent: true, exclusive: true },
  { key: 'average_fraud_loss', label: 'Average loss per fraud', value: '850', unit: 'USD per fraudulent transaction', min: 0 },
  { key: 'false_positive_cost', label: 'Cost per false positive', value: '25', unit: 'USD per false positive', min: 0 },
  { key: 'manual_review_cost', label: 'Manual review cost', value: '15', unit: 'USD per reviewed transaction', min: 0 },
  { key: 'baseline_recall', label: 'Baseline recall', value: '30', unit: '% · 0–100', min: 0, max: 100, percent: true },
  { key: 'baseline_false_positive_rate', label: 'Baseline false-positive rate', value: '5', unit: '% · 0–100', min: 0, max: 100, percent: true },
];

export function initROI(root) {
  const form = root.querySelector('[data-slot="roi-form"]');
  const inputsSlot = root.querySelector('[data-slot="roi-inputs"]');
  const outputSlot = root.querySelector('[data-slot="roi-output"]');
  const statusSlot = root.querySelector('[data-slot="roi-status"]');
  const submit = form.querySelector('[type="submit"]');
  const inputs = new Map();

  inputsSlot.replaceChildren(
    ...ASSUMPTIONS.map((assumption) => {
      const id = `roi-${assumption.key}`;
      const input = h('input', {
        id,
        class: 'input',
        name: assumption.key,
        type: 'text',
        inputmode: assumption.integer ? 'numeric' : 'decimal',
        value: assumption.value,
        autocomplete: 'off',
        'aria-describedby': `${id}-hint ${id}-error`,
      });
      inputs.set(assumption.key, input);
      return h(
        'div',
        { class: 'field' },
        h('label', { for: id }, assumption.label),
        input,
        h('p', { class: 'hint', id: `${id}-hint` }, assumption.unit),
        h('p', { class: 'error', id: `${id}-error` }),
      );
    }),
  );

  function readAssumptions() {
    const payload = {};
    const errors = [];
    for (const assumption of ASSUMPTIONS) {
      const input = inputs.get(assumption.key);
      const raw = input.value.trim().replace(/,/g, '');
      const number = Number(raw);
      let message = '';
      if (!raw || !Number.isFinite(number)) message = 'Enter a valid number.';
      else if (assumption.integer && (!Number.isSafeInteger(number) || number < assumption.min || number > assumption.max)) {
        message = `Enter a whole number from ${assumption.min.toLocaleString()} to ${assumption.max.toLocaleString()}.`;
      } else if (assumption.exclusive && (number <= assumption.min || number >= assumption.max)) {
        message = 'Enter a percentage greater than 0 and less than 100.';
      } else if (assumption.min != null && number < assumption.min) message = `Must be at least ${assumption.min}.`;
      else if (assumption.max != null && number > assumption.max) message = `Must be at most ${assumption.max}.`;

      input.setAttribute('aria-invalid', message ? 'true' : 'false');
      root.querySelector(`#${input.id}-error`).textContent = message;
      if (message) errors.push({ input, key: assumption.key });
      payload[assumption.key] = assumption.percent ? number / 100 : number;
    }
    return { payload, errors };
  }

  function renderResult(data) {
    if (data.status !== 'estimate' || !Number.isFinite(data.annual_savings) || !Number.isFinite(data.baseline_cost) || !Number.isFinite(data.model_cost)) {
      outputSlot.replaceChildren(notice({ kind: 'warn', title: 'Unexpected ROI response', body: ['The backend did not return a complete estimate. Refresh the system status and try again.'] }));
      return;
    }
    const metrics = data.model_metrics || {};
    outputSlot.replaceChildren(
      h('section', null,
        h('p', { class: 'eyebrow' }, 'Server-calculated scenario'),
        h('h3', { class: 'section-title' }, 'Estimated annual impact'),
        h('ul', { class: 'rows' },
          row({ label: 'Annual savings', value: fmt.money(data.annual_savings), tone: data.annual_savings >= 0 ? 'green' : 'red', note: 'Estimate from the assumptions you submitted.' }),
          row({ label: 'Baseline annual cost', value: fmt.money(data.baseline_cost), small: true, compact: true }),
          row({ label: 'Annual cost with FraudShield', value: fmt.money(data.model_cost), small: true, compact: true }),
          row({ label: 'Measured model recall', value: fmt.pct(metrics.recall), small: true, compact: true }),
          row({ label: 'Measured model precision', value: fmt.pct(metrics.precision), small: true, compact: true }),
        ),
        h('p', { class: 'hint' }, 'This is a what-if estimate, not a guarantee. The backend uses the saved holdout precision and recall for its ensemble.'),
      ),
    );
    revealChildren(outputSlot);
  }

  form.addEventListener('input', () => {
    statusSlot.textContent = 'Assumptions changed. Recalculate to refresh the estimate.';
    outputSlot.replaceChildren(h('p', { class: 'muted' }, 'Your assumptions changed. Recalculate to show the current backend estimate.'));
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const { payload, errors } = readAssumptions();
    if (errors.length) {
      statusSlot.textContent = `${errors.length} assumption${errors.length === 1 ? '' : 's'} need attention.`;
      errors[0].input.focus();
      return;
    }

    submit.disabled = true;
    submit.textContent = 'Calculating…';
    statusSlot.textContent = 'Sending your assumptions to POST /roi…';
    outputSlot.setAttribute('aria-busy', 'true');
    try {
      const data = await api.roi(payload);
      renderResult(data);
      statusSlot.textContent = data.status === 'estimate' ? 'Estimate updated from the backend.' : 'The backend could not calculate an estimate.';
    } catch (error) {
      outputSlot.replaceChildren(
        error.kind === 'unreachable'
          ? offlineNotice(error)
          : notice({ kind: 'offline', title: `ROI request failed (HTTP ${error.status || '—'})`, body: [error.message] }),
      );
      statusSlot.textContent = 'The estimate could not be calculated.';
    } finally {
      submit.disabled = false;
      submit.textContent = 'Calculate ROI estimate';
      outputSlot.removeAttribute('aria-busy');
    }
  });

  outputSlot.replaceChildren(h('p', { class: 'muted' }, 'Enter your scenario assumptions and calculate an estimate using the backend’s measured ensemble metrics.'));
}
