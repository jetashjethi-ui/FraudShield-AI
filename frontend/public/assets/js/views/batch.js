import { h, fmt, tierByKey, RISK_TIERS, revealChildren, downloadText } from '../util.js';
import { api } from '../api.js';
import { addSessionResult } from '../store.js';
import { FIELDS, CSV_COLUMNS, EXTRA_KEY, coerceTransaction } from '../schema.js';
import { parseCSV, toCSV } from '../csv.js';
import { notice, offlineNotice } from '../components.js';

const MAX_ROWS = 500;
const MAX_BYTES = 5 * 1024 * 1024;
const PREVIEW_ROWS = 25;

const TEMPLATE = [
  CSV_COLUMNS,
  ['599.99', 'W', '10409', '', '', 'visa', '', 'debit', '', '', '', 'gmail.com', '', 'desktop', '3', '', ''],
  ['42.50', 'C', '7919', '194', '150', 'mastercard', '226', 'credit', '315', '87', '12', 'yahoo.com', 'Windows', 'desktop', '14', '', '{"C1": 1, "V1": 0.42}'],
];

function validate(text) {
  const fileErrors = [];
  let parsed;
  try {
    parsed = parseCSV(text);
  } catch (error) {
    return { fileErrors: [error.message], header: [], records: [], blankLines: [] };
  }
  const [headRow, ...dataRows] = parsed.rows;
  if (!headRow) return { fileErrors: ['The file is empty.'], header: [], records: [], blankLines: [] };

  const header = headRow.cells.map((cell) => cell.trim());
  const unknown = header.filter((name) => name && !CSV_COLUMNS.includes(name));
  const dupes = header.filter((name, i) => name && header.indexOf(name) !== i);
  if (header.some((name) => !name)) fileErrors.push('The header row contains an empty column name.');
  if (unknown.length) {
    fileErrors.push(
      `Unsupported column${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}. POST /batch accepts ${CSV_COLUMNS.join(', ')}. Remove ${unknown.length > 1 ? 'them' : 'it'} so no input is silently discarded.`,
    );
  }
  if (dupes.length) fileErrors.push(`Duplicate column${dupes.length > 1 ? 's' : ''}: ${[...new Set(dupes)].join(', ')}.`);
  if (!header.includes('TransactionAmt')) fileErrors.push('Missing required column TransactionAmt.');
  if (dataRows.length === 0) fileErrors.push('No data rows found below the header.');
  if (dataRows.length > MAX_ROWS) fileErrors.push(`${dataRows.length} rows found. The limit is ${MAX_ROWS} per batch; split the file.`);

  const records = dataRows.map((dataRow, index) => {
    const raw = {};
    header.forEach((name, i) => (raw[name] = dataRow.cells[i] ?? ''));
    const errors = [];
    if (dataRow.cells.length !== header.length) {
      errors.push({ field: '(row)', message: `has ${dataRow.cells.length} cells but the header has ${header.length}` });
    }
    const coerced = coerceTransaction(raw);
    errors.push(...coerced.errors);
    return { index: index + 1, line: dataRow.line, raw, value: coerced.value, errors };
  });

  const timestampRows = records.filter((record) => record.value.transaction_timestamp).length;
  if (timestampRows > 0 && timestampRows < records.length) {
    fileErrors.push('Provide transaction_timestamp for every row, or leave it empty for every row. The backend cannot mix timestamped and untimestamped transactions in one batch.');
  }

  return { fileErrors, header, records, blankLines: parsed.blankLines };
}

function resultsTable(records, results) {
  return h(
    'div',
    { class: 'table-wrap' },
    h(
      'table',
      { class: 'data-table' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          ['Row', 'Amount', 'Transaction ID', 'Score', 'Category', 'Probability', 'Time', 'Context signals / error'].map((t) =>
            h('th', { scope: 'col' }, t),
          ),
        ),
      ),
      h(
        'tbody',
        null,
        records.map((record, i) => {
          const r = results[i];
          if (!r || r.error) {
            return h(
              'tr',
              { class: 'is-invalid' },
              h('td', null, String(record.index)),
              h('td', { class: 'num' }, fmt.dec(record.value.TransactionAmt, 2)),
              h('td', { colspan: 5 }, '—'),
              h('td', { class: 'wrap' }, r ? `Error: ${r.error}` : 'No result returned for this row'),
            );
          }
          const tier = tierByKey(r.risk_category);
          return h(
            'tr',
            null,
            h('td', null, String(record.index)),
            h('td', { class: 'num' }, fmt.dec(record.value.TransactionAmt, 2)),
            h('td', null, r.transaction_id),
            h('td', { class: 'num tone-text', dataset: tier ? { tone: tier.tone } : undefined }, fmt.dec(r.risk_score, 1)),
            h('td', null, r.risk_category),
            h('td', { class: 'num' }, fmt.pct(r.fraud_probability, 2)),
            h('td', { class: 'num' }, fmt.ms(r.processing_time_ms)),
            h('td', { class: 'wrap' }, (r.context_signals || []).join(' · ')),
          );
        }),
      ),
    ),
  );
}

function exportCSV(records, results) {
  const head = [
    'row',
    ...CSV_COLUMNS.map((k) => `input_${k}`),
    'transaction_id',
    'risk_score',
    'risk_category',
    'fraud_probability',
    'auth_recommendation',
    'context_signals',
    'base_model_probabilities',
    'processing_time_ms',
    'timestamp',
    'error',
  ];
  const lines = records.map((record, i) => {
    const r = results[i] || { error: 'No result returned for this row' };
    return [
      record.index,
      ...CSV_COLUMNS.map((k) => {
        const value = record.value[k];
        return k === EXTRA_KEY && value && typeof value === 'object' ? JSON.stringify(value) : value ?? '';
      }),
      r.transaction_id ?? '',
      r.risk_score ?? '',
      r.risk_category ?? '',
      r.fraud_probability ?? '',
      r.auth_recommendation ?? '',
      (r.context_signals || []).join(' | '),
      r.base_model_probabilities ? JSON.stringify(r.base_model_probabilities) : '',
      r.processing_time_ms ?? '',
      r.timestamp ?? '',
      r.error ?? '',
    ];
  });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  downloadText(`fraudshield-batch-${stamp}.csv`, toCSV([head, ...lines]));
}

export function initBatch(root) {
  const input = root.querySelector('#batch-file');
  const zone = root.querySelector('.dropzone');
  const fileLabel = root.querySelector('[data-slot="batch-file-name"]');
  const validationSlot = root.querySelector('[data-slot="batch-validation"]');
  const previewSlot = root.querySelector('[data-slot="batch-preview"]');
  const runSlot = root.querySelector('[data-slot="batch-run"]');
  const resultsSlot = root.querySelector('[data-slot="batch-results"]');
  const submit = root.querySelector('[data-action="batch-submit"]');
  const clear = root.querySelector('[data-action="batch-clear"]');
  const schemaSlot = root.querySelector('[data-slot="batch-schema"]');

  schemaSlot.replaceChildren(
    ...FIELDS.map((f) => h('li', { class: 'tag', title: f.hint }, f.required ? `${f.key} *` : f.key)),
    h('li', { class: 'tag', title: 'Optional JSON object with up to 500 named features' }, `${EXTRA_KEY} · JSON`),
  );

  root.querySelector('[data-action="batch-template"]').addEventListener('click', () => {
    downloadText('fraudshield-batch-template.csv', toCSV(TEMPLATE));
  });

  let current = null;

  function reset() {
    current = null;
    input.value = '';
    fileLabel.textContent = 'No file selected';
    validationSlot.replaceChildren();
    previewSlot.replaceChildren();
    runSlot.replaceChildren();
    resultsSlot.replaceChildren();
    submit.disabled = true;
    clear.disabled = true;
  }

  function renderValidation(name, check) {
    const invalid = check.records.filter((r) => r.errors.length);
    const valid = check.records.length - invalid.length;
    const ready = !check.fileErrors.length && !invalid.length && check.records.length > 0;
    submit.disabled = !ready;
    clear.disabled = false;
    fileLabel.textContent = name;

    const stats = h(
      'div',
      { class: 'stat-strip', 'data-reveal': true },
      h('div', { class: 'stat' }, h('span', { class: 'stat-label' }, 'Rows'), h('span', { class: 'stat-value' }, fmt.int(check.records.length))),
      h(
        'div',
        { class: 'stat', dataset: { tone: 'green' } },
        h('span', { class: 'stat-label' }, 'Valid'),
        h('span', { class: 'stat-value' }, fmt.int(valid)),
      ),
      h(
        'div',
        { class: 'stat', dataset: invalid.length ? { tone: 'red' } : undefined },
        h('span', { class: 'stat-label' }, 'Invalid'),
        h('span', { class: 'stat-value' }, fmt.int(invalid.length)),
      ),
      h(
        'div',
        { class: 'stat' },
        h('span', { class: 'stat-label' }, 'Columns'),
        h('span', { class: 'stat-value' }, fmt.int(check.header.length)),
      ),
    );

    const parts = [stats];
    if (check.blankLines.length) {
      parts.push(
        h('p', { class: 'table-caption' }, `Ignored ${check.blankLines.length} blank line(s) at line ${check.blankLines.slice(0, 8).join(', ')}${check.blankLines.length > 8 ? '…' : ''}.`),
      );
    }
    if (check.fileErrors.length) {
      parts.push(notice({ kind: 'offline', title: 'File cannot be submitted', body: check.fileErrors }));
    }
    if (invalid.length) {
      parts.push(
        h(
          'div',
          { class: 'stack-sm', 'data-reveal': true },
          h('h3', { class: 'sub-title' }, `${invalid.length} invalid row${invalid.length > 1 ? 's' : ''} · fix and re-upload`),
          h(
            'ul',
            { class: 'error-list' },
            invalid.map((r) => h('li', null, `Row ${r.index} (line ${r.line}): ${r.errors.map((e) => `${e.field} ${e.message}`).join('; ')}`)),
          ),
        ),
      );
    }
    if (ready) {
      parts.push(
        notice({
          kind: 'info',
          title: `${fmt.int(valid)} rows ready`,
          body: ['All rows passed validation against the TransactionRequest schema. Submitting sends them in one POST /batch request.'],
        }),
      );
    }
    validationSlot.replaceChildren(...parts);

    const visible = check.records.slice(0, PREVIEW_ROWS);
    previewSlot.replaceChildren(
      h('h3', { class: 'sub-title' }, 'Preview'),
      h(
        'div',
        { class: 'table-wrap' },
        h(
          'table',
          { class: 'data-table' },
          h(
            'thead',
            null,
            h('tr', null, h('th', { scope: 'col' }, 'Row'), h('th', { scope: 'col' }, 'Status'), check.header.map((c) => h('th', { scope: 'col' }, c))),
          ),
          h(
            'tbody',
            null,
            visible.map((r) =>
              h(
                'tr',
                { class: r.errors.length ? 'is-invalid' : '' },
                h('td', null, String(r.index)),
                h('td', null, r.errors.length ? `${r.errors.length} error${r.errors.length > 1 ? 's' : ''}` : 'OK'),
                check.header.map((c) => h('td', null, r.raw[c] ?? '')),
              ),
            ),
          ),
        ),
      ),
      h(
        'p',
        { class: 'table-caption' },
        check.records.length > PREVIEW_ROWS
          ? `Showing ${PREVIEW_ROWS} of ${fmt.int(check.records.length)} rows. All rows are validated.`
          : `Showing all ${fmt.int(check.records.length)} rows.`,
      ),
    );
    revealChildren(root.querySelector('[data-slot="batch-body"]'));
  }

  async function handleFile(file) {
    if (!file) return;
    resultsSlot.replaceChildren();
    runSlot.replaceChildren();
    if (file.size > MAX_BYTES) {
      reset();
      clear.disabled = false;
      fileLabel.textContent = file.name;
      validationSlot.replaceChildren(notice({ kind: 'offline', title: 'File too large', body: ['Files over 5 MB are not accepted. A 500-row batch is far smaller.'] }));
      return;
    }
    const text = await file.text();
    const check = validate(text);
    current = { name: file.name, ...check };
    renderValidation(file.name, check);
  }

  input.addEventListener('change', () => handleFile(input.files && input.files[0]));
  ['dragenter', 'dragover'].forEach((type) =>
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add('is-over');
    }),
  );
  ['dragleave', 'drop'].forEach((type) => zone.addEventListener(type, () => zone.classList.remove('is-over')));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer && e.dataTransfer.files[0]);
  });

  clear.addEventListener('click', reset);

  submit.addEventListener('click', async () => {
    if (!current) return;
    const records = current.records;
    const started = performance.now();
    const label = h('p', { class: 'progress-label', role: 'status' }, `Submitting ${records.length} rows to POST /batch…`);
    runSlot.replaceChildren(h('div', { class: 'progress', 'aria-hidden': 'true' }), label);
    const ticker = setInterval(() => {
      label.textContent = `Submitting ${records.length} rows to POST /batch… ${((performance.now() - started) / 1000).toFixed(1)}s`;
    }, 200);
    submit.disabled = true;
    clear.disabled = true;
    input.disabled = true;
    resultsSlot.replaceChildren();

    try {
      const response = await api.batch(records.map((r) => r.value));
      const results = Array.isArray(response.results) ? response.results : [];
      const elapsed = (performance.now() - started) / 1000;
      const failed = results.filter((r) => r && r.error).length + Math.max(0, records.length - results.length);
      const counts = Object.fromEntries(RISK_TIERS.map((t) => [t.key, 0]));
      results.forEach((r) => {
        if (r && !r.error && r.risk_category in counts) counts[r.risk_category]++;
      });

      runSlot.replaceChildren(
        h('p', { class: 'progress-label', role: 'status' }, `Completed in ${elapsed.toFixed(1)}s · API count ${fmt.int(response.count)}.`),
      );

      const parts = [];
      if (response.count !== records.length || results.length !== records.length) {
        parts.push(
          notice({
            kind: 'warn',
            title: 'Result count mismatch',
            body: [`Submitted ${records.length} rows, API returned ${results.length} results (count ${response.count}). Unmatched rows are marked below.`],
          }),
        );
      }
      parts.push(
        h(
          'div',
          { class: 'stat-strip', 'data-reveal': true },
          RISK_TIERS.map((t) =>
            h(
              'div',
              { class: 'stat', dataset: { tone: t.tone } },
              h('span', { class: 'stat-label' }, t.label),
              h('span', { class: 'stat-value' }, fmt.int(counts[t.key])),
            ),
          ),
          h(
            'div',
            { class: 'stat', dataset: failed ? { tone: 'red' } : undefined },
            h('span', { class: 'stat-label' }, 'Row errors'),
            h('span', { class: 'stat-value' }, fmt.int(failed)),
          ),
        ),
        h(
          'div',
          { class: 'dist', role: 'img', 'aria-label': RISK_TIERS.map((t) => `${t.label} ${counts[t.key]}`).join(', '), 'data-reveal': true },
          RISK_TIERS.map((t) => h('span', { dataset: { tone: t.tone }, style: { 'flex-grow': String(counts[t.key]) } })),
        ),
        h(
          'div',
          { class: 'btn-row', 'data-reveal': true },
          h('button', { class: 'btn', type: 'button', onclick: () => exportCSV(records, results) }, 'Download results CSV'),
        ),
        resultsTable(records, results),
      );
      resultsSlot.replaceChildren(...parts);
      revealChildren(resultsSlot);
      addSessionResult({ source: 'batch', at: new Date(), rows: records.length, counts, failed });
    } catch (error) {
      runSlot.replaceChildren();
      resultsSlot.replaceChildren(
        error.kind === 'unreachable'
          ? offlineNotice(error)
          : notice({
              kind: 'offline',
              title: `Batch failed (HTTP ${error.status || '—'})`,
              body: [error.message, 'Row numbers in the message refer to zero-based positions in the transactions array.'],
            }),
      );
    } finally {
      clearInterval(ticker);
      submit.disabled = false;
      clear.disabled = false;
      input.disabled = false;
    }
  });

  reset();
  return {};
}
