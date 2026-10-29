// Mirrors TransactionRequest in api.py exactly (extra="forbid").
// Only these keys are ever sent as top-level fields.

export const FIELDS = [
  { key: 'TransactionAmt', type: 'number', required: true, min: 0, max: 1_000_000_000, hint: 'Required · number 0 – 1,000,000,000' },
  { key: 'ProductCD', type: 'string', hint: 'String · API default "W"' },
  { key: 'card1', type: 'int', hint: 'Optional integer' },
  { key: 'card2', type: 'number', hint: 'Optional number' },
  { key: 'card3', type: 'number', hint: 'Optional number' },
  { key: 'card4', type: 'string', hint: 'Optional string · e.g. visa' },
  { key: 'card5', type: 'number', hint: 'Optional number' },
  { key: 'card6', type: 'string', hint: 'Optional string · e.g. debit' },
  { key: 'addr1', type: 'number', hint: 'Optional number' },
  { key: 'addr2', type: 'number', hint: 'Optional number' },
  { key: 'dist1', type: 'number', hint: 'Optional number' },
  { key: 'P_emaildomain', type: 'string', hint: 'Optional string · e.g. gmail.com' },
  { key: 'DeviceInfo', type: 'string', hint: 'Optional string' },
  { key: 'DeviceType', type: 'string', hint: 'Optional string · e.g. desktop, mobile' },
  { key: 'hour_of_day', type: 'int', min: 0, max: 23, hint: 'Integer 0–23 · API default 12' },
  { key: 'transaction_timestamp', type: 'datetime', hint: 'Optional ISO-8601 datetime · no offset = UTC' },
];

export const FIELD_KEYS = FIELDS.map((field) => field.key);
export const EXTRA_KEY = 'extra_features';
// Columns accepted in a batch CSV: every declared field plus a JSON extra_features cell.
export const CSV_COLUMNS = [...FIELD_KEYS, EXTRA_KEY];
export const MAX_EXTRA_FEATURES = 500;

const NUMBER_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const INT_RE = /^[-+]?\d+$/;
const ISO_RE = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:?\d{2})?$/i;

export function parseExtraFeatures(input) {
  if (input == null) return { value: null, errors: [] };
  let parsed = input;
  if (typeof input === 'string') {
    const text = input.trim();
    if (!text) return { value: null, errors: [] };
    try {
      parsed = JSON.parse(text);
    } catch {
      return { value: null, errors: [{ field: EXTRA_KEY, message: 'must be a valid JSON object' }] };
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { value: null, errors: [{ field: EXTRA_KEY, message: 'must be a JSON object of feature → value' }] };
  }
  const errors = [];
  const keys = Object.keys(parsed);
  if (keys.length > MAX_EXTRA_FEATURES) errors.push({ field: EXTRA_KEY, message: `may contain at most ${MAX_EXTRA_FEATURES} keys` });
  const overlap = keys.filter((k) => FIELD_KEYS.includes(k) || k === EXTRA_KEY);
  if (overlap.length) errors.push({ field: EXTRA_KEY, message: `cannot override declared fields (${overlap.join(', ')})` });
  const badValues = keys.filter((k) => {
    const v = parsed[k];
    return !(v === null || typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v)));
  });
  if (badValues.length) errors.push({ field: EXTRA_KEY, message: `values must be string, number, or null (check ${badValues.slice(0, 5).join(', ')})` });
  return { value: errors.length || !keys.length ? null : parsed, errors };
}

export function coerceTransaction(raw) {
  const value = {};
  const errors = [];
  for (const field of FIELDS) {
    let input = raw[field.key];
    input = input == null ? '' : String(input).trim();
    if (input === '') {
      if (field.required) errors.push({ field: field.key, message: 'is required' });
      continue;
    }
    if (field.type === 'string') {
      value[field.key] = input;
      continue;
    }
    if (field.type === 'datetime') {
      if (!ISO_RE.test(input) || Number.isNaN(Date.parse(input.replace(' ', 'T')))) {
        errors.push({ field: field.key, message: 'must be ISO-8601, e.g. 2026-10-28T03:15:00Z' });
      } else {
        value[field.key] = input;
      }
      continue;
    }
    const pattern = field.type === 'int' ? INT_RE : NUMBER_RE;
    if (!pattern.test(input)) {
      errors.push({ field: field.key, message: field.type === 'int' ? 'must be a whole number' : 'must be a number' });
      continue;
    }
    const n = Number(input);
    if (!Number.isFinite(n) || (field.type === 'int' && !Number.isSafeInteger(n))) {
      errors.push({ field: field.key, message: 'is out of range' });
    } else if (field.min != null && n < field.min) {
      errors.push({ field: field.key, message: `must be ≥ ${field.min}` });
    } else if (field.max != null && n > field.max) {
      errors.push({ field: field.key, message: `must be ≤ ${field.max.toLocaleString('en-US')}` });
    } else {
      value[field.key] = n;
    }
  }
  const extra = parseExtraFeatures(raw[EXTRA_KEY]);
  errors.push(...extra.errors);
  if (extra.value) value[EXTRA_KEY] = extra.value;
  return { value, errors };
}

// A demo input for trying the form. It is not a real transaction and not taken from api.py.
export const SAMPLE_INPUT = {
  TransactionAmt: '599.99',
  ProductCD: 'W',
  card1: '10409',
  card4: 'visa',
  card6: 'debit',
  P_emaildomain: 'gmail.com',
  DeviceType: 'desktop',
  hour_of_day: '3',
};
