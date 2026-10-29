// Minimal RFC 4180 CSV parser / writer.

export function parseCSV(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  let line = 1;
  let rowLine = 1;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        if (c === '\n') line++;
        field += c;
      }
      continue;
    }
    if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      rows.push({ cells: row, line: rowLine });
      row = [];
      field = '';
      line++;
      rowLine = line;
    } else field += c;
  }
  if (quoted) throw new Error(`Unterminated quoted field starting near line ${rowLine}.`);
  if (field !== '' || row.length) {
    row.push(field);
    rows.push({ cells: row, line: rowLine });
  }

  const isBlank = (r) => r.cells.every((cell) => cell.trim() === '');
  const blankLines = rows.filter(isBlank).map((r) => r.line);
  return { rows: rows.filter((r) => !isBlank(r)), blankLines };
}

function escapeCell(value) {
  const s = value == null ? '' : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCSV(rows) {
  return rows.map((row) => row.map(escapeCell).join(',')).join('\r\n') + '\r\n';
}
