// Pyodide + Streamlit component for client-side redaction
const streamlit = window.parent.Streamlit;
let pyodide = null;

function showStatus(msg, color='black') {
  streamlit.setComponentValue({status: msg, color: color});
}

async function loadPyodideAndPackages() {
  showStatus('Loading Pyodide...');
  pyodide = await window.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
  });
  showStatus('Installing packages...');
  await pyodide.loadPackage(['pandas', 'numpy']);
  showStatus('Pyodide ready!', 'green');
  return pyodide;
}

async function redactData(pyodide, csvString) {
  const code = `
import pandas as pd
import hashlib
import datetime
from io import StringIO

df = pd.read_csv(StringIO(csv_string))
redacted = df.copy()
if 'patient_id' in redacted.columns:
    del redacted['patient_id']
if 'patient_name' in redacted.columns:
    del redacted['patient_name']
if 'dob' in redacted.columns and 'city' in redacted.columns:
    def hash_row(row):
        return hashlib.sha256((str(row['dob']) + str(row['city'])).encode()).hexdigest()
    redacted['dob_city_hash'] = redacted.apply(hash_row, axis=1)
    del redacted['dob']
    del redacted['city']
# Generate privacy proof
proof = {
    'timestamp': datetime.datetime.now().isoformat(),
    'columns_removed': ['patient_id', 'patient_name'],
    'columns_hashed': ['dob', 'city'],
    'hash_examples': redacted['dob_city_hash'].head(3).tolist(),
    'method': 'SHA-256 on dob+city',
    'row_count': len(redacted)
}
`;
  pyodide.globals.set('csv_string', csvString);
  await pyodide.runPythonAsync(code);
  const redacted = pyodide.globals.get('redacted');
  const proof = pyodide.globals.get('proof').toJs();
  return {redacted: redacted.to_csv(index=false), proof};
}

window.addEventListener('message', async (event) => {
  const {type, csvString} = event.data;
  if (type === 'init') {
    try {
      await loadPyodideAndPackages();
      streamlit.setComponentValue({status: 'ready'});
    } catch (e) {
      streamlit.setComponentValue({status: 'pyodide_error', error: e.toString()});
    }
  } else if (type === 'redact' && pyodide) {
    try {
      showStatus('Redacting data...', 'blue');
      const result = await redactData(pyodide, csvString);
      streamlit.setComponentValue({status: 'redacted', ...result});
    } catch (e) {
      streamlit.setComponentValue({status: 'redact_error', error: e.toString()});
    }
  }
});
