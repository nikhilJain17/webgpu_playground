import { reflect, parameters } from './core.js';
import { KernelRunner } from './runner.js';
const $ = id => document.getElementById(id);
let baseline = '', sourceName = '', downloaded = false, result = null, loading = false;
const runner = new KernelRunner(message => {
  result = null; $('outputs').replaceChildren(); $('download-results').disabled = true;
  $('execution').textContent = 'Execution: FAILED'; $('execution').className = 'bad';
  $('correctness').textContent = 'Correctness: unavailable'; $('errors').textContent = message;
  $('status').textContent = message;
});
function message(error) { $('errors').textContent = `${error.stage ? `[${error.stage}] ` : ''}${error.message}`; }
function sourceStatus() {
  const dirty = $('editor').value !== baseline;
  $('source-status').textContent = `${sourceName || 'Editor'} — ${dirty ? 'UNSAVED editor changes' : downloaded ? 'downloaded snapshot (disk file not overwritten)' : 'matches loaded file'}`;
}
function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function addResource(resource = { group: 0, binding: 0, kind: 'storage', access: 'read', role: 'input', type: 'f32', length: 'N', init: 'deterministic', readback: false }) {
  const div = document.createElement('div'); div.className = 'resource';
  div.innerHTML = `<div class="grid"></div><details><summary>Initial values / packed fields (JSON)</summary><textarea class="extra" rows="3" spellcheck="false"></textarea><p>Values: {"values":[1,2]}. Fields: {"fields":[{"offset":0,"type":"u32","value":"channels"}]}. Offsets are bytes; follow the reflected WGSL layout, including padding.</p></details><button class="remove">Remove resource</button>`;
  const choices = { kind: ['storage','uniform'], access: ['read','read_write'], role: ['input','output','debug','uniform'], type: ['f32','u32','i32'], init: ['deterministic','zero','values','fields'] };
  for (const field of ['group','binding','kind','access','role','type','length','init','readback']) {
    const label = document.createElement('label'); label.textContent = field;
    const control = document.createElement(choices[field] ? 'select' : 'input'); control.dataset.field = field;
    if (choices[field]) for (const value of choices[field]) control.add(new Option(value, value));
    if (field === 'readback') { control.type = 'checkbox'; control.checked = !!resource[field]; }
    else { control.value = resource[field]; if (['group','binding'].includes(field)) { control.type = 'number'; control.min = 0; } }
    label.append(control); div.querySelector('.grid').append(label);
  }
  const extra = {};
  if (resource.values) extra.values = resource.values;
  if (resource.fields) extra.fields = resource.fields;
  div.querySelector('.extra').value = JSON.stringify(extra);
  div.querySelector('.remove').onclick = () => { div.remove(); };
  $('resources').append(div);
}
function readConfig() {
  return {
    entryPoint: $('entry').value.trim(), parameters: JSON.parse($('parameters').value),
    resources: [...document.querySelectorAll('.resource')].map(div => {
      const r = { ...JSON.parse(div.querySelector('.extra').value) };
      for (const control of div.querySelectorAll('[data-field]')) {
        const k = control.dataset.field;
        r[k] = k === 'readback' ? control.checked : ['group','binding'].includes(k) ? Number(control.value) : control.value;
      }
      if (r.length.trim() !== '' && Number.isFinite(Number(r.length))) r.length = Number(r.length);
      return r;
    }),
    dispatch: { mode: $('dispatch').value, workgroups: ['dx','dy','dz'].map(id => Number($(id).value)) },
    check: { mode: $('check').value, input: $('check-input').value, output: $('check-output').value, atol: Number($('atol').value), rtol: Number($('rtol').value) },
    constants: JSON.parse($('constants').value)
  };
}
function applyConfig(config) {
  $('entry').value = config.entryPoint;
  $('parameters').value = JSON.stringify(config.parameters, null, 2);
  $('resources').replaceChildren(); config.resources.forEach(addResource);
  $('dispatch').value = config.dispatch.mode;
  ['dx','dy','dz'].forEach((id, i) => $(id).value = config.dispatch.workgroups?.[i] ?? 1);
  $('check').value = config.check.mode;
  $('check-input').value = config.check.input ?? '0:0'; $('check-output').value = config.check.output ?? '0:2';
  $('atol').value = config.check.atol ?? 0.00001; $('rtol').value = config.check.rtol ?? 0.0001;
  $('constants').value = JSON.stringify(config.constants || {});
  updateParameters(); updateDispatch();
}
function updateDispatch() { $('explicit').hidden = $('dispatch').value !== 'explicit'; $('explicit').style.display = $('explicit').hidden ? 'none' : 'flex'; }
function updateParameters() {
  try { const p = parameters(JSON.parse($('parameters').value)); $('parameter-summary').textContent = `N = ${p.N}${p.batch ? `; ${p.batch * p.groups} normalization groups, ${p.channels / p.groups * p.spatialSize} elements each; epsilon = ${p.epsilon}. Choose explicit dispatch to match your algorithm.` : ''}`; }
  catch (e) { $('parameter-summary').textContent = e.message; }
}
function layout(type, prefix = '') {
  if (!type.members) return '';
  return type.members.map(m => `${prefix}${m.name}: ${m.type.getTypeName()}, offset ${m.offset}, size ${m.size}${m.type.isArray ? `, stride ${m.type.stride}` : ''}\n${layout(m.type, prefix + '  ')}`).join('');
}
function inspect() {
  try {
    const r = reflect($('editor').value, $('entry').value.trim());
    $('reflection').textContent = `Entry points: ${r.entries.join(', ')}\nWorkgroup: ${r.workgroup?.join(' × ') || 'unresolved / overrides: use explicit dispatch'}\n` +
      r.resources.map(b => `@group(${b.group}) @binding(${b.binding}) ${b.name}: ${b.kind} ${b.access} ${b.type}; minimum ${b.size} bytes${b.info.type.isArray ? `; stride ${b.info.type.stride}` : ''}\n${layout(b.info.type)}`).join('\n');
  } catch (e) { $('reflection').textContent = `Reflection failed: ${e.message}`; }
}
async function loadSource() {
  if ($('editor').value !== baseline && !confirm('Discard unsaved editor changes and load the selected file?')) return;
  loading = true; $('run').disabled = true;
  try {
    const response = await fetch($('source').value, { cache: 'no-store' });
    if (!response.ok) throw Error(`Could not load shader: HTTP ${response.status}`);
    baseline = await response.text(); sourceName = $('source').selectedOptions[0].textContent; downloaded = false;
    $('editor').value = baseline; sourceStatus(); inspect();
  } catch (e) { message(e); }
  finally { loading = false; $('run').disabled = runner.busy; }
}
function format(value) { return JSON.stringify(value, (_, v) => typeof v === 'number' && !Number.isFinite(v) ? String(v) : v, 2); }
async function run() {
  if (runner.busy || loading) return;
  result = null; $('outputs').replaceChildren(); $('errors').textContent = ''; $('diagnostics').textContent = ''; $('download-results').disabled = true;
  $('execution').textContent = 'Execution: running'; $('execution').className = 'pending'; $('correctness').textContent = 'Correctness: pending'; $('correctness').className = '';
  const code = $('editor').value;
  $('executed').textContent = `${sourceName} — ${code === baseline ? 'loaded source' : 'unsaved editor snapshot'} — ${new Date().toLocaleTimeString()}`;
  // Freeze all editable run inputs, preventing the display from diverging from the snapshot.
  const controls = [...document.querySelectorAll('input, select, textarea, button')];
  const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
  try {
    const config = readConfig();
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(code));
    const sourceHash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    $('executed').textContent += ` — SHA-256 ${sourceHash}`;
    result = await runner.run(code, config, stage => $('status').textContent = stage);
    $('execution').textContent = `Execution: SUCCESS — dispatched ${result.dispatch.join(' × ')} workgroups; shader workgroup ${result.workgroup?.join(' × ') || 'override-dependent'}`; $('execution').className = 'good';
    $('correctness').textContent = !result.check ? 'Correctness: DISABLED' : result.check.passed ? 'Correctness: PASS' : `Correctness: FAIL — ${result.check.count} mismatches, ${result.check.nonfinite} nonfinite pairs`;
    $('correctness').className = !result.check ? 'pending' : result.check.passed ? 'good' : 'bad';
    if (result.check && !result.check.passed) $('errors').textContent = `First mismatches (maximum 32):\n${format(result.check.mismatches)}`;
    $('diagnostics').textContent = result.diagnostics.map(m => `${m.type} ${m.line}:${m.column}: ${m.message}`).join('\n');
    for (const [key, data] of Object.entries(result.outputs)) {
      const pre = document.createElement('pre');
      const nonfinite = data.reduce((n, v) => n + !Number.isFinite(v), 0);
      pre.textContent = `Binding ${key} — ${data.length} values; ${nonfinite} nonfinite\n` + Array.from(data.slice(0, 256), (v, i) => `${i}: ${v}`).join('\n') + (data.length > 256 ? '\nPreview truncated. Download all readbacks for every index.' : '');
      $('outputs').append(pre);
    }
    if (result.reference?.means) {
      const pre = document.createElement('pre'); pre.textContent = `CPU reference means / variances (first 256):\n${format({ means: result.reference.means.slice(0,256), variances: result.reference.variances.slice(0,256) })}`; $('outputs').append(pre);
    }
    $('status').textContent = 'Ready';
  } catch (e) {
    result = null; message(e); $('outputs').replaceChildren();
    $('execution').textContent = `Execution: FAILED — ${e.stage || 'configuration'}`; $('execution').className = 'bad';
    $('correctness').textContent = 'Correctness: NOT CHECKED'; $('status').textContent = 'Run failed';
  } finally {
    controls.forEach((c, i) => c.disabled = disabled[i]); $('download-results').disabled = !result;
  }
}
$('run').onclick = run;
$('reload').onclick = loadSource;
$('editor').addEventListener('input', () => { sourceStatus(); $('reflection').textContent = 'Source changed — inspect again for current requirements.'; });
$('parameters').addEventListener('input', () => { updateParameters(); });
$('dispatch').addEventListener('change', updateDispatch);
$('reflect').onclick = inspect;
$('add').onclick = () => { addResource(); };
$('save').onclick = () => { download(sourceName.split('/').pop() || 'shader.wgsl', $('editor').value, 'text/plain'); downloaded = true; sourceStatus(); };
$('open').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  if ($('editor').value !== baseline && !confirm('Discard unsaved editor changes?')) return;
  baseline = await file.text(); sourceName = file.name; downloaded = false; $('editor').value = baseline; sourceStatus(); inspect(); event.target.value = '';
};
$('export').onclick = () => { try { download('kernel-config.json', JSON.stringify(readConfig(), null, 2)); } catch (e) { message(e); } };
$('import').onchange = async event => { try { const file = event.target.files[0]; if (file) applyConfig(JSON.parse(await file.text())); } catch (e) { message(e); } finally { event.target.value = ''; } };
$('download-results').onclick = () => { if (result) download('readbacks.json', format(Object.fromEntries(Object.entries(result.outputs).map(([k,v]) => [k, Array.from(v)])))); };
$('load-preset').onclick = async () => {
  try { const response = await fetch(`configs/${$('preset').value}.json`); if (!response.ok) throw Error('Preset fetch failed'); applyConfig(await response.json()); } catch (e) { message(e); }
};
try {
  const query = new URLSearchParams(location.search);
  const preset = query.get('config');
  if (preset && [...$('preset').options].some(o => o.value === preset)) $('preset').value = preset;
  const source = query.get('source');
  if (source && [...$('source').options].some(o => o.value === source)) $('source').value = source;
  const response = await fetch(`configs/${$('preset').value}.json`); if (!response.ok) throw Error('Preset fetch failed');
  const initial = await response.json();
  if (!preset) initial.check.mode = 'disabled';
  applyConfig(initial);
  await loadSource(); await runner.init(); $('status').textContent = 'WebGPU ready';
} catch (e) { message(e); $('status').textContent = 'Initialization failed'; }
window.addEventListener('beforeunload', event => {
  if ($('editor').value !== baseline) { event.preventDefault(); event.returnValue = ''; }
});
