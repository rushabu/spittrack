const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#video-file');
const button = document.querySelector('#run-button');
const statusText = document.querySelector('#status');
let sourceUrl;
let timer;
fileInput.addEventListener('change', () => {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  const video = document.querySelector('#source-video');
  video.hidden = !fileInput.files.length;
  if (fileInput.files.length) {
    sourceUrl = URL.createObjectURL(fileInput.files[0]);
    video.src = sourceUrl;
  }
});
function fileUrl(id, name) { return `/runs/${id}/files/${name}`; }
function showResults(id, data) {
  document.querySelector('#waiting').hidden = true;
  document.querySelector('#results').hidden = false;
  document.querySelector('#result-video').src = fileUrl(id, 'annotated_browser.mp4');
  const summary = data.summary;
  const metrics = document.querySelector('#metrics');
  metrics.replaceChildren();
  for (const [label, value] of [['Events', summary.event_count], ['Windows analyzed', summary.window_count], ['Threshold', Number(summary.frozen_threshold).toFixed(4)], ['Total inference', `${Number(summary.end_to_end_performance.wall_time_s).toFixed(1)}s`]]) {
    const box = document.createElement('div'); box.className = 'metric';
    const strong = document.createElement('strong'); strong.textContent = value;
    const caption = document.createElement('span'); caption.textContent = label;
    box.append(strong, caption); metrics.append(box);
  }
  const downloads = document.querySelector('#downloads'); downloads.replaceChildren();
  for (const [name, label] of [['annotated_browser.mp4', 'Annotated video ↓'], ['events.csv', 'Events CSV ↓'], ['events.json', 'Events JSON ↓'], ['inference_summary.json', 'Run summary ↓']]) {
    const link = document.createElement('a'); link.href = fileUrl(id, name) + '?download=1'; link.textContent = label; downloads.append(link);
  }
  const events = document.querySelector('#events'); events.replaceChildren();
  if (!data.events.length) events.textContent = 'No spitting events met the frozen threshold and temporal aggregation rules in this video.';
  for (const event of data.events) {
    const card = document.createElement('article'); card.className = 'event';
    const caption = document.createElement('p');
    caption.textContent = `${event.event_id} · Person ${event.track_id} · ${Number(event.start_s).toFixed(2)}–${Number(event.end_s).toFixed(2)} s · Confidence ${(100 * event.confidence).toFixed(1)}%`;
    const clip = document.createElement('video'); clip.controls = true; clip.playsInline = true; clip.preload = 'metadata';
    clip.src = fileUrl(id, `evidence/${event.event_id}_browser.mp4`);
    card.append(caption, clip); events.append(card);
  }
}
async function poll(id) {
  try {
    const response = await fetch(`/api/runs/${id}`);
    if (!response.ok) throw new Error('Could not read run status.');
    const data = await response.json(); statusText.textContent = data.message;
    document.querySelector('#log-details').hidden = false;
    document.querySelector('#log').textContent = data.log || 'Starting inference process…';
    document.querySelector('#log-download').href = fileUrl(id, 'process.log') + '?download=1';
    if (data.state === 'complete') { showResults(id, data); button.disabled = false; }
    else if (data.state === 'failed') { statusText.className = 'error'; button.disabled = false; }
    else timer = setTimeout(() => poll(id), 2000);
  } catch (error) {
    statusText.textContent = `${error.message} Reconnecting…`;
    timer = setTimeout(() => poll(id), 4000);
  }
}
form.addEventListener('submit', async event => {
  event.preventDefault(); clearTimeout(timer); button.disabled = true;
  statusText.className = ''; statusText.textContent = 'Uploading video…';
  document.querySelector('#results').hidden = true;
  document.querySelector('#waiting').hidden = false;
  document.querySelector('#log-details').hidden = true;
  try {
    const response = await fetch('/api/runs', {method: 'POST', body: new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed.');
    history.replaceState(null, '', `/#${data.id}`); poll(data.id);
  } catch (error) { statusText.textContent = error.message; statusText.className = 'error'; button.disabled = false; }
});
if (/^#[a-f0-9]{32}$/.test(location.hash)) { button.disabled = true; poll(location.hash.slice(1)); }
