console.log('[script] loaded');  // should appear in console immediately

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const bars = document.getElementById('bars');
const topLabel = document.getElementById('top-label');
const alphaSlider = document.getElementById('alpha');
const alphaVal = document.getElementById('alpha-val');
const startBtn = document.getElementById('startCam');

const LABELS = ["angry", "happy", "sad", "neutral"];
let prevProbs = null;

alphaSlider?.addEventListener('input', () => {
  alphaVal.textContent = Number(alphaSlider.value).toFixed(2);
});

function setStatus(msg) {
  console.log('[ui]', msg);
  if (statusEl) statusEl.textContent = msg;
}

async function getCameraStream() {
  const tries = [
    { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false },
    { video: { facingMode: 'user' }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr = null;
  for (const c of tries) {
    try {
      console.log('[gUM] requesting', c);
      const s = await navigator.mediaDevices.getUserMedia(c);
      console.log('[gUM] OK', s);
      return s;
    } catch (e) {
      console.warn('[gUM] failed', c, e);
      lastErr = e;
    }
  }
  throw lastErr || new Error('Camera not available');
}

async function setupCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia not supported in this browser');
    }
    setStatus('Requesting camera…');
    const stream = await getCameraStream();
    video.srcObject = stream;

    // wait for metadata then ensure playback
    await new Promise(res => video.addEventListener('loadedmetadata', res, { once: true }));
    await video.play().catch(e => console.warn('video.play() warn:', e));

    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;

    setStatus('Camera ready');
    if (startBtn) startBtn.style.display = 'none'; // hide after success
    tick(); // start loop
  } catch (e) {
    console.error('Camera error:', e);
    setStatus('Camera error: ' + (e?.message || e));
  }
}

function drawBox(face) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!face) return;
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#7aa2f7';
  ctx.strokeRect(face.x, face.y, face.w, face.h);
}

function renderBars(probs) {
  bars.innerHTML = '';
  LABELS.forEach(lbl => {
    const val = (probs[lbl] || 0) * 100;
    const row = document.createElement('div');
    row.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = val.toFixed(1) + '%';

    const lab = document.createElement('div'); lab.className = 'label'; lab.textContent = lbl;
    const score = document.createElement('div'); score.className = 'score'; score.textContent = val.toFixed(1) + '%';

    row.appendChild(fill); row.appendChild(lab); row.appendChild(score);
    bars.appendChild(row);
  });
}

function ema(curr, prev, alpha) {
  if (!prev) return curr;
  const out = {};
  LABELS.forEach(lbl => { out[lbl] = alpha * curr[lbl] + (1 - alpha) * prev[lbl]; });
  return out;
}

async function captureFrameBlob() {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640; c.height = video.videoHeight || 480;
  c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(res, 'image/jpeg', 0.8));
}

async function tick() {
  try {
    const blob = await captureFrameBlob();
    const form = new FormData(); form.append('frame', blob, 'frame.jpg');
    const res = await fetch('/predict', { method: 'POST', body: form });
    const json = await res.json();
    if (!json.success) throw new Error(json.error || 'Unknown error');

    drawBox(json.face);

    const alpha = Number(alphaSlider?.value || 0.6);
    prevProbs = ema(json.probs, prevProbs, alpha);
    renderBars(prevProbs);

    if (json.top) {
      let topLbl = '—', topScore = 0;
      LABELS.forEach(lbl => { if (prevProbs[lbl] > topScore) { topScore = prevProbs[lbl]; topLbl = lbl; } });
      topLabel.textContent = `${topLbl} (${(topScore * 100).toFixed(1)}%)`;
    } else {
      topLabel.textContent = 'No face detected'; prevProbs = null;
    }
  } catch (e) {
    console.error('Predict error:', e);
    setStatus('Predict error: ' + (e?.message || e));
  } finally {
    setTimeout(tick, 250);
  }
}

// Try auto-start on load; if autoplay/permissions block, the button works
window.addEventListener('DOMContentLoaded', () => {
  setStatus('Initializing camera…');
  setupCamera().catch(() => {
    if (startBtn) startBtn.style.display = 'inline-block';
  });
});
if (startBtn) startBtn.addEventListener('click', () => setupCamera());
