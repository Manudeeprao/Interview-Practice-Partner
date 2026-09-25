/**
 * Interview Practice Partner — Frontend Application
 *
 * Architecture:
 *  - SpeechRecognition (STT) for voice input → feeds into sendMessage()
 *  - SpeechSynthesis (TTS) for agent voice output
 *  - Text input always visible; works identically to voice input downstream
 *  - POST /chat for each turn, POST /feedback for final feedback
 *  - Session ID stored in memory (new ID each page load)
 *
 * State machine:
 *  idle → listening (mic on) → thinking (awaiting LLM) → speaking (TTS) → idle
 *  Any state → done (feedback rendered)
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────
const BACKEND_URL = 'http://localhost:5000';

// ── DOM References ─────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  transcript:         $('transcript'),
  transcriptEmpty:    $('transcript-empty'),
  statusBadge:        $('status-badge'),
  statusText:         $('status-text'),
  sessionInfo:        $('session-info'),
  textInput:          $('text-input'),
  btnSend:            $('btn-send'),
  btnMic:             $('btn-mic'),
  interimText:        $('interim-text'),
  btnEnd:             $('btn-end-interview'),
  btnUploadResume:    $('btn-upload-resume'),
  resumeFileInput:    $('resume-file'),
  progressSection:    $('progress-section'),
  progressFill:       $('progress-fill'),
  progressLabel:      $('progress-label'),
  progressBar:        document.querySelector('.progress-bar'),
  stageLabel:         $('stage-label'),
  permissionWarn:     $('permission-warning'),
  feedbackPanel:      $('feedback-panel'),
  feedbackGrid:       $('feedback-grid'),
  feedbackTitle:      $('feedback-title'),
  feedbackSubtitle:   $('feedback-subtitle'),
  scoreRing:          $('score-ring'),
  scoreValue:         $('score-value'),
  btnRestart:         $('btn-restart'),
  btnCopyFeedback:    $('btn-copy-feedback'),
  btnClearTranscript: $('btn-clear-transcript'),
  toastContainer:     $('toast-container'),
  backendError:       $('backend-error'),
};

// ── App State ──────────────────────────────────────────────────
const state = {
  sessionId:        null,
  voiceAvailable:   false,
  isRecording:      false,
  isThinking:       false,
  isSpeaking:       false,
  interviewStarted: false,
  interviewDone:    false,
  typingMsgEl:      null,
  speechSynth:      window.speechSynthesis || null,
  currentUtterance: null,
  recognition:      null,
  lastFeedback:     null,   // stored for copy-to-clipboard
  backendHealthy:   false,
};

let backendHealthRetryTimer = null;

// ── Session Initialisation ─────────────────────────────────────
function initSession() {
  state.sessionId = 'ipp-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
  dom.sessionInfo.textContent = `Session: ${state.sessionId.slice(0, 16)}…`;
}

// ── Status Badge ───────────────────────────────────────────────
function setStatus(statusKey, label) {
  dom.statusBadge.className = `status-badge ${statusKey}`;
  dom.statusText.textContent = label;
}

// ── Toast Notifications ────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s ease';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

function setUploadButtonState(state) {
  const button = dom.btnUploadResume;
  if (!button) return;

  switch (state) {
    case 'uploading':
      button.disabled = true;
      button.textContent = 'Uploading…';
      break;
    case 'success':
      button.disabled = true;
      button.textContent = 'Uploaded';
      break;
    case 'error':
      button.disabled = false;
      button.textContent = 'Retry';
      break;
    default:
      button.disabled = false;
      button.textContent = 'Upload';
  }
}

// ── Transcript Rendering ───────────────────────────────────────
function hideEmpty() {
  if (dom.transcriptEmpty) dom.transcriptEmpty.style.display = 'none';
}

function addMessage(role, content, isTyping = false) {
  hideEmpty();

  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'agent' ? '🤖' : '👤';
  avatar.setAttribute('aria-hidden', 'true');

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const label = document.createElement('div');
  label.className = 'bubble-label';
  label.textContent = role === 'agent' ? 'Interviewer' : 'You';
  bubble.appendChild(label);

  if (isTyping) {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('span');
      dot.className = 'typing-dot';
      indicator.appendChild(dot);
    }
    bubble.appendChild(indicator);
    wrapper.dataset.typing = 'true';
    state.typingMsgEl = wrapper;
  } else {
    const text = document.createElement('div');
    text.className = 'bubble-text';
    text.textContent = content;
    bubble.appendChild(text);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  dom.transcript.appendChild(wrapper);
  dom.transcript.scrollTop = dom.transcript.scrollHeight;
  return wrapper;
}

function removeTypingIndicator() {
  if (state.typingMsgEl) {
    state.typingMsgEl.remove();
    state.typingMsgEl = null;
  }
}

// ── Progress + Stage ───────────────────────────────────────────
// ALWAYS reads from backend state_info — never maintains local counter.
const STAGE_LABELS = {
  'introduction':            '👋 Introduction',
  'project_discussion':      '💼 Project Discussion',
  'technical_fundamentals':  '⚙️ Technical Fundamentals',
  'system_design':           '🏗️ System Design',
  'behavioral':              '💬 Behavioral',
  'feedback':                '📊 Feedback',
  'INTERVIEW_ACTIVE':        '🎤 Interview',
  'ROLE_SELECTION':          '🎯 Setup',
};

function updateProgress(qCount, maxQ, stage, difficulty) {
  const pct = maxQ > 0 ? Math.min(100, Math.round((qCount / maxQ) * 100)) : 0;
  dom.progressFill.style.width = `${pct}%`;
  dom.progressLabel.textContent = `Q ${qCount} / ${maxQ}`;
  dom.progressBar.setAttribute('aria-valuenow', pct);

  if (dom.stageLabel) {
    const difficultyText = difficulty ? ` · ${String(difficulty).toUpperCase()}` : '';
    const stageText = (STAGE_LABELS[stage] || stage || '') + difficultyText;
    dom.stageLabel.textContent = stageText;
    dom.stageLabel.style.display = stageText ? 'inline-flex' : 'none';
  }

  dom.progressSection.classList.toggle('visible', qCount > 0 || state.interviewStarted);
}

// ── Input Controls ─────────────────────────────────────────────
function setInputEnabled(enabled) {
  dom.textInput.disabled = !enabled;
  dom.btnSend.disabled = !enabled || dom.textInput.value.trim() === '';
  if (state.voiceAvailable) dom.btnMic.disabled = !enabled;
}

dom.textInput.addEventListener('input', () => {
  dom.textInput.style.height = 'auto';
  dom.textInput.style.height = Math.min(dom.textInput.scrollHeight, 120) + 'px';
  dom.btnSend.disabled = dom.textInput.value.trim() === '' || state.isThinking;
});

dom.textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const msg = dom.textInput.value.trim();
    if (msg && !state.isThinking && !state.interviewDone) sendMessage(msg);
  }
});

dom.btnSend.addEventListener('click', () => {
  const msg = dom.textInput.value.trim();
  if (msg && !state.isThinking && !state.interviewDone) sendMessage(msg);
});

dom.btnClearTranscript.addEventListener('click', () => {
  Array.from(dom.transcript.children).forEach(child => {
    if (child.id !== 'transcript-empty') child.remove();
  });
  if (dom.transcriptEmpty) dom.transcriptEmpty.style.display = '';
});

async function uploadResume() {
  const file = dom.resumeFileInput?.files?.[0];
  if (!file) {
    showToast('Please choose a PDF or DOCX resume first.', 'warning');
    return;
  }

  if (!state.sessionId) initSession();

  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', state.sessionId);

  setUploadButtonState('uploading');

  try {
    const response = await fetch(`${BACKEND_URL}/resume/upload`, {
      method: 'POST',
      body: formData,
    });

    const rawText = await response.text();
    let data = {};
    if (rawText) {
      try {
        data = JSON.parse(rawText);
      } catch {
        data = { message: rawText };
      }
    }

    if (!response.ok) {
      throw new Error(data.error || data.message || 'Resume upload failed');
    }

    const successMessage = data.message || `Resume uploaded${data.filename ? `: ${data.filename}` : ''}`;
    showToast(successMessage, 'success');
    setUploadButtonState('success');
  } catch (err) {
    showToast(err.message || 'Resume upload failed', 'error');
    setUploadButtonState('error');
  } finally {
    setTimeout(() => setUploadButtonState('default'), 1500);
  }
}

dom.btnUploadResume?.addEventListener('click', uploadResume);

// ── Backend error banner ───────────────────────────────────────
function showBackendError(msg) {
  if (!dom.backendError) return;
  dom.backendError.textContent = msg;
  dom.backendError.classList.add('visible');
}
function hideBackendError() {
  if (dom.backendError) dom.backendError.classList.remove('visible');
}

// ── Core: sendMessage ──────────────────────────────────────────
async function sendMessage(text) {
  if (!text || state.isThinking || state.interviewDone) return;

  stopSpeaking();
  addMessage('user', text);

  dom.textInput.value = '';
  dom.textInput.style.height = 'auto';
  dom.btnSend.disabled = true;
  setInputEnabled(false);
  addMessage('agent', '', true);
  setStatus('thinking', 'Thinking…');
  state.isThinking = true;
  hideBackendError();

  try {
    const response = await fetchWithRetry(`${BACKEND_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, message: text }),
    });

    // Surface non-2xx responses clearly
    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`Server ${response.status}: ${body.slice(0, 120)}`);
    }

    const data = await response.json();
    removeTypingIndicator();
    state.isThinking = false;

    const agentText = data.response || "I'm here. Please go ahead.";
    addMessage('agent', agentText);

    // ── Always read progress from backend (never drift from local counter) ──
    if (data.state_info) {
      const info = data.state_info;
      const qCount = info.main_question_count ?? 0;
      const maxQ   = info.max_questions ?? 6;
      const stage  = info.interview_stage ?? '';

      updateProgress(qCount, maxQ, stage, info.difficulty);

      console.debug(
        `[IPP] stage=${stage} | q=${qCount}/${maxQ} | difficulty=${info.difficulty ?? 'medium'} | ` +
        `classification=${info.classification ?? 'N/A'} | ` +
        `next_node=${info.current_node ?? '???'}`
      );

      if (info.role && !state.interviewStarted && info.role_confirmed) {
        state.interviewStarted = true;
        dom.btnEnd.disabled = false;
        dom.progressSection.classList.add('visible');
      }
    }

    // Interview complete → show feedback
    if (data.done && data.feedback) {
      state.interviewDone = true;
      renderFeedback(data.feedback);
      setStatus('done', 'Interview Complete');
      setInputEnabled(false);
      dom.btnEnd.disabled = true;
      return;
    }

    speakText(agentText);
    setStatus('idle', 'Ready');
    setInputEnabled(true);

  } catch (err) {
    console.error('sendMessage error:', err);
    removeTypingIndicator();
    state.isThinking = false;

    const isNetworkErr = err instanceof TypeError && err.message.includes('fetch');
    const friendlyMsg = isNetworkErr
      ? "Can't reach the backend — make sure the Flask server is running on port 5000."
      : `Error: ${err.message}`;

    showBackendError(friendlyMsg);
    addMessage('agent', "I'm sorry, something went wrong. Please try again.");
    showToast(friendlyMsg, 'error', 8000);
    setStatus('idle', 'Ready');
    setInputEnabled(true);
  }
}

// ── HTTP with Retry ────────────────────────────────────────────
async function fetchWithRetry(url, options, retries = 2, backoff = 1200) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, options);
      if (res.status === 429 && attempt < retries) {
        await delay(backoff * Math.pow(2, attempt));
        continue;
      }
      return res;
    } catch (err) {
      if (attempt < retries) await delay(backoff * Math.pow(2, attempt));
      else throw err;
    }
  }
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Text-to-Speech ─────────────────────────────────────────────
function speakText(text) {
  if (!state.speechSynth) return;
  const cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1');
  stopSpeaking();
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 0.95; utterance.pitch = 1.0; utterance.volume = 1.0;
  utterance.onstart = () => { state.isSpeaking = true; setStatus('speaking', 'Speaking…'); };
  utterance.onend = () => { state.isSpeaking = false; setStatus('idle', 'Ready'); };
  utterance.onerror = () => { state.isSpeaking = false; setStatus('idle', 'Ready'); };
  state.currentUtterance = utterance;
  state.speechSynth.speak(utterance);
}

function stopSpeaking() {
  if (state.speechSynth && state.speechSynth.speaking) state.speechSynth.cancel();
  state.isSpeaking = false;
}

// ── Speech Recognition (STT) ───────────────────────────────────
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    state.voiceAvailable = false;
    dom.btnMic.disabled = true;
    dom.btnMic.title = 'Voice not supported in this browser. Use Chrome for voice input.';
    // Make text input visually primary when mic is unavailable
    dom.textInput.placeholder = 'Type your answer here… (Shift+Enter for new line)';
    showToast('Voice requires Chrome. Use the text box to answer.', 'warning', 6000);
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    state.isRecording = true;
    dom.btnMic.classList.add('recording');
    dom.btnMic.setAttribute('aria-pressed', 'true');
    dom.btnMic.setAttribute('aria-label', 'Stop recording');
    setStatus('listening', 'Listening…');
    dom.interimText.textContent = '';
  };

  recognition.onresult = (event) => {
    let interim = '', final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    dom.interimText.textContent = interim ? `"${interim}"` : '';
    if (final) {
      dom.interimText.textContent = '';
      dom.textInput.value = (dom.textInput.value + ' ' + final).trim();
      dom.textInput.style.height = 'auto';
      dom.textInput.style.height = Math.min(dom.textInput.scrollHeight, 120) + 'px';
      dom.btnSend.disabled = false;
      sendMessage(dom.textInput.value.trim());
    }
  };

  recognition.onerror = (event) => {
    console.error('STT error:', event.error);
    state.isRecording = false;
    resetMicUI();
    if (event.error === 'not-allowed' || event.error === 'permission-denied') {
      state.voiceAvailable = false;
      dom.btnMic.disabled = true;
      dom.permissionWarn.classList.add('visible');
      // Make text input visually primary
      dom.textInput.placeholder = 'Mic blocked — type your answer here and press Enter.';
      dom.textInput.focus();
    } else if (event.error === 'no-speech') {
      showToast("Didn't catch that — please try again or type your answer.", 'warning');
    } else {
      showToast('Mic error. Please type your answer instead.', 'error');
    }
    setStatus('idle', 'Ready');
  };

  recognition.onend = () => {
    state.isRecording = false;
    resetMicUI();
    dom.interimText.textContent = '';
    if (!state.isThinking) setStatus('idle', 'Ready');
  };

  state.recognition = recognition;
  state.voiceAvailable = true;
}

function resetMicUI() {
  dom.btnMic.classList.remove('recording');
  dom.btnMic.setAttribute('aria-pressed', 'false');
  dom.btnMic.setAttribute('aria-label', 'Start voice recording');
}

dom.btnMic.addEventListener('click', () => {
  if (!state.voiceAvailable || state.isThinking || state.interviewDone) return;
  if (state.isRecording) {
    state.recognition.stop();
  } else {
    stopSpeaking();
    try { state.recognition.start(); }
    catch (e) { console.warn('Recognition already active:', e); }
  }
});

// ── End Interview Button ───────────────────────────────────────
dom.btnEnd.addEventListener('click', async () => {
  if (state.interviewDone) return;
  dom.btnEnd.disabled = true;
  stopSpeaking();
  await sendMessage("I'm done with the interview, please generate my feedback.");
});

// ── Feedback Rendering ─────────────────────────────────────────
function renderFeedback(feedback) {
  state.lastFeedback = feedback;
  dom.feedbackGrid.innerHTML = '';

  // Role title
  if (dom.feedbackTitle) {
    dom.feedbackTitle.textContent =
      feedback.interviewRole ? `${feedback.interviewRole} Interview Feedback` : 'Interview Feedback';
  }

  // Readiness score ring
  if (dom.scoreRing && dom.scoreValue) {
    const score = feedback.readinessScore;
    if (score != null) {
      dom.scoreValue.textContent = score;
      // Color: 1-4 red, 5-6 amber, 7-10 green
      const colorClass = score >= 7 ? 'score-green' : score >= 5 ? 'score-amber' : 'score-red';
      dom.scoreRing.className = `score-ring ${colorClass}`;
      dom.scoreRing.setAttribute('aria-label', `Readiness score: ${score} out of 10`);
      dom.scoreRing.style.display = 'flex';
      if (dom.feedbackSubtitle) {
        const label = score >= 7 ? 'Strong performance!' : score >= 5 ? 'Good progress!' : 'Room to improve.';
        dom.feedbackSubtitle.textContent = `Job readiness: ${score}/10 — ${label}`;
      }
    } else {
      dom.scoreRing.style.display = 'none';
    }
  }

  // Feedback cards
  const cards = [
    { icon: '💡', iconClass: 'blue',   title: 'Overall Impression',  content: feedback.overallImpression },
    { icon: '🗣️', iconClass: 'teal',   title: 'Communication',       content: feedback.communication },
    { icon: '🧠', iconClass: 'purple', title: 'Technical Knowledge', content: feedback.technicalKnowledge },
    { icon: '⭐', iconClass: 'green',  title: 'Strengths',           content: feedback.strengths },
  ];

  cards.forEach(card => {
    if (!card.content) return;
    const el = document.createElement('div');
    el.className = 'feedback-card';
    el.innerHTML = `
      <div class="card-icon-row">
        <div class="card-icon ${card.iconClass}" aria-hidden="true">${card.icon}</div>
        <span class="card-title">${card.title}</span>
      </div>
      <p class="card-text">${escapeHtml(card.content)}</p>
    `;
    dom.feedbackGrid.appendChild(el);
  });

  // Improvement areas
  if (feedback.improvementAreas?.length > 0) {
    const impCard = document.createElement('div');
    impCard.className = 'feedback-card full-width';
    impCard.setAttribute('aria-label', 'Areas for improvement');
    const itemsHtml = feedback.improvementAreas.map(area => `
      <div class="improvement-item">
        <span class="improvement-bullet" aria-hidden="true">▸</span>
        <span>${escapeHtml(area)}</span>
      </div>
    `).join('');
    impCard.innerHTML = `
      <div class="card-icon-row">
        <div class="card-icon amber" aria-hidden="true">📈</div>
        <span class="card-title">Areas for Improvement</span>
      </div>
      <div class="improvement-list" role="list">${itemsHtml}</div>
    `;
    dom.feedbackGrid.appendChild(impCard);
  }

  dom.feedbackPanel.classList.add('visible');
  dom.feedbackPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Copy Feedback ──────────────────────────────────────────────
if (dom.btnCopyFeedback) {
  dom.btnCopyFeedback.addEventListener('click', async () => {
    const fb = state.lastFeedback;
    if (!fb) return;
    const lines = [
      `${fb.interviewRole ? fb.interviewRole + ' ' : ''}Interview Feedback`,
      fb.readinessScore != null ? `Job Readiness Score: ${fb.readinessScore}/10` : '',
      '',
      '== Overall Impression ==',
      fb.overallImpression || '',
      '',
      '== Communication ==',
      fb.communication || '',
      '',
      '== Technical Knowledge ==',
      fb.technicalKnowledge || '',
      '',
      '== Strengths ==',
      fb.strengths || '',
      '',
      '== Areas for Improvement ==',
      ...(fb.improvementAreas || []).map(a => `• ${a}`),
    ];
    const text = lines.filter(l => l !== undefined).join('\n');
    try {
      await navigator.clipboard.writeText(text);
      showToast('Feedback copied to clipboard!', 'success');
    } catch {
      showToast('Could not copy — please select and copy the text manually.', 'warning');
    }
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

// ── Restart ────────────────────────────────────────────────────
dom.btnRestart.addEventListener('click', () => {
  state.interviewDone = false;
  state.interviewStarted = false;
  state.lastFeedback = null;
  initSession();

  Array.from(dom.transcript.children).forEach(child => {
    if (child.id !== 'transcript-empty') child.remove();
  });
  if (dom.transcriptEmpty) dom.transcriptEmpty.style.display = '';

  dom.feedbackPanel.classList.remove('visible');
  dom.feedbackGrid.innerHTML = '';
  if (dom.scoreRing) dom.scoreRing.style.display = 'none';

  updateProgress(0, 6, '');
  dom.progressSection.classList.remove('visible');
  hideBackendError();

  setStatus('idle', 'Ready');
  setInputEnabled(true);
  dom.btnEnd.disabled = true;
  dom.textInput.focus();

  showToast('New session started! Tell me what role you want to practice.', 'success');
});

// ── Keyboard shortcut: Ctrl/Cmd + M to toggle mic ─────────────
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
    e.preventDefault();
    dom.btnMic.click();
  }
});

// ── Health check on load ───────────────────────────────────────
async function checkBackendHealth(retries = 3, delayMs = 1000) {
  const message =
    'Cannot reach the backend. Make sure the Flask server is running on port 5000 (python app.py).';

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        console.log('[IPP] Backend healthy ✓');
        state.backendHealthy = true;
        hideBackendError();
        return true;
      }
      throw new Error('Backend responded with non-OK status');
    } catch (err) {
      state.backendHealthy = false;
      if (attempt === retries) {
        showBackendError(message);
        return false;
      }
      await delay(delayMs);
    }
  }
}

// ── Bootstrap ─────────────────────────────────────────────────
function init() {
  initSession();
  initSpeechRecognition();
  setInputEnabled(true);
  dom.textInput.focus();

  setTimeout(() => {
    addMessage('agent',
      "Hello! I'm your AI Interview Practice Partner. 🎯\n\n" +
      "Tell me what job role you'd like to practice for — for example: " +
      "'Software Engineer', 'Product Manager', 'Data Analyst', or any role you have in mind.\n\n" +
      "You can speak using the mic button 🎤 or type below."
    );
  }, 300);

  checkBackendHealth().then((healthy) => {
    if (!healthy) {
      backendHealthRetryTimer = setInterval(async () => {
        const ok = await checkBackendHealth(0);
        if (ok && backendHealthRetryTimer) {
          clearInterval(backendHealthRetryTimer);
          backendHealthRetryTimer = null;
        }
      }, 3000);
    }
  });
}

document.addEventListener('DOMContentLoaded', init);
