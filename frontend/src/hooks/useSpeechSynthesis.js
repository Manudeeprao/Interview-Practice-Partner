import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * useSpeechSynthesis — speaks interviewer replies out loud.
 *
 * - `enabled`: master TTS toggle (persisted in localStorage).
 * - `speak(text)`: cancels anything in progress, then speaks.
 * - `cancel()`: stops speech immediately — called when the user sends
 *   a message or starts recording so the interviewer never talks over them.
 * - `speaking`: true while an utterance is playing.
 *
 * Markdown markers (**bold**) are stripped before speaking.
 */
const STORAGE_KEY = 'ipp-tts-enabled';

export function useSpeechSynthesis() {
  const [supported] = useState(
    () => typeof window !== 'undefined' && 'speechSynthesis' in window,
  );
  const [enabled, setEnabled] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored === null ? true : stored === '1';
    } catch {
      return true;
    }
  });
  const [speaking, setSpeaking] = useState(false);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');
    } catch {
      /* storage unavailable */
    }
    if (!enabled) {
      window.speechSynthesis?.cancel();
      setSpeaking(false);
    }
  }, [enabled]);

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  const speak = useCallback(
    (text) => {
      if (!supported || !enabledRef.current || !text) return;
      const synth = window.speechSynthesis;
      synth.cancel();
      const clean = text.replace(/\*\*([^*]+)\*\*/g, '$1').replace(/\*/g, '');
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      // Prefer a natural English voice when one is available.
      const voices = synth.getVoices();
      const voice =
        voices.find((v) => v.lang?.startsWith('en') && v.localService) ||
        voices.find((v) => v.lang?.startsWith('en'));
      if (voice) utterance.voice = voice;
      synth.speak(utterance);
    },
    [supported],
  );

  // Some browsers load voices asynchronously; warm the list.
  useEffect(() => {
    if (!supported) return undefined;
    const synth = window.speechSynthesis;
    const load = () => synth.getVoices();
    load();
    synth.addEventListener('voiceschanged', load);
    return () => {
      synth.removeEventListener('voiceschanged', load);
      synth.cancel();
    };
  }, [supported]);

  return { supported, enabled, setEnabled, speaking, speak, cancel };
}
