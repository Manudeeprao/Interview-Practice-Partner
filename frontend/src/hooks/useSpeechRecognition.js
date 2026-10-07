import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * useSpeechRecognition — Web Speech API wrapper.
 *
 * - `supported`: false in browsers without the API (mic must then be
 *   disabled with a note that Chrome is required).
 * - `listening` / `interim`: live recording state + interim transcript,
 *   so the interim text can be shown inside the input box.
 * - `micError`: human-readable error message when recognition fails
 *   (no-speech, audio-capture, network…). Cleared on the next start().
 * - `onFinal`: called with the FULL accumulated transcript every time a new
 *   final chunk arrives. The caller decides what to do with it (we append
 *   it to the answer box — we never auto-send fragments).
 * - `onError`: called with the raw SpeechRecognition error code.
 *
 * Uses continuous mode so long interview answers are not cut off at the
 * first pause, and auto-restarts when Chrome ends the session on its own
 * (it stops after ~60s or long silence) while the user still wants to talk.
 */
export function useSpeechRecognition({ onFinal, onError } = {}) {
  const [supported] = useState(
    () =>
      typeof window !== 'undefined' &&
      (window.SpeechRecognition || window.webkitSpeechRecognition),
  );
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState('');
  const [permissionError, setPermissionError] = useState(false);
  const [micError, setMicError] = useState(null);
  const recognitionRef = useRef(null);
  const wantListeningRef = useRef(false);
  const restartCountRef = useRef(0);
  const finalTextRef = useRef('');
  const onFinalRef = useRef(onFinal);
  const onErrorRef = useRef(onError);
  onFinalRef.current = onFinal;
  onErrorRef.current = onError;

  const stop = useCallback(() => {
    wantListeningRef.current = false;
    try {
      recognitionRef.current?.stop();
    } catch {
      /* already stopped */
    }
  }, []);

  const start = useCallback(() => {
    if (!supported || recognitionRef.current) return;
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new Ctor();
    // Continuous: keep listening across pauses so long answers survive.
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    wantListeningRef.current = true;
    restartCountRef.current = 0;
    finalTextRef.current = '';
    setMicError(null);
    setInterim('');

    recognition.onstart = () => {
      setListening(true);
      setInterim('');
    };

    recognition.onresult = (event) => {
      let interimText = '';
      let finalText = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += transcript;
        else interimText += transcript;
      }
      setInterim(interimText);
      if (finalText.trim()) {
        finalTextRef.current +=
          (finalTextRef.current ? ' ' : '') + finalText.trim();
        if (onFinalRef.current) onFinalRef.current(finalTextRef.current);
      }
    };

    recognition.onerror = (event) => {
      const code = event.error;
      if (code === 'not-allowed' || code === 'permission-denied') {
        setPermissionError(true);
        wantListeningRef.current = false;
        setMicError('Microphone access was blocked. Allow mic access and try again.');
      } else if (code === 'audio-capture') {
        wantListeningRef.current = false;
        setMicError('No microphone found. Check your mic and try again.');
      } else if (code === 'network') {
        wantListeningRef.current = false;
        setMicError('Voice recognition needs an internet connection.');
      } else if (code === 'no-speech') {
        // Common on silence — keep listening, just nudge the user.
        setMicError('No speech detected — keep talking.');
      } else if (code !== 'aborted') {
        setMicError('Voice input stopped unexpectedly. Try again.');
      }
      if (onErrorRef.current) onErrorRef.current(code);
    };

    recognition.onend = () => {
      // Chrome ends the session on its own after ~60s or long silence.
      // Resume transparently while the user still wants to record.
      if (wantListeningRef.current && restartCountRef.current < 5) {
        restartCountRef.current += 1;
        try {
          recognition.start();
          return;
        } catch {
          /* fall through to full stop */
        }
      }
      wantListeningRef.current = false;
      restartCountRef.current = 0;
      setListening(false);
      setInterim('');
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
      wantListeningRef.current = false;
    }
  }, [supported]);

  const clearError = useCallback(() => setMicError(null), []);

  // Clean up the recogniser if the component unmounts mid-recording.
  useEffect(() => () => stop(), [stop]);

  return {
    supported,
    listening,
    interim,
    permissionError,
    micError,
    clearError,
    start,
    stop,
  };
}
