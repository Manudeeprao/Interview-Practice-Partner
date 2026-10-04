import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * useSpeechRecognition — Web Speech API wrapper.
 *
 * - `supported`: false in browsers without the API (mic must then be
 *   disabled with a note that Chrome is required).
 * - `listening` / `interim`: live recording state + interim transcript,
 *   so the interim text can be shown inside the input box.
 * - `onFinalRef`: callback invoked with each final transcript chunk.
 */
export function useSpeechRecognition({ onFinal } = {}) {
  const [supported] = useState(
    () =>
      typeof window !== 'undefined' &&
      (window.SpeechRecognition || window.webkitSpeechRecognition),
  );
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState('');
  const [permissionError, setPermissionError] = useState(false);
  const recognitionRef = useRef(null);
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  const stop = useCallback(() => {
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
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

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
      if (finalText && onFinalRef.current) onFinalRef.current(finalText.trim());
    };

    recognition.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'permission-denied') {
        setPermissionError(true);
      }
    };

    recognition.onend = () => {
      setListening(false);
      setInterim('');
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
    }
  }, [supported]);

  // Clean up the recogniser if the component unmounts mid-recording.
  useEffect(() => () => stop(), [stop]);

  return { supported, listening, interim, permissionError, start, stop };
}
