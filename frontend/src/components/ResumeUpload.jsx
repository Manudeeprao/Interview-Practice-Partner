import { useRef, useState } from 'react';
import { deleteResume, uploadResume } from '../api/client';

const MAX_BYTES = 5 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ['.pdf', '.docx'];

function validateFile(file) {
  const name = file.name || '';
  const ext = name.slice(name.lastIndexOf('.')).toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return 'Only PDF and DOCX resumes are supported.';
  }
  if (file.size > MAX_BYTES) {
    return 'Resume is too large — please keep it under 5 MB.';
  }
  return null;
}

/**
 * ResumeUpload — drag-and-drop + file picker resume upload.
 * Shows upload progress, the uploaded filename + chunk count,
 * and a remove button (DELETE /resume/<session_id>).
 */
export function ResumeUpload({ sessionId, resume, setResume, notify }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const dragCounter = useRef(0);

  const startUpload = async (file) => {
    if (!file || uploading || !sessionId) return;
    const error = validateFile(file);
    if (error) {
      notify(error, 'error');
      return;
    }
    setUploading(true);
    setProgress(0);
    try {
      const data = await uploadResume(sessionId, file, setProgress);
      setResume({
        uploaded: true,
        filename: data.filename || file.name,
        chunks: data.chunks ?? 0,
      });
      notify(data.message || 'Resume uploaded successfully.', 'success');
    } catch (err) {
      notify(err.message || 'Resume upload failed.', 'error');
    } finally {
      setUploading(false);
      setProgress(0);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleRemove = async () => {
    if (!sessionId) return;
    try {
      await deleteResume(sessionId);
      setResume({ uploaded: false, filename: null, chunks: 0 });
      notify('Resume removed.', 'success');
    } catch (err) {
      notify(err.message || 'Could not remove the resume.', 'error');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) startUpload(file);
  };

  return (
    <div
      className={`resume-upload${dragOver ? ' drag-over' : ''}`}
      onDragEnter={(e) => {
        e.preventDefault();
        dragCounter.current += 1;
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        dragCounter.current = Math.max(0, dragCounter.current - 1);
        if (dragCounter.current === 0) setDragOver(false);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      aria-label="Resume upload area"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx"
        className="sr-only"
        aria-label="Choose a resume file (PDF or DOCX, max 5 MB)"
        onChange={(e) => startUpload(e.target.files?.[0])}
      />

      {resume.uploaded ? (
        <>
          <span className="resume-file-name" title={resume.filename || 'Resume'}>
            📄 {resume.filename || 'Resume'}
          </span>
          {resume.chunks > 0 && (
            <span className="resume-chunks">{resume.chunks} chunks indexed</span>
          )}
          <button
            type="button"
            className="resume-remove-btn"
            onClick={handleRemove}
            disabled={uploading}
            aria-label="Remove uploaded resume"
          >
            Remove
          </button>
        </>
      ) : (
        <>
          <span className="resume-upload-idle">
            {dragOver ? 'Drop your resume here' : '📄 Drop resume or'}
          </span>
          <button
            type="button"
            className="resume-upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || !sessionId}
          >
            {uploading ? `Uploading… ${progress}%` : 'Choose file'}
          </button>
        </>
      )}

      {uploading && (
        <div
          className="resume-progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
          aria-label="Resume upload progress"
        >
          <div className="resume-progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}
