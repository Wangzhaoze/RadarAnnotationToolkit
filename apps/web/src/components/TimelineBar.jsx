export default function TimelineBar({ frameIndex, totalFrames, isPlaying, onTogglePlay, onFrameChange }) {
  const frameCount = Math.max(0, totalFrames || 0);
  const maxFrameIndex = Math.max(0, frameCount - 1);
  const displayFrame = frameCount > 0 ? Math.min(frameIndex + 1, frameCount) : 0;

  return (
    <div className="timeline-shell">
      <button
        className="timeline-play-button"
        onClick={onTogglePlay}
        title={isPlaying ? "Pause" : "Play"}
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? <span className="timeline-pause-icon" aria-hidden="true" /> : <span className="timeline-play-icon" aria-hidden="true" />}
      </button>
      <input
        className="timeline-slider"
        type="range"
        min="0"
        max={maxFrameIndex}
        value={frameIndex}
        onChange={(event) => onFrameChange(Number(event.target.value))}
      />
      <div className="timeline-meta" aria-label={`Frame ${displayFrame} of ${frameCount}`}>
        <strong>{displayFrame} / {frameCount}</strong>
      </div>
    </div>
  );
}
