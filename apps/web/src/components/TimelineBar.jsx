export default function TimelineBar({ frameIndex, totalFrames, isPlaying, onTogglePlay, onFrameChange, fps }) {
  return (
    <div className="timeline-shell">
      <button className="primary-button" onClick={onTogglePlay}>
        {isPlaying ? "Pause" : "Play"}
      </button>
      <input
        className="timeline-slider"
        type="range"
        min="0"
        max={Math.max(0, totalFrames - 1)}
        value={frameIndex}
        onChange={(event) => onFrameChange(Number(event.target.value))}
      />
      <div className="timeline-meta">
        <strong>
          Frame {frameIndex} / {Math.max(0, totalFrames - 1)}
        </strong>
        <span>{fps.toFixed(2)} Hz</span>
      </div>
    </div>
  );
}
