import { getAnnotationColor, hexToRgba } from "../lib/annotationColors";
import NumberField from "./NumberField";
import OptionalNumberField from "./OptionalNumberField";
import SidebarSection from "./SidebarSection";

const TRACKING_STAGE_LABELS = {
  preparing_inputs: "Preparing inputs",
  associating_tracks: "Associating tracks",
  writing_results: "Writing results"
};

function clampProgressPercent(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return 0;
  }
  return Math.max(0, Math.min(100, numericValue));
}

function formatProgressPercent(value, precise = false) {
  const clamped = clampProgressPercent(value);
  return precise ? `${clamped.toFixed(1)}%` : `${Math.round(clamped)}%`;
}

export default function AnnotateMenuPanel({
  hasWorkspace,
  modelProfiles,
  classNames,
  sequence,
  selectedModelProfile,
  selectedModelProfileId,
  onSelectModelProfile,
  scoreThreshold,
  startFrame,
  endFrame,
  setStartFrame,
  setEndFrame,
  onRunAutoLabel,
  onRunTracking,
  trackingMethod,
  onTrackingMethodChange,
  trackingInputScoreThreshold,
  onTrackingInputScoreThresholdChange,
  trackingMaxGapFrames,
  onTrackingMaxGapFramesChange,
  trackingMaxTrackLostFrames,
  onTrackingMaxTrackLostFramesChange,
  trackingMatchDistanceXY,
  onTrackingMatchDistanceXYChange,
  trackingMatchZDelta,
  onTrackingMatchZDeltaChange,
  onPauseAutoLabel,
  onResumeAutoLabel,
  onCancelAutoLabel,
  jobStatus,
  jobLaunchState,
  hasActiveJob,
  trackingReadiness,
  annotations,
  totalAnnotationCount,
  hiddenAnnotationCount,
  selectedAnnotationId,
  onSelectAnnotation,
  onUpdateAnnotation,
  onCreateAnnotation,
  onDeleteAnnotation,
  onSaveSequence,
  selectedAnnotation,
  capabilityReport,
  transformMode,
  onChangeTransformMode,
  expandedSectionId,
  onToggleSection
}) {
  const classOptions = classNames?.length
    ? classNames
    : sequence?.class_names?.length
      ? sequence.class_names
      : ["Car", "Pedestrian", "Cyclist"];
  const datasetReady = hasWorkspace;
  const canAnnotateBoxes = datasetReady && (capabilityReport?.can_annotate_3d_box ?? true);
  const autoLabelJob = jobStatus?.task_name === "3D Tracking" ? null : jobStatus;
  const trackingJob = jobStatus?.task_name === "3D Tracking" ? jobStatus : null;
  const autoLabelLaunchState = jobLaunchState?.type === "auto-label" ? jobLaunchState : null;
  const trackingLaunchState = jobLaunchState?.type === "tracking" ? jobLaunchState : null;
  const autoLabelJobIsActive = Boolean(autoLabelJob && ["queued", "running", "paused"].includes(autoLabelJob.status));
  const trackingJobIsActive = Boolean(trackingJob && ["queued", "running"].includes(trackingJob.status));
  const canPause =
    autoLabelJobIsActive &&
    ["queued", "running"].includes(autoLabelJob.status) &&
    !autoLabelJob.pause_requested &&
    !autoLabelJob.cancel_requested;
  const canResume = autoLabelJobIsActive && autoLabelJob.status === "paused";
  const canCancel = autoLabelJobIsActive && !autoLabelJob.cancel_requested;
  const canCancelTracking = trackingJobIsActive && !trackingJob.cancel_requested;
  const rawJobPercent = autoLabelJob ? clampProgressPercent((autoLabelJob.progress || 0) * 100) : 0;
  const isWarmupPhase =
    autoLabelJob &&
    autoLabelJob.status === "running" &&
    autoLabelJob.current_frame != null &&
    autoLabelJob.processed_frames === 0 &&
    (autoLabelJob.progress || 0) === 0;
  const displayJobPercent = isWarmupPhase ? 4 : rawJobPercent;
  const displayJobPercentLabel = formatProgressPercent(displayJobPercent, autoLabelJobIsActive);
  const trackingReady = Boolean(trackingReadiness?.ready);
  const trackingProgressPercent = trackingJob
    ? clampProgressPercent((trackingJob.progress || 0) * 100)
    : 0;
  const trackingProgressPercentLabel = formatProgressPercent(trackingProgressPercent, trackingJobIsActive);
  const trackingStageLabel = trackingJob?.stage ? TRACKING_STAGE_LABELS[trackingJob.stage] : null;
  const trackingStageProgressLabel =
    trackingJob?.stage === "preparing_inputs"
      ? "frames prepared"
      : trackingJob?.stage === "associating_tracks"
        ? "frames associated"
        : trackingJob?.stage === "writing_results"
          ? "frames written"
          : "frames handled";

  const autoLabelSummary = !hasWorkspace
    ? "Load a dataset first"
    : autoLabelLaunchState
      ? "Preparing..."
    : autoLabelJob
      ? `${autoLabelJob.status} - ${displayJobPercentLabel}`
      : `${selectedModelProfile?.name || "No model"} - ${startFrame}-${endFrame}`;
  const trackingSummary = !hasWorkspace
    ? "Load a dataset first"
    : trackingLaunchState
      ? "Preparing..."
    : trackingJob
      ? `${trackingJob.status} - ${trackingProgressPercentLabel}`
      : trackingReady
        ? `${trackingReadiness.annotated_frames}/${trackingReadiness.total_frames} ready`
        : trackingReadiness?.next_missing_frame != null
          ? `Locked - next ${trackingReadiness.next_missing_frame}`
          : "Locked";
  const manualLabelSummary = canAnnotateBoxes ? `Mode ${transformMode}` : "Locked";
  const annotationSummary = hasWorkspace
    ? `${annotations.length}/${totalAnnotationCount} visible`
    : "No dataset loaded";

  return (
    <>
      <SidebarSection
        id="annotate-autolabel"
        title="Auto Label"
        summary={autoLabelSummary}
        expanded={expandedSectionId === "annotate-autolabel"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Task</div>
          <strong>3D Object Detection</strong>
          <div>Framework</div>
          <strong>{selectedModelProfile?.framework || "-"}</strong>
          <div>Network</div>
          <strong>{selectedModelProfile?.architecture || "-"}</strong>
          <div>Score Min</div>
          <strong>{scoreThreshold.toFixed(2)}</strong>
        </div>
        <label className="field-row">
          <span>Weight</span>
          <select
            className="model-select"
            value={selectedModelProfileId}
            onChange={(event) => onSelectModelProfile(event.target.value)}
            disabled={!modelProfiles.length || !canAnnotateBoxes}
          >
            {modelProfiles.length ? (
              modelProfiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} / {profile.weight_name}
                </option>
              ))
            ) : (
              <option value="">No models available</option>
            )}
          </select>
        </label>
        <NumberField label="Start" value={startFrame} step={1} onChange={setStartFrame} disabled={!canAnnotateBoxes} />
        <NumberField label="End" value={endFrame} step={1} onChange={setEndFrame} disabled={!canAnnotateBoxes} />
        <button
          className="primary-button"
          onClick={onRunAutoLabel}
          disabled={!selectedModelProfile || hasActiveJob || !canAnnotateBoxes}
        >
          {autoLabelLaunchState ? "Preparing..." : hasActiveJob ? "Job Running" : "Run Auto Label"}
        </button>
        {!hasWorkspace ? <p className="panel-hint">Auto-label is unavailable until a dataset is loaded into the workspace.</p> : null}
        {hasWorkspace && !canAnnotateBoxes ? <p className="panel-hint">Auto-label is unavailable until the dataset provides LiDAR frames.</p> : null}
        <p className="panel-hint">Tracking unlocks only after this whole frame range has been processed by frame-wise detection, including frames with zero boxes.</p>
        {autoLabelLaunchState ? (
          <div className="job-box">
            <div className="job-status job-status-preparing">preparing</div>
            <div>{autoLabelLaunchState.message}</div>
            {autoLabelLaunchState.detail ? <div className="job-subtle">{autoLabelLaunchState.detail}</div> : null}
            <div className="job-progress-header">
              <span>Preparing...</span>
              <span>Request pending</span>
            </div>
            <div className="job-progress-track is-active is-indeterminate" role="progressbar" aria-valuemin="0" aria-valuemax="100">
              <div className="job-progress-fill" style={{ width: "28%" }} />
            </div>
          </div>
        ) : null}
        {!autoLabelLaunchState && autoLabelJob ? (
          <div className="job-box">
            <div className={`job-status job-status-${autoLabelJob.status}`}>{autoLabelJob.status}</div>
            <div>{autoLabelJob.message}</div>
            <div className="job-meta">
              {autoLabelJob.processed_frames} / {autoLabelJob.total_frames}
            </div>
            {autoLabelJob.skipped_frames > 0 ? <div className="job-subtle">Reused {autoLabelJob.skipped_frames} existing processed frames.</div> : null}
            {autoLabelJob.current_frame != null ? <div className="job-subtle">Current frame: {autoLabelJob.current_frame}</div> : null}
            {autoLabelJob.pause_requested && autoLabelJob.status !== "paused" ? <div className="job-subtle">Pause will apply after the current frame finishes.</div> : null}
            {autoLabelJob.cancel_requested && autoLabelJob.status !== "cancelled" ? <div className="job-subtle">Stop will apply after the current frame finishes.</div> : null}
            <div className="job-progress-header">
              <span>{displayJobPercentLabel}</span>
              <span>{Math.max(0, autoLabelJob.processed_frames - (autoLabelJob.skipped_frames || 0))} new frames done</span>
            </div>
            <div
              className={`job-progress-track ${autoLabelJobIsActive ? "is-active" : ""} ${isWarmupPhase ? "is-warmup" : ""}`}
              role="progressbar"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(displayJobPercent)}
            >
              <div className="job-progress-fill" style={{ width: `${displayJobPercent}%` }} />
            </div>
            {isWarmupPhase ? <div className="job-subtle">The first frame includes model warm-up, so progress may sit at 0% briefly.</div> : null}
            {autoLabelJob.error ? <div className="job-error">{autoLabelJob.error}</div> : null}
            {autoLabelJobIsActive ? (
              <div className="button-row job-controls">
                <button onClick={onPauseAutoLabel} disabled={!canPause}>
                  Pause
                </button>
                <button onClick={onResumeAutoLabel} disabled={!canResume}>
                  Resume
                </button>
                <button className="danger-button" onClick={onCancelAutoLabel} disabled={!canCancel}>
                  Stop
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </SidebarSection>

      <SidebarSection
        id="annotate-tracking"
        title="Tracking"
        summary={trackingSummary}
        expanded={expandedSectionId === "annotate-tracking"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Range Ready</div>
          <strong>{trackingReadiness?.total_frames ? `${trackingReadiness.annotated_frames} / ${trackingReadiness.total_frames}` : "-"}</strong>
          <div>Next Missing</div>
          <strong>{trackingReadiness?.next_missing_frame ?? "-"}</strong>
          <div>Status</div>
          <strong>{trackingReady ? "Unlocked" : "Locked"}</strong>
          <div>Stage</div>
          <strong>{trackingStageLabel || "-"}</strong>
        </div>
        <label className="field-row">
          <span>Method</span>
          <select className="model-select" value={trackingMethod} onChange={(event) => onTrackingMethodChange(event.target.value)} disabled={!canAnnotateBoxes || hasActiveJob}>
            <option value="greedy_cv">Greedy CV</option>
          </select>
        </label>
        <OptionalNumberField
          label="Score Gate"
          value={trackingInputScoreThreshold}
          step={0.01}
          min={0}
          placeholder="Off"
          disabled={!canAnnotateBoxes || hasActiveJob}
          onChange={onTrackingInputScoreThresholdChange}
        />
        <NumberField label="Max Gap" value={trackingMaxGapFrames} step={1} disabled={!canAnnotateBoxes || hasActiveJob} onChange={onTrackingMaxGapFramesChange} />
        <NumberField label="Lost Frames" value={trackingMaxTrackLostFrames} step={1} disabled={!canAnnotateBoxes || hasActiveJob} onChange={onTrackingMaxTrackLostFramesChange} />
        <OptionalNumberField
          label="Match XY"
          value={trackingMatchDistanceXY}
          step={0.1}
          min={0}
          placeholder="Per class"
          disabled={!canAnnotateBoxes || hasActiveJob}
          onChange={onTrackingMatchDistanceXYChange}
        />
        <OptionalNumberField
          label="Match Z"
          value={trackingMatchZDelta}
          step={0.1}
          min={0}
          placeholder="Per class"
          disabled={!canAnnotateBoxes || hasActiveJob}
          onChange={onTrackingMatchZDeltaChange}
        />
        <button className="primary-button" onClick={onRunTracking} disabled={!trackingReady || hasActiveJob || !canAnnotateBoxes}>
          {trackingLaunchState ? "Preparing..." : hasActiveJob ? "Job Running" : "Run Tracking"}
        </button>
        <p className="panel-hint">{trackingReadiness?.message || "Tracking is unavailable until frame-wise detection finishes for the selected range."}</p>
        <p className="panel-hint">Blank score or match thresholds fall back to no filtering or the built-in per-class gates.</p>
        {!trackingReady && trackingReadiness?.next_missing_frame != null ? (
          <p className="panel-hint">Finish detection for frame {trackingReadiness.next_missing_frame} and the remaining missing frames in this range first.</p>
        ) : null}
        {trackingLaunchState ? (
          <div className="job-box">
            <div className="job-status job-status-preparing">preparing</div>
            <div>{trackingLaunchState.message}</div>
            {trackingLaunchState.detail ? <div className="job-subtle">{trackingLaunchState.detail}</div> : null}
            <div className="job-progress-header">
              <span>Preparing...</span>
              <span>Request pending</span>
            </div>
            <div className="job-progress-track is-active is-indeterminate" role="progressbar" aria-valuemin="0" aria-valuemax="100">
              <div className="job-progress-fill" style={{ width: "28%" }} />
            </div>
          </div>
        ) : null}
        {!trackingLaunchState && trackingJob ? (
          <div className="job-box">
            <div className={`job-status job-status-${trackingJob.status}`}>{trackingJob.status}</div>
            <div>{trackingJob.message}</div>
            <div className="job-meta">
              {trackingJob.processed_frames} / {trackingJob.total_frames}
            </div>
            {trackingJob.stage ? <div className="job-subtle">Stage: {TRACKING_STAGE_LABELS[trackingJob.stage]}</div> : null}
            {trackingJob.current_frame != null ? <div className="job-subtle">Current frame: {trackingJob.current_frame}</div> : null}
            {trackingJob.cancel_requested && trackingJob.status !== "cancelled" ? <div className="job-subtle">Stop will apply after the current tracking stage finishes.</div> : null}
            <div className="job-progress-header">
              <span>{trackingProgressPercentLabel}</span>
              <span>{trackingJob.processed_frames} {trackingStageProgressLabel}</span>
            </div>
            <div
              className={`job-progress-track ${trackingJobIsActive ? "is-active" : ""}`}
              role="progressbar"
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(trackingProgressPercent)}
            >
              <div className="job-progress-fill" style={{ width: `${trackingProgressPercent}%` }} />
            </div>
            {trackingJob.error ? <div className="job-error">{trackingJob.error}</div> : null}
            {trackingJobIsActive ? (
              <div className="button-row job-controls">
                <button className="danger-button" onClick={onCancelAutoLabel} disabled={!canCancelTracking}>
                  Stop
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </SidebarSection>

      <SidebarSection
        id="annotate-transform"
        title="Manual Label"
        summary={manualLabelSummary}
        expanded={expandedSectionId === "annotate-transform"}
        onToggle={onToggleSection}
      >
        <div className="button-row">
          {["translate", "rotate", "scale"].map((mode) => (
            <button
              key={mode}
              className={transformMode === mode ? "active-toggle" : ""}
              disabled={!canAnnotateBoxes}
              onClick={() => onChangeTransformMode(mode)}
            >
              {mode}
            </button>
          ))}
        </div>
        <p className="panel-hint">Visualize mode will pause auto-label and save the current annotation state before leaving this menu.</p>
      </SidebarSection>

      <SidebarSection
        id="annotate-annotations"
        title="Annotations"
        summary={annotationSummary}
        expanded={expandedSectionId === "annotate-annotations"}
        onToggle={onToggleSection}
      >
        <div className="section-head">
          <div className="button-row">
            <button onClick={onCreateAnnotation} disabled={!canAnnotateBoxes}>
              Add
            </button>
            <button onClick={onDeleteAnnotation} disabled={!selectedAnnotationId}>
              Delete
            </button>
            <button onClick={onSaveSequence} disabled={!canAnnotateBoxes}>
              Save
            </button>
          </div>
        </div>
        <p className="annotation-summary">
          Showing {annotations.length} / {totalAnnotationCount}
          {hiddenAnnotationCount > 0 ? ` (${hiddenAnnotationCount} hidden below threshold)` : ""}
        </p>
        {!hasWorkspace ? <p className="panel-hint">No sequence is loaded yet, so there are no annotations to review.</p> : null}
        <div className="annotation-list">
          {annotations.map((annotation) => {
            const annotationColor = getAnnotationColor(annotation.class_name, classOptions);
            const isActive = annotation.id === selectedAnnotationId;
            return (
              <button
                key={annotation.id}
                className={`annotation-chip ${isActive ? "active" : ""}`}
                style={{
                  borderColor: hexToRgba(annotationColor, isActive ? 0.82 : 0.42),
                  background: hexToRgba(annotationColor, isActive ? 0.22 : 0.08)
                }}
                onClick={() => onSelectAnnotation(annotation.id)}
              >
                <span className="annotation-chip-label">
                  <span className="annotation-color-dot" style={{ backgroundColor: annotationColor }} />
                  <span>{annotation.class_name}</span>
                </span>
                <span className="annotation-chip-meta">
                  {annotation.track_id ? `${annotation.track_id} | ` : ""}
                  {annotation.source_stage === "interpolated" ? "interp | " : ""}
                  {annotation.score.toFixed(2)}
                </span>
              </button>
            );
          })}
        </div>
        {selectedAnnotation ? (
          <div className="editor-form">
            <label className="field-row">
              <span>Class</span>
              <select className="model-select" value={selectedAnnotation.class_name} onChange={(event) => onUpdateAnnotation(selectedAnnotation.id, { class_name: event.target.value })}>
                {classOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <NumberField label="Score" value={selectedAnnotation.score} step={0.01} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { score: value })} />
            <NumberField label="Center X" value={selectedAnnotation.center.x} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { center: { ...selectedAnnotation.center, x: value } })} />
            <NumberField label="Center Y" value={selectedAnnotation.center.y} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { center: { ...selectedAnnotation.center, y: value } })} />
            <NumberField label="Center Z" value={selectedAnnotation.center.z} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { center: { ...selectedAnnotation.center, z: value } })} />
            <NumberField label="Size X" value={selectedAnnotation.size.x} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { size: { ...selectedAnnotation.size, x: value } })} />
            <NumberField label="Size Y" value={selectedAnnotation.size.y} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { size: { ...selectedAnnotation.size, y: value } })} />
            <NumberField label="Size Z" value={selectedAnnotation.size.z} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { size: { ...selectedAnnotation.size, z: value } })} />
            <NumberField label="Yaw" value={selectedAnnotation.yaw} disabled={!canAnnotateBoxes} onChange={(value) => onUpdateAnnotation(selectedAnnotation.id, { yaw: value })} />
            <div className="field-row">
              <span>Track</span>
              <strong>{selectedAnnotation.track_id || "-"}</strong>
            </div>
            <div className="field-row">
              <span>Stage</span>
              <strong>{selectedAnnotation.source_stage || "-"}</strong>
            </div>
          </div>
        ) : null}
      </SidebarSection>
    </>
  );
}
