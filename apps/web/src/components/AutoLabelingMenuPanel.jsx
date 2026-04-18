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

function CompactJobBox({ launchState, job, active, canCancel, onCancel }) {
  if (launchState) {
    return (
      <div className="job-box">
        <div className="job-status job-status-preparing">preparing</div>
        <div>{launchState.message}</div>
        {launchState.detail ? <div className="job-subtle">{launchState.detail}</div> : null}
        <div className="job-progress-header">
          <span>Preparing...</span>
          <span>Request pending</span>
        </div>
        <div className="job-progress-track is-active is-indeterminate" role="progressbar" aria-valuemin="0" aria-valuemax="100">
          <div className="job-progress-fill" style={{ width: "28%" }} />
        </div>
      </div>
    );
  }
  if (!job) {
    return null;
  }
  const percent = clampProgressPercent((job.progress || 0) * 100);
  return (
    <div className="job-box">
      <div className={`job-status job-status-${job.status}`}>{job.status}</div>
      <div>{job.message}</div>
      <div className="job-meta">
        {job.processed_frames} / {job.total_frames}
      </div>
      {job.current_frame != null ? <div className="job-subtle">Current frame: {job.current_frame}</div> : null}
      {job.error ? <div className="job-error">{job.error}</div> : null}
      <div className="job-progress-header">
        <span>{formatProgressPercent(percent, active)}</span>
        <span>{job.processed_frames} frames done</span>
      </div>
      <div
        className={`job-progress-track ${active ? "is-active" : ""}`}
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(percent)}
      >
        <div className="job-progress-fill" style={{ width: `${percent}%` }} />
      </div>
      {active ? (
        <div className="button-row job-controls">
          <button className="danger-button" onClick={onCancel} disabled={!canCancel}>
            Stop
          </button>
        </div>
      ) : null}
    </div>
  );
}

export default function AutoLabelingMenuPanel({
  hasWorkspace,
  modelProfiles,
  yolo2dDetectionModelProfiles,
  selected2dDetectionModelProfile,
  selected2dDetectionModelProfileId,
  onSelect2dDetectionModelProfile,
  yolo2dSegmentationModelProfiles,
  selected2dSegmentationModelProfile,
  selected2dSegmentationModelProfileId,
  onSelect2dSegmentationModelProfile,
  selectedModelProfile,
  selectedModelProfileId,
  onSelectModelProfile,
  scoreThreshold,
  startFrame,
  endFrame,
  setStartFrame,
  setEndFrame,
  onRunAutoLabel,
  onRun2dObjectDetection,
  onRun2dSemanticSegmentation,
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
  capabilityReport,
  expandedSectionId,
  onToggleSection
}) {
  const canAnnotateBoxes = hasWorkspace && (capabilityReport?.can_annotate_3d_box ?? true);
  const canAnnotate2d = hasWorkspace && Boolean(capabilityReport?.camera_available);
  const autoLabelJob = jobStatus?.task_name === "3D Bounding Box" ? jobStatus : null;
  const box2dJob = jobStatus?.task_name === "2D Object Detection" ? jobStatus : null;
  const segmentationJob = jobStatus?.task_name === "2D Semantic Segmentation" ? jobStatus : null;
  const trackingJob = jobStatus?.task_name === "3D Tracking" ? jobStatus : null;
  const autoLabelLaunchState = jobLaunchState?.type === "auto-label" ? jobLaunchState : null;
  const box2dLaunchState = jobLaunchState?.type === "2d-detection" ? jobLaunchState : null;
  const segmentationLaunchState = jobLaunchState?.type === "2d-segmentation" ? jobLaunchState : null;
  const trackingLaunchState = jobLaunchState?.type === "tracking" ? jobLaunchState : null;
  const autoLabelJobIsActive = Boolean(autoLabelJob && ["queued", "running", "paused"].includes(autoLabelJob.status));
  const box2dJobIsActive = Boolean(box2dJob && ["queued", "running", "paused"].includes(box2dJob.status));
  const segmentationJobIsActive = Boolean(segmentationJob && ["queued", "running", "paused"].includes(segmentationJob.status));
  const trackingJobIsActive = Boolean(trackingJob && ["queued", "running"].includes(trackingJob.status));
  const canPause =
    autoLabelJobIsActive &&
    ["queued", "running"].includes(autoLabelJob.status) &&
    !autoLabelJob.pause_requested &&
    !autoLabelJob.cancel_requested;
  const canResume = autoLabelJobIsActive && autoLabelJob.status === "paused";
  const canCancel = autoLabelJobIsActive && !autoLabelJob.cancel_requested;
  const canCancelBox2d = box2dJobIsActive && !box2dJob.cancel_requested;
  const canCancelSegmentation = segmentationJobIsActive && !segmentationJob.cancel_requested;
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

  const detectionSummary = !hasWorkspace
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
  const box2dSummary = !hasWorkspace
    ? "Load a dataset first"
    : box2dLaunchState
      ? "Preparing..."
      : box2dJob
        ? `${box2dJob.status} - ${formatProgressPercent((box2dJob.progress || 0) * 100, box2dJobIsActive)}`
        : selected2dDetectionModelProfile?.name || "No YOLO model";
  const segmentationSummary = !hasWorkspace
    ? "Load a dataset first"
    : segmentationLaunchState
      ? "Preparing..."
      : segmentationJob
        ? `${segmentationJob.status} - ${formatProgressPercent((segmentationJob.progress || 0) * 100, segmentationJobIsActive)}`
        : selected2dSegmentationModelProfile?.name || "No YOLO model";

  return (
    <>
      <SidebarSection
        id="auto-labeling-3d-detection"
        title="3D Object Detection"
        summary={detectionSummary}
        expanded={expandedSectionId === "auto-labeling-3d-detection"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
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
        {!hasWorkspace ? <p className="panel-hint">Auto-labeling is unavailable until a dataset is loaded into the workspace.</p> : null}
        {hasWorkspace && !canAnnotateBoxes ? <p className="panel-hint">3D detection is unavailable until the dataset provides LiDAR frames.</p> : null}
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
        <div className="section-action-row">
          <button
            className="primary-button"
            onClick={onRunAutoLabel}
            disabled={!selectedModelProfile || hasActiveJob || !canAnnotateBoxes}
          >
            Run
          </button>
        </div>
      </SidebarSection>

      <SidebarSection
        id="auto-labeling-tracking"
        title="Tracking"
        summary={trackingSummary}
        expanded={expandedSectionId === "auto-labeling-tracking"}
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
        <div className="section-action-row">
          <button className="primary-button" onClick={onRunTracking} disabled={!trackingReady || hasActiveJob || !canAnnotateBoxes}>
            Run
          </button>
        </div>
      </SidebarSection>

      <SidebarSection
        id="auto-labeling-2d-detection"
        title="2D Object Detection"
        summary={box2dSummary}
        expanded={expandedSectionId === "auto-labeling-2d-detection"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Provider</div>
          <strong>YOLO</strong>
          <div>Status</div>
          <strong>{selected2dDetectionModelProfile ? "Ready" : "No model"}</strong>
          <div>Output</div>
          <strong>2D boxes</strong>
        </div>
        <label className="field-row">
          <span>Weight</span>
          <select
            className="model-select"
            value={selected2dDetectionModelProfileId}
            onChange={(event) => onSelect2dDetectionModelProfile(event.target.value)}
            disabled={!yolo2dDetectionModelProfiles.length || !canAnnotate2d || hasActiveJob}
          >
            {yolo2dDetectionModelProfiles.length ? (
              yolo2dDetectionModelProfiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} / {profile.weight_name}
                </option>
              ))
            ) : (
              <option value="">No YOLO detection model available</option>
            )}
          </select>
        </label>
        <NumberField label="Start" value={startFrame} step={1} onChange={setStartFrame} disabled={!canAnnotate2d || hasActiveJob} />
        <NumberField label="End" value={endFrame} step={1} onChange={setEndFrame} disabled={!canAnnotate2d || hasActiveJob} />
        {!hasWorkspace ? <p className="panel-hint">2D detection is unavailable until a dataset is loaded into the workspace.</p> : null}
        {hasWorkspace && !canAnnotate2d ? <p className="panel-hint">2D detection requires camera frames.</p> : null}
        <CompactJobBox
          launchState={box2dLaunchState}
          job={box2dJob}
          active={box2dJobIsActive}
          canCancel={canCancelBox2d}
          onCancel={onCancelAutoLabel}
        />
        <div className="section-action-row">
          <button className="primary-button" onClick={onRun2dObjectDetection} disabled={!selected2dDetectionModelProfile || hasActiveJob || !canAnnotate2d}>
            Run
          </button>
        </div>
      </SidebarSection>

      <SidebarSection
        id="auto-labeling-2d-segmentation"
        title="2D Semantic Segmentation"
        summary={segmentationSummary}
        expanded={expandedSectionId === "auto-labeling-2d-segmentation"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Provider</div>
          <strong>YOLO</strong>
          <div>Status</div>
          <strong>{selected2dSegmentationModelProfile ? "Ready" : "No model"}</strong>
          <div>Output</div>
          <strong>Camera semantic mask</strong>
        </div>
        <label className="field-row">
          <span>Weight</span>
          <select
            className="model-select"
            value={selected2dSegmentationModelProfileId}
            onChange={(event) => onSelect2dSegmentationModelProfile(event.target.value)}
            disabled={!yolo2dSegmentationModelProfiles.length || !canAnnotate2d || hasActiveJob}
          >
            {yolo2dSegmentationModelProfiles.length ? (
              yolo2dSegmentationModelProfiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} / {profile.weight_name}
                </option>
              ))
            ) : (
              <option value="">No YOLO segmentation model available</option>
            )}
          </select>
        </label>
        <NumberField label="Start" value={startFrame} step={1} onChange={setStartFrame} disabled={!canAnnotate2d || hasActiveJob} />
        <NumberField label="End" value={endFrame} step={1} onChange={setEndFrame} disabled={!canAnnotate2d || hasActiveJob} />
        <p className="panel-hint">Successful runs write a camera semantic layer that can be shown from Show / Layers / Camera Semantic.</p>
        <CompactJobBox
          launchState={segmentationLaunchState}
          job={segmentationJob}
          active={segmentationJobIsActive}
          canCancel={canCancelSegmentation}
          onCancel={onCancelAutoLabel}
        />
        <div className="section-action-row">
          <button className="primary-button" onClick={onRun2dSemanticSegmentation} disabled={!selected2dSegmentationModelProfile || hasActiveJob || !canAnnotate2d}>
            Run
          </button>
        </div>
      </SidebarSection>
    </>
  );
}
