import { useEffect, useMemo, useState } from "react";
import NumberField from "./NumberField";
import OptionalNumberField from "./OptionalNumberField";
import ParameterLabel from "./ParameterLabel";
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

function uniqueValues(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

function profileNetworkLabel(profile) {
  return profile?.network_label || profile?.architecture || profile?.name || "";
}

function profileConfigLabel(profile) {
  return profile?.config_label || profile?.cfg_file?.split(/[\\/]/).pop()?.replace(/\.ya?ml$/i, "") || "default";
}

function frameworkSortKey(framework) {
  if (framework === "OpenPCDet") {
    return "0";
  }
  if (framework === "WildDet3D") {
    return "1";
  }
  return `2-${framework}`;
}

function formatAvailabilityLabel(label, isAvailable) {
  return isAvailable ? label : `${label} (Unavailable)`;
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
  modelCatalogStatus = "idle",
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
  trackingInputScoreThreshold,
  onTrackingInputScoreThresholdChange,
  trackingMaxMissedFrames,
  onTrackingMaxMissedFramesChange,
  trackingAssociationDistanceM,
  onTrackingAssociationDistanceMChange,
  trackingProcessNoisePos,
  onTrackingProcessNoisePosChange,
  trackingProcessNoiseSize,
  onTrackingProcessNoiseSizeChange,
  trackingProcessNoiseVel,
  onTrackingProcessNoiseVelChange,
  trackingMeasurementNoisePos,
  onTrackingMeasurementNoisePosChange,
  trackingMeasurementNoiseSize,
  onTrackingMeasurementNoiseSizeChange,
  trackingYawSmoothingAlpha,
  onTrackingYawSmoothingAlphaChange,
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
  const [selected3dFramework, setSelected3dFramework] = useState("");
  const [selected3dNetwork, setSelected3dNetwork] = useState("");
  const [selected3dConfig, setSelected3dConfig] = useState("");

  const frameworkOptions = useMemo(
    () =>
      uniqueValues(modelProfiles.map((profile) => profile.framework || profile.provider)).sort((a, b) =>
        frameworkSortKey(a).localeCompare(frameworkSortKey(b))
      ),
    [modelProfiles]
  );
  const frameworkProfiles = useMemo(
    () => modelProfiles.filter((profile) => selected3dFramework && (profile.framework || profile.provider) === selected3dFramework),
    [modelProfiles, selected3dFramework]
  );
  const networkOptions = useMemo(
    () => uniqueValues(frameworkProfiles.map(profileNetworkLabel)),
    [frameworkProfiles]
  );
  const networkProfiles = useMemo(
    () => frameworkProfiles.filter((profile) => profileNetworkLabel(profile) === selected3dNetwork),
    [frameworkProfiles, selected3dNetwork]
  );
  const configOptions = useMemo(
    () => uniqueValues(networkProfiles.map(profileConfigLabel)),
    [networkProfiles]
  );

  useEffect(() => {
    if (!selectedModelProfile) {
      return;
    }
    setSelected3dFramework(selectedModelProfile.framework || selectedModelProfile.provider || "");
    setSelected3dNetwork(profileNetworkLabel(selectedModelProfile));
    setSelected3dConfig(profileConfigLabel(selectedModelProfile));
  }, [selectedModelProfile]);

  useEffect(() => {
    if (!selected3dFramework) {
      setSelected3dNetwork("");
      setSelected3dConfig("");
      return;
    }
    if (!frameworkOptions.includes(selected3dFramework)) {
      setSelected3dFramework("");
      setSelected3dNetwork("");
      setSelected3dConfig("");
      onSelectModelProfile("");
    }
  }, [frameworkOptions, onSelectModelProfile, selected3dFramework]);

  function resolve3dProfile(framework, network, config) {
    return (
      modelProfiles.find(
        (profile) =>
          (profile.framework || profile.provider) === framework &&
          profileNetworkLabel(profile) === network &&
          profileConfigLabel(profile) === config
      ) || null
    );
  }

  function handleFrameworkChange(value) {
    setSelected3dFramework(value);
    setSelected3dNetwork("");
    setSelected3dConfig("");
    onSelectModelProfile("");
    if (!value) {
      return;
    }
    const nextFrameworkProfiles = modelProfiles.filter((profile) => (profile.framework || profile.provider) === value);
    const nextNetworkOptions = uniqueValues(nextFrameworkProfiles.map(profileNetworkLabel));
    if (nextNetworkOptions.length !== 1) {
      return;
    }
    const nextNetwork = nextNetworkOptions[0];
    const nextNetworkProfiles = nextFrameworkProfiles.filter((profile) => profileNetworkLabel(profile) === nextNetwork);
    const nextConfigOptions = uniqueValues(nextNetworkProfiles.map(profileConfigLabel));
    setSelected3dNetwork(nextNetwork);
    if (nextConfigOptions.length === 1) {
      const nextConfig = nextConfigOptions[0];
      setSelected3dConfig(nextConfig);
      onSelectModelProfile(resolve3dProfile(value, nextNetwork, nextConfig)?.id || "");
    }
  }

  function handleNetworkChange(value) {
    setSelected3dNetwork(value);
    setSelected3dConfig("");
    onSelectModelProfile("");
    if (!selected3dFramework || !value) {
      return;
    }
    const nextNetworkProfiles = modelProfiles.filter(
      (profile) =>
        (profile.framework || profile.provider) === selected3dFramework &&
        profileNetworkLabel(profile) === value
    );
    const nextConfigOptions = uniqueValues(nextNetworkProfiles.map(profileConfigLabel));
    if (nextConfigOptions.length === 1) {
      const nextConfig = nextConfigOptions[0];
      setSelected3dConfig(nextConfig);
      onSelectModelProfile(resolve3dProfile(selected3dFramework, value, nextConfig)?.id || "");
    }
  }

  function handleConfigChange(value) {
    setSelected3dConfig(value);
    if (!selected3dFramework || !selected3dNetwork || !value) {
      onSelectModelProfile("");
      return;
    }
    onSelectModelProfile(resolve3dProfile(selected3dFramework, selected3dNetwork, value)?.id || "");
  }

  const selected3dProvider = selectedModelProfile?.provider || (selected3dFramework === "WildDet3D" ? "wilddet3d" : "openpcdet");
  const selected3dTask = selectedModelProfile?.task || "";
  const selected3dModelAvailable = Boolean(selectedModelProfile?.is_available);
  const isCameraOnly3dModel = selected3dFramework === "WildDet3D" || selected3dProvider === "wilddet3d" || selected3dTask === "3d_camera_bbox";
  const canRun3dDetection = hasWorkspace && (
    isCameraOnly3dModel
      ? Boolean(capabilityReport?.can_camera_3d_box)
      : (capabilityReport?.can_annotate_3d_box ?? true)
  );
  const canChoose3dModel = hasWorkspace && Boolean(modelProfiles.length) && !hasActiveJob;
  const canTrack3d = hasWorkspace && (capabilityReport?.can_annotate_3d_box ?? true);
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
  const selected2dDetectionModelAvailable = Boolean(selected2dDetectionModelProfile?.is_available);
  const selected2dSegmentationModelAvailable = Boolean(selected2dSegmentationModelProfile?.is_available);
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
  const modelCatalogIsLoading = modelCatalogStatus === "loading";
  const modelCatalogHasError = modelCatalogStatus === "error";

  const detectionSummary = !hasWorkspace
    ? "Load a dataset first"
    : modelCatalogIsLoading
      ? "Checking models..."
    : autoLabelLaunchState
      ? "Preparing..."
      : autoLabelJob
        ? `${autoLabelJob.status} - ${displayJobPercentLabel}`
      : selectedModelProfile
          ? `${selectedModelProfile.name}${selected3dModelAvailable ? "" : " - unavailable"} - ${startFrame}-${endFrame}`
          : selected3dFramework
            ? `${selected3dFramework} - choose config`
            : "Choose framework";
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
    : modelCatalogIsLoading
      ? "Checking models..."
    : box2dLaunchState
      ? "Preparing..."
      : box2dJob
        ? `${box2dJob.status} - ${formatProgressPercent((box2dJob.progress || 0) * 100, box2dJobIsActive)}`
        : selected2dDetectionModelProfile
          ? `${selected2dDetectionModelProfile.name}${selected2dDetectionModelAvailable ? "" : " - unavailable"}`
          : "No YOLO model";
  const segmentationSummary = !hasWorkspace
    ? "Load a dataset first"
    : modelCatalogIsLoading
      ? "Checking models..."
    : segmentationLaunchState
      ? "Preparing..."
      : segmentationJob
        ? `${segmentationJob.status} - ${formatProgressPercent((segmentationJob.progress || 0) * 100, segmentationJobIsActive)}`
        : selected2dSegmentationModelProfile
          ? `${selected2dSegmentationModelProfile.name}${selected2dSegmentationModelAvailable ? "" : " - unavailable"}`
          : "No YOLO model";

  return (
    <>
      <SidebarSection
        id="auto-labeling-3d-detection"
        title="3D Object Detection"
        summary={detectionSummary}
        expanded={expandedSectionId === "auto-labeling-3d-detection"}
        onToggle={onToggleSection}
      >
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight3d">Framework</ParameterLabel>
          <select
            className="model-select"
            value={selected3dFramework}
            onChange={(event) => handleFrameworkChange(event.target.value)}
            disabled={!canChoose3dModel}
          >
            <option value="">--</option>
            {frameworkOptions.map((framework) => {
              const frameworkAvailable = modelProfiles.some(
                (profile) => (profile.framework || profile.provider) === framework && profile.is_available
              );
              return (
                <option key={framework} value={framework}>
                  {formatAvailabilityLabel(framework, frameworkAvailable)}
                </option>
              );
            })}
          </select>
        </label>
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight3d">Network</ParameterLabel>
          <select
            className="model-select"
            value={selected3dNetwork}
            onChange={(event) => handleNetworkChange(event.target.value)}
            disabled={!canChoose3dModel || !selected3dFramework || !networkOptions.length}
          >
            <option value="">--</option>
            {networkOptions.map((network) => {
              const networkAvailable = frameworkProfiles.some(
                (profile) => profileNetworkLabel(profile) === network && profile.is_available
              );
              return (
                <option key={network} value={network}>
                  {formatAvailabilityLabel(network, networkAvailable)}
                </option>
              );
            })}
          </select>
        </label>
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight3d">Config</ParameterLabel>
          <select
            className="model-select"
            value={selected3dConfig}
            onChange={(event) => handleConfigChange(event.target.value)}
            disabled={!canChoose3dModel || !selected3dFramework || !selected3dNetwork || !configOptions.length}
          >
            <option value="">--</option>
            {configOptions.map((config) => {
              const configAvailable = networkProfiles.some(
                (profile) => profileConfigLabel(profile) === config && profile.is_available
              );
              return (
                <option key={config} value={config} disabled={!configAvailable}>
                  {formatAvailabilityLabel(config, configAvailable)}
                </option>
              );
            })}
          </select>
        </label>
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight3d">Weight</ParameterLabel>
          <select className="model-select" value={selectedModelProfileId} disabled>
            {selectedModelProfile ? (
              <option value={selectedModelProfile.id}>
                {formatAvailabilityLabel(selectedModelProfile.weight_name, selected3dModelAvailable)}
              </option>
            ) : (
              <option value="">--</option>
            )}
          </select>
        </label>
        <NumberField label="Start" helpKey="frameStart" value={startFrame} step={1} onChange={setStartFrame} disabled={!canRun3dDetection} />
        <NumberField label="End" helpKey="frameEnd" value={endFrame} step={1} onChange={setEndFrame} disabled={!canRun3dDetection} />
        {!hasWorkspace ? <p className="panel-hint">Auto-labeling is unavailable until a dataset is loaded into the workspace.</p> : null}
        {hasWorkspace && modelCatalogIsLoading ? <p className="panel-hint">Model availability is being checked in the background.</p> : null}
        {hasWorkspace && modelCatalogHasError ? <p className="panel-hint">Model availability could not be checked yet. Data viewing is unaffected.</p> : null}
        {hasWorkspace && selectedModelProfile && !selected3dModelAvailable ? (
          <p className="panel-hint">This model is registered but unavailable because its checkpoint is not mounted into the worker container.</p>
        ) : null}
        {hasWorkspace && !canRun3dDetection && isCameraOnly3dModel ? <p className="panel-hint">WildDet3D requires camera frames, camera intrinsics, and a LiDAR/Camera transform.</p> : null}
        {hasWorkspace && !canRun3dDetection && !isCameraOnly3dModel ? <p className="panel-hint">This 3D detector is unavailable until the dataset provides LiDAR frames.</p> : null}
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
            disabled={!selected3dModelAvailable || hasActiveJob || !canRun3dDetection}
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
          <div>Method</div>
          <strong>Kalman Filter</strong>
        </div>
        <OptionalNumberField
          label="Score Gate"
          helpKey="trackingScoreGate"
          value={trackingInputScoreThreshold}
          step={0.01}
          min={0}
          placeholder="Off"
          disabled={!canTrack3d || hasActiveJob}
          onChange={onTrackingInputScoreThresholdChange}
        />
        <NumberField label="Max Missed" helpKey="trackingMaxMissed" value={trackingMaxMissedFrames} step={1} min={0} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingMaxMissedFramesChange} />
        <NumberField
          label="Assoc Gate"
          helpKey="trackingAssociationGate"
          value={trackingAssociationDistanceM}
          step={0.1}
          min={0}
          disabled={!canTrack3d || hasActiveJob}
          onChange={onTrackingAssociationDistanceMChange}
        />
        <NumberField label="Process Pos" helpKey="trackingProcessPos" value={trackingProcessNoisePos} step={0.05} min={0.001} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingProcessNoisePosChange} />
        <NumberField label="Process Size" helpKey="trackingProcessSize" value={trackingProcessNoiseSize} step={0.05} min={0.001} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingProcessNoiseSizeChange} />
        <NumberField label="Process Vel" helpKey="trackingProcessVel" value={trackingProcessNoiseVel} step={0.05} min={0.001} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingProcessNoiseVelChange} />
        <NumberField label="Measure Pos" helpKey="trackingMeasurePos" value={trackingMeasurementNoisePos} step={0.05} min={0.001} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingMeasurementNoisePosChange} />
        <NumberField label="Measure Size" helpKey="trackingMeasureSize" value={trackingMeasurementNoiseSize} step={0.05} min={0.001} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingMeasurementNoiseSizeChange} />
        <NumberField label="Yaw Smooth" helpKey="trackingYawSmooth" value={trackingYawSmoothingAlpha} step={0.05} min={0} max={1} disabled={!canTrack3d || hasActiveJob} onChange={onTrackingYawSmoothingAlphaChange} />
        <p className="panel-hint">{trackingReadiness?.message || "Tracking is unavailable until frame-wise detection finishes for the selected range."}</p>
        <p className="panel-hint">Blank score leaves detections unfiltered; Kalman settings are applied to the whole selected frame range.</p>
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
          <button className="primary-button" onClick={onRunTracking} disabled={!trackingReady || hasActiveJob || !canTrack3d}>
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
          <strong>{selected2dDetectionModelProfile ? (selected2dDetectionModelAvailable ? "Ready" : "Unavailable") : "No model"}</strong>
          <div>Output</div>
          <strong>2D boxes</strong>
        </div>
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight2d">Weight</ParameterLabel>
          <select
            className="model-select"
            value={selected2dDetectionModelProfileId}
            onChange={(event) => onSelect2dDetectionModelProfile(event.target.value)}
            disabled={!yolo2dDetectionModelProfiles.length || !canAnnotate2d || hasActiveJob}
          >
            {yolo2dDetectionModelProfiles.length ? (
              yolo2dDetectionModelProfiles.map((profile) => (
                <option key={profile.id} value={profile.id} disabled={!profile.is_available}>
                  {profile.name} / {formatAvailabilityLabel(profile.weight_name, profile.is_available)}
                </option>
              ))
            ) : (
              <option value="">No YOLO detection model available</option>
            )}
          </select>
        </label>
        <NumberField label="Start" helpKey="frameStart" value={startFrame} step={1} onChange={setStartFrame} disabled={!canAnnotate2d || hasActiveJob} />
        <NumberField label="End" helpKey="frameEnd" value={endFrame} step={1} onChange={setEndFrame} disabled={!canAnnotate2d || hasActiveJob} />
        {!hasWorkspace ? <p className="panel-hint">2D detection is unavailable until a dataset is loaded into the workspace.</p> : null}
        {hasWorkspace && modelCatalogIsLoading ? <p className="panel-hint">Model availability is being checked in the background.</p> : null}
        {hasWorkspace && modelCatalogHasError ? <p className="panel-hint">Model availability could not be checked yet. Data viewing is unaffected.</p> : null}
        {hasWorkspace && !canAnnotate2d ? <p className="panel-hint">2D detection requires camera frames.</p> : null}
        {hasWorkspace && selected2dDetectionModelProfile && !selected2dDetectionModelAvailable ? (
          <p className="panel-hint">This model is registered but unavailable because its checkpoint is not mounted into the worker container.</p>
        ) : null}
        <CompactJobBox
          launchState={box2dLaunchState}
          job={box2dJob}
          active={box2dJobIsActive}
          canCancel={canCancelBox2d}
          onCancel={onCancelAutoLabel}
        />
        <div className="section-action-row">
          <button className="primary-button" onClick={onRun2dObjectDetection} disabled={!selected2dDetectionModelAvailable || hasActiveJob || !canAnnotate2d}>
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
          <strong>{selected2dSegmentationModelProfile ? (selected2dSegmentationModelAvailable ? "Ready" : "Unavailable") : "No model"}</strong>
          <div>Output</div>
          <strong>Camera semantic mask</strong>
        </div>
        <label className="field-row">
          <ParameterLabel helpKey="modelWeight2d">Weight</ParameterLabel>
          <select
            className="model-select"
            value={selected2dSegmentationModelProfileId}
            onChange={(event) => onSelect2dSegmentationModelProfile(event.target.value)}
            disabled={!yolo2dSegmentationModelProfiles.length || !canAnnotate2d || hasActiveJob}
          >
            {yolo2dSegmentationModelProfiles.length ? (
              yolo2dSegmentationModelProfiles.map((profile) => (
                <option key={profile.id} value={profile.id} disabled={!profile.is_available}>
                  {profile.name} / {formatAvailabilityLabel(profile.weight_name, profile.is_available)}
                </option>
              ))
            ) : (
              <option value="">No YOLO segmentation model available</option>
            )}
          </select>
        </label>
        <NumberField label="Start" helpKey="frameStart" value={startFrame} step={1} onChange={setStartFrame} disabled={!canAnnotate2d || hasActiveJob} />
        <NumberField label="End" helpKey="frameEnd" value={endFrame} step={1} onChange={setEndFrame} disabled={!canAnnotate2d || hasActiveJob} />
        {hasWorkspace && modelCatalogIsLoading ? <p className="panel-hint">Model availability is being checked in the background.</p> : null}
        {hasWorkspace && modelCatalogHasError ? <p className="panel-hint">Model availability could not be checked yet. Data viewing is unaffected.</p> : null}
        {hasWorkspace && selected2dSegmentationModelProfile && !selected2dSegmentationModelAvailable ? (
          <p className="panel-hint">This model is registered but unavailable because its checkpoint is not mounted into the worker container.</p>
        ) : null}
        <p className="panel-hint">Successful runs write a camera semantic layer that can be shown from Show / Layers / Camera Semantic.</p>
        <CompactJobBox
          launchState={segmentationLaunchState}
          job={segmentationJob}
          active={segmentationJobIsActive}
          canCancel={canCancelSegmentation}
          onCancel={onCancelAutoLabel}
        />
        <div className="section-action-row">
          <button className="primary-button" onClick={onRun2dSemanticSegmentation} disabled={!selected2dSegmentationModelAvailable || hasActiveJob || !canAnnotate2d}>
            Run
          </button>
        </div>
      </SidebarSection>
    </>
  );
}
