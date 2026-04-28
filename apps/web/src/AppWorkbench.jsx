import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, request } from "./lib/api";
import { clampScoreThreshold, isAnnotationVisible } from "./lib/annotationColors";
import BottomTerminal from "./components/BottomTerminal";
import MediaPanel from "./components/MediaPanel";
import PointCloudViewer from "./components/PointCloudViewer";
import TimelineBar from "./components/TimelineBar";
import TopMenuBar from "./components/TopMenuBar";
import WorkspaceSidebar from "./components/WorkspaceSidebar";

const ACTIVE_JOB_STATUSES = ["queued", "running", "paused"];
const REFRESHABLE_JOB_STATUSES = ["completed", "cancelled"];
const ACTIVE_JOB_STORAGE_KEY = "rat-active-job-id";
const SIDEBAR_WIDTH_STORAGE_KEY = "rat-sidebar-width";
const RIGHT_PANEL_WIDTH_STORAGE_KEY = "rat-right-panel-width";
const DEFAULT_SIDEBAR_WIDTH = 340;
const MIN_SIDEBAR_WIDTH = 260;
const MAX_SIDEBAR_WIDTH = 520;
const DEFAULT_RIGHT_PANEL_WIDTH = 400;
const MIN_RIGHT_PANEL_WIDTH = 300;
const MAX_RIGHT_PANEL_WIDTH = 720;
const DESKTOP_LAYOUT_MIN_WIDTH = 1100;
const FIXED_VISUALIZATION_CLASS_NAMES = ["Vehicle", "Pedestrian", "Bicycle", "truck", "bus"];
const WORKSPACE_BOOTSTRAP_ATTEMPTS = 8;
const WORKSPACE_BOOTSTRAP_DELAY_MS = 500;
const DEFAULT_TRACKING_METHOD = "kalman_filter";
const DEFAULT_TRACKING_MAX_MISSED_FRAMES = 3;
const DEFAULT_TRACKING_ASSOCIATION_DISTANCE_M = 4.0;
const DEFAULT_TRACKING_PROCESS_NOISE_POS = 0.5;
const DEFAULT_TRACKING_PROCESS_NOISE_SIZE = 0.1;
const DEFAULT_TRACKING_PROCESS_NOISE_VEL = 1.0;
const DEFAULT_TRACKING_MEASUREMENT_NOISE_POS = 1.0;
const DEFAULT_TRACKING_MEASUREMENT_NOISE_SIZE = 0.5;
const DEFAULT_TRACKING_YAW_SMOOTHING_ALPHA = 0.5;
const DEFAULT_LAYER_MODES = {
  lidar: "intensity",
  camera: "intensity",
  radar: "intensity"
};
const DEFAULT_BBX_VISIBILITY = {
  box3d: true,
  box2d: false
};
const DEFAULT_FILTER_TARGET = "box3d";

const FALLBACK_MODEL_PROFILES = [
  {
    id: "pv_rcnn_kitti",
    name: "OpenPCDet PV-RCNN (KITTI)",
    architecture: "PV-RCNN",
    framework: "OpenPCDet",
    weight_name: "pv_rcnn_kitti.pth",
    description: "PV-RCNN checkpoint pretrained on KITTI",
    default_score_threshold: 0.1,
    is_available: true
  },
  {
    id: "pointpillar_kitti",
    name: "OpenPCDet PointPillar (KITTI)",
    architecture: "PointPillar",
    framework: "OpenPCDet",
    weight_name: "pointpillar_kitti.pth",
    description: "PointPillar checkpoint pretrained on KITTI",
    default_score_threshold: 0.1,
    is_available: true
  },
  {
    id: "second_kitti",
    name: "OpenPCDet SECOND (KITTI)",
    architecture: "SECOND",
    framework: "OpenPCDet",
    weight_name: "second_kitti.pth",
    description: "SECOND checkpoint pretrained on KITTI",
    default_score_threshold: 0.1,
    is_available: true
  },
  {
    id: "pointpillar_multihead_nuscenes",
    name: "OpenPCDet PointPillar Multi-Head (nuScenes)",
    architecture: "PointPillar",
    framework: "OpenPCDet",
    weight_name: "pointpillar_multihead_nuscenes.pth",
    description: "Multi-head PointPillar checkpoint pretrained on nuScenes",
    default_score_threshold: 0.1,
    is_available: true
  },
  {
    id: "centerpoint_nuscenes",
    name: "OpenPCDet CenterPoint (nuScenes)",
    architecture: "CenterPoint",
    framework: "OpenPCDet",
    weight_name: "centerpoint_nuscenes.pth",
    description: "CenterPoint checkpoint pretrained on nuScenes",
    default_score_threshold: 0.1,
    is_available: true
  }
];

function createEmptyPointCloud() {
  return {
    values: new Float32Array(0),
    count: 0
  };
}

function bufferToPointCloud(buffer) {
  const values = buffer instanceof Float32Array ? buffer : new Float32Array(buffer);
  return {
    values,
    count: Math.floor(values.length / 4)
  };
}

function bufferToSemanticLabels(buffer) {
  return buffer instanceof Int32Array ? buffer : new Int32Array(buffer);
}

function getJobSocketUrl(jobId) {
  const wsBase = API_BASE.replace(/^http/i, "ws");
  return `${wsBase}/jobs/${jobId}/events`;
}

function hasCameraTransform(calibration) {
  return Object.entries(calibration || {}).some(([key, value]) => {
    if (key === "camera_rgb_intrinsics") {
      return false;
    }
    return Array.isArray(value) && value.length === 4 && value.every((row) => Array.isArray(row) && row.length === 4);
  });
}

function buildCapabilityReport(timeline, frameDetail) {
  if (!timeline?.frames?.length) {
    return null;
  }
  const lidarAvailable = timeline.frames.some((frame) => Boolean(frame.lidar_path));
  const cameraAvailable = timeline.frames.some((frame) => Object.values(frame.camera || {}).some((match) => match.available));
  const radarAvailable = timeline.frames.some((frame) => Object.values(frame.radar || {}).some((match) => match.available));
  const calibration = frameDetail?.calibration || {};
  const hasIntrinsics = Boolean(calibration.camera_rgb_intrinsics);
  const canProjectToCamera = lidarAvailable && cameraAvailable && hasIntrinsics && hasCameraTransform(calibration);
  const canCamera3dBox = cameraAvailable && hasIntrinsics && hasCameraTransform(calibration);
  const missingRequirements = [];
  if (!lidarAvailable) {
    missingRequirements.push("LiDAR data is missing, so 3D box annotation and auto-label are unavailable.");
  }
  if (!cameraAvailable) {
    missingRequirements.push("Camera data is missing, so camera visualization is unavailable.");
  }
  if (!radarAvailable) {
    missingRequirements.push("Radar data is missing, so radar preview is unavailable.");
  }
  if (cameraAvailable && !hasIntrinsics) {
    missingRequirements.push("Camera intrinsics are missing, so camera projection is unavailable.");
  }
  if (cameraAvailable && lidarAvailable && !hasCameraTransform(calibration)) {
    missingRequirements.push("LiDAR/Camera transform is missing, so camera projection is unavailable.");
  }
  return {
    lidar_available: lidarAvailable,
    camera_available: cameraAvailable,
    radar_available: radarAvailable,
    can_annotate_3d_box: lidarAvailable,
    can_camera_3d_box: canCamera3dBox,
    can_project_to_camera: canProjectToCamera,
    missing_requirements: missingRequirements
  };
}

function createLogEntry(level, message) {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    timestamp: new Date().toISOString(),
    level,
    message,
    repeat: 1
  };
}

function createTrackingReadinessFallback(message = "Tracking is unavailable until frame-wise detection finishes for the selected range.") {
  return {
    ready: false,
    annotated_frames: 0,
    total_frames: 0,
    next_missing_frame: null,
    message
  };
}

function createJobLaunchState(type, message, detail = null) {
  return {
    type,
    message,
    detail
  };
}

function clampSidebarWidth(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return DEFAULT_SIDEBAR_WIDTH;
  }
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, Math.round(numericValue)));
}

function readInitialSidebarWidth() {
  if (typeof window === "undefined") {
    return DEFAULT_SIDEBAR_WIDTH;
  }
  return clampSidebarWidth(window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY) || DEFAULT_SIDEBAR_WIDTH);
}

function getSidebarScale(width) {
  const ratio = width / DEFAULT_SIDEBAR_WIDTH;
  return Math.max(0.92, Math.min(1.08, ratio));
}

function clampRightPanelWidth(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return DEFAULT_RIGHT_PANEL_WIDTH;
  }
  return Math.min(MAX_RIGHT_PANEL_WIDTH, Math.max(MIN_RIGHT_PANEL_WIDTH, Math.round(numericValue)));
}

function readInitialRightPanelWidth() {
  if (typeof window === "undefined") {
    return DEFAULT_RIGHT_PANEL_WIDTH;
  }
  return clampRightPanelWidth(window.localStorage.getItem(RIGHT_PANEL_WIDTH_STORAGE_KEY) || DEFAULT_RIGHT_PANEL_WIDTH);
}

function chooseSceneId(dataset, preferredSceneId = null) {
  if (!dataset?.scenes?.length) {
    return null;
  }
  if (preferredSceneId && dataset.scenes.some((scene) => scene.id === preferredSceneId && scene.available)) {
    return preferredSceneId;
  }
  if (dataset.default_scene_id) {
    const defaultScene = dataset.scenes.find((scene) => scene.id === dataset.default_scene_id && scene.available);
    if (defaultScene) {
      return defaultScene.id;
    }
  }
  return dataset.scenes.find((scene) => scene.available)?.id || dataset.default_scene_id || dataset.scenes[0]?.id || null;
}

function mergeClassNames(...groups) {
  const seen = new Set();
  const merged = [];
  groups.flat().forEach((entry) => {
    const normalized = String(entry || "").trim();
    if (!normalized) {
      return;
    }
    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    merged.push(normalized);
  });
  return merged;
}

function mapVisualizationClassName(className) {
  const normalized = String(className || "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === "vehicle" || normalized === "car") {
    return "Vehicle";
  }
  if (normalized === "pedestrian") {
    return "Pedestrian";
  }
  if (normalized === "bicycle") {
    return "Bicycle";
  }
  if (normalized === "truck") {
    return "truck";
  }
  if (normalized === "bus") {
    return "bus";
  }
  return null;
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function normalizeLayerCapability(raw, fallback) {
  return {
    intensity: Boolean(raw?.intensity ?? fallback?.intensity),
    semantic: Boolean(raw?.semantic ?? fallback?.semantic),
    defaultMode: raw?.default_mode || null,
    intensityReason: raw?.intensity_reason || fallback?.intensityReason || null,
    semanticReason: raw?.semantic_reason || fallback?.semanticReason || null
  };
}

function resolveLayerDefault(capability, fallbackMode = "intensity") {
  if (capability?.defaultMode && capability[capability.defaultMode]) {
    return capability.defaultMode;
  }
  if (capability?.intensity) {
    return "intensity";
  }
  if (capability?.semantic) {
    return "semantic";
  }
  return fallbackMode;
}

function isLayerModeAvailable(capability, mode) {
  return Boolean(capability?.[mode]);
}

export default function AppWorkbench() {
  const [activeMenu, setActiveMenu] = useState("file");
  const [sequences, setSequences] = useState([]);
  const [datasetCatalog, setDatasetCatalog] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [selectedSceneId, setSelectedSceneId] = useState(null);
  const [currentSequenceId, setCurrentSequenceId] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameDetail, setFrameDetail] = useState(null);
  const [points, setPoints] = useState(createEmptyPointCloud);
  const [lidarSemanticLabels, setLidarSemanticLabels] = useState(null);
  const [layerModes, setLayerModes] = useState(DEFAULT_LAYER_MODES);
  const [ramapCacheStatus, setRamapCacheStatus] = useState(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState(null);
  const [transformMode, setTransformMode] = useState("translate");
  const [isPlaying, setIsPlaying] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobLaunchState, setJobLaunchState] = useState(null);
  const [startFrame, setStartFrame] = useState(0);
  const [endFrame, setEndFrame] = useState(0);
  const [trackingReadiness, setTrackingReadiness] = useState(createTrackingReadinessFallback);
  const [trackingInputScoreThreshold, setTrackingInputScoreThreshold] = useState(null);
  const [trackingMaxMissedFrames, setTrackingMaxMissedFrames] = useState(DEFAULT_TRACKING_MAX_MISSED_FRAMES);
  const [trackingAssociationDistanceM, setTrackingAssociationDistanceM] = useState(DEFAULT_TRACKING_ASSOCIATION_DISTANCE_M);
  const [trackingProcessNoisePos, setTrackingProcessNoisePos] = useState(DEFAULT_TRACKING_PROCESS_NOISE_POS);
  const [trackingProcessNoiseSize, setTrackingProcessNoiseSize] = useState(DEFAULT_TRACKING_PROCESS_NOISE_SIZE);
  const [trackingProcessNoiseVel, setTrackingProcessNoiseVel] = useState(DEFAULT_TRACKING_PROCESS_NOISE_VEL);
  const [trackingMeasurementNoisePos, setTrackingMeasurementNoisePos] = useState(DEFAULT_TRACKING_MEASUREMENT_NOISE_POS);
  const [trackingMeasurementNoiseSize, setTrackingMeasurementNoiseSize] = useState(DEFAULT_TRACKING_MEASUREMENT_NOISE_SIZE);
  const [trackingYawSmoothingAlpha, setTrackingYawSmoothingAlpha] = useState(DEFAULT_TRACKING_YAW_SMOOTHING_ALPHA);
  const [modelProfiles, setModelProfiles] = useState(FALLBACK_MODEL_PROFILES);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState(FALLBACK_MODEL_PROFILES[0].id);
  const [selected2dDetectionModelProfileId, setSelected2dDetectionModelProfileId] = useState("");
  const [selected2dSegmentationModelProfileId, setSelected2dSegmentationModelProfileId] = useState("");
  const [scoreThreshold, setScoreThreshold] = useState(FALLBACK_MODEL_PROFILES[0].default_score_threshold);
  const [hiddenClassNames, setHiddenClassNames] = useState([]);
  const [bbxVisibility, setBbxVisibility] = useState(DEFAULT_BBX_VISIBILITY);
  const [filterTarget, setFilterTarget] = useState(DEFAULT_FILTER_TARGET);
  const [statusMessage, setStatusMessage] = useState("Choose a dataset and scene in File, then click Load.");
  const [isFrameLoading, setIsFrameLoading] = useState(false);
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [isMenuBusy, setIsMenuBusy] = useState(false);
  const [lastLoadedAt, setLastLoadedAt] = useState(null);
  const [terminalEntries, setTerminalEntries] = useState([]);
  const [isTerminalExpanded, setIsTerminalExpanded] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(readInitialSidebarWidth);
  const [rightPanelWidth, setRightPanelWidth] = useState(readInitialRightPanelWidth);
  const [expandedSidebarSection, setExpandedSidebarSection] = useState(null);
  const [viewResetToken, setViewResetToken] = useState(0);
  const updateTimersRef = useRef(new Map());
  const pendingPatchesRef = useRef(new Map());
  const frameRequestIdRef = useRef(0);
  const frameAbortControllerRef = useRef(null);
  const jobLaunchLockRef = useRef(false);
  const jobSocketRef = useRef(null);
  const previousJobRef = useRef(null);
  const layerModeTouchedRef = useRef({ lidar: false, camera: false, radar: false });
  const appShellRef = useRef(null);
  const sidebarResizeStateRef = useRef({ active: false, pointerOffset: 0 });
  const rightPanelResizeStateRef = useRef({ active: false, shellRight: 0, pointerOffset: 0 });
  const [jobSocketConnected, setJobSocketConnected] = useState(false);
  const previousRamapCacheRunningRef = useRef(false);

  const currentSequence = useMemo(
    () => sequences.find((sequence) => sequence.id === currentSequenceId) || null,
    [sequences, currentSequenceId]
  );
  const currentDataset = useMemo(
    () => datasetCatalog.find((dataset) => dataset.id === selectedDatasetId) || null,
    [datasetCatalog, selectedDatasetId]
  );
  const currentScene = useMemo(
    () => currentDataset?.scenes?.find((scene) => scene.id === selectedSceneId) || null,
    [currentDataset, selectedSceneId]
  );
  const currentAnnotations = frameDetail?.annotations || [];
  const visualizableAnnotations = useMemo(
    () =>
      currentAnnotations
        .map((annotation) => {
          const visualClassName = mapVisualizationClassName(annotation.class_name);
          if (!visualClassName) {
            return null;
          }
          return {
            ...annotation,
            class_name: visualClassName
          };
        })
        .filter(Boolean),
    [currentAnnotations]
  );
  const visibleAnnotations = useMemo(
    () => {
      if (filterTarget !== "box3d") {
        return visualizableAnnotations;
      }
      return visualizableAnnotations.filter(
        (annotation) =>
          isAnnotationVisible(annotation, scoreThreshold) && !hiddenClassNames.includes(annotation.class_name)
      );
    },
    [filterTarget, visualizableAnnotations, scoreThreshold, hiddenClassNames]
  );
  const visible2dAnnotations = useMemo(
    () => {
      const annotations2d = frameDetail?.annotations_2d || [];
      if (filterTarget !== "box2d") {
        return annotations2d;
      }
      return annotations2d.filter(
        (annotation) =>
          isAnnotationVisible(annotation, scoreThreshold) && !hiddenClassNames.includes(annotation.class_name)
      );
    },
    [filterTarget, frameDetail?.annotations_2d, hiddenClassNames, scoreThreshold]
  );
  const hiddenAnnotationCount = filterTarget === "box3d" ? currentAnnotations.length - visibleAnnotations.length : 0;
  const drawnAnnotations = useMemo(
    () => (bbxVisibility.box3d ? visibleAnnotations : []),
    [bbxVisibility.box3d, visibleAnnotations]
  );
  const drawn2dAnnotations = useMemo(
    () => (bbxVisibility.box2d ? visible2dAnnotations : []),
    [bbxVisibility.box2d, visible2dAnnotations]
  );
  const selectedAnnotation =
    visibleAnnotations.find((annotation) => annotation.id === selectedAnnotationId) || null;
  const openpcdetModelProfiles = useMemo(
    () => modelProfiles.filter((profile) => profile.provider !== "yolo"),
    [modelProfiles]
  );
  const yolo2dDetectionModelProfiles = useMemo(
    () => modelProfiles.filter((profile) => profile.provider === "yolo" && profile.task === "2d_object_detection"),
    [modelProfiles]
  );
  const yolo2dSegmentationModelProfiles = useMemo(
    () => modelProfiles.filter((profile) => profile.provider === "yolo" && profile.task === "2d_semantic_segmentation"),
    [modelProfiles]
  );
  const selectedModelProfile =
    openpcdetModelProfiles.find((profile) => profile.id === selectedModelProfileId) || openpcdetModelProfiles[0] || null;
  const selected2dDetectionModelProfile =
    yolo2dDetectionModelProfiles.find((profile) => profile.id === selected2dDetectionModelProfileId) || yolo2dDetectionModelProfiles[0] || null;
  const selected2dSegmentationModelProfile =
    yolo2dSegmentationModelProfiles.find((profile) => profile.id === selected2dSegmentationModelProfileId) || yolo2dSegmentationModelProfiles[0] || null;
  const visualizationClassNames = FIXED_VISUALIZATION_CLASS_NAMES;
  const layerCapabilities = useMemo(() => {
    const layerAvailability = frameDetail?.layer_availability || {};
    return {
      lidar: normalizeLayerCapability(layerAvailability.lidar, {
        intensity: Boolean(frameDetail?.asset_urls?.lidar),
        semantic: Boolean(frameDetail?.semantic_available?.lidar && frameDetail?.asset_urls?.semantic?.lidar),
        intensityReason: "No LiDAR frame for this sample",
        semanticReason: "No semantic labels for this frame"
      }),
      camera: normalizeLayerCapability(layerAvailability.camera?.rgb, {
        intensity: Boolean(frameDetail?.asset_urls?.camera?.rgb),
        semantic: Boolean(frameDetail?.semantic_available?.camera?.rgb && frameDetail?.asset_urls?.semantic?.camera?.rgb),
        intensityReason: "No camera frame for this sample",
        semanticReason: "No semantic mask for this frame"
      }),
      radar: normalizeLayerCapability(layerAvailability.radar?.heatmap, {
        intensity: Boolean(frameDetail?.asset_urls?.radar_preview?.heatmap),
        semantic: Boolean(frameDetail?.semantic_available?.radar?.heatmap && frameDetail?.asset_urls?.semantic?.radar?.heatmap),
        intensityReason: "No RAMap intensity for this frame",
        semanticReason: "No semantic radar layer for this frame"
      })
    };
  }, [frameDetail]);
  const defaultLayerModes = useMemo(
    () => ({
      lidar: resolveLayerDefault(layerCapabilities.lidar),
      camera: resolveLayerDefault(layerCapabilities.camera),
      radar: resolveLayerDefault(layerCapabilities.radar)
    }),
    [layerCapabilities]
  );
  const semanticAvailable = useMemo(
    () => ({
      lidar: layerCapabilities.lidar.semantic,
      camera: layerCapabilities.camera.semantic,
      radar: layerCapabilities.radar.semantic
    }),
    [layerCapabilities]
  );
  const intensityAvailable = useMemo(
    () => ({
      lidar: layerCapabilities.lidar.intensity,
      camera: layerCapabilities.camera.intensity,
      radar: layerCapabilities.radar.intensity
    }),
    [layerCapabilities]
  );
  const isAutoLabelingMode = activeMenu === "auto-labeling";
  const isManualLabelingMode = activeMenu === "manual-labeling";
  const isLabelingMenu = isAutoLabelingMode || isManualLabelingMode;
  const isAnnotationMode = isManualLabelingMode;
  const hasRunningJob = jobStatus ? ACTIVE_JOB_STATUSES.includes(jobStatus.status) : false;
  const hasActiveJob = hasRunningJob || Boolean(jobLaunchState);
  const capabilityReport = useMemo(() => buildCapabilityReport(timeline, frameDetail), [timeline, frameDetail]);
  const bbxAvailability = useMemo(
    () => ({
      box3d: Boolean(currentSequenceId && (capabilityReport?.can_annotate_3d_box ?? true)),
      box2d: Boolean(frameDetail?.annotations_2d?.length)
    }),
    [capabilityReport, currentSequenceId, frameDetail?.annotations_2d]
  );
  const sidebarScale = useMemo(() => getSidebarScale(sidebarWidth), [sidebarWidth]);
  const appShellStyle = useMemo(
    () => ({
      "--sidebar-width": `${sidebarWidth}px`,
      "--right-panel-width": `${rightPanelWidth}px`,
      "--sidebar-scale": sidebarScale.toFixed(3)
    }),
    [rightPanelWidth, sidebarScale, sidebarWidth]
  );

  function updateStatus(message) {
    if (message) {
      setStatusMessage(message);
    }
  }

  function logMessage(level, message) {
    const trimmed = String(message || "").trim();
    if (!trimmed) {
      return;
    }
    updateStatus(trimmed);
    setTerminalEntries((current) => {
      const next = [...current];
      const last = next[next.length - 1];
      if (last && last.level === level && last.message === trimmed) {
        next[next.length - 1] = {
          ...last,
          repeat: last.repeat + 1,
          timestamp: new Date().toISOString()
        };
        return next;
      }
      return [...next.slice(-249), createLogEntry(level, trimmed)];
    });
    if (level === "error") {
      setIsTerminalExpanded(true);
    }
  }

  async function refreshSequences() {
    const payload = await request("/datasets");
    const nextSequences = payload.sequences || [];
    setSequences(nextSequences);
    if (!nextSequences.length) {
      setCurrentSequenceId(null);
      setTimeline(null);
      setFrameDetail(null);
      setPoints(createEmptyPointCloud());
      setLidarSemanticLabels(null);
      setLayerModes(DEFAULT_LAYER_MODES);
      layerModeTouchedRef.current = { lidar: false, camera: false, radar: false };
      setRamapCacheStatus(null);
      return [];
    }
    setCurrentSequenceId((current) =>
      nextSequences.some((sequence) => sequence.id === current) ? current : nextSequences[0].id
    );
    return nextSequences;
  }

  async function refreshDatasetCatalog({ preferredDatasetId = null, preferredSceneId = null } = {}) {
    const payload = await request("/datasets/catalog");
    const nextCatalog = payload.datasets || [];
    setDatasetCatalog(nextCatalog);
    const nextDatasetId =
      preferredDatasetId && nextCatalog.some((dataset) => dataset.id === preferredDatasetId)
        ? preferredDatasetId
        : payload.default_dataset_id || nextCatalog[0]?.id || null;
    const nextDataset = nextCatalog.find((dataset) => dataset.id === nextDatasetId) || null;
    const nextSceneId =
      chooseSceneId(nextDataset, preferredSceneId) || payload.default_scene_id || null;
    if (nextDatasetId) {
      setSelectedDatasetId(nextDatasetId);
    }
    if (nextSceneId) {
      setSelectedSceneId(nextSceneId);
    }
    return {
      catalog: nextCatalog,
      datasetId: nextDatasetId,
      sceneId: nextSceneId
    };
  }

  async function refreshTrackingReadiness({
    sequenceId = currentSequenceId,
    start = startFrame,
    end = endFrame
  } = {}) {
    if (!sequenceId) {
      setTrackingReadiness(createTrackingReadinessFallback());
      return null;
    }
    try {
      const payload = await request(
        `/jobs/tracking/readiness?sequence_id=${encodeURIComponent(sequenceId)}&start_frame=${start}&end_frame=${end}`
      );
      setTrackingReadiness(payload);
      return payload;
    } catch (error) {
      const fallback = createTrackingReadinessFallback(error.message);
      setTrackingReadiness(fallback);
      return fallback;
    }
  }

  async function loadSelectedDataset({ datasetId = selectedDatasetId, sceneId = selectedSceneId, announce = true } = {}) {
    setIsDatasetLoading(true);
    if (announce) {
      logMessage("info", `Loading dataset ${datasetId || "dataset"}${sceneId ? ` / ${sceneId}` : ""}...`);
    }
    try {
      const payload = await request("/datasets/load", {
        method: "POST",
        body: JSON.stringify({ dataset_id: datasetId, scene_id: sceneId })
      });
      setRamapCacheStatus(payload?.ramap_cache || null);
      const loadedSequences = await refreshSequences();
      const loadedSequence =
        payload?.sequence || loadedSequences.find((sequence) => sequence.id === payload?.sequence?.id) || null;
      if (loadedSequence) {
        setSequences([loadedSequence]);
      }
      if (payload?.sequence?.id) {
        setCurrentSequenceId(payload.sequence.id);
      }
      setFrameIndex(0);
      setSelectedAnnotationId(null);
      setLidarSemanticLabels(null);
      setLayerModes(DEFAULT_LAYER_MODES);
      layerModeTouchedRef.current = { lidar: false, camera: false, radar: false };
      setIsPlaying(false);
      setLastLoadedAt(new Date().toISOString());
      const loadedName = loadedSequence?.name;
      logMessage("success", `Dataset loaded${loadedName ? ` (${loadedName})` : ""}.`);
    } catch (error) {
      logMessage("error", `Dataset load failed: ${error.message}`);
    } finally {
      setIsDatasetLoading(false);
    }
  }

  async function flushPendingAnnotationUpdates() {
    if (!currentSequenceId) {
      return;
    }
    const pendingEntries = Array.from(pendingPatchesRef.current.entries());
    if (!pendingEntries.length) {
      return;
    }
    for (const [annotationId, timerId] of updateTimersRef.current.entries()) {
      if (timerId) {
        window.clearTimeout(timerId);
      }
      updateTimersRef.current.delete(annotationId);
    }
    await Promise.all(
      pendingEntries.map(([annotationId, patch]) =>
        request(`/annotations/${currentSequenceId}/${frameIndex}/${annotationId}`, {
          method: "PATCH",
          body: JSON.stringify(patch)
        })
      )
    );
    pendingPatchesRef.current.clear();
  }

  async function persistAnnotationProgress(successMessage = "Annotations saved.") {
    if (!currentSequenceId) {
      return;
    }
    await flushPendingAnnotationUpdates();
    await request(`/annotations/${currentSequenceId}/save`, {
      method: "POST",
      body: JSON.stringify({})
    });
    logMessage("success", successMessage);
  }

  async function handleMenuSelect(nextMenu) {
    if (nextMenu === activeMenu || isMenuBusy) {
      return;
    }
    const leavingLabelingMenu = isLabelingMenu && !["auto-labeling", "manual-labeling"].includes(nextMenu);
    const leavingAutoLabeling = isAutoLabelingMode && nextMenu !== "auto-labeling";
    const leavingManualLabeling = isManualLabelingMode && nextMenu !== "manual-labeling";
    if (leavingLabelingMenu || leavingAutoLabeling || leavingManualLabeling) {
      setIsMenuBusy(true);
      updateStatus("Saving annotation progress before leaving labeling tools...");
      try {
        if (leavingAutoLabeling && jobStatus?.id && jobStatus.task_name !== "3D Tracking" && ["queued", "running"].includes(jobStatus.status)) {
          const payload = await request(`/jobs/${jobStatus.id}/pause`, { method: "POST" });
          setJobStatus(payload);
          logMessage("info", "Auto-label pause requested while leaving Auto Labeling.");
        }
        if (currentSequenceId) {
          await persistAnnotationProgress("Saved annotation progress before leaving labeling tools.");
        }
      } catch (error) {
        logMessage("error", `Failed to preserve labeling state: ${error.message}`);
      } finally {
        setIsMenuBusy(false);
      }
    }
    setActiveMenu(nextMenu);
    setExpandedSidebarSection(null);
  }

  function resetViewState() {
    setTransformMode("translate");
    setHiddenClassNames([]);
    setFrameIndex(0);
    setViewResetToken((current) => current + 1);
    logMessage("info", "View state reset.");
  }

  function clearLogs() {
    setTerminalEntries([]);
    updateStatus("Terminal cleared.");
  }

  function toggleClassVisibility(className) {
    setHiddenClassNames((current) =>
      current.includes(className) ? current.filter((item) => item !== className) : [...current, className]
    );
  }

  function toggleBbxVisibility(kind) {
    if (!bbxAvailability[kind]) {
      return;
    }
    setBbxVisibility((current) => ({
      ...current,
      [kind]: !current[kind]
    }));
  }

  function handleDatasetSelection(nextDatasetId) {
    setSelectedDatasetId(nextDatasetId);
    const nextDataset = datasetCatalog.find((dataset) => dataset.id === nextDatasetId) || null;
    setSelectedSceneId(chooseSceneId(nextDataset));
  }

  function handleSceneSelection(nextSceneId) {
    setSelectedSceneId(nextSceneId);
  }

  function handleLayerModeChange(sensor, mode) {
    if (mode === "intensity" && !intensityAvailable?.[sensor]) {
      return;
    }
    if (mode === "semantic" && !semanticAvailable?.[sensor]) {
      return;
    }
    layerModeTouchedRef.current = {
      ...layerModeTouchedRef.current,
      [sensor]: true
    };
    setLayerModes((current) => ({
      ...current,
      [sensor]: mode
    }));
  }

  function handleSidebarSectionToggle(sectionId) {
    setExpandedSidebarSection((current) => (current === sectionId ? null : sectionId));
  }

  function handleSidebarResizeStart(event) {
    if (window.innerWidth <= DESKTOP_LAYOUT_MIN_WIDTH) {
      return;
    }
    event.preventDefault();
    sidebarResizeStateRef.current = {
      active: true,
      pointerOffset: event.clientX - sidebarWidth
    };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
  }

  function handleRightPanelResizeStart(event) {
    if (window.innerWidth <= DESKTOP_LAYOUT_MIN_WIDTH || !appShellRef.current) {
      return;
    }
    event.preventDefault();
    const shellRect = appShellRef.current.getBoundingClientRect();
    rightPanelResizeStateRef.current = {
      active: true,
      shellRight: shellRect.right,
      pointerOffset: shellRect.right - event.clientX - rightPanelWidth
    };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
  }

  useEffect(() => {
    let cancelled = false;

    async function bootstrapWorkspace() {
      await request("/health");
      await refreshDatasetCatalog();
      const loadedSequences = await refreshSequences();
      if (loadedSequences.length) {
        return;
      }
      updateStatus("Choose a dataset and scene in File, then click Load.");
    }

    (async () => {
      let lastError = null;
      for (let attempt = 1; attempt <= WORKSPACE_BOOTSTRAP_ATTEMPTS; attempt += 1) {
        if (cancelled) {
          return;
        }
        try {
          await bootstrapWorkspace();
          return;
        } catch (error) {
          lastError = error;
          if (attempt === WORKSPACE_BOOTSTRAP_ATTEMPTS) {
            break;
          }
          await wait(WORKSPACE_BOOTSTRAP_DELAY_MS * attempt);
        }
      }
      if (!cancelled && lastError) {
        logMessage("error", `Dataset bootstrap failed: ${lastError.message}`);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const savedJobId = window.localStorage.getItem(ACTIVE_JOB_STORAGE_KEY);
    if (!savedJobId) {
      return;
    }
    let cancelled = false;
    request(`/jobs/${savedJobId}`)
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setJobStatus(payload);
        updateStatus(payload.message || "Recovered auto-label job");
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        if (error.message.includes("404")) {
          window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
          return;
        }
        logMessage("error", `Job restore failed: ${error.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    request("/jobs/auto-label/models")
      .then((payload) => {
        const nextProfiles = (payload.profiles || []).filter((profile) => profile.is_available);
        if (!nextProfiles.length) {
          setModelProfiles([]);
          setSelectedModelProfileId("");
          logMessage("warn", "No available auto-label models were reported by the worker.");
          return;
        }
        setModelProfiles(nextProfiles);
        const nextOpenpcdetProfiles = nextProfiles.filter((profile) => profile.provider !== "yolo");
        const nextYolo2dDetectionProfiles = nextProfiles.filter((profile) => profile.provider === "yolo" && profile.task === "2d_object_detection");
        const nextYolo2dSegmentationProfiles = nextProfiles.filter((profile) => profile.provider === "yolo" && profile.task === "2d_semantic_segmentation");
        setSelectedModelProfileId((current) => {
          if (nextOpenpcdetProfiles.some((profile) => profile.id === current)) {
            return current;
          }
          if (payload.default_model_profile_id && nextOpenpcdetProfiles.some((profile) => profile.id === payload.default_model_profile_id)) {
            return payload.default_model_profile_id;
          }
          return nextOpenpcdetProfiles[0]?.id || "";
        });
        setSelected2dDetectionModelProfileId((current) => {
          if (nextYolo2dDetectionProfiles.some((profile) => profile.id === current)) {
            return current;
          }
          return nextYolo2dDetectionProfiles[0]?.id || "";
        });
        setSelected2dSegmentationModelProfileId((current) => {
          if (nextYolo2dSegmentationProfiles.some((profile) => profile.id === current)) {
            return current;
          }
          return nextYolo2dSegmentationProfiles[0]?.id || "";
        });
        logMessage("success", `Loaded ${nextProfiles.length} auto-label model${nextProfiles.length === 1 ? "" : "s"}.`);
      })
      .catch((error) => logMessage("error", `Model catalog load failed: ${error.message}`));
  }, []);

  useEffect(() => {
    if (!selectedModelProfile?.default_score_threshold && selectedModelProfile?.default_score_threshold !== 0) {
      return;
    }
    setScoreThreshold(clampScoreThreshold(selectedModelProfile.default_score_threshold));
  }, [selectedModelProfile]);

  useEffect(() => {
    setHiddenClassNames([]);
  }, [currentSequenceId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidth));
  }, [sidebarWidth]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(RIGHT_PANEL_WIDTH_STORAGE_KEY, String(rightPanelWidth));
  }, [rightPanelWidth]);

  useEffect(() => {
    const handlePointerMove = (event) => {
      if (!sidebarResizeStateRef.current.active) {
        if (rightPanelResizeStateRef.current.active) {
          const nextWidth = clampRightPanelWidth(
            rightPanelResizeStateRef.current.shellRight - event.clientX - rightPanelResizeStateRef.current.pointerOffset
          );
          setRightPanelWidth(nextWidth);
        }
        return;
      }
      const nextWidth = clampSidebarWidth(event.clientX - sidebarResizeStateRef.current.pointerOffset);
      setSidebarWidth(nextWidth);
    };

    const handlePointerUp = () => {
      if (!sidebarResizeStateRef.current.active) {
        if (!rightPanelResizeStateRef.current.active) {
          return;
        }
        rightPanelResizeStateRef.current.active = false;
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        return;
      }
      sidebarResizeStateRef.current.active = false;
      rightPanelResizeStateRef.current.active = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };

    window.addEventListener("mousemove", handlePointerMove);
    window.addEventListener("mouseup", handlePointerUp);
    return () => {
      window.removeEventListener("mousemove", handlePointerMove);
      window.removeEventListener("mouseup", handlePointerUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, []);

  useEffect(() => {
    if (!currentSequenceId) {
      setRamapCacheStatus(null);
      return;
    }
    request(`/sequences/${currentSequenceId}/timeline`)
      .then((payload) => {
        setTimeline(payload);
        setStartFrame(0);
        setEndFrame(Math.max(0, payload.frames.length - 1));
        setFrameIndex((current) => Math.min(current, Math.max(0, payload.frames.length - 1)));
        updateStatus(`Timeline loaded for ${payload.sequence.name}.`);
      })
      .catch((error) => logMessage("error", `Timeline load failed: ${error.message}`));
  }, [currentSequenceId]);

  useEffect(() => {
    if (!currentSequenceId) {
      setRamapCacheStatus(null);
      return undefined;
    }

    let cancelled = false;
    let timerId = null;

    const poll = async () => {
      try {
        const payload = await request(`/sequences/${currentSequenceId}/ramap-cache`);
        if (cancelled) {
          return;
        }
        const nextStatus = payload?.available ? payload : null;
        setRamapCacheStatus(nextStatus);
        if (nextStatus?.running) {
          timerId = window.setTimeout(poll, 1500);
        }
      } catch (error) {
        if (!cancelled) {
          setRamapCacheStatus(null);
        }
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timerId) {
        window.clearTimeout(timerId);
      }
    };
  }, [currentSequenceId]);

  useEffect(() => {
    refreshTrackingReadiness().catch(() => null);
  }, [currentSequenceId, startFrame, endFrame, jobStatus?.id, jobStatus?.status]);

  async function refreshFrame(sequenceId, nextFrameIndex) {
    const requestId = frameRequestIdRef.current + 1;
    frameRequestIdRef.current = requestId;
    frameAbortControllerRef.current?.abort();
    const controller = new AbortController();
    frameAbortControllerRef.current = controller;
    setIsFrameLoading(true);
    setLidarSemanticLabels(null);
    try {
      const detail = await request(`/sequences/${sequenceId}/frames/${nextFrameIndex}`, {
        signal: controller.signal
      });
      let nextPoints = createEmptyPointCloud();
      if (detail.asset_urls.lidar) {
        const lidarBuffer = await request(detail.asset_urls.lidar, {
          headers: {},
          signal: controller.signal
        });
        nextPoints = bufferToPointCloud(lidarBuffer);
      }
      if (controller.signal.aborted || frameRequestIdRef.current !== requestId) {
        return;
      }
      setFrameDetail(detail);
      setPoints(nextPoints);
      setSelectedAnnotationId((current) =>
        detail.annotations.some((annotation) => annotation.id === current) ? current : detail.annotations[0]?.id || null
      );
      updateStatus(`Frame ${nextFrameIndex} loaded.`);
    } catch (error) {
      if (error?.name === "AbortError") {
        return;
      }
      throw error;
    } finally {
      if (frameAbortControllerRef.current === controller) {
        frameAbortControllerRef.current = null;
      }
      if (frameRequestIdRef.current === requestId) {
        setIsFrameLoading(false);
      }
    }
  }

  useEffect(() => {
    if (!currentSequenceId || !timeline) {
      return;
    }
    refreshFrame(currentSequenceId, frameIndex).catch((error) => logMessage("error", `Frame load failed: ${error.message}`));
  }, [currentSequenceId, frameIndex, timeline]);

  useEffect(() => {
    setLayerModes((current) => {
      const next = { ...current };
      ["lidar", "camera", "radar"].forEach((sensor) => {
        const currentMode = current[sensor] || defaultLayerModes[sensor];
        const capability = layerCapabilities[sensor];
        const defaultMode = defaultLayerModes[sensor];
        if (!isLayerModeAvailable(capability, currentMode)) {
          next[sensor] = defaultMode;
          return;
        }
        if (!layerModeTouchedRef.current[sensor] && currentMode !== defaultMode) {
          next[sensor] = defaultMode;
          return;
        }
        next[sensor] = currentMode;
      });
      return next.lidar === current.lidar && next.camera === current.camera && next.radar === current.radar
        ? current
        : next;
    });
  }, [defaultLayerModes, layerCapabilities]);

  useEffect(() => {
    const wasRunning = previousRamapCacheRunningRef.current;
    const isRunning = Boolean(ramapCacheStatus?.running);
    previousRamapCacheRunningRef.current = isRunning;
    if (!currentSequenceId || !timeline || !wasRunning || isRunning || ramapCacheStatus?.error) {
      return;
    }
    logMessage("success", "RAMap intensity cache is ready.");
    refreshFrame(currentSequenceId, frameIndex).catch((error) =>
      logMessage("error", `Frame refresh after RAMap cache failed: ${error.message}`)
    );
  }, [currentSequenceId, frameIndex, ramapCacheStatus?.error, ramapCacheStatus?.running, timeline]);

  useEffect(() => {
    setLidarSemanticLabels(null);
    const semanticUrl = frameDetail?.asset_urls?.semantic?.lidar || null;
    if (layerModes.lidar !== "semantic" || !semanticUrl) {
      return undefined;
    }

    const controller = new AbortController();
    request(semanticUrl, {
      headers: {},
      signal: controller.signal
    })
      .then((buffer) => {
        if (!controller.signal.aborted) {
          setLidarSemanticLabels(bufferToSemanticLabels(buffer));
        }
      })
      .catch((error) => {
        if (error?.name !== "AbortError") {
          logMessage("error", `LiDAR semantic labels failed to load: ${error.message}`);
        }
      });

    return () => controller.abort();
  }, [frameDetail?.asset_urls?.semantic?.lidar, layerModes.lidar]);

  useEffect(() => {
    setSelectedAnnotationId((current) => {
      if (!visibleAnnotations.length) {
        return null;
      }
      return visibleAnnotations.some((annotation) => annotation.id === current)
        ? current
        : visibleAnnotations[0].id;
    });
  }, [visibleAnnotations]);

  useEffect(() => {
    if (!isPlaying || !timeline?.sequence?.fps || isFrameLoading) {
      return;
    }
    const interval = Math.max(50, Math.round(1000 / timeline.sequence.fps));
    const timer = window.setTimeout(() => {
      setFrameIndex((current) => {
        if (!timeline || current >= timeline.frames.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, interval);
    return () => window.clearTimeout(timer);
  }, [isPlaying, frameIndex, timeline, isFrameLoading]);

  useEffect(() => {
    if (!jobStatus?.id || !ACTIVE_JOB_STATUSES.includes(jobStatus.status) || jobSocketConnected) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await request(`/jobs/${jobStatus.id}`);
        setJobStatus(nextJob);
      } catch (error) {
        logMessage("error", `Job polling failed: ${error.message}`);
      }
    }, 500);
    return () => window.clearInterval(timer);
  }, [jobStatus?.id, jobStatus?.status, jobSocketConnected]);

  useEffect(() => {
    if (!jobStatus?.id || !ACTIVE_JOB_STATUSES.includes(jobStatus.status)) {
      if (jobSocketRef.current) {
        jobSocketRef.current.close();
        jobSocketRef.current = null;
      }
      setJobSocketConnected(false);
      return;
    }
    const socket = new window.WebSocket(getJobSocketUrl(jobStatus.id));
    jobSocketRef.current = socket;
    setJobSocketConnected(false);
    socket.onopen = () => {
      if (jobSocketRef.current === socket) {
        setJobSocketConnected(true);
      }
    };
    socket.onmessage = (event) => {
      try {
        const nextJob = JSON.parse(event.data);
        setJobStatus(nextJob);
      } catch (error) {
        console.error("Failed to parse job event", error);
      }
    };
    socket.onerror = () => {
      if (jobSocketRef.current === socket) {
        setJobSocketConnected(false);
      }
    };
    socket.onclose = () => {
      if (jobSocketRef.current === socket) {
        jobSocketRef.current = null;
      }
      setJobSocketConnected(false);
    };
    return () => {
      if (jobSocketRef.current === socket) {
        jobSocketRef.current = null;
      }
      setJobSocketConnected(false);
      socket.close();
    };
  }, [jobStatus?.id, jobStatus?.status]);

  useEffect(
    () => () => {
      frameAbortControllerRef.current?.abort();
      for (const timerId of updateTimersRef.current.values()) {
        if (timerId) {
          window.clearTimeout(timerId);
        }
      }
      updateTimersRef.current.clear();
      pendingPatchesRef.current.clear();
    },
    []
  );

  useEffect(() => {
    if (!jobStatus) {
      window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
      previousJobRef.current = null;
      return;
    }
    if (ACTIVE_JOB_STATUSES.includes(jobStatus.status)) {
      window.localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, jobStatus.id);
    } else {
      window.localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    }
    const previousJob = previousJobRef.current;
    previousJobRef.current = jobStatus;
    if (jobStatus.error) {
      logMessage("error", `${jobStatus.message}: ${jobStatus.error}`);
    } else if (!previousJob || previousJob.status !== jobStatus.status) {
      const level =
        jobStatus.status === "completed"
          ? "success"
          : jobStatus.status === "failed"
            ? "error"
            : jobStatus.status === "cancelled"
              ? "warn"
              : "info";
      logMessage(level, jobStatus.message || `Job ${jobStatus.status}`);
    } else if (jobStatus.message) {
      updateStatus(jobStatus.message);
    }
    if (
      REFRESHABLE_JOB_STATUSES.includes(jobStatus.status) &&
      previousJob?.status !== jobStatus.status &&
      currentSequenceId === jobStatus.sequence_id
    ) {
      refreshFrame(jobStatus.sequence_id, frameIndex).catch(() => {});
    }
  }, [jobStatus, currentSequenceId, frameIndex]);

  async function runAutoLabelTask({ profile, taskName, launchType, label }) {
    if (!currentSequenceId || !profile || !isAutoLabelingMode || hasActiveJob || jobLaunchLockRef.current) {
      return;
    }
    jobLaunchLockRef.current = true;
    setJobLaunchState(
      createJobLaunchState(
        launchType,
        `Preparing ${label} job...`,
        `Queueing frames ${startFrame}-${endFrame}.`
      )
    );
    updateStatus(`Preparing ${label} job...`);
    try {
      const payload = await request("/jobs/auto-label", {
        method: "POST",
        body: JSON.stringify({
          sequence_id: currentSequenceId,
          start_frame: startFrame,
          end_frame: endFrame,
          model_name: profile.name,
          model_profile_id: profile.id,
          score_threshold: scoreThreshold,
          task_name: taskName
        })
      });
      setJobLaunchState(null);
      setJobStatus(payload);
      updateStatus(payload.message || `${label} job started`);
    } catch (error) {
      setJobLaunchState(null);
      logMessage("error", `${label} failed to start: ${error.message}`);
    } finally {
      jobLaunchLockRef.current = false;
    }
  }

  async function runAutoLabel() {
    return runAutoLabelTask({
      profile: selectedModelProfile,
      taskName: "3D Bounding Box",
      launchType: "auto-label",
      label: selectedModelProfile?.name || "3D object detection"
    });
  }

  async function run2dObjectDetection() {
    return runAutoLabelTask({
      profile: selected2dDetectionModelProfile,
      taskName: "2D Object Detection",
      launchType: "2d-detection",
      label: selected2dDetectionModelProfile?.name || "2D object detection"
    });
  }

  async function run2dSemanticSegmentation() {
    return runAutoLabelTask({
      profile: selected2dSegmentationModelProfile,
      taskName: "2D Semantic Segmentation",
      launchType: "2d-segmentation",
      label: selected2dSegmentationModelProfile?.name || "2D semantic segmentation"
    });
  }

  async function runTracking() {
    if (!currentSequenceId || !isAutoLabelingMode || hasActiveJob || jobLaunchLockRef.current) {
      return;
    }
    jobLaunchLockRef.current = true;
    setJobLaunchState(
      createJobLaunchState(
        "tracking",
        "Preparing tracking job...",
        "Saving pending annotation edits."
      )
    );
    updateStatus("Preparing tracking job...");
    try {
      setJobLaunchState(
        createJobLaunchState(
          "tracking",
          "Preparing tracking job...",
          "Saving pending annotation edits."
        )
      );
      await flushPendingAnnotationUpdates();
      setJobLaunchState(
        createJobLaunchState(
          "tracking",
          "Preparing tracking job...",
          "Checking whether the selected frame range is ready."
        )
      );
      const readiness = await refreshTrackingReadiness();
      if (!readiness?.ready) {
        setJobLaunchState(null);
        logMessage("warn", readiness?.message || "Tracking is locked until frame-wise detection finishes for the selected range.");
        return;
      }
      setJobLaunchState(
        createJobLaunchState(
          "tracking",
          "Preparing tracking job...",
          "Submitting the tracking request to the API."
        )
      );
      const payload = await request("/jobs/tracking", {
        method: "POST",
        body: JSON.stringify({
          sequence_id: currentSequenceId,
          start_frame: startFrame,
          end_frame: endFrame,
          method: DEFAULT_TRACKING_METHOD,
          input_score_threshold:
            trackingInputScoreThreshold == null
              ? null
              : Math.max(0, Math.min(1, trackingInputScoreThreshold)),
          max_missed_frames: Math.max(0, Math.round(trackingMaxMissedFrames)),
          association_distance_m: Math.max(0.1, trackingAssociationDistanceM),
          process_noise_pos: Math.max(0.001, trackingProcessNoisePos),
          process_noise_size: Math.max(0.001, trackingProcessNoiseSize),
          process_noise_vel: Math.max(0.001, trackingProcessNoiseVel),
          measurement_noise_pos: Math.max(0.001, trackingMeasurementNoisePos),
          measurement_noise_size: Math.max(0.001, trackingMeasurementNoiseSize),
          yaw_smoothing_alpha: Math.max(0, Math.min(1, trackingYawSmoothingAlpha))
        })
      });
      setJobLaunchState(null);
      setJobStatus(payload);
      updateStatus(payload.message || "Tracking job started");
    } catch (error) {
      setJobLaunchState(null);
      logMessage("error", `Tracking failed to start: ${error.message}`);
    } finally {
      jobLaunchLockRef.current = false;
    }
  }

  async function pauseAutoLabel() {
    if (!jobStatus?.id) {
      return;
    }
    try {
      const payload = await request(`/jobs/${jobStatus.id}/pause`, { method: "POST" });
      setJobStatus(payload);
    } catch (error) {
      logMessage("error", `Pause failed: ${error.message}`);
    }
  }

  async function resumeAutoLabel() {
    if (!jobStatus?.id) {
      return;
    }
    try {
      const payload = await request(`/jobs/${jobStatus.id}/resume`, { method: "POST" });
      setJobStatus(payload);
    } catch (error) {
      logMessage("error", `Resume failed: ${error.message}`);
    }
  }

  async function cancelAutoLabel() {
    if (!jobStatus?.id) {
      return;
    }
    try {
      const payload = await request(`/jobs/${jobStatus.id}/cancel`, { method: "POST" });
      setJobStatus(payload);
    } catch (error) {
      logMessage("error", `Stop failed: ${error.message}`);
    }
  }

  function queueAnnotationPatch(annotationId, patch) {
    if (!currentSequenceId || frameDetail == null || !isAnnotationMode) {
      return;
    }
    pendingPatchesRef.current.set(annotationId, {
      ...(pendingPatchesRef.current.get(annotationId) || {}),
      ...patch
    });
    setFrameDetail((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        annotations: current.annotations.map((annotation) =>
          annotation.id === annotationId ? { ...annotation, ...patch } : annotation
        )
      };
    });
    const existingTimer = updateTimersRef.current.get(annotationId);
    if (existingTimer) {
      window.clearTimeout(existingTimer);
    }
    const nextTimer = window.setTimeout(async () => {
      try {
        const pendingPatch = pendingPatchesRef.current.get(annotationId);
        if (!pendingPatch) {
          return;
        }
        await request(`/annotations/${currentSequenceId}/${frameIndex}/${annotationId}`, {
          method: "PATCH",
          body: JSON.stringify(pendingPatch)
        });
        pendingPatchesRef.current.delete(annotationId);
      } catch (error) {
        logMessage("error", `Annotation update failed: ${error.message}`);
      } finally {
        updateTimersRef.current.delete(annotationId);
      }
    }, 250);
    updateTimersRef.current.set(annotationId, nextTimer);
  }

  async function createAnnotation() {
    if (!currentSequenceId || !isAnnotationMode) {
      return;
    }
    try {
      const annotation = await request(`/annotations/${currentSequenceId}/frames/${frameIndex}`, {
        method: "POST",
        body: JSON.stringify({})
      });
      setFrameDetail((current) =>
        current ? { ...current, annotations: [...current.annotations, annotation] } : current
      );
      setSelectedAnnotationId(annotation.id);
      refreshTrackingReadiness().catch(() => null);
      logMessage("success", `Created annotation ${annotation.id}.`);
    } catch (error) {
      logMessage("error", `Annotation creation failed: ${error.message}`);
    }
  }

  async function deleteSelectedAnnotation() {
    if (!currentSequenceId || !selectedAnnotationId || !isAnnotationMode) {
      return;
    }
    try {
      await request(`/annotations/${currentSequenceId}/${frameIndex}/${selectedAnnotationId}`, { method: "DELETE" });
      setFrameDetail((current) =>
        current
          ? {
              ...current,
              annotations: current.annotations.filter((annotation) => annotation.id !== selectedAnnotationId)
            }
          : current
      );
      refreshTrackingReadiness().catch(() => null);
      logMessage("warn", `Deleted annotation ${selectedAnnotationId}.`);
      setSelectedAnnotationId(null);
    } catch (error) {
      logMessage("error", `Annotation delete failed: ${error.message}`);
    }
  }

  async function saveSequence() {
    if (!currentSequenceId || !isAnnotationMode) {
      return;
    }
    try {
      await persistAnnotationProgress("Annotations saved.");
      refreshTrackingReadiness().catch(() => null);
    } catch (error) {
      logMessage("error", `Save failed: ${error.message}`);
    }
  }

  return (
    <div className="workspace-page">
      <TopMenuBar
        activeMenu={activeMenu}
        onSelectMenu={handleMenuSelect}
        isBusy={isMenuBusy}
      />
      <div className="app-shell" style={appShellStyle} ref={appShellRef}>
        <aside className="left-panel">
          <WorkspaceSidebar
            activeMenu={activeMenu}
            capabilityReport={capabilityReport}
            hasWorkspace={Boolean(currentSequenceId)}
            sequences={sequences}
            currentSequenceId={currentSequenceId}
            onSelectSequence={setCurrentSequenceId}
            sequence={currentSequence}
            classNames={currentSequence?.class_names || []}
            visualizationClassNames={visualizationClassNames}
            frameIndex={frameIndex}
            totalFrames={timeline?.frames.length || 0}
            startFrame={startFrame}
            endFrame={endFrame}
            setStartFrame={setStartFrame}
            setEndFrame={setEndFrame}
            isAnnotationMode={isAnnotationMode}
            modelProfiles={openpcdetModelProfiles}
            selectedModelProfile={selectedModelProfile}
            selectedModelProfileId={selectedModelProfileId}
            onSelectModelProfile={setSelectedModelProfileId}
            yolo2dDetectionModelProfiles={yolo2dDetectionModelProfiles}
            selected2dDetectionModelProfile={selected2dDetectionModelProfile}
            selected2dDetectionModelProfileId={selected2dDetectionModelProfileId}
            onSelect2dDetectionModelProfile={setSelected2dDetectionModelProfileId}
            yolo2dSegmentationModelProfiles={yolo2dSegmentationModelProfiles}
            selected2dSegmentationModelProfile={selected2dSegmentationModelProfile}
            selected2dSegmentationModelProfileId={selected2dSegmentationModelProfileId}
            onSelect2dSegmentationModelProfile={setSelected2dSegmentationModelProfileId}
            scoreThreshold={scoreThreshold}
            onScoreThresholdChange={(value) => setScoreThreshold(clampScoreThreshold(value))}
            filterTarget={filterTarget}
            onFilterTargetChange={setFilterTarget}
            bbxVisibility={bbxVisibility}
            bbxAvailability={bbxAvailability}
            onToggleBbxVisibility={toggleBbxVisibility}
            onRunAutoLabel={runAutoLabel}
            onRun2dObjectDetection={run2dObjectDetection}
            onRun2dSemanticSegmentation={run2dSemanticSegmentation}
            onRunTracking={runTracking}
            trackingInputScoreThreshold={trackingInputScoreThreshold}
            onTrackingInputScoreThresholdChange={setTrackingInputScoreThreshold}
            trackingMaxMissedFrames={trackingMaxMissedFrames}
            onTrackingMaxMissedFramesChange={setTrackingMaxMissedFrames}
            trackingAssociationDistanceM={trackingAssociationDistanceM}
            onTrackingAssociationDistanceMChange={setTrackingAssociationDistanceM}
            trackingProcessNoisePos={trackingProcessNoisePos}
            onTrackingProcessNoisePosChange={setTrackingProcessNoisePos}
            trackingProcessNoiseSize={trackingProcessNoiseSize}
            onTrackingProcessNoiseSizeChange={setTrackingProcessNoiseSize}
            trackingProcessNoiseVel={trackingProcessNoiseVel}
            onTrackingProcessNoiseVelChange={setTrackingProcessNoiseVel}
            trackingMeasurementNoisePos={trackingMeasurementNoisePos}
            onTrackingMeasurementNoisePosChange={setTrackingMeasurementNoisePos}
            trackingMeasurementNoiseSize={trackingMeasurementNoiseSize}
            onTrackingMeasurementNoiseSizeChange={setTrackingMeasurementNoiseSize}
            trackingYawSmoothingAlpha={trackingYawSmoothingAlpha}
            onTrackingYawSmoothingAlphaChange={setTrackingYawSmoothingAlpha}
            onPauseAutoLabel={pauseAutoLabel}
            onResumeAutoLabel={resumeAutoLabel}
            onCancelAutoLabel={cancelAutoLabel}
            jobStatus={jobStatus}
            jobLaunchState={jobLaunchState}
            hasActiveJob={hasActiveJob}
            trackingReadiness={trackingReadiness}
            annotations={visibleAnnotations}
            totalAnnotationCount={currentAnnotations.length}
            hiddenAnnotationCount={hiddenAnnotationCount}
            selectedAnnotationId={selectedAnnotationId}
            onSelectAnnotation={setSelectedAnnotationId}
            onUpdateAnnotation={queueAnnotationPatch}
            onCreateAnnotation={createAnnotation}
            onDeleteAnnotation={deleteSelectedAnnotation}
            onSaveSequence={saveSequence}
            selectedAnnotation={selectedAnnotation}
            transformMode={transformMode}
            onChangeTransformMode={setTransformMode}
            currentDataset={currentDataset}
            selectedDatasetId={selectedDatasetId}
            selectedSceneId={selectedSceneId}
            currentScene={currentScene}
            datasetCatalog={datasetCatalog}
            onSelectDataset={handleDatasetSelection}
            onSelectScene={handleSceneSelection}
            onLoadSelectedDataset={() => loadSelectedDataset()}
            layerModes={layerModes}
            layerCapabilities={layerCapabilities}
            intensityAvailable={intensityAvailable}
            semanticAvailable={semanticAvailable}
            ramapCacheStatus={ramapCacheStatus}
            onLayerModeChange={handleLayerModeChange}
            hiddenClassNames={hiddenClassNames}
            onToggleClassVisibility={toggleClassVisibility}
            onShowAllClasses={() => setHiddenClassNames([])}
            isDatasetLoading={isDatasetLoading}
            lastLoadedLabel={lastLoadedAt ? new Date(lastLoadedAt).toLocaleTimeString() : null}
            onResetView={resetViewState}
            onClearLogs={clearLogs}
            logCount={terminalEntries.length}
            expandedSectionId={expandedSidebarSection}
            onToggleSection={handleSidebarSectionToggle}
          />
        </aside>
        <div
          className="panel-resizer"
          onMouseDown={handleSidebarResizeStart}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left sidebar"
        />

        <main className="center-panel">
          {currentSequenceId ? (
            <>
              <div className="viewer-panel">
                <PointCloudViewer
                  points={points}
                  annotations={drawnAnnotations}
                  classNames={visualizationClassNames}
                  selectedAnnotationId={selectedAnnotationId}
                  onSelectAnnotation={setSelectedAnnotationId}
                  onUpdateAnnotation={queueAnnotationPatch}
                  transformMode={transformMode}
                  viewResetToken={viewResetToken}
                  isAnnotationMode={isAnnotationMode}
                  layerMode={layerModes.lidar}
                  semanticLabels={lidarSemanticLabels}
                />
                {!frameDetail?.asset_urls?.lidar ? <div className="viewer-overlay-message">LiDAR unavailable for this dataset configuration.</div> : null}
              </div>
              <TimelineBar
                frameIndex={frameIndex}
                totalFrames={timeline?.frames.length || 0}
                isPlaying={isPlaying}
                onTogglePlay={() => setIsPlaying((current) => !current)}
                onFrameChange={setFrameIndex}
                fps={timeline?.sequence?.fps || 10}
              />
            </>
          ) : (
            <section className="workspace-empty-state">
              <div className="launcher-kicker">Dataset Workspace</div>
              <h1>Select a dataset and scene in File before entering the workspace.</h1>
              <p>
                The app now starts without loading data by default. Open `File`, choose one of the supported datasets and
                scenes, click `Load`, then switch to `Show`, `Auto Labeling`, or `Manual Labeling`. Status, warnings, and request errors stay
                docked in the bottom terminal.
              </p>
            </section>
          )}
        </main>

        <div
          className="panel-resizer panel-resizer-right"
          onMouseDown={handleRightPanelResizeStart}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize right media panel"
        />

        <aside className="right-panel">
          {currentSequenceId ? (
            <MediaPanel
              apiBase={API_BASE}
              cameraUrl={frameDetail?.asset_urls?.camera?.rgb || null}
              radarUrl={frameDetail?.asset_urls?.radar_preview?.heatmap || null}
              cameraSemanticUrl={frameDetail?.asset_urls?.semantic?.camera?.rgb || null}
              radarSemanticUrl={frameDetail?.asset_urls?.semantic?.radar?.heatmap || null}
              cameraLayerMode={layerModes.camera}
              radarLayerMode={layerModes.radar}
              frameDetail={frameDetail}
              annotations={drawnAnnotations}
              annotations2d={drawn2dAnnotations}
              classNames={visualizationClassNames}
            />
          ) : (
            <section className="panel-card workspace-side-empty">
              <h2>Workspace</h2>
              <p>{statusMessage}</p>
            </section>
          )}
        </aside>
      </div>
      <BottomTerminal
        entries={terminalEntries}
        isExpanded={isTerminalExpanded}
        onToggle={() => setIsTerminalExpanded((current) => !current)}
        onClear={clearLogs}
      />
    </div>
  );
}
