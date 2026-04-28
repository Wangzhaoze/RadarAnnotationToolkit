import { getAnnotationColor, hexToRgba } from "../lib/annotationColors";
import NumberField from "./NumberField";
import ParameterLabel from "./ParameterLabel";
import SidebarSection from "./SidebarSection";

function LayerToggleRow({
  label,
  mode,
  intensityAvailable = true,
  intensityReason,
  semanticAvailable,
  semanticReason,
  disabled,
  onChange
}) {
  return (
    <div className="layer-toggle-row">
      <span>{label}</span>
      <div className="layer-toggle-group" role="group" aria-label={`${label} layer mode`}>
        <button
          className={mode === "intensity" ? "is-active" : ""}
          onClick={() => onChange("intensity")}
          disabled={disabled || !intensityAvailable}
          title={!intensityAvailable ? intensityReason : undefined}
          type="button"
        >
          Intensity
        </button>
        <button
          className={mode === "semantic" ? "is-active" : ""}
          onClick={() => onChange("semantic")}
          disabled={disabled || !semanticAvailable}
          title={!semanticAvailable ? semanticReason : undefined}
          type="button"
        >
          Semantic
        </button>
      </div>
    </div>
  );
}

const FILTER_TARGET_OPTIONS = [
  { id: "box3d", label: "3D BBX" },
  { id: "box2d", label: "2D BBX" },
  { id: "segmentation", label: "Segmentation" }
];

function BbxVisibilityRow({ label, visible, available, reason, onToggle }) {
  const stateLabel = !available ? "Unavailable" : visible ? "Visible" : "Hidden";
  return (
    <div className={`bbx-visibility-row ${visible ? "is-visible" : "is-hidden"} ${!available ? "is-unavailable" : ""}`}>
      <div>
        <strong>{label}</strong>
        <span>{stateLabel}</span>
      </div>
      <button
        type="button"
        className="bbx-eye-button"
        onClick={onToggle}
        disabled={!available}
        title={!available ? reason : visible ? `Hide ${label}` : `Show ${label}`}
        aria-label={`${stateLabel}: ${label}`}
      >
        <span className="bbx-eye-icon" aria-hidden="true" />
      </button>
    </div>
  );
}

export default function ShowMenuPanel({
  hasWorkspace,
  sequence,
  classNames,
  layerModes,
  layerCapabilities,
  intensityAvailable,
  semanticAvailable,
  ramapCacheStatus,
  onLayerModeChange,
  scoreThreshold,
  onScoreThresholdChange,
  filterTarget,
  onFilterTargetChange,
  bbxVisibility,
  bbxAvailability,
  onToggleBbxVisibility,
  hiddenClassNames,
  onToggleClassVisibility,
  onShowAllClasses,
  expandedSectionId,
  onToggleSection
}) {
  const classOptions = classNames?.length ? classNames : sequence?.class_names?.length ? sequence.class_names : [];
  const activeLayerCount = ["lidar", "camera", "radar"].filter((sensor) => layerModes?.[sensor] === "semantic").length;
  const layerSummary = activeLayerCount ? `${activeLayerCount} semantic layer${activeLayerCount === 1 ? "" : "s"}` : "Intensity layers";
  const scoreSummary = `Score >= ${scoreThreshold.toFixed(2)}`;
  const classSummary = classOptions.length
    ? `${classOptions.length - hiddenClassNames.length}/${classOptions.length} classes`
    : "No classes";
  const visibleBbxCount = ["box3d", "box2d"].filter((kind) => bbxVisibility?.[kind] && bbxAvailability?.[kind]).length;
  const bbxSummary = visibleBbxCount ? `${visibleBbxCount} visible` : "Hidden";
  const filterTargetAvailability = {
    box3d: Boolean(bbxAvailability?.box3d),
    box2d: Boolean(bbxAvailability?.box2d),
    segmentation: false
  };
  const activeFilterTarget = FILTER_TARGET_OPTIONS.find((option) => option.id === filterTarget) || FILTER_TARGET_OPTIONS[0];

  return (
    <>
      <SidebarSection
        id="show-layers"
        title="Layers"
        summary={layerSummary}
        expanded={expandedSectionId === "show-layers"}
        onToggle={onToggleSection}
      >
        <div className="layer-toggle-list">
          <LayerToggleRow
            label="LiDAR"
            mode={layerModes?.lidar || "intensity"}
            intensityAvailable={Boolean(layerCapabilities?.lidar?.intensity ?? intensityAvailable?.lidar)}
            intensityReason={layerCapabilities?.lidar?.intensityReason || "No LiDAR frame for this sample"}
            semanticAvailable={Boolean(layerCapabilities?.lidar?.semantic ?? semanticAvailable?.lidar)}
            semanticReason={layerCapabilities?.lidar?.semanticReason || "No semantic labels for this frame"}
            disabled={!hasWorkspace}
            onChange={(mode) => onLayerModeChange("lidar", mode)}
          />
          <LayerToggleRow
            label="Camera"
            mode={layerModes?.camera || "intensity"}
            intensityAvailable={Boolean(layerCapabilities?.camera?.intensity ?? intensityAvailable?.camera)}
            intensityReason={layerCapabilities?.camera?.intensityReason || "No camera frame for this sample"}
            semanticAvailable={Boolean(layerCapabilities?.camera?.semantic ?? semanticAvailable?.camera)}
            semanticReason={layerCapabilities?.camera?.semanticReason || "No semantic mask for this frame"}
            disabled={!hasWorkspace}
            onChange={(mode) => onLayerModeChange("camera", mode)}
          />
          <LayerToggleRow
            label="Radar"
            mode={layerModes?.radar || "intensity"}
            intensityAvailable={Boolean(layerCapabilities?.radar?.intensity ?? intensityAvailable?.radar)}
            intensityReason={
              layerCapabilities?.radar?.intensityReason ||
              (ramapCacheStatus?.running
                ? "RAMap intensity cache is being prepared"
                : "RAMap intensity cache is not available for this frame")
            }
            semanticAvailable={Boolean(layerCapabilities?.radar?.semantic ?? semanticAvailable?.radar)}
            semanticReason={layerCapabilities?.radar?.semanticReason || "No semantic radar layer for this frame"}
            disabled={!hasWorkspace}
            onChange={(mode) => onLayerModeChange("radar", mode)}
          />
        </div>
        {hasWorkspace && !(layerCapabilities?.radar?.intensity ?? intensityAvailable?.radar) && (layerCapabilities?.radar?.semantic ?? semanticAvailable?.radar) ? (
          <p className="panel-hint">
            Radar intensity is disabled until <code>RAMaps</code> are ready. Semantic radar is shown from <code>radar_gt</code>
            {ramapCacheStatus?.running ? ` while RAMaps build in the background (${Math.round((ramapCacheStatus.progress || 0) * 100)}%).` : "."}
          </p>
        ) : null}
        <p className="panel-hint">Semantic mode uses point-level or pixel-level labels only. 3D bounding boxes are not used as a fallback.</p>
      </SidebarSection>

      <SidebarSection
        id="show-bbxs"
        title="BBXs"
        summary={bbxSummary}
        expanded={expandedSectionId === "show-bbxs"}
        onToggle={onToggleSection}
      >
        <div className="bbx-visibility-list">
          <BbxVisibilityRow
            label="3D Bounding Box"
            visible={Boolean(bbxVisibility?.box3d)}
            available={Boolean(bbxAvailability?.box3d)}
            reason="Load a LiDAR-capable dataset before drawing 3D boxes"
            onToggle={() => onToggleBbxVisibility("box3d")}
          />
          <BbxVisibilityRow
            label="2D Bounding Box"
            visible={Boolean(bbxVisibility?.box2d)}
            available={Boolean(bbxAvailability?.box2d)}
            reason="2D object detection annotations are not available yet"
            onToggle={() => onToggleBbxVisibility("box2d")}
          />
        </div>
        <p className="panel-hint">These switches only control whether bounding boxes are drawn in the viewers. They do not delete or edit annotations.</p>
      </SidebarSection>

      <SidebarSection
        id="show-filters"
        title="Filters"
        summary={`${activeFilterTarget.label} - ${scoreSummary} - ${classSummary}`}
        expanded={expandedSectionId === "show-filters"}
        onToggle={onToggleSection}
      >
        <label className="field-row">
          <ParameterLabel helpKey="filterTarget">Target</ParameterLabel>
          <select
            className="model-select"
            value={filterTarget}
            onChange={(event) => onFilterTargetChange(event.target.value)}
            disabled={!hasWorkspace}
          >
            {FILTER_TARGET_OPTIONS.map((option) => (
              <option key={option.id} value={option.id} disabled={!filterTargetAvailability[option.id]}>
                {option.label}{filterTargetAvailability[option.id] ? "" : " (Reserved)"}
              </option>
            ))}
          </select>
        </label>
        <NumberField label="Score >=" helpKey="scoreFilter" value={scoreThreshold} step={0.01} onChange={onScoreThresholdChange} disabled={!hasWorkspace} />
        <p className="panel-hint">Filters apply to the selected annotation target. Segmentation filtering is reserved until pixel-level annotation editing is added.</p>
        {!hasWorkspace ? <p className="panel-hint">Open File, choose a dataset scene, and load it before showing frames.</p> : null}

        <div className="section-head filter-section-head">
          <strong>Classes</strong>
          <button onClick={onShowAllClasses} disabled={!classOptions.length || !hiddenClassNames.length}>
            Show All
          </button>
        </div>
        <div className="button-row">
          {classOptions.length ? (
            classOptions.map((className) => {
              const isVisible = !hiddenClassNames.includes(className);
              const classColor = getAnnotationColor(className, classOptions);
              return (
                <button
                  key={className}
                  className={`class-filter-button ${isVisible ? "is-visible" : ""}`}
                  style={{
                    borderColor: classColor,
                    background: isVisible ? hexToRgba(classColor, 0.18) : undefined,
                    color: isVisible ? classColor : undefined
                  }}
                  onClick={() => onToggleClassVisibility(className)}
                  disabled={!hasWorkspace}
                >
                  <span className="class-color-swatch" style={{ backgroundColor: classColor }} />
                  {className}
                </button>
              );
            })
          ) : (
            <span className="panel-hint">No class metadata is available yet.</span>
          )}
        </div>
        <p className="panel-hint">Use filters to inspect annotations without changing saved labels.</p>
      </SidebarSection>
    </>
  );
}
