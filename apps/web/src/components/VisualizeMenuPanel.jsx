import { getAnnotationColor, hexToRgba } from "../lib/annotationColors";
import NumberField from "./NumberField";
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

export default function VisualizeMenuPanel({
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
  hiddenClassNames,
  onToggleClassVisibility,
  onShowAllClasses,
  expandedSectionId,
  onToggleSection
}) {
  const classOptions = classNames?.length ? classNames : sequence?.class_names?.length ? sequence.class_names : [];
  const activeLayerCount = ["lidar", "camera", "radar"].filter((sensor) => layerModes?.[sensor] === "semantic").length;
  const layerSummary = activeLayerCount ? `${activeLayerCount} semantic layer${activeLayerCount === 1 ? "" : "s"}` : "Intensity layers";
  const displaySummary = `Score >= ${scoreThreshold.toFixed(2)}`;
  const filterSummary = classOptions.length
    ? `${classOptions.length - hiddenClassNames.length}/${classOptions.length} visible`
    : "No classes";

  return (
    <>
      <SidebarSection
        id="visualize-layers"
        title="Layers"
        summary={layerSummary}
        expanded={expandedSectionId === "visualize-layers"}
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
        id="visualize-display"
        title="Score Filter"
        summary={displaySummary}
        expanded={expandedSectionId === "visualize-display"}
        onToggle={onToggleSection}
      >
        <NumberField label="Score >=" value={scoreThreshold} step={0.01} onChange={onScoreThresholdChange} disabled={!hasWorkspace} />
        <p className="panel-hint">Low-score auto-label boxes are hidden from the list and 3D view. Manual boxes stay visible.</p>
        {!hasWorkspace ? <p className="panel-hint">Open File, choose a dataset scene, and load it before visualizing frames.</p> : null}
        <p className="panel-hint">Point-cloud navigation stays in the shared 3D perspective workspace.</p>
      </SidebarSection>

      <SidebarSection
        id="visualize-filters"
        title="Class Filters"
        summary={filterSummary}
        expanded={expandedSectionId === "visualize-filters"}
        onToggle={onToggleSection}
      >
        <div className="section-head">
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
        <p className="panel-hint">Use these toggles to hide selected bounding-box classes while staying in Visualize mode.</p>
      </SidebarSection>
    </>
  );
}
