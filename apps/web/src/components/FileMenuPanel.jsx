import ParameterLabel from "./ParameterLabel";
import SidebarSection from "./SidebarSection";

export default function FileMenuPanel({
  hasWorkspace,
  sequence,
  currentDataset,
  selectedDatasetId,
  selectedSceneId,
  currentScene,
  datasetCatalog,
  totalFrames,
  lastLoadedLabel,
  onSelectDataset,
  onSelectScene,
  onLoadSelectedDataset,
  isDatasetLoading,
  onResetView,
  onClearLogs,
  logCount,
  expandedSectionId,
  onToggleSection
}) {
  const datasetSummary = hasWorkspace
    ? `${currentDataset?.name || selectedDatasetId || "Dataset"} / ${sequence?.name || selectedSceneId || "Scene"}`
    : currentScene?.available
      ? `${currentDataset?.name || selectedDatasetId || "Dataset"} / ${selectedSceneId || "Scene"}`
      : "No dataset loaded";
  const workspaceSummary = hasWorkspace
    ? `${totalFrames || 0} frames${lastLoadedLabel ? ` - loaded ${lastLoadedLabel}` : ""}`
    : `Logs ${logCount}`;

  return (
    <>
      <SidebarSection
        id="file-dataset"
        title="Select Dataset"
        summary={datasetSummary}
        expanded={expandedSectionId === "file-dataset"}
        onToggle={onToggleSection}
      >
        <label className="field-stack">
          <ParameterLabel helpKey="dataset">Dataset</ParameterLabel>
          <select
            className="model-select"
            value={selectedDatasetId || ""}
            onChange={(event) => onSelectDataset(event.target.value)}
            disabled={!datasetCatalog.length || isDatasetLoading}
          >
            {datasetCatalog.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field-stack">
          <ParameterLabel helpKey="scene">Scene</ParameterLabel>
          <select
            className="model-select"
            value={selectedSceneId || ""}
            onChange={(event) => onSelectScene(event.target.value)}
            disabled={!currentDataset?.scenes?.length || isDatasetLoading}
          >
            {(currentDataset?.scenes || []).map((scene) => (
              <option key={scene.id} value={scene.id} disabled={!scene.available}>
                {scene.name}{scene.available ? "" : " (Unavailable)"}
              </option>
            ))}
          </select>
        </label>
        {currentScene && !currentScene.available ? (
          <p className="panel-hint">Unavailable: {currentScene.reason || "This scene is not available in the current mount."}</p>
        ) : null}
        <div className="meta-grid">
          <div>Current Dataset</div>
          <strong>{currentDataset?.name || "-"}</strong>
          <div>Current Scene</div>
          <strong>{sequence?.name || selectedSceneId || "-"}</strong>
          <div>Frames</div>
          <strong>{totalFrames || 0}</strong>
          <div>Last Loaded</div>
          <strong>{lastLoadedLabel || "-"}</strong>
        </div>
        {!hasWorkspace ? (
          <p className="panel-hint">Choose a supported dataset and scene here, then click Load to start the workspace.</p>
        ) : null}
        <div className="section-action-row">
          <button
            className="primary-button"
            onClick={onLoadSelectedDataset}
            disabled={isDatasetLoading || !selectedDatasetId || !selectedSceneId || !currentScene?.available}
          >
            {isDatasetLoading ? "Loading..." : "Load"}
          </button>
        </div>
      </SidebarSection>

      <SidebarSection
        id="file-tools"
        title="Workspace Tools"
        summary={workspaceSummary}
        expanded={expandedSectionId === "file-tools"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Sequence</div>
          <strong>{sequence?.name || "-"}</strong>
          <div>Total Frames</div>
          <strong>{totalFrames || 0}</strong>
          <div>Logs</div>
          <strong>{logCount}</strong>
        </div>
        <div className="button-row">
          <button onClick={onResetView} disabled={!hasWorkspace}>
            Reset View State
          </button>
          <button onClick={onClearLogs}>
            Clear Logs ({logCount})
          </button>
        </div>
        <p className="panel-hint">Use these controls to reset the 3D workspace and clear the terminal backlog.</p>
      </SidebarSection>
    </>
  );
}
