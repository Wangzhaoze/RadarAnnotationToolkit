import { getAnnotationColor, hexToRgba } from "../lib/annotationColors";
import NumberField from "./NumberField";
import SidebarSection from "./SidebarSection";

export default function ManualLabelingMenuPanel({
  hasWorkspace,
  classNames,
  sequence,
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
  const canAnnotateBoxes = hasWorkspace && (capabilityReport?.can_annotate_3d_box ?? true);
  const manualSummary = !hasWorkspace
    ? "Load a dataset first"
    : canAnnotateBoxes
      ? `${annotations.length}/${totalAnnotationCount} visible - ${transformMode}`
      : "Locked";

  return (
    <>
      <SidebarSection
        id="manual-labeling-3d-detection"
        title="3D Object Detection"
        summary={manualSummary}
        expanded={expandedSectionId === "manual-labeling-3d-detection"}
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
        <p className="panel-hint">Manual 3D labels use the shared LiDAR annotation schema and are saved into the active annotation set.</p>

        <div className="section-head manual-labeling-actions">
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
        {hasWorkspace && !canAnnotateBoxes ? <p className="panel-hint">3D manual labeling is unavailable until the dataset provides LiDAR frames.</p> : null}
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

      <SidebarSection
        id="manual-labeling-tracking"
        title="Tracking"
        summary="Reserved"
        expanded={expandedSectionId === "manual-labeling-tracking"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Status</div>
          <strong>Not connected</strong>
          <div>Planned</div>
          <strong>Manual track association edits</strong>
        </div>
        <button disabled>Edit Track Association</button>
        <p className="panel-hint">Manual tracking correction will be added after the track editing schema and conflict rules are defined.</p>
      </SidebarSection>

      <SidebarSection
        id="manual-labeling-2d-detection"
        title="2D Object Detection"
        summary="Reserved"
        expanded={expandedSectionId === "manual-labeling-2d-detection"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Status</div>
          <strong>Not connected</strong>
          <div>Required</div>
          <strong>2D annotation schema/API</strong>
        </div>
        <button disabled>Add 2D Box</button>
        <p className="panel-hint">Manual 2D labeling is intentionally a placeholder until 2D annotations have a first-class storage format.</p>
      </SidebarSection>
    </>
  );
}
