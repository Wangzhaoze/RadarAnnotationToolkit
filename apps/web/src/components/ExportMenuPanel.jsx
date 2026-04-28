import NumberField from "./NumberField";
import ParameterLabel from "./ParameterLabel";
import SidebarSection from "./SidebarSection";

export default function ExportMenuPanel({
  hasWorkspace,
  sequence,
  startFrame,
  endFrame,
  totalFrames,
  expandedSectionId,
  onToggleSection
}) {
  const rangeSummary = hasWorkspace ? `${startFrame}-${endFrame} of ${Math.max(0, totalFrames - 1)}` : "Load a dataset first";

  return (
    <>
      <SidebarSection
        id="export-annotations"
        title="Annotations"
        summary="Reserved"
        expanded={expandedSectionId === "export-annotations"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Sequence</div>
          <strong>{sequence?.name || "-"}</strong>
          <div>Range</div>
          <strong>{rangeSummary}</strong>
          <div>Status</div>
          <strong>Not connected</strong>
        </div>
        <label className="field-stack">
          <ParameterLabel helpKey="exportFormat">Format</ParameterLabel>
          <select className="model-select" value="radar-annotation-toolkit-json" disabled>
            <option value="radar-annotation-toolkit-json">Radar Annotation Toolkit JSON</option>
            <option value="kitti">KITTI 3D</option>
            <option value="coco">COCO 2D</option>
          </select>
        </label>
        <NumberField label="Start" helpKey="frameStart" value={startFrame} step={1} disabled onChange={() => {}} />
        <NumberField label="End" helpKey="frameEnd" value={endFrame} step={1} disabled onChange={() => {}} />
        <label className="field-stack">
          <ParameterLabel helpKey="exportDestination">Destination</ParameterLabel>
          <input value="Select export folder in a future version" disabled readOnly />
        </label>
        <button disabled>Export</button>
        <p className="panel-hint">Export is reserved for batch annotation management and fixed-format dataset output. No files are written from this placeholder.</p>
      </SidebarSection>
    </>
  );
}
