import SidebarSection from "./SidebarSection";

export default function HelpMenuPanel({ logCount, expandedSectionId, onToggleSection }) {
  return (
    <>
      <SidebarSection
        id="help-workflow"
        title="Workflow Tips"
        summary="Core navigation and annotation tips"
        expanded={expandedSectionId === "help-workflow"}
        onToggle={onToggleSection}
      >
        <ul className="plain-list">
          <li>`File` is now the first stop for choosing a supported dataset and scene.</li>
          <li>`Visualize` is the inspection mode for thresholding, class filtering, and camera/radar checks.</li>
          <li>`Annotate` exposes auto-label, tracking, transform tools, and the selected 3D box parameters.</li>
          <li>Leaving `Annotate` automatically saves progress and requests a pause for any running auto-label job.</li>
          <li>The bottom terminal collects API errors, job progress, and system messages.</li>
        </ul>
      </SidebarSection>

      <SidebarSection
        id="help-system"
        title="System Status"
        summary={`Terminal backlog ${logCount}`}
        expanded={expandedSectionId === "help-system"}
        onToggle={onToggleSection}
      >
        <div className="meta-grid">
          <div>Terminal Backlog</div>
          <strong>{logCount}</strong>
        </div>
        <p className="panel-hint">Use the terminal drawer at the bottom to inspect recent API errors, job state changes, and system messages.</p>
      </SidebarSection>
    </>
  );
}
