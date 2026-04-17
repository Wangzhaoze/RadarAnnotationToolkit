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
          <li>`Show` is the inspection mode for sensor layers, score filters, and class filters.</li>
          <li>`Auto Labeling` runs model-driven tasks such as 3D object detection and tracking.</li>
          <li>`Manual Labeling` keeps hand-authored 3D box creation, transforms, and box parameter editing.</li>
          <li>`Export` is reserved for batch annotation export and fixed-format output.</li>
          <li>Leaving labeling menus automatically saves progress; leaving `Auto Labeling` also requests a pause for running 3D detection jobs.</li>
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
