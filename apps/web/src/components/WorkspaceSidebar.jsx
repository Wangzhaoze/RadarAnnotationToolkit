import AutoLabelingMenuPanel from "./AutoLabelingMenuPanel";
import ExportMenuPanel from "./ExportMenuPanel";
import FileMenuPanel from "./FileMenuPanel";
import HelpMenuPanel from "./HelpMenuPanel";
import ManualLabelingMenuPanel from "./ManualLabelingMenuPanel";
import ShowMenuPanel from "./ShowMenuPanel";

export default function WorkspaceSidebar(props) {
  const capabilityReport = props.capabilityReport || null;
  const showProps = {
    ...props,
    classNames: props.visualizationClassNames || props.classNames || []
  };

  return (
    <div className="sidebar-content">
      {props.activeMenu === "file" ? <FileMenuPanel {...props} /> : null}
      {props.activeMenu === "show" ? <ShowMenuPanel {...showProps} /> : null}
      {props.activeMenu === "auto-labeling" ? <AutoLabelingMenuPanel {...props} capabilityReport={capabilityReport} /> : null}
      {props.activeMenu === "manual-labeling" ? <ManualLabelingMenuPanel {...props} capabilityReport={capabilityReport} /> : null}
      {props.activeMenu === "export" ? <ExportMenuPanel {...props} /> : null}
      {props.activeMenu === "help" ? <HelpMenuPanel {...props} logCount={props.logCount} /> : null}
    </div>
  );
}
