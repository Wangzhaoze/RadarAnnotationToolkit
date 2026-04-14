import AnnotateMenuPanel from "./AnnotateMenuPanel";
import FileMenuPanel from "./FileMenuPanel";
import HelpMenuPanel from "./HelpMenuPanel";
import VisualizeMenuPanel from "./VisualizeMenuPanel";

export default function WorkspaceSidebar(props) {
  const capabilityReport = props.capabilityReport || null;
  const visualizeProps = {
    ...props,
    classNames: props.visualizationClassNames || props.classNames || []
  };

  return (
    <div className="sidebar-content">
      {props.activeMenu === "file" ? <FileMenuPanel {...props} /> : null}
      {props.activeMenu === "annotate" ? <AnnotateMenuPanel {...props} capabilityReport={capabilityReport} /> : null}
      {props.activeMenu === "visualize" ? <VisualizeMenuPanel {...visualizeProps} /> : null}
      {props.activeMenu === "help" ? <HelpMenuPanel {...props} logCount={props.logCount} /> : null}
    </div>
  );
}
