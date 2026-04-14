export default function SidebarSection({
  id,
  title,
  summary = null,
  expanded = false,
  onToggle,
  children,
  disabled = false
}) {
  return (
    <section className={`panel-card sidebar-section ${expanded ? "is-expanded" : ""} ${disabled ? "is-disabled" : ""}`}>
      <button
        type="button"
        className="sidebar-section-toggle"
        onClick={() => onToggle(id)}
        aria-expanded={expanded}
        disabled={disabled}
      >
        <span className="sidebar-section-copy">
          <span className="sidebar-section-title">{title}</span>
          {summary ? <span className="sidebar-section-summary">{summary}</span> : null}
        </span>
        <span className="sidebar-section-caret" aria-hidden="true">
          {expanded ? "−" : "+"}
        </span>
      </button>
      {expanded ? <div className="sidebar-section-body">{children}</div> : null}
    </section>
  );
}
