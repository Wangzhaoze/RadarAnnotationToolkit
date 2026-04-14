const MENUS = [
  { id: "file", label: "File" },
  { id: "visualize", label: "Visualize" },
  { id: "annotate", label: "Annotate" },
  { id: "help", label: "Help" }
];

export default function TopMenuBar({ activeMenu, onSelectMenu, isBusy = false }) {
  return (
    <header className="top-menu-bar">
      <nav className="top-menu-tabs" aria-label="Workspace menu">
        {MENUS.map((menu) => (
          <button
            key={menu.id}
            className={`top-menu-tab ${activeMenu === menu.id ? "is-active" : ""}`}
            onClick={() => onSelectMenu(menu.id)}
            disabled={isBusy}
          >
            {menu.label}
          </button>
        ))}
      </nav>
      <div className="top-menu-brand">
        <strong>Radar Annotation Toolkit</strong>
      </div>
    </header>
  );
}
