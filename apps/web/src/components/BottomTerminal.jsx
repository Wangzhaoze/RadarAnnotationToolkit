function formatTimestamp(timestamp) {
  return new Date(timestamp).toLocaleTimeString();
}

export default function BottomTerminal({
  entries,
  isExpanded,
  onToggle,
  onClear
}) {
  const latestEntry = entries[entries.length - 1] || null;

  return (
    <section className={`terminal-shell ${isExpanded ? "is-expanded" : "is-collapsed"}`}>
      <div className="terminal-toolbar">
        <div className="terminal-title-group">
          <strong>Terminal</strong>
          <span>{entries.length} messages</span>
          {!isExpanded && latestEntry ? <span className="terminal-latest">{latestEntry.message}</span> : null}
        </div>
        <div className="button-row">
          <button onClick={onClear} disabled={!entries.length}>
            Clear
          </button>
          <button onClick={onToggle}>{isExpanded ? "Collapse" : "Expand"}</button>
        </div>
      </div>
      {isExpanded ? (
        <div className="terminal-log-list">
          {entries.length ? (
            entries.map((entry) => (
              <div key={entry.id} className={`terminal-entry terminal-entry-${entry.level}`}>
                <span className="terminal-time">{formatTimestamp(entry.timestamp)}</span>
                <span className="terminal-level">{entry.level.toUpperCase()}</span>
                <span className="terminal-message">{entry.message}</span>
                {entry.repeat > 1 ? <span className="terminal-repeat">x{entry.repeat}</span> : null}
              </div>
            ))
          ) : (
            <div className="terminal-empty">No messages yet.</div>
          )}
        </div>
      ) : null}
    </section>
  );
}
