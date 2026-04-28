import { getParameterHelp } from "../lib/parameterHelp";

export default function ParameterLabel({ children, help, helpKey }) {
  const text = help || getParameterHelp(helpKey);
  if (!text) {
    return <span>{children}</span>;
  }
  return (
    <span className="parameter-label" title={text} tabIndex={0} aria-label={`${children}: ${text}`}>
      {children}
    </span>
  );
}
