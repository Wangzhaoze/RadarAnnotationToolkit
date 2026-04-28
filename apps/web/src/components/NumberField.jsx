import ParameterLabel from "./ParameterLabel";

export default function NumberField({ label, value, step = 0.1, min, max, help, helpKey, onChange, disabled = false }) {
  return (
    <label className="field-row">
      <ParameterLabel help={help} helpKey={helpKey}>{label}</ParameterLabel>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
