import ParameterLabel from "./ParameterLabel";

export default function OptionalNumberField({
  label,
  value,
  step = 0.1,
  onChange,
  disabled = false,
  placeholder = "Auto",
  min,
  max,
  help,
  helpKey
}) {
  return (
    <label className="field-row">
      <ParameterLabel help={help} helpKey={helpKey}>{label}</ParameterLabel>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={value ?? ""}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => {
          const rawValue = event.target.value.trim();
          if (!rawValue) {
            onChange(null);
            return;
          }
          const numericValue = Number(rawValue);
          onChange(Number.isFinite(numericValue) ? numericValue : null);
        }}
      />
    </label>
  );
}
