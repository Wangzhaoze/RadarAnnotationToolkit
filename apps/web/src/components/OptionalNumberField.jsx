export default function OptionalNumberField({
  label,
  value,
  step = 0.1,
  onChange,
  disabled = false,
  placeholder = "Auto",
  min
}) {
  return (
    <label className="field-row">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={min}
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
