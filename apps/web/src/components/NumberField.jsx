export default function NumberField({ label, value, step = 0.1, onChange, disabled = false }) {
  return (
    <label className="field-row">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
