const ANNOTATION_PALETTE = [
  "#35d07f",
  "#3a7bd5",
  "#ff8c42",
  "#f4d35e",
  "#ff5d8f",
  "#8f7cff",
  "#2ec4b6",
  "#f07167"
];

const SEMANTIC_CLASS_COLOR_ALIASES = {
  vehicle: "#f59664",
  car: "#f59664",
  truck: "#b41e50",
  bus: "#fa5064",
  van: "#f59664",
  pedestrian: "#1e1eff",
  person: "#1e1eff",
  bicyclist: "#c828ff",
  cyclist: "#c828ff",
  bicycle: "#f5e664",
  motorcycle: "#963c1e",
  motorcyclist: "#5a1e96",
  road: "#ff00ff",
  sidewalk: "#4b004b",
  building: "#4b2000",
  vegetation: "#00af00",
  terrain: "#96ff00",
  pole: "#96f050",
  traffic_sign: "#ff5032",
  traffic_light: "#ff5032",
  unknown: "#7f8c9a"
};

const SEMANTIC_KITTI_LABEL_COLORS = {
  0: "#000000",
  1: "#0000ff",
  10: "#f59664",
  11: "#f5e664",
  13: "#fa5064",
  15: "#963c1e",
  16: "#ff0000",
  18: "#b41e50",
  20: "#ff0000",
  30: "#1e1eff",
  31: "#c828ff",
  32: "#5a1e96",
  40: "#ff00ff",
  44: "#ff96ff",
  48: "#4b004b",
  49: "#4b0000",
  50: "#4b2000",
  51: "#f0f096",
  52: "#960000",
  60: "#ff0000",
  70: "#00af00",
  71: "#873c00",
  72: "#96ff00",
  80: "#96f050",
  81: "#ff5032",
  99: "#0000ff",
  252: "#ff0000",
  253: "#c828ff",
  254: "#1e1eff",
  255: "#5a1e96",
  256: "#ff0000",
  257: "#fa5064",
  258: "#b41e50",
  259: "#ff0000"
};

function normalizeClassName(className) {
  return String(className || "Unknown").trim();
}

function normalizeClassKey(className) {
  return normalizeClassName(className).toLowerCase().replace(/[\s-]+/g, "_");
}

function hashClassName(className) {
  let hash = 0;
  const normalized = normalizeClassName(className).toLowerCase();
  for (let index = 0; index < normalized.length; index += 1) {
    hash = (hash * 31 + normalized.charCodeAt(index)) | 0;
  }
  return Math.abs(hash);
}

export function getAnnotationColor(className, classNames = []) {
  const semanticColor = getSemanticColor(className, classNames);
  if (semanticColor) {
    return semanticColor;
  }
  const normalized = normalizeClassName(className).toLowerCase();
  const classIndex = classNames.findIndex((entry) => normalizeClassName(entry).toLowerCase() === normalized);
  const paletteIndex = classIndex >= 0 ? classIndex % ANNOTATION_PALETTE.length : hashClassName(className) % ANNOTATION_PALETTE.length;
  return ANNOTATION_PALETTE[paletteIndex];
}

export function getSemanticColor(labelOrClassName, classNames = []) {
  if (typeof labelOrClassName === "number" && Number.isFinite(labelOrClassName)) {
    const label = Math.trunc(labelOrClassName);
    if (SEMANTIC_KITTI_LABEL_COLORS[label]) {
      return SEMANTIC_KITTI_LABEL_COLORS[label];
    }
    if (classNames[label]) {
      return getSemanticColor(classNames[label], classNames);
    }
    if (label > 0 && classNames[label - 1]) {
      return getSemanticColor(classNames[label - 1], classNames);
    }
    return ANNOTATION_PALETTE[Math.abs(label) % ANNOTATION_PALETTE.length];
  }

  const key = normalizeClassKey(labelOrClassName);
  if (SEMANTIC_CLASS_COLOR_ALIASES[key]) {
    return SEMANTIC_CLASS_COLOR_ALIASES[key];
  }
  const classIndex = classNames.findIndex((entry) => normalizeClassKey(entry) === key);
  if (classIndex >= 0) {
    return ANNOTATION_PALETTE[classIndex % ANNOTATION_PALETTE.length];
  }
  return ANNOTATION_PALETTE[hashClassName(labelOrClassName) % ANNOTATION_PALETTE.length];
}

export function hexToRgbFloats(hexColor) {
  const normalized = hexColor.replace("#", "");
  const value = Number.parseInt(normalized, 16);
  return {
    r: ((value >> 16) & 255) / 255,
    g: ((value >> 8) & 255) / 255,
    b: (value & 255) / 255
  };
}

export function hexToRgba(hexColor, alpha) {
  const normalized = hexColor.replace("#", "");
  const value = Number.parseInt(normalized, 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

export function clampScoreThreshold(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

export function isAnnotationVisible(annotation, scoreThreshold) {
  if (!annotation) {
    return false;
  }
  if (annotation.source_model === "manual") {
    return true;
  }
  return Number(annotation.score || 0) >= clampScoreThreshold(scoreThreshold);
}
