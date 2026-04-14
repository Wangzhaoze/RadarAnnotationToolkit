import { useEffect, useRef, useState } from "react";
import { getAnnotationColor, hexToRgba } from "../lib/annotationColors";
import { getCameraProjectionCalibration, projectAnnotationToImage } from "../lib/cameraProjection";

const FALLBACK_FRAME_WIDTH = 1600;
const FALLBACK_FRAME_HEIGHT = 1000;

function getFrameDimensions(source) {
  return {
    width: source.width || source.naturalWidth || FALLBACK_FRAME_WIDTH,
    height: source.height || source.naturalHeight || FALLBACK_FRAME_HEIGHT
  };
}

function drawFrameToCanvas(canvas, source) {
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }

  const { width, height } = getFrameDimensions(source);
  canvas.width = width;
  canvas.height = height;
  context.clearRect(0, 0, width, height);
  context.drawImage(source, 0, 0, width, height);
  return { context, sourceWidth: width, sourceHeight: height };
}

async function loadFrameSource(src, signal) {
  const response = await fetch(src, {
    signal,
    cache: "default"
  });
  if (!response.ok) {
    throw new Error(`Failed to load ${src}`);
  }

  const blob = await response.blob();
  if ("createImageBitmap" in window) {
    const bitmap = await createImageBitmap(blob);
    return {
      source: bitmap,
      cleanup: () => bitmap.close?.()
    };
  }

  const objectUrl = URL.createObjectURL(blob);
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () =>
      resolve({
        source: image,
        cleanup: () => URL.revokeObjectURL(objectUrl)
      });
    image.onerror = (error) => {
      URL.revokeObjectURL(objectUrl);
      reject(error);
    };
    image.src = objectUrl;
  });
}

function MediaCanvas({ alt, placeholder, src }) {
  const canvasRef = useRef(null);
  const requestIdRef = useRef(0);
  const hasFrameRef = useRef(false);
  const [hasFrame, setHasFrame] = useState(false);

  const setFrameVisible = (value) => {
    hasFrameRef.current = value;
    setHasFrame(value);
  };

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    if (!src) {
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (canvas && context) {
        context.clearRect(0, 0, canvas.width || FALLBACK_FRAME_WIDTH, canvas.height || FALLBACK_FRAME_HEIGHT);
      }
      setFrameVisible(false);
      return undefined;
    }

    const controller = new AbortController();
    let cleanup = null;

    const render = async () => {
      try {
        const frame = await loadFrameSource(src, controller.signal);
        cleanup = frame.cleanup;
        if (controller.signal.aborted || requestIdRef.current !== requestId || !canvasRef.current) {
          cleanup?.();
          cleanup = null;
          return;
        }
        drawFrameToCanvas(canvasRef.current, frame.source);
        setFrameVisible(true);
        cleanup?.();
        cleanup = null;
      } catch (error) {
        if (!controller.signal.aborted && requestIdRef.current === requestId && !hasFrameRef.current) {
          setFrameVisible(false);
        }
      }
    };

    render();

    return () => {
      controller.abort();
      cleanup?.();
    };
  }, [src]);

  return (
    <div className="media-frame">
      {!hasFrame && <div className="media-placeholder">{placeholder}</div>}
      <canvas aria-label={alt} className={`media-canvas ${hasFrame ? "visible" : ""}`} ref={canvasRef} role="img" />
    </div>
  );
}

function mapImagePointToCanvas(point, metrics) {
  return {
    x: point.u,
    y: point.v
  };
}

function drawProjectedBoxes(metrics, annotations, classNames, calibration) {
  if (!metrics?.context || !annotations?.length) {
    return 0;
  }

  if (!getCameraProjectionCalibration(calibration)) {
    return 0;
  }

  const projectedAnnotations = annotations
    .map((annotation) => ({
      annotation,
      projection: projectAnnotationToImage(annotation, calibration)
    }))
    .filter((entry) => entry.projection)
    .sort((left, right) => right.projection.depth - left.projection.depth);

  projectedAnnotations.forEach(({ annotation, projection }) => {
    const color = getAnnotationColor(annotation.class_name, classNames);
    const strokeColor = color;
    const fillColor = hexToRgba(color, 0.16);
    const labelText = annotation.class_name;
    const labelX = Math.max(0, Math.min(metrics.sourceWidth, projection.bounds.minX));
    const labelY = Math.max(18, Math.min(metrics.sourceHeight, projection.bounds.minY - 8));
    const labelCanvas = mapImagePointToCanvas({ u: labelX, v: labelY }, metrics);

    metrics.context.save();
    metrics.context.lineWidth = 2.2;
    metrics.context.strokeStyle = strokeColor;
    metrics.context.fillStyle = fillColor;
    metrics.context.lineJoin = "round";
    metrics.context.lineCap = "round";

    projection.segments.forEach((segment) => {
      const start = mapImagePointToCanvas(segment.start, metrics);
      const end = mapImagePointToCanvas(segment.end, metrics);
      metrics.context.beginPath();
      metrics.context.moveTo(start.x, start.y);
      metrics.context.lineTo(end.x, end.y);
      metrics.context.stroke();
    });

    metrics.context.font = "600 14px 'Segoe UI', sans-serif";
    const textWidth = metrics.context.measureText(labelText).width;
    metrics.context.fillStyle = "rgba(7, 16, 24, 0.82)";
    metrics.context.fillRect(labelCanvas.x - 6, labelCanvas.y - 16, textWidth + 12, 20);
    metrics.context.strokeStyle = strokeColor;
    metrics.context.strokeRect(labelCanvas.x - 6, labelCanvas.y - 16, textWidth + 12, 20);
    metrics.context.fillStyle = strokeColor;
    metrics.context.fillText(labelText, labelCanvas.x, labelCanvas.y - 2);
    metrics.context.restore();
  });

  return projectedAnnotations.length;
}

function CameraProjectionCanvas({ alt, placeholder, src, annotations, classNames, calibration, onProjectionCount }) {
  const canvasRef = useRef(null);
  const frameSourceRef = useRef(null);
  const requestIdRef = useRef(0);
  const hasFrameRef = useRef(false);
  const [hasFrame, setHasFrame] = useState(false);
  const [frameRevision, setFrameRevision] = useState(0);

  const setFrameVisible = (value) => {
    hasFrameRef.current = value;
    setHasFrame(value);
  };

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    if (!src) {
      onProjectionCount?.(0);
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (canvas && context) {
        context.clearRect(0, 0, canvas.width || FALLBACK_FRAME_WIDTH, canvas.height || FALLBACK_FRAME_HEIGHT);
      }
      if (frameSourceRef.current) {
        frameSourceRef.current.cleanup?.();
        frameSourceRef.current = null;
      }
      setFrameVisible(false);
      return undefined;
    }

    const controller = new AbortController();
    let disposed = false;

    const render = async () => {
      try {
        const frame = await loadFrameSource(src, controller.signal);
        if (disposed || controller.signal.aborted || requestIdRef.current !== requestId) {
          frame.cleanup?.();
          return;
        }
        if (frameSourceRef.current) {
          frameSourceRef.current.cleanup?.();
        }
        frameSourceRef.current = frame;
        setFrameVisible(true);
        setFrameRevision((current) => current + 1);
      } catch (error) {
        if (!controller.signal.aborted && requestIdRef.current === requestId && !hasFrameRef.current) {
          setFrameVisible(false);
          onProjectionCount?.(0);
        }
      }
    };

    render();

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [src, onProjectionCount]);

  useEffect(
    () => () => {
      if (frameSourceRef.current) {
        frameSourceRef.current.cleanup?.();
        frameSourceRef.current = null;
      }
    },
    []
  );

  useEffect(() => {
    if (!frameSourceRef.current?.source || !canvasRef.current) {
      return;
    }
    const metrics = drawFrameToCanvas(canvasRef.current, frameSourceRef.current.source);
    const projectedCount = drawProjectedBoxes(metrics, annotations, classNames, calibration);
    onProjectionCount?.(projectedCount);
  }, [annotations, calibration, classNames, frameRevision, onProjectionCount]);

  return (
    <div className="media-frame">
      {!hasFrame && <div className="media-placeholder">{placeholder}</div>}
      <canvas aria-label={alt} className={`media-canvas ${hasFrame ? "visible" : ""}`} ref={canvasRef} role="img" />
    </div>
  );
}

export default function MediaPanel({
  apiBase,
  cameraUrl,
  radarUrl,
  cameraSemanticUrl = null,
  radarSemanticUrl = null,
  cameraLayerMode = "intensity",
  radarLayerMode = "intensity",
  frameDetail,
  annotations = [],
  classNames = []
}) {
  const resolvedCameraUrl = cameraLayerMode === "semantic" && cameraSemanticUrl ? cameraSemanticUrl : cameraUrl;
  const resolvedRadarUrl =
    radarLayerMode === "semantic" && radarSemanticUrl ? radarSemanticUrl : radarUrl || radarSemanticUrl;
  const cameraSrc = resolvedCameraUrl ? `${apiBase}${resolvedCameraUrl}` : null;
  const radarSrc = resolvedRadarUrl ? `${apiBase}${resolvedRadarUrl}` : null;
  const isCameraSemantic = cameraLayerMode === "semantic" && Boolean(cameraSemanticUrl);

  return (
    <div className="media-stack">
      <section className="panel-card media-card">
        <h2>Camera RGB</h2>
        {cameraSrc && isCameraSemantic ? (
          <MediaCanvas alt="Camera semantic" placeholder="No camera semantic frame" src={cameraSrc} />
        ) : cameraSrc ? (
          <CameraProjectionCanvas
            alt="Camera RGB"
            placeholder="No camera frame"
            src={cameraSrc}
            annotations={annotations}
            classNames={classNames}
            calibration={frameDetail?.calibration}
          />
        ) : (
          <div className="media-placeholder">No camera frame</div>
        )}
      </section>
      <section className="panel-card media-card">
        <h2>Radar</h2>
        {radarSrc ? (
          <MediaCanvas alt="Radar" placeholder="No radar frame" src={radarSrc} />
        ) : (
          <div className="media-placeholder">No radar frame</div>
        )}
      </section>
    </div>
  );
}
