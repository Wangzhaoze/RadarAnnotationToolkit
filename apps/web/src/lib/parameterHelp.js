export const PARAMETER_HELP = {
  dataset:
    "Selects the dataset adapter and mounted data root. Changing it switches available scenes, classes, sensors, and calibration.",
  scene:
    "Selects the sequence/scene inside the dataset. Changing it reloads timeline frames and annotations for that scene.",
  modelWeight3d:
    "Selects the 3D detector checkpoint. Stronger matching weights improve detections; mismatched weights can produce wrong classes or box sizes.",
  modelWeight2d:
    "Selects the YOLO camera model. Different weights change 2D boxes or masks and can affect downstream camera workflows.",
  frameStart:
    "First frame in the batch range. Increasing it skips earlier frames; decreasing it processes or tracks more history.",
  frameEnd:
    "Last frame in the batch range. Increasing it processes more future frames and takes longer; decreasing it limits runtime and output.",
  trackingScoreGate:
    "Minimum detection score accepted by tracking. Higher removes weak detections but can miss objects; lower keeps more objects but may create false tracks.",
  trackingMaxMissed:
    "How many consecutive missed frames a track can be predicted. Higher bridges longer dropouts but may leave stale boxes; lower removes lost objects sooner.",
  trackingAssociationGate:
    "Maximum 3D distance in meters for matching a detection to a track. Higher tolerates fast motion but risks ID switches; lower is stricter but may split tracks.",
  trackingProcessPos:
    "Kalman process noise for position. Higher follows abrupt motion faster but is less smooth; lower is steadier but may lag moving objects.",
  trackingProcessSize:
    "Kalman process noise for box size. Higher adapts to size changes faster; lower keeps dimensions stable but may lag real changes.",
  trackingProcessVel:
    "Kalman process noise for velocity. Higher allows acceleration and turns; lower assumes steadier motion and can oversmooth dynamics.",
  trackingMeasurePos:
    "Kalman measurement noise for detected center. Higher trusts the prediction more and reduces jitter; lower follows detections more closely.",
  trackingMeasureSize:
    "Kalman measurement noise for detected size. Higher stabilizes box dimensions; lower lets size estimates react faster.",
  trackingYawSmooth:
    "Yaw smoothing factor from 0 to 1. Higher keeps previous heading and reduces flips; lower follows detected yaw more quickly.",
  filterTarget:
    "Chooses which annotation type the visibility filters affect. Changing it switches score/class filtering between available annotation layers.",
  scoreFilter:
    "Minimum score shown in the viewer. Higher hides low-confidence boxes; lower shows more detections and possible false positives.",
  annotationClass:
    "Object class label for the selected annotation. Changing it affects color, filtering, and class-specific evaluation/export meaning.",
  annotationScore:
    "Confidence score stored with the annotation. Higher means more confidence; lower may be hidden by score filters.",
  centerX:
    "Box center X in LiDAR coordinates. Increasing moves the box forward along the dataset X axis; decreasing moves it backward.",
  centerY:
    "Box center Y in LiDAR coordinates. Increasing moves the box left/right depending on the dataset convention; decreasing moves the opposite way.",
  centerZ:
    "Box center Z in LiDAR coordinates. Increasing raises the box; decreasing lowers it.",
  sizeX:
    "Box length/extent on the X axis. Increasing makes the box longer; decreasing can crop the object.",
  sizeY:
    "Box width/extent on the Y axis. Increasing makes the box wider; decreasing can crop side points.",
  sizeZ:
    "Box height/extent on the Z axis. Increasing makes the box taller; decreasing can crop vertical points.",
  yaw:
    "Box heading around the vertical Z axis in radians. Increasing rotates counterclockwise; decreasing rotates clockwise.",
  exportFormat:
    "Target output format for future exports. Different formats preserve different fields, classes, and coordinate conventions.",
  exportDestination:
    "Output folder for future exports. Changing it will decide where generated annotation files are written.",
};

export function getParameterHelp(key) {
  return key ? PARAMETER_HELP[key] || "" : "";
}
