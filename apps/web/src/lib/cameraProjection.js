const BOX_EDGE_INDICES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7]
];

const EPSILON = 1e-5;

function reshapeMatrix(values, rows, cols) {
  if (Array.isArray(values) && values.length === rows && values.every((row) => Array.isArray(row) && row.length === cols)) {
    return values.map((row) => row.map((entry) => Number(entry)));
  }
  if (Array.isArray(values) && values.length === rows * cols) {
    const matrix = [];
    for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
      matrix.push(values.slice(rowIndex * cols, rowIndex * cols + cols).map((entry) => Number(entry)));
    }
    return matrix;
  }
  return null;
}

function multiplyMatrixVector(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
}

function rotateYaw(x, y, yaw) {
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return {
    x: x * cosine - y * sine,
    y: x * sine + y * cosine
  };
}

export function getCameraProjectionCalibration(calibration) {
  const intrinsics = calibration?.camera_rgb_intrinsics || {};
  const projectionMatrix = reshapeMatrix(calibration?.camera_rgb_projection_matrix || intrinsics.P, 3, 4);
  const lidarToCameraMatrix = reshapeMatrix(calibration?.camera_rgb_lidar_to_camera, 4, 4);
  if (!projectionMatrix || !lidarToCameraMatrix) {
    return null;
  }
  return {
    projectionMatrix,
    lidarToCameraMatrix
  };
}

export function getBoxCorners(annotation) {
  const halfX = Number(annotation?.size?.x || 0) / 2;
  const halfY = Number(annotation?.size?.y || 0) / 2;
  const halfZ = Number(annotation?.size?.z || 0) / 2;
  const centerX = Number(annotation?.center?.x || 0);
  const centerY = Number(annotation?.center?.y || 0);
  const centerZ = Number(annotation?.center?.z || 0);
  const yaw = Number(annotation?.yaw || 0);

  const localCorners = [
    [halfX, halfY, halfZ],
    [halfX, -halfY, halfZ],
    [-halfX, -halfY, halfZ],
    [-halfX, halfY, halfZ],
    [halfX, halfY, -halfZ],
    [halfX, -halfY, -halfZ],
    [-halfX, -halfY, -halfZ],
    [-halfX, halfY, -halfZ]
  ];

  return localCorners.map(([x, y, z]) => {
    const rotated = rotateYaw(x, y, yaw);
    return {
      x: centerX + rotated.x,
      y: centerY + rotated.y,
      z: centerZ + z
    };
  });
}

export function projectLidarPointToImage(point, projectionMatrix, lidarToCameraMatrix) {
  const cameraPoint = multiplyMatrixVector(lidarToCameraMatrix, [point.x, point.y, point.z, 1]);
  const depth = cameraPoint[2];
  if (!Number.isFinite(depth) || depth <= EPSILON) {
    return null;
  }
  const projected = multiplyMatrixVector(projectionMatrix, cameraPoint);
  if (!Number.isFinite(projected[2]) || Math.abs(projected[2]) <= EPSILON) {
    return null;
  }
  return {
    u: projected[0] / projected[2],
    v: projected[1] / projected[2],
    depth
  };
}

export function projectAnnotationToImage(annotation, calibration) {
  const cameraCalibration = getCameraProjectionCalibration(calibration);
  if (!cameraCalibration) {
    return null;
  }

  const corners = getBoxCorners(annotation);
  const projectedCorners = corners.map((corner) =>
    projectLidarPointToImage(corner, cameraCalibration.projectionMatrix, cameraCalibration.lidarToCameraMatrix)
  );
  const visibleCorners = projectedCorners.filter(Boolean);
  if (visibleCorners.length < 2) {
    return null;
  }

  const segments = BOX_EDGE_INDICES.map(([startIndex, endIndex]) => {
    const start = projectedCorners[startIndex];
    const end = projectedCorners[endIndex];
    if (!start || !end) {
      return null;
    }
    return { start, end };
  }).filter(Boolean);

  if (!segments.length) {
    return null;
  }

  const xs = visibleCorners.map((point) => point.u);
  const ys = visibleCorners.map((point) => point.v);
  const centerProjection = projectLidarPointToImage(
    annotation.center,
    cameraCalibration.projectionMatrix,
    cameraCalibration.lidarToCameraMatrix
  );

  return {
    segments,
    bounds: {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    },
    depth: centerProjection?.depth ?? Math.min(...visibleCorners.map((point) => point.depth))
  };
}
