import { useEffect, useRef } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { getAnnotationColor, getSemanticColor, hexToRgbFloats } from "../lib/annotationColors";

function findAnnotationMesh(object) {
  let current = object;
  while (current) {
    if (current.userData?.annotationId) {
      return current;
    }
    current = current.parent;
  }
  return null;
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach((entry) => entry?.dispose?.());
    return;
  }
  material?.dispose?.();
}

function disposeObject3D(object) {
  object.traverse((node) => {
    node.geometry?.dispose?.();
    disposeMaterial(node.material);
  });
}

function computePointBounds(points) {
  const values = points?.values;
  const count = points?.count || 0;
  if (!values || count <= 0) {
    return {
      center: new THREE.Vector3(18, 0, 0),
      extent: new THREE.Vector3(24, 24, 8),
      distance: 28
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < count; index += 1) {
    const offset = index * 4;
    const x = values[offset];
    const y = values[offset + 1];
    const z = values[offset + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  const center = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
  const extent = new THREE.Vector3(
    Math.max(maxX - minX, 8),
    Math.max(maxY - minY, 8),
    Math.max(maxZ - minZ, 4)
  );
  const distance = Math.max(18, Math.max(extent.x, extent.y) * 0.65, extent.z * 2.4);

  return { center, extent, distance };
}

function getDefaultViewPreset(bounds) {
  const { center, distance } = bounds;
  return {
    position: new THREE.Vector3(center.x - distance * 1.35, center.y - distance * 0.45, center.z + distance * 0.55),
    target: center.clone(),
    up: new THREE.Vector3(0, 0, 1)
  };
}

export default function PointCloudViewer({
  points,
  annotations,
  classNames,
  selectedAnnotationId,
  onSelectAnnotation,
  onUpdateAnnotation,
  transformMode,
  viewResetToken,
  isAnnotationMode,
  layerMode = "intensity",
  semanticLabels = null
}) {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const orbitControlsRef = useRef(null);
  const transformControlsRef = useRef(null);
  const pointCloudRef = useRef(null);
  const annotationMeshesRef = useRef({});
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const onSelectAnnotationRef = useRef(onSelectAnnotation);
  const onUpdateAnnotationRef = useRef(onUpdateAnnotation);
  const boundsRef = useRef(computePointBounds([]));
  const hasInitializedViewRef = useRef(false);
  const pointerAnchorRef = useRef(new THREE.Vector3());
  const panPlaneRef = useRef(new THREE.Plane());
  const pointerStateRef = useRef({ isMiddlePanning: false, isTransformDragging: false });

  function applyViewPreset() {
    const camera = cameraRef.current;
    const controls = orbitControlsRef.current;
    if (!camera || !controls) {
      return;
    }
    const preset = getDefaultViewPreset(boundsRef.current);
    camera.up.copy(preset.up);
    camera.position.copy(preset.position);
    controls.target.copy(preset.target);
    camera.lookAt(preset.target);
    controls.update();
  }

  useEffect(() => {
    onSelectAnnotationRef.current = onSelectAnnotation;
    onUpdateAnnotationRef.current = onUpdateAnnotation;
  }, [onSelectAnnotation, onUpdateAnnotation]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return undefined;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c1118);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 1000);
    camera.up.set(0, 0, 1);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const orbitControls = new TrackballControls(camera, renderer.domElement);
    orbitControls.rotateSpeed = 4.2;
    orbitControls.zoomSpeed = 1.3;
    orbitControls.panSpeed = 0.9;
    orbitControls.dynamicDampingFactor = 0.12;
    orbitControls.noZoom = true;
    orbitControls.noPan = true;
    orbitControls.target.set(18, 0, 0);
    orbitControls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: null,
      RIGHT: THREE.MOUSE.ROTATE
    };
    orbitControlsRef.current = orbitControls;
    applyViewPreset();

    const syncNavigationEnabled = () => {
      orbitControls.enabled = !pointerStateRef.current.isMiddlePanning && !pointerStateRef.current.isTransformDragging;
    };

    const transformControls = new TransformControls(camera, renderer.domElement);
    transformControls.setSpace("local");
    transformControls.addEventListener("dragging-changed", (event) => {
      pointerStateRef.current.isTransformDragging = event.value;
      syncNavigationEnabled();
    });
    transformControls.addEventListener("objectChange", () => {
      const target = transformControls.object;
      if (!target?.userData?.annotationId) {
        return;
      }
      onUpdateAnnotationRef.current?.(target.userData.annotationId, {
        center: {
          x: Number(target.position.x.toFixed(4)),
          y: Number(target.position.y.toFixed(4)),
          z: Number(target.position.z.toFixed(4))
        },
        size: {
          x: Number(target.scale.x.toFixed(4)),
          y: Number(target.scale.y.toFixed(4)),
          z: Number(target.scale.z.toFixed(4))
        },
        yaw: Number(target.rotation.z.toFixed(4))
      });
    });
    scene.add(transformControls);
    transformControlsRef.current = transformControls;

    const grid = new THREE.GridHelper(120, 24, 0x3a7bd5, 0x18324b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(8));
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.7);
    keyLight.position.set(8, 10, 6);
    scene.add(keyLight);

    const handleResize = () => {
      if (!mount || !rendererRef.current || !cameraRef.current) {
        return;
      }
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      orbitControls.handleResize?.();
    };
    window.addEventListener("resize", handleResize);
    const resizeObserver =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => handleResize()) : null;
    resizeObserver?.observe(mount);

    const handleClick = (event) => {
      if (!rendererRef.current || !cameraRef.current) {
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      const hits = raycasterRef.current.intersectObjects(Object.values(annotationMeshesRef.current), true);
      if (!hits.length) {
        return;
      }
      const mesh = findAnnotationMesh(hits[0].object);
      if (mesh?.userData?.annotationId) {
        onSelectAnnotationRef.current?.(mesh.userData.annotationId);
      }
    };
    const handleContextMenu = (event) => {
      event.preventDefault();
    };
    const screenToNdc = (clientX, clientY) => {
      const rect = renderer.domElement.getBoundingClientRect();
      return new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1
      );
    };
    const intersectPointerPlane = (clientX, clientY, plane, fallbackPoint) => {
      const pointer = screenToNdc(clientX, clientY);
      raycasterRef.current.setFromCamera(pointer, cameraRef.current);
      const intersection = new THREE.Vector3();
      return raycasterRef.current.ray.intersectPlane(plane, intersection)
        ? intersection.clone()
        : fallbackPoint.clone();
    };
    const getPointerAnchor = (clientX, clientY) => {
      const pointer = screenToNdc(clientX, clientY);
      raycasterRef.current.params.Points.threshold = 0.45;
      raycasterRef.current.setFromCamera(pointer, cameraRef.current);

      if (pointCloudRef.current) {
        const pointHits = raycasterRef.current.intersectObject(pointCloudRef.current, false);
        if (pointHits.length) {
          return pointHits[0].point.clone();
        }
      }

      const annotationHits = raycasterRef.current.intersectObjects(Object.values(annotationMeshesRef.current), true);
      if (annotationHits.length) {
        return annotationHits[0].point.clone();
      }

      const viewDirection = new THREE.Vector3();
      cameraRef.current.getWorldDirection(viewDirection);
      panPlaneRef.current.setFromNormalAndCoplanarPoint(viewDirection, orbitControls.target);
      return intersectPointerPlane(clientX, clientY, panPlaneRef.current, orbitControls.target);
    };
    const handleMouseDown = (event) => {
      if (event.button !== 1 || pointerStateRef.current.isTransformDragging) {
        return;
      }
      event.preventDefault();
      pointerStateRef.current.isMiddlePanning = true;
      pointerAnchorRef.current.copy(getPointerAnchor(event.clientX, event.clientY));
      const viewDirection = new THREE.Vector3();
      cameraRef.current.getWorldDirection(viewDirection);
      panPlaneRef.current.setFromNormalAndCoplanarPoint(viewDirection, pointerAnchorRef.current);
      syncNavigationEnabled();
    };
    const handleMouseMove = (event) => {
      if (!pointerStateRef.current.isMiddlePanning) {
        return;
      }
      event.preventDefault();
      const nextAnchor = intersectPointerPlane(
        event.clientX,
        event.clientY,
        panPlaneRef.current,
        pointerAnchorRef.current
      );
      const delta = pointerAnchorRef.current.clone().sub(nextAnchor);
      camera.position.add(delta);
      orbitControls.target.add(delta);
      orbitControls.update();
    };
    const stopMiddlePan = () => {
      if (!pointerStateRef.current.isMiddlePanning) {
        return;
      }
      pointerStateRef.current.isMiddlePanning = false;
      syncNavigationEnabled();
    };
    const handleWheel = (event) => {
      event.preventDefault();
      if (pointerStateRef.current.isMiddlePanning || pointerStateRef.current.isTransformDragging) {
        return;
      }
      const anchor = getPointerAnchor(event.clientX, event.clientY);
      const zoomFactor = event.deltaY < 0 ? 0.88 : 1 / 0.88;

      const nextCameraPosition = camera.position.clone().sub(anchor).multiplyScalar(zoomFactor).add(anchor);
      const nextTarget = orbitControls.target.clone().sub(anchor).multiplyScalar(zoomFactor).add(anchor);

      const nextDistance = nextCameraPosition.distanceTo(nextTarget);
      if (nextDistance < 1.2 || nextDistance > 400) {
        return;
      }

      camera.position.copy(nextCameraPosition);
      orbitControls.target.copy(nextTarget);
      orbitControls.update();
    };

    renderer.domElement.addEventListener("click", handleClick);
    renderer.domElement.addEventListener("contextmenu", handleContextMenu);
    renderer.domElement.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", stopMiddlePan);
    renderer.domElement.addEventListener("mouseleave", stopMiddlePan);
    renderer.domElement.addEventListener("wheel", handleWheel, { passive: false });

    let animationFrame = 0;
    const render = () => {
      animationFrame = window.requestAnimationFrame(render);
      orbitControls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      renderer.domElement.removeEventListener("click", handleClick);
      renderer.domElement.removeEventListener("contextmenu", handleContextMenu);
      renderer.domElement.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", stopMiddlePan);
      renderer.domElement.removeEventListener("mouseleave", stopMiddlePan);
      renderer.domElement.removeEventListener("wheel", handleWheel);
      window.removeEventListener("resize", handleResize);
      resizeObserver?.disconnect();
      transformControls.dispose();
      orbitControls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) {
      return;
    }
    if (pointCloudRef.current) {
      disposeObject3D(pointCloudRef.current);
      scene.remove(pointCloudRef.current);
      pointCloudRef.current = null;
    }
    const pointValues = points?.values;
    const pointCount = points?.count || 0;
    if (!pointValues || pointCount <= 0) {
      return;
    }
    boundsRef.current = computePointBounds(points);
    if (!hasInitializedViewRef.current) {
      applyViewPreset();
      hasInitializedViewRef.current = true;
    }

    const colors = new Float32Array(pointCount * 3);
    const color = new THREE.Color();
    const useSemanticColors = layerMode === "semantic" && semanticLabels?.length >= pointCount;
    if (useSemanticColors) {
      for (let index = 0; index < pointCount; index += 1) {
        const rgb = hexToRgbFloats(getSemanticColor(Number(semanticLabels[index]), classNames));
        colors[index * 3] = rgb.r;
        colors[index * 3 + 1] = rgb.g;
        colors[index * 3 + 2] = rgb.b;
      }
    } else {
      let maxIntensity = 1;
      for (let index = 0; index < pointCount; index += 1) {
        const intensity = pointValues[index * 4 + 3];
        if (Number.isFinite(intensity)) {
          maxIntensity = Math.max(maxIntensity, intensity);
        }
      }

      for (let index = 0; index < pointCount; index += 1) {
        const intensity = Math.min(1, Math.max(0, pointValues[index * 4 + 3] / maxIntensity));
        color.setHSL(0.6 - intensity * 0.5, 0.9, 0.55);
        colors[index * 3] = color.r;
        colors[index * 3 + 1] = color.g;
        colors[index * 3 + 2] = color.b;
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(pointValues, 4), 3, 0)
    );
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();

    const material = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      sizeAttenuation: true
    });
    const pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    pointCloudRef.current = pointCloud;
  }, [classNames, layerMode, points, semanticLabels]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) {
      return;
    }
    Object.values(annotationMeshesRef.current).forEach((mesh) => {
      disposeObject3D(mesh);
      scene.remove(mesh);
    });
    const nextMeshes = {};

    annotations.forEach((annotation) => {
      const isSelected = annotation.id === selectedAnnotationId;
      const baseColor = new THREE.Color(getAnnotationColor(annotation.class_name, classNames));
      const fillColor = isSelected
        ? baseColor.clone().lerp(new THREE.Color(0xffffff), 0.18)
        : baseColor.clone();
      const edgeColor = isSelected
        ? baseColor.clone().lerp(new THREE.Color(0xffffff), 0.42)
        : baseColor.clone().lerp(new THREE.Color(0xffffff), 0.15);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshPhongMaterial({
          color: fillColor,
          transparent: true,
          opacity: isSelected ? 0.34 : 0.24
        })
      );
      mesh.scale.set(annotation.size.x, annotation.size.y, annotation.size.z);
      mesh.position.set(annotation.center.x, annotation.center.y, annotation.center.z);
      mesh.rotation.z = annotation.yaw;
      mesh.userData.annotationId = annotation.id;

      const wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ color: edgeColor })
      );
      mesh.add(wireframe);
      scene.add(mesh);
      nextMeshes[annotation.id] = mesh;
    });

    annotationMeshesRef.current = nextMeshes;
  }, [annotations, classNames, selectedAnnotationId]);

  useEffect(() => {
    if (!transformControlsRef.current) {
      return;
    }
    transformControlsRef.current.setMode(transformMode);
  }, [transformMode]);

  useEffect(() => {
    if (!transformControlsRef.current) {
      return;
    }
    transformControlsRef.current.enabled = isAnnotationMode;
    if (!isAnnotationMode) {
      transformControlsRef.current.detach();
      return;
    }
    const mesh = annotationMeshesRef.current[selectedAnnotationId];
    if (mesh) {
      transformControlsRef.current.attach(mesh);
    } else {
      transformControlsRef.current.detach();
    }
  }, [selectedAnnotationId, annotations, isAnnotationMode]);

  useEffect(() => {
    applyViewPreset();
  }, [viewResetToken]);

  return <div className="viewer-canvas" ref={mountRef} />;
}
