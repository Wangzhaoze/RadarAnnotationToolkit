from __future__ import annotations

import unittest

import numpy as np

from service import TrackingBox3D, TrackingConfig, TrackingFrameInput, TrackingService, Vector3


def make_box(
    annotation_id: str,
    *,
    class_name: str = "Car",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    size_x: float = 4.0,
    size_y: float = 1.8,
    size_z: float = 1.6,
    yaw: float = 0.0,
    score: float = 0.9,
) -> TrackingBox3D:
    return TrackingBox3D(
        id=annotation_id,
        class_name=class_name,
        score=score,
        center=Vector3(x=x, y=y, z=z),
        size=Vector3(x=size_x, y=size_y, z=size_z),
        yaw=yaw,
        source_model="detector",
        coord_frame="lidar",
        source_stage="detected",
    )


def make_pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> list[list[float]]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [tx, ty, tz]
    return pose.tolist()


class KalmanTrackingServiceTest(unittest.TestCase):
    def test_static_world_object_survives_one_missed_frame(self) -> None:
        service = TrackingService()
        frames = [
            TrackingFrameInput(frame_index=0, timestamp=0.0, ego_pose=make_pose(0.0), annotations=[make_box("car-0", x=10.0)]),
            TrackingFrameInput(frame_index=1, timestamp=1.0, ego_pose=make_pose(1.0), annotations=[]),
            TrackingFrameInput(frame_index=2, timestamp=2.0, ego_pose=make_pose(2.0), annotations=[make_box("car-2", x=8.0)]),
        ]

        results = service.track(frames, TrackingConfig(max_missed_frames=3))

        self.assertEqual(results[0][0].source_stage, "detected")
        self.assertEqual(results[1][0].source_stage, "predicted")
        self.assertEqual(results[2][0].source_stage, "detected")
        self.assertEqual(results[0][0].track_id, results[1][0].track_id)
        self.assertEqual(results[0][0].track_id, results[2][0].track_id)
        self.assertAlmostEqual(results[1][0].center.x, 9.0, places=3)
        self.assertAlmostEqual(results[2][0].center.x, 8.0, places=3)

    def test_noisy_measurement_is_smoothed(self) -> None:
        service = TrackingService()
        frames = [
            TrackingFrameInput(frame_index=0, timestamp=0.0, annotations=[make_box("car-0", x=0.0)]),
            TrackingFrameInput(frame_index=1, timestamp=1.0, annotations=[make_box("car-1", x=2.0)]),
        ]

        results = service.track(
            frames,
            TrackingConfig(
                process_noise_pos=0.01,
                process_noise_size=0.01,
                process_noise_vel=0.01,
                measurement_noise_pos=2.0,
            ),
        )

        self.assertEqual(results[0][0].track_id, results[1][0].track_id)
        self.assertGreater(results[1][0].center.x, 0.0)
        self.assertLess(results[1][0].center.x, 2.0)

    def test_class_association_is_separate(self) -> None:
        service = TrackingService()
        frames = [
            TrackingFrameInput(
                frame_index=0,
                timestamp=0.0,
                annotations=[
                    make_box("car-0", class_name="Car", x=0.0),
                    make_box("ped-0", class_name="Pedestrian", x=0.0, size_x=0.8, size_y=0.8, size_z=1.7),
                ],
            ),
            TrackingFrameInput(
                frame_index=1,
                timestamp=1.0,
                annotations=[
                    make_box("car-1", class_name="Car", x=0.2),
                    make_box("ped-1", class_name="Pedestrian", x=0.2, size_x=0.8, size_y=0.8, size_z=1.7),
                ],
            ),
        ]

        results = service.track(frames, TrackingConfig())
        frame0_by_class = {annotation.class_name: annotation.track_id for annotation in results[0]}
        frame1_by_class = {annotation.class_name: annotation.track_id for annotation in results[1]}

        self.assertEqual(frame0_by_class["Car"], frame1_by_class["Car"])
        self.assertEqual(frame0_by_class["Pedestrian"], frame1_by_class["Pedestrian"])
        self.assertNotEqual(frame1_by_class["Car"], frame1_by_class["Pedestrian"])

    def test_track_expires_after_max_missed_frames(self) -> None:
        service = TrackingService()
        frames = [
            TrackingFrameInput(frame_index=0, timestamp=0.0, annotations=[make_box("car-0", x=0.0)]),
            TrackingFrameInput(frame_index=1, timestamp=1.0, annotations=[]),
            TrackingFrameInput(frame_index=2, timestamp=2.0, annotations=[]),
            TrackingFrameInput(frame_index=3, timestamp=3.0, annotations=[]),
        ]

        results = service.track(frames, TrackingConfig(max_missed_frames=2))

        self.assertEqual(results[1][0].source_stage, "predicted")
        self.assertEqual(results[2][0].source_stage, "predicted")
        self.assertEqual(results[3], [])

    def test_predicted_stage_validates(self) -> None:
        payload = make_box("pred-1").model_dump()
        payload["source_stage"] = "predicted"
        annotation = TrackingBox3D.model_validate(payload)

        self.assertEqual(annotation.source_stage, "predicted")


if __name__ == "__main__":
    unittest.main()
