import os
import argparse
import numpy as np
from tqdm import tqdm

# optional imports
try:
    from pypcd import pypcd
except:
    pypcd = None

try:
    import open3d as o3d
except:
    o3d = None


def load_pcd(file_path):
    assert pypcd is not None, "pypcd not installed"
    pc = pypcd.PointCloud.from_path(file_path)

    x = np.array(pc.pc_data['x'], dtype=np.float32)
    y = np.array(pc.pc_data['y'], dtype=np.float32)
    z = np.array(pc.pc_data['z'], dtype=np.float32)

    if 'intensity' in pc.fields:
        i = np.array(pc.pc_data['intensity'], dtype=np.float32)
    elif 'Intensity' in pc.fields:
        i = np.array(pc.pc_data['Intensity'], dtype=np.float32)
    else:
        i = np.ones_like(x)

    points = np.stack([x, y, z, i], axis=1)
    return points


def load_ply(file_path):
    assert o3d is not None, "open3d not installed"
    pcd = o3d.io.read_point_cloud(file_path)

    xyz = np.asarray(pcd.points, dtype=np.float32)

    # ply usually has no intensity
    i = np.ones((xyz.shape[0], 1), dtype=np.float32)

    points = np.hstack((xyz, i))
    return points


def load_npy(file_path):
    data = np.load(file_path)

    # 支持 (N,3) 或 (N,4+)
    if data.shape[1] == 3:
        i = np.ones((data.shape[0], 1), dtype=np.float32)
        data = np.hstack((data, i))
    else:
        data = data[:, :4]

    return data.astype(np.float32)


def convert_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pcd":
        return load_pcd(file_path)
    elif ext == ".ply":
        return load_ply(file_path)
    elif ext == ".npy":
        return load_npy(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(args.input_dir):
        for f in filenames:
            if f.endswith((".pcd", ".ply", ".npy")):
                files.append(os.path.join(root, f))

    files.sort()
    print(f"Found {len(files)} files")

    for file_path in tqdm(files):
        try:
            points = convert_file(file_path)

            out_name = os.path.basename(file_path).split(".")[0] + ".bin"
            out_path = os.path.join(args.output_dir, out_name)

            points.astype(np.float32).tofile(out_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()