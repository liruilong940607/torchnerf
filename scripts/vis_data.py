"""Plot functions."""
import vedo
import numpy as np


def plot_multiview(intrinsics, extrinsics, images,
                   ray_subsample=1./200, ray_near=0, ray_far=500,
                   image_dist=100):
    """Plot NeRF data in 3D: multiview images with cameras.

    Args:
        intrinsics: list of np.array [3, 3] or just one np.array [3, 3].
        extrinsics: list of np.array [3, 4].
        images: list of np.array [height, width, 3].
    """
    # intrinsics is just one np.array [3, 3], shared by all images.
    if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
        intrinsics = [intrinsics] * len(images)
    assert len(extrinsics) == len(intrinsics) == len(images), (
        "all inputs must have same length!")

    vis_list = []
    for intrin, extrin, image in zip(intrinsics, extrinsics, images):
        vis_list += plot_camera(intrin, extrin)
        vis_list += plot_rays(intrin, extrin, image.shape[0], image.shape[1],
                              ray_subsample, ray_near, ray_far)
        vis_list += plot_image(intrin, extrin, image, image_dist)
    vedo.show(*vis_list, interactive=True)


def plot_camera(intrinsic, extrinsic):
    """Plot a camera in 3D with view direction.

    Args:
        intrinsic: np.array [3, 3].
        extrinsic: np.array [3, 4].
    Returns:
        list of vedo objs to be plotted.
    """
    points_world = np.array([
        [40., 0., 0.],  # arrow x: red
        [0., 40., 0.],  # arrow y: green
        [0., 0., 40.],  # arrow z: blue
    ])
    colors = ['r', 'g', 'b']

    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3]
    position = - np.dot(np.linalg.inv(rot), trans)
    points_cam = np.einsum('ij,kj->ki', np.linalg.inv(rot), points_world)
    return [
        vedo.Arrows([position], [position + points_cam[i]]).c(colors[i])
        for i in range(3)]


def plot_rays(intrinsic, extrinsic, image_height, image_width,
              subsample=1./200, near=0, far=500):
    """Plot a set of rays based on camera and image plane.

    Args:
        intrinsic: np.array [3, 3].
        extrinsic: np.array [3, 4].
    Returns:
        list of vedo objs to be plotted.
    """
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3]
    ray_origin = - np.dot(np.linalg.inv(rot), trans)

    gridx, gridy = np.meshgrid(
        np.linspace(0, image_width, int(image_width * subsample), dtype=np.float32),
        np.linspace(0, image_height, int(image_height * subsample), dtype=np.float32),
    )
    gridx, gridy = gridx.reshape(-1), gridy.reshape(-1)

    ray_dirs = np.stack([gridx, gridy, np.ones_like(gridx)], axis=1)  # in image coords
    ray_dirs = np.einsum('ij,kj->ki', np.linalg.inv(intrinsic), ray_dirs)  # in cam coords
    ray_dirs = np.einsum('ij,kj->ki', np.linalg.inv(rot), ray_dirs)  # in world coords
    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / norms
    return [
        vedo.Line(ray_origin + near * ray_dir, ray_origin + far * ray_dir)
        for ray_dir in ray_dirs]


def plot_image(intrinsic, extrinsic, image, dist=100):
    """Plot a image in front of a camera.

    Args:
        intrinsic: np.array [3, 3].
        extrinsic: np.array [3, 4].
        image: np.array [height, width, 3].
    Returns:
        list of vedo objs to be plotted.
    """
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3]
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
    ray_origin = - np.dot(np.linalg.inv(rot), trans)

    # scale
    img_scale = dist / ((f_x + f_y) / 2)
    # position: bottom-right corner of the image
    ray_dir = np.array([image.shape[1], image.shape[0], 1], dtype=np.float32)
    ray_dir = np.dot(np.linalg.inv(intrinsic), ray_dir)
    ray_dir = np.dot(np.linalg.inv(rot), ray_dir)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    img_pos = ray_origin + dist * ray_dir * np.sqrt((c_x / f_x)**2 + (c_y / f_y)**2 + 1)
    # orientation: center of the image
    img_orient = np.dot(np.linalg.inv(rot), np.array([0, 0, 1], dtype=np.float32))
    return [vedo.Picture(image).pos(img_pos).orientation(img_orient).scale(img_scale)]



if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R

    intrinsic = np.array([
        [1600, 0, 1920 / 2],
        [0, 1600, 1080 / 2],
        [0, 0, 1],
    ], dtype=np.float32)

    extrinsics, images = [], []
    for i in range(8):
        r1 = R.from_euler('y', 180 - 360 // 8 * i, degrees=True)
        r2 = R.from_euler('z', 180, degrees=True)
        rvec = (r1 * r2).as_matrix()
        tvec = np.array([[0, 180, 500]], dtype=np.float32)
        extrinsics.append(np.concatenate([rvec, tvec.T], axis=1))
        images.append(np.zeros((1080, 1920, 3), dtype=np.uint8))

    plot_multiview(intrinsic, extrinsics, images)
