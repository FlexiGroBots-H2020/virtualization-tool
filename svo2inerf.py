import argparse
import cv2
import numpy as np
import json
from pyzed import sl
import sys
import os

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()
    
    
def laplacian_sharpness(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = np.var(laplacian)
    return sharpness

    
def add_frame_data(frames, file_path, sharpness, transform_matrix):
    frame_data = {
        "file_path": file_path,
        "sharpness": sharpness,
        "transform_matrix": transform_matrix.tolist()
    }
    frames.append(frame_data)
    return frames
    
    
def generate_transforms_json(zed_camera: sl.Camera, frame_data: dict, out_path):
    calibration_params = zed_camera.get_camera_information().calibration_parameters
    left_cam = calibration_params.left_cam
    resolution = left_cam.image_size

    transforms = {
        "camera_angle_x": left_cam.h_fov,
        "camera_angle_y": left_cam.v_fov,
        "fl_x": left_cam.fx,
        "fl_y": left_cam.fy,
        "k1": left_cam.disto[0],
        "k2": left_cam.disto[1],
        "p1": left_cam.disto[2],
        "p2": left_cam.disto[3],
        "cx": left_cam.cx,
        "cy": left_cam.cy,
        "w": resolution.width,
        "h": resolution.height,
        "aabb_scale": 4,
        "frames": frame_data
    }
    with open(out_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    

def create_extrinsic_matrix(rotation_matrix, translation_vector):
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation_vector
    return extrinsic_matrix


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    n = w**2 + x**2 + y**2 + z**2
    if n == 0.0:
        raise ValueError("Quaternion must be non-zero.")
    
    s = 2.0 / n
    wx, wy, wz = s * x * w, s * y * w, s * z * w
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    rotation_matrix = np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)]
    ])
    
    return rotation_matrix

def get_pose_matrix(rotation, translation):
    pose_matrix = np.zeros((4, 4))
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation
    pose_matrix[3, 3] = 1
    return pose_matrix

def svo_to_transforms_json(input_svo, output_path, target_fps):
    zed = sl.Camera()
    input_params = sl.InitParameters()
    input_params.set_from_svo_file(input_svo)
    input_params.coordinate_units = sl.UNIT.METER
    err = zed.open(input_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error al abrir el archivo SVO: ", err)
        return
    
    # Enable positional tracking with default parameters
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    svo_length = zed.get_svo_number_of_frames()
    print(f"Number of frames: {svo_length}")
    
    zed_pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()
    
    # Calculate the frame skip factor
    current_fps = zed.get_camera_information().camera_fps
    skip_factor = round(current_fps / target_fps)
    print(f"original fps: {current_fps}, skip factor: {skip_factor}")

    frames_data = []
    
    # Prepare single image containers
    left_image = sl.Mat()
    output_json = os.path.join(output_path, "transforms.json")
    output_imgs = os.path.join(output_path, "images")
    if not os.path.exists(output_imgs):
        os.mkdir(output_imgs)

    num_frames_use = svo_length // skip_factor
    #num_frames_use = 300
    for i in range(num_frames_use):
        # skip frames
        for _ in range(skip_factor):
            zed.grab(runtime_parameters)
        print(f"frame id: {i}")
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Get the pose of the left eye of the camera with reference to the world frame
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            
            svo_position = zed.get_svo_position()
            
            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            
            # Generate file names
            filename = os.path.join(output_imgs , ("%s.png" % str(svo_position).zfill(6)))
            img_json_path = os.path.join("images" , ("%s.png" % str(svo_position).zfill(6)))
            # Save Left images
            cv2.imwrite(str(filename), left_image.get_data())
            
            # Display the translation and timestamp
            py_translation = sl.Translation()
            tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
            ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
            tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
            print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))
            translation_vector = (tx, ty, tz)
            
            # Display the orientation quaternion
            py_orientation = sl.Orientation()
            ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
            oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
            oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
            ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
            print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))
            
            q = (ow, ox, oy, oz)
            rotation_matrix = quaternion_to_rotation_matrix(q)
            
            world_pose_matrix = create_extrinsic_matrix(rotation_matrix, translation_vector)
            
            sharpness = laplacian_sharpness(left_image.get_data())

            frames_data = add_frame_data(frames_data, img_json_path, sharpness, world_pose_matrix)
            
            # Display progress
            progress_bar((svo_position + 1) / svo_length * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (svo_length - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    generate_transforms_json(zed, frames_data, output_json)

    # Disable positional tracking and close the camera
    zed.disable_positional_tracking();
    zed.close()

def main():
    parser = argparse.ArgumentParser(description="Convierte un archivo SVO de ZED2i en un archivo transforms.json para Nerf")
    parser.add_argument("input_svo", type=str, help="Ruta del archivo de entrada en formato .svo")
    parser.add_argument("output_path", type=str, help="Ruta del fichero de salida")
    parser.add_argument("target_fps", type=int, help="Número de frames por segundo que se desean procesar")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    svo_to_transforms_json(args.input_svo, args.output_path, args.target_fps)
    print(f"Conversión completada. Se guardó el set en {args.output_path}")

if __name__ == "__main__":
    main()
