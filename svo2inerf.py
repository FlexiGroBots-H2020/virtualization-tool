import cv2
import numpy as np
import json
from pyzed import sl

def svo_to_transforms_json(input_svo, output_json):
    zed = sl.Camera()
    input_params = sl.InitParameters()
    input_params.set_from_svo_file(input_svo)
    input_params.coordinate_units = sl.UNIT.METER
    err = zed.open(input_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error al abrir el archivo SVO: ", err)
        return

    svo_length = zed.get_svo_number_of_frames()
    runtime = sl.RuntimeParameters()

    poses = []
    for i in range(svo_length):
        zed_pose = sl.Pose()
        zed.grab(runtime)
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

        translation = np.array(zed_pose.get_translation().get())
        rotation = np.array(zed_pose.get_rotation().get())

        pose_matrix = np.zeros((4, 4))
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = translation
        pose_matrix[3, 3] = 1

        poses.append(pose_matrix.tolist())

    with open(output_json, 'w') as f:
        json.dump(poses, f, indent=4)

    zed.close()

if __name__ == "__main__":
    input_svo = "input/HD720_SN37655529_15-46-40.svo"
    output_json = "transforms.json"
    svo_to_transforms_json(input_svo, output_json)
    print("Conversión completada. Se guardó el archivo transforms.json")