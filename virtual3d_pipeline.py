from colmap2nerf import colmap2Nerf
from svo_export import svo2vid
from background_substraction import bckg_subs
from virtual3d_pipeline_utils import parse_args, get_video_resolution

import os
import mimetypes
import logging
import numpy as np 
import glob
import json

def main():
    # Parse the arguments
    args, extra_args = parse_args()

    # Generate experiment folder 
    list_existing_exp = glob.glob(os.path.join(args.output, "exp*"))
    exist_exp_idx = np.zeros(len(list_existing_exp),dtype=int)
    for ii in range(len(list_existing_exp)):
        exist_exp_idx[ii] = int(list_existing_exp[ii].split("exp")[1])
    for jj in range(len(list_existing_exp)+1):
        if jj not in exist_exp_idx:
            exp_name= "exp" + str(jj)
    exp_folder = os.path.join(args.output, exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    # Generate experiment json file
    variables = {
        'general_args': str(args),
        'functional_args': str(extra_args),
    }
    json_path = os.path.join(exp_folder,'input_parameters.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(variables))

    # Read the input file
    input_path = args.input

    if os.path.isdir(input_path):
        # The input is a directory
        logging.info(f"Directory: please give the path to a video file")
 
    elif os.path.isfile(input_path):
        # The input is a file
        mimetype = mimetypes.guess_type(input_path)[0]

        if mimetype == "video/avi" or mimetype == "video/mp4":
            # The input is a video file
            logging.info(f"Video file: {input_path}")
            video_path = input_path
     
        elif mimetype == "application/octet-stream" or input_path.endswith(".svo"):
            # The input is an SVO file
            logging.info(f"SVO file: {input_path}")
            video_path = svo2vid(input_path, os.path.join(exp_folder,"aux.avi"))
            logging.info(f"SVO file converted")

        else:
            logging.warning(f"Unknown file type: {input_path}")
    else:
        logging.error(f"Invalid input: {input_path}")

    resolution = get_video_resolution(video_path)

    tranforms_json, nerf_imgs_path = colmap2Nerf(video_path, exp_folder, resolution)

    bckg_subs(nerf_imgs_path, exp_folder)

    print(f"Use {exp_folder} as input to instant-ngp: ~/virtualization-tool/instant-ngp$ ./instant-ngp ../{exp_folder}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception as e:
        logging.exception("Exception occurred", exc_info=True)
