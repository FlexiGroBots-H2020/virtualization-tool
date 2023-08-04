import glob
import os
import sys
sys.path.insert(0, 'X_Decoder/')

import numpy as np
import json
from PIL import Image
import cv2 

from xdcoder_utils import semseg_single_im, load_xdecoder, get_parser
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color

import mimetypes

def is_video_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path[0])
    return mime_type.startswith("video") if mime_type else False


def save_video_frames(video_path, output_dir, desired_fps=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    if video_fps < desired_fps:
        print("El FPS deseado es mayor que el FPS del video. Guardando todos los frames.")
        desired_fps = video_fps

    frame_interval = video_fps // desired_fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_dir, f"{saved_count:04d}.png")
            cv2.imwrite(frame_file, frame)
            saved_count += 1


        frame_count += 1

    cap.release()
    print(f"{saved_count} frames guardados en '{output_dir}' con {desired_fps} FPS.")


def remove_background(img_np):
    # Convertir la imagen a RGBA
    img_rgba = cv2.cvtColor(img_np, cv2.COLOR_BGR2BGRA)

    # Create a boolean mask for the pixels with RGB values (0, 0, 0)
    mask = (img_rgba[:, :, :3] == (0, 0, 0)).all(axis=2)

    # Set the alpha channel of the masked pixels to 0 (transparent)
    img_rgba[mask, 3] = 0
    return img_rgba


if __name__ == "__main__":
    
    # Set-up models and variables
    setup_logger(name="fvcore")
    logger = setup_logger()
    args = get_parser().parse_args()
    
    logger.info("Arguments: " + str(args))
    
    # Instanciate X-DECODER
    model, transform, metadata, vocabulary_xdec = load_xdecoder(args, logger)
    list_bckgrd_clss = args.bckgrd_xdec
    
    stuff_classes = args.vocabulary_xdec
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    
    if is_video_file(args.input):
        video_path = args.input[0]
        images_path = os.path.join(video_path.split('.')[0],"frames")
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        save_video_frames(video_path, images_path, desired_fps=5)
        args.input = [images_path + "/*.png"]
    
    # List all images to be proccessed
    list_images_paths = [] 
    for input in args.input:
        list_images_paths = list_images_paths + glob.glob(input)
        
            
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
        'path': exp_folder,
        'xdec_img_size': args.xdec_img_size,
        'vocabulary_xdec': args.vocabulary_xdec,
    }
    json_path = os.path.join(exp_folder,'variables.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(variables))
      
    # Proccess images
    for img_path in list_images_paths:
        # Set the paths to the images/outputs and GT data
        file_name = img_path.split('/')[-1]
        base_name = file_name.split('.')[-2]
        file_name = file_name[-10:]
                 
        output_folder = exp_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Load img in PILL and CV2
        img = Image.open(img_path).convert("RGB")
        img_ori_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        img_seg, img_bckgrd = semseg_single_im(img, transform, model, metadata, vocabulary_xdec, bckgrd_clss=list_bckgrd_clss)
        out_img = cv2.bitwise_and(img_ori_np, img_ori_np, mask=img_bckgrd.cpu().numpy().astype('int8'))
    
        cv2.imwrite(os.path.join(output_folder, "seg_"+ file_name), img_seg)
        cv2.imwrite(os.path.join(output_folder, file_name), remove_background(out_img))
    
    MetadataCatalog.remove('demo')
     
def bckg_subs(imgs_path, exp_folder=""):
    # Set-up models and variables
    setup_logger(name="fvcore")
    logger = setup_logger()
    args, _ = get_parser().parse_known_args()

    args.input = imgs_path.strip('"').strip("'")
    
    logger.info("Arguments: " + str(args))
    
    # Instanciate X-DECODER
    model, transform, metadata, vocabulary_xdec = load_xdecoder(args, logger)
    list_bckgrd_clss = args.bckgrd_xdec
    
    stuff_classes = args.vocabulary_xdec
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    
    if is_video_file(args.input):
        video_path = args.input[0]
        images_path = os.path.join(video_path.split('.')[0],"frames")
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        save_video_frames(video_path, images_path, desired_fps=5)
        args.input = [images_path + "/*.png"]
    
    # List all images to be proccessed
    list_images_paths = glob.glob(args.input + "/*.jpg") + glob.glob(args.input + "/*.png") 
        
      
    # Proccess images
    for img_path in list_images_paths:
        # Set the paths to the images/outputs and GT data
        file_name = img_path.split('/')[-1]
        base_name = file_name.split('.')[-2]
        file_name = file_name[-10:]
                 
        output_folder = os.path.join(exp_folder,"images_full")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Load img in PILL and CV2
        img = Image.open(img_path).convert("RGB")
        img_ori_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        img_seg, img_bckgrd = semseg_single_im(img, transform, model, metadata, vocabulary_xdec, bckgrd_clss=list_bckgrd_clss)
        out_img = cv2.bitwise_and(img_ori_np, img_ori_np, mask=img_bckgrd.cpu().numpy().astype('int8'))
    
        cv2.imwrite(os.path.join(output_folder, "seg_"+ file_name), img_seg)
        cv2.imwrite(os.path.join(output_folder, file_name), remove_background(out_img))
    
    MetadataCatalog.remove('demo')
    return output_folder