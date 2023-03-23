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
                 
        output_folder = exp_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Load img in PILL and CV2
        img = Image.open(img_path).convert("RGB")
        img_ori_np = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
        
        img_seg, img_bckgrd = semseg_single_im(img, transform, model, metadata, vocabulary_xdec, bckgrd_clss=list_bckgrd_clss)
        out_img = cv2.bitwise_and(img_ori_np, img_ori_np, mask=img_bckgrd.cpu().numpy().astype('int8'))
    
        cv2.imwrite(os.path.join(output_folder, "seg_"+ file_name), img_seg)
        cv2.imwrite(os.path.join(output_folder, file_name), out_img)
    
    
    MetadataCatalog.remove('demo')
     