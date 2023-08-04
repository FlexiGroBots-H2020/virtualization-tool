install instant-ngp: https://github.com/NVlabs/instant-ngp

docker pull ghcr.io/flexigrobots-h2020/virtualization-tool:v0

docker run it -v "$(pwd)":/wd/shared --name 3dv ghcr.io/flexigrobots-h2020/virtualization-tool:v0 --input input/vids_chema/out2.svo --xdec_img_size 512 --vocabulary_xdec vine soil ground building road sky --bckgrd_xdec building background road sky --video_shape 1280:720 --video_fps 3 --run_colmap --aabb_scale 8 --skip_early 10 --overwrite

cd intant-ngp
./instant-ngp ../output/exp0