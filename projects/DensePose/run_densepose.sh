#python apply_net.py all configs/densepose_rcnn_R_101_FPN_s1x.yaml models/densepose_rcnn_R_101_FPN_s1x.pkl "/ssd2/swheo/db/lg_project/test/images/*.jpg" --output "/ssd2/swheo/db/lg_project/test/DensePose/result" --visualizer dp_u
#work_path="/ssd2/swheo/db/lg_project/synthetic_test/"
work_path="/ssd2/swheo/db/lg_project/test/"
python apply_net.py all configs/densepose_rcnn_R_101_FPN_s1x.yaml models/densepose_rcnn_R_101_FPN_s1x.pkl "${work_path}images/*.jpg" --output "${work_path}DensePose/result" --visualizer dp_u --gpu_id 4
python plot_IUV_from_pkl.py --input_path "${work_path}DensePose/result.pkl" --output_path "${work_path}DensePose/"
