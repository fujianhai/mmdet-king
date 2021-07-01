python /public/home/hpc8201180303/competition/round1/to_test_coco_onecls.py

./tools/dist_test.sh configs/universenet/universenet101_ppai.py work_dirs/universenet101_ppai/latest.pth 1 --eval bbox --out test_predA.pkl

python /public/home/hpc8201180303/competition/round1/test_bbox_info.py

python /public/home/hpc8201180303/competition/round1/feature_net/src_infer/extract_feature_ppd.py

python /public/home/hpc8201180303/competition/round1/feature_net/src_infer/extract_feature_ppd_effv1.py

python /public/home/hpc8201180303/competition/round1/feature_net/src_infer/extract_feature_snet200.py

python /public/home/hpc8201180303/competition/round1/feature_net/src_infer/sub_ppd_faiss_ensemble_refine.py


