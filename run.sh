python ./code/test.py model=RCAC_augnum0_test comment=analysis_times_RCAC_aug0 \
dataroot_FSC_147=/root/dataset/FSC-147 dataroot_FSC_147_edge=/root/dataset/FSC-147/edges_gen_by_RCF \
dataset=FSC_147_edge_test train_split=[train] test_split=[val] \
num_workers=16 train_batch_size=1 test_batch_size=1 \
model_for_load=/root/RCAC/RCAC_20.94.pth \
vision_each_epoch=0 vision_runtime=True
