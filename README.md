# SpoofFormerNet_Pytorch
Pytorch Implementation on SpoofFormerNet 




python export.py --checkpoint checkpoints/best_model.pt --export-to torchscript


python export.py --checkpoint checkpoints/best_model.pt --export-to onnx




python infer.py --image data/test_img/color/1_1.avi_25_real.jpg --infer-type torch --model-path checkpoints/best_model.pt


python infer.py --image data/test_img/color/5_8.avi_125_fake.jpg --infer-type torchscript --model-path checkpoints/spoof_former_net.torchscript.pt


python infer.py --image data/test_img/color/20_8.avi_175_fake.jpg --infer-type onnx --model-path checkpoints/spoof_former_net.onnx