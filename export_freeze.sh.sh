python slim/export_inference_graph.py \
  --alsologtostderr \
  --model_name="mobilenet_v2_140" \
  --image_size=500 \
  --output_file=slim/satellite/export/mobilenet_v2_140_inf_graph.pb \
  --dataset_name satellite

python freeze_graph.py \
  --input_graph slim/satellite/export/mobilenet_v2_140_inf_graph.pb \
  --input_checkpoint slim/satellite/train_log/model.ckpt-10000 \
  --input_binary true \
  --output_node_names MobilenetV2/Predictions/Reshape_1 \
  --output_graph slim/satellite/freeze/mobilenet_v2_140.pb
