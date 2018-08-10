python eval_image_classifier.py \
  --checkpoint_path=satellite/train_log \
  --eval_dir=satellite/eval_log \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name="mobilenet_v2_140" \
  --batch_size=32 \
  --num_preprocessing_threads=2 \
  --eval_image_size=500

