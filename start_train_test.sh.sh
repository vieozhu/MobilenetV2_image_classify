python slim/train_image_classifier.py \
  --train_dir=slim/satellite/train_log \
  --dataset_name=satellite \
  --train_image_size=500 \
  --dataset_split_name=train \
  --dataset_dir=slim/satellite/data \
  --model_name="mobilenet_v2_140" \
  --checkpoint_path=slim/satellite/pretrained/mobilenet_v2_1.4_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV2/Logits,MobilenetV2/AuxLogits \
  --trainable_scopes=MobilenetV2/Logits,MobilenetV2/AuxLogits \
  --max_number_of_steps=10000 \
  --batch_size=16 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --label_smoothing=0.1 \
  --num_clones=1 \
  --num_epochs_per_decay=2.5 \
  --moving_average_decay=0.9999 \
  --learning_rate_decay_factor=0.98 \
  --preprocessing_name="inception_v2"

python slim/eval_image_classifier.py \
  --checkpoint_path=slim/satellite/train_log \
  --eval_dir=satellite/eval_log \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=slim/satellite/data \
  --model_name="mobilenet_v2_140" \
  --batch_size=32 \
  --num_preprocessing_threads=2 \
  --eval_image_size=500

