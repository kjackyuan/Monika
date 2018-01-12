python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path faster_rcnn.config \
    --trained_checkpoint_prefix faster_rcnn_training/model.ckpt-52138 \
    --output_directory deep-finger