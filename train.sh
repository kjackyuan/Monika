if [ $# -ne 1 ]; then
	echo "what model brother?"
	exit 1
fi

TRAINING_DIR=$1_training

python train.py --logtostderr --train_dir=$TRAINING_DIR/ --pipeline_config_path=$1.config