if [ $# -ne 1 ]; then 
	echo "Where's the logdir?" 
	exit 1
fi

tensorboard --logdir=$1/