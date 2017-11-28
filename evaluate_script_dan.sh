echo "DAN-imagenet on CIFAR evaluation"
python evaluate_dan.py --dataset cifar --dataset_path /work/05147/srama/shareddir/data/cifar-10/ --batch_size 100 --num_workers 4 --load_model models/dan_models_100batch/dan_imagenet-cifar/model_best.net --num_fc 1

echo "DAN-imagenet on SVHN evaluation"
python evaluate_dan.py --dataset svhn --dataset_path /work/05147/srama/shareddir/data/svhn/ --batch_size 100 --num_workers 4 --load_model models/dan_models_100batch/dan_imagenet-svhn/model_best.net --num_fc 1

echo "DAN-imagenet on SKETCHES evaluation"
python evaluate_dan.py --dataset sketches --dataset_path /work/05147/srama/shareddir/data/sketches/ --batch_size 100 --num_workers 4 --load_model models/dan_models_100batch/dan_imagenet-sketches/model_best.net --num_fc 1

echo "DAN-imagenet on CALTECH evaluation"
python evaluate_dan.py --dataset caltech --dataset_path /work/05147/srama/shareddir/data/caltech-256/ --batch_size 100 --num_workers 4 --load_model models/dan_models_100batch/dan_imagenet-caltech/model_best.net --num_fc 1

