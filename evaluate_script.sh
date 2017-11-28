#echo "CIFAR scratch evaluation"
#python evaluate.py --dataset 'cifar' --dataset_path /work/05147/srama/shareddir/data/cifar-10/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-scratch-cifar/model_best.net --num_fc 1

#echo "CIFAR imagenet evaluation"
#python evaluate.py --dataset 'cifar' --dataset_path /work/05147/srama/shareddir/data/cifar-10/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-imagenet-cifar/model_best.net --num_fc 1

#echo "SVHN scratch evaluation"
#python evaluate.py --dataset 'svhn' --dataset_path /work/05147/srama/shareddir/data/svhn/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-scratch-svhn/model_best.net --num_fc 1

#echo "SVHN imagenet evaluation"
#python evaluate.py --dataset 'svhn' --dataset_path /work/05147/srama/shareddir/data/svhn/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-imagenet-svhn/model_best.net --num_fc 1

echo "SKETCHES scratch evaluation"
python evaluate.py --dataset 'sketches' --dataset_path /work/05147/srama/shareddir/data/sketches/ --batch_size 100 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-scratch-sketches/model_best.net --num_fc 1

#echo "SKETCHES imagenet evaluation"
#python evaluate.py --dataset 'sketches' --dataset_path /work/05147/srama/shareddir/data/sketches/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-imagenet-sketches/model_best.net --num_fc 1

#echo "CALTECH scratch evaluation"
#python evaluate.py --dataset 'caltech' --dataset_path /work/05147/srama/shareddir/data/caltech-256/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-scratch-caltech/model_best.net --num_fc 1

#echo "CALTECH imagenet evaluation"
#python evaluate.py --dataset 'caltech' --dataset_path /work/05147/srama/shareddir/data/caltech-256/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/models_1layer/vggb-imagenet-caltech/model_best.net --num_fc 1
