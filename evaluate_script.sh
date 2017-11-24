#echo "CIFAR scratch evaluation"
#python evaluate.py --dataset 'cifar' --dataset_path /work/05147/srama/shareddir/data/cifar-10/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-scratch-cifar/model_best.net

#echo "CIFAR imagenet evaluation"
#python evaluate.py --dataset 'cifar' --dataset_path /work/05147/srama/shareddir/data/cifar-10/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-imagenet-cifar/model_best.net

#echo "SVHN scratch evaluation"
#python evaluate.py --dataset 'svhn' --dataset_path /work/05147/srama/shareddir/data/svhn/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-scratch-svhn/model_best.net

#echo "SVHN imagenet evaluation"
#python evaluate.py --dataset 'svhn' --dataset_path /work/05147/srama/shareddir/data/svhn/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-imagenet-svhn/model_best.net

echo "SKETCHES scratch evaluation"
python evaluate.py --dataset 'sketches' --dataset_path /work/05147/srama/shareddir/data/sketches/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-scratch-sketches/model_best.net

#echo "SKETCHES imagenet evaluation"
#python evaluate.py --dataset 'sketches' --dataset_path /work/05147/srama/shareddir/data/sketches/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-imagenet-sketches/model_best.net

echo "CALTECH scratch evaluation"
python evaluate.py --dataset 'caltech' --dataset_path /work/05147/srama/shareddir/data/caltech-256/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-scratch-caltech/model_best.net

echo "CALTECH imagenet evaluation"
python evaluate.py --dataset 'caltech' --dataset_path /work/05147/srama/shareddir/data/caltech-256/ --batch_size 32 --num_workers 2 --load_model ../deep-adaptation-networks-dl/models/vggb-imagenet-caltech/model_best.net


