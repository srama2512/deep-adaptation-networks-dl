# Caltech 256 dataset
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar
mv 256_ObjectCategories "caltech-256"
rm 256_ObjectCategories.tar

# Visual decathlon challenge
wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz
tar -xvf decathlon-1.0-data.tar.gz
tar -xvf aircraft.tar
tar -xvf cifar100.tar
tar -xvf daimlerpedcls.tar
tar -xvf dtd.tar
tar -xvf gtsrb.tar
tar -xvf omniglot.tar
tar -xvf svhn.tar
tar -xvf ucf101.tar
tar -xvf vgg-flowers.tar

rm decathlon-1.0-data.tar.gz
rm aircraft.tar
rm cifar100.tar
rm daimlerpedcls.tar
rm dtd.tar
rm gtsrb.tar
rm omniglot.tar
rm svhn.tar
rm ucf101.tar
rm vgg-flowers.tar

# Annotations for Visual decathlon
wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-devkit.tar.gz
tar -xvf decathlon-1.0-devkit.tar.gz
rm decathlon-1.0-devkit.tar.gz

# Obtain CIFAR10 from pytorch itself

# Sketch dataset
wget http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
unzip sketches_png.zip
mv png sketches
rm sketches_png.zip
