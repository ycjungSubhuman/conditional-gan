# cGAN-cDCGAN training

## Prerequisite

* Download CelebA dataset.
* Unzip and put all .jpg in `dataset/CelebA/Img/img_align_celeba/`
* Downlaod `list_attr_celeba.txt` and put it in `dataset/CelebA/`
* Run `code/tensorflow-MNIST-cGAN-cDCGAN/tools/celeba_resize.py`

## Running

* Install `nvidia-docker`
* Run `./builddocker.sh`
* Run `./rundocker.sh`
* In your web browser, connect to localhost:6006 to launch Jupyter
* Click `code/tensorflow-MNIST-cGAN-cDCGAN`
* Create any notebook and launch `%run -i '(python script name)' (any run name you want)` to start training 
* Check out the training intermediate results in tensorboard at `https://localhost:8888/`

Here are the list of python scripts

* `tensorflow_CelebA_cDCGAN.py` : train CelebA using cDCGAN
* `tensorflow_MNIST_cDCGAN.py` : train MNIST using cDCGAN
* `tensorflow_MNIST_cGAN.py` : train MNIST using cGAN

## Warning
CelebA training requires very high VRAM. 12GB would suffice.

## Models
You can download pre-trained models from https://mega.nz/#F!OzgwHL5S

## Checking FID of model
unzip the model and put each folder to `checkpoints/`

`%run -i 'tensorflow_MNIST_cGAN.py' cgan-fid-every-epoch`
`%run -i 'tensorflow_MNIST_cDCGAN.py' cdcgan-mnist-base2`
`%run -i 'tensorflow_CelebA_cDCGAN.py' cdcgan-celebA-fid-every-epoch`

