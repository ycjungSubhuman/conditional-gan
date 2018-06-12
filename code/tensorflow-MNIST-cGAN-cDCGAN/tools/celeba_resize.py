from PIL import Image
import os,sys

for i in range(202599):
    p = '../../../dataset/CelebA/Img/img_align_celeba/{:06d}.jpg'.format(i+1)
    im = Image.open(p)
    im = im.resize((64, 64), Image.ANTIALIAS)
    im.save('../../../dataset/CelebA/Img/img_align_celeba/{:06d}_re.jpg'.format(i+1))
    im.close()
    print ('Processed image {:06d}'.format(i+1))
