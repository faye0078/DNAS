try:
    import Image
    import ImageDraw
except:
    from PIL import Image
    from PIL import ImageDraw
import glob
import numpy as np
import os
import sys


def image_clip(img_path, size):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1]
    img_dir = "../../data/GID-15/512/image/" + img_name
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    imarray = np.array(Image.open(img_path))
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            clipImg = Image.fromarray(clipArray)

            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_img.tif"
            clipImg.save(img_filepath)


    # 两个for循环分割能完整分割的图像，并保存图像、坐标转换文件
    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_img.tif"
        clipImg.save(img_filepath)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_img.tif"
        clipImg.save(img_filepath)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_img.tif"
        clipImg.save(img_filepath)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_img.tif"
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 1) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('3drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 2) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('4drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 1) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('5drong!!')
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('6drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 2) + "_img.tif"
    clipImg.save(img_filepath)
def label_clip(img_path, size):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1].replace('_15label', '')
    img_dir = "../../../data/GID-15/512/label/" + img_name
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    imarray = np.array(Image.open(img_path)) - 1
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            clipImg = Image.fromarray(clipArray)

            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_label.png"
            clipImg.save(img_filepath)


    # 两个for循环分割能完整分割的图像，并保存图像、坐标转换文件
    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_label.png"
        clipImg.save(img_filepath)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_label.png"
        clipImg.save(img_filepath)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_label.png"
        clipImg.save(img_filepath)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_label.png"
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 1) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('3drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 2) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('4drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 1) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('5drong!!')
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('6drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 2) + "_label.png"
    clipImg.save(img_filepath)
if __name__=='__main__':
    img_path = '/media/dell/DATA/wy/data/gid-15/GID/ann_dir/train/GF2_PMS1__L1A0000647767-MSS1_15label.png'
    folder = os.path.exists("../../../data/GID-15/512/label")
    if not folder:
        os.makedirs("../../../data/GID-15/512/label")

    img_dir = '../../../data/gid-15/GID/ann_dir/train/'
    imgs = glob.glob('{}*.png'.format(img_dir))
    for img in imgs:
        label_clip(img, [512, 512])
    # if not folder:
    #     os.makedirs("../../data/GID-15/512/image")
    #
    # img_dir = '../../data/Large-scale Classification_5classes/image_NirRGB/'
    # imgs = glob.glob('{}*.tif'.format(img_dir))
    # for img in imgs:
    #     image_clip(img, [512, 512])
