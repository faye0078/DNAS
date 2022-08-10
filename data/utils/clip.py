from PIL import Image
import glob
import numpy as np
import os

def image_clip(img_path, size):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1]
    img_dir = "../../../data/GID-5/512/image/" + img_name
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

def get_gid_labels():
    return np.array([
        [255,0,0],    #buildup
        [0,255,0],   #farmland
        [0,255,255],  #forest
        [255,255,0],  #meadow
        [0,0,255] ])  #water
def label_clip(img_path, size):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1].replace('_label', '')
    img_dir = "../../../data/GID-5/4096/label/" + img_name
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    mask = np.array(Image.open(img_path))
    imarray = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for ii, label in enumerate(get_gid_labels()):
        imarray[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    imarray = np.uint8(imarray)
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
    folder = os.path.exists("../../../data/GID-5/512/image")
    # if not folder:
    #     os.makedirs("../../../data/GID-5/4096/label")
    #
    # img_dir = '/media/dell/DATA/wy/data/Large-scale Classification_5classes/label_5classes/'
    #
    # imgs = glob.glob('{}*.tif'.format(img_dir))
    # for img in imgs:
    #     label_clip(img, [4096, 4096])
    if not folder:
        os.makedirs("../../../data/GID-5/512/image")

    img_dir = '/media/dell/DATA/wy/data/Large-scale Classification_5classes/image_RGB/'
    imgs = glob.glob('{}*.tif'.format(img_dir))
    for img in imgs:
        image_clip(img, [512, 512])

