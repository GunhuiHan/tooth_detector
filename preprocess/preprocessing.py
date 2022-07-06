
import os 
import shutil
import cv2
import json
import glob
from PIL import Image
from tqdm import tqdm

def getGroundTruthCoordinateHardcoding(originalImg, gtImg):
    # 테스트용, 느려서 안씀
    Y, X  = originalImg.shape
    y, x = gtImg.shape[:2]

    stopy = Y - y + 1
    stopx = X - x + 1

    for yi in range(0, stopy):
        y1 = yi + y
        for xi in range(0, stopx):
            x1 = xi + x
            pic = originalImg[yi:y1, xi:x1]
            test = (pic == gtImg)
            if test.all():
                print(x,y)

def getGroundTruthBoundingBox(image):
    originalPath = 'data/%s/panorama/Originals/%s_PANO_0_1.bmp' % (image, image)
    gtPath = 'data/%s/panorama/%s_PANO_0_1.bmp' % (image, image)
    originalImg = cv2.imread(originalPath, 0) # 0 means cv2.IMREAD_GRAYSCALE
    gtImg = cv2.imread(gtPath, 0)

    if originalImg is None or gtImg is None:
        print("이미지 로드가 실패 했습니다.")
        # sys.exit()
        return []

    result = cv2.matchTemplate(originalImg, gtImg, cv2.TM_SQDIFF)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc
    h, w = gtImg.shape

    return [x,y,x+w,y+h]

    # show image with gt-bbox
    # dst = cv2.rectangle(originalImg, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
    # cv2.imshow("dst", dst)
    # # cv2.imshow('original image', originalImg)
    # # cv2.imshow('ground truth image', gtImg)
    # cv2.waitKey()
    # cv2.destroyWindow()

def changeFolderName():
    for path in tqdm(list(set(glob.glob("data/*/panorama/Original", recursive=True)))):
        print(path+'s')
        os.rename(path, path+'s')

def jpgTobmp():
    for jpg_path in tqdm(list(set(glob.glob("data/*/panorama/Originals/"+"*.jpg", recursive=True)))):
        img = Image.open(jpg_path)
        file_name = jpg_path.split('/')[1]
        bmp_name = file_name + '_PANO_0_1.bmp'
        bmp_path = 'data/%s/panorama/Originals/' % (file_name)
        img.save(bmp_path+bmp_name)

    for jpg_path in tqdm(list(set(glob.glob("data/*/panorama/"+"*.jpg", recursive=True)))):
        img = Image.open(jpg_path)
        file_name = jpg_path.split('/')[1]
        bmp_name = file_name + '_PANO_0_' + jpg_path.split('.')[0][-1] + '.bmp'
        bmp_path = 'data/%s/panorama/' % (file_name)
        img.save(bmp_path+bmp_name)

def changeFileNames():
    for path in tqdm(list(set(glob.glob("data/*/panorama/*.bmp", recursive=True)))):
        file = path.split('/')[-1]
        if "PANOVIEW" in file or "pano" in file: 
            newFile = file.replace("PANOVIEW", "PANO")
            newFile = newFile.replace("pano", "PANO")
            newPath = path.replace(file, newFile)
            os.rename(path, newPath)

def deleteUnNecessary():
    for path in tqdm(list(set(glob.glob("data/*/", recursive=True)))):
        if os.path.exists(path + 'axial/'):
            shutil.rmtree(path + 'axial/')
        if os.path.exists(path + 'cross section/'):
            shutil.rmtree(path + 'cross section/')
        if os.path.exists(path + 'panoramic view/'):
            shutil.rmtree(path + 'panoramic view/')
        if os.path.exists(path + 'periapical/'):
            shutil.rmtree(path + 'periapical/')

        file_name = path.split('/')[1]
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_2.bmp'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_2.bmp')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_3.bmp'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_3.bmp')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_4.bmp'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_4.bmp')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_5.bmp'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_5.bmp')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_1.jpg'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_1.jpg')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_2.jpg'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_2.jpg')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_3.jpg'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_3.jpg')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_4.jpg'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_4.jpg')
        if os.path.exists(path + 'panorama/' + file_name + '_PANO_0_5.jpg'):
            os.remove(path + 'panorama/' + file_name + '_PANO_0_5.jpg')
        if os.path.exists(path + 'panorama/Originals/' + file_name + '_PANO_0_1.jpg'):
            os.remove(path + 'panorama/Originals/' + file_name + '_PANO_0_1.jpg')

def makeJsonFile():
    path = './data/' 
    file_list = os.listdir(path)

    ground_truth_file_path = './ground_truth.json'
    error_path = './error.json'
    data = []
    error_data = []

    for i in file_list:
        box = getGroundTruthBoundingBox(i)
        if(len(box) == 0):
            error_data.append(i) 
        else :
            data.append({'box' : box, 'filename' : i}) # data format :  boxes [x1, y1, x2, y2], filename
            print('data added : {box : ', box, 'filename : ', i, '}')


    with open(ground_truth_file_path, 'w') as outfile:
        json.dump(data, outfile)
    with open(error_path, 'w') as error_outfile:
        json.dump(error_data, error_outfile)

def preprocess_data():
    changeFolderName()
    jpgTobmp()
    changeFileNames()
    deleteUnNecessary()
    makeJsonFile()