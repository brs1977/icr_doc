from logger import logger
from pytesseract import image_to_osd, Output
import numpy as np
import cv2
from logger import logger

def orient_doc(image, max_area = 200):
    ''' Находим в какой четверти маски изображения больше черных пикселей, и считаем что это верхний угол страницы,
        можно покрутить параметр area - размер контура'''
    
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)    
    adaptive = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,4)
    
    cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < max_area and area > 20:
          cv2.drawContours(mask, [c], -1, (255,255,255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(mask)
    h, w = mask.shape

    top_left_pixels = cv2.countNonZero(mask[0:h//2, 0:w//2])
    bottom_left_pixels = cv2.countNonZero(mask[h//2:, 0:w//2])
    top_right_pixels = cv2.countNonZero(mask[0:h//2, w//2:])
    bottom_right_pixels = cv2.countNonZero(mask[h//2:, w//2])

    pixels = [top_left_pixels,bottom_left_pixels,top_right_pixels,bottom_right_pixels]
    if h>w:
      angle = 0 if top_left_pixels == max(pixels) else \
            -90 if bottom_left_pixels == max(pixels) else \
            90 if top_right_pixels == max(pixels) else \
            180 if bottom_right_pixels == max(pixels) else 0
    else:
      angle = 0 if top_left_pixels == max(pixels) else \
            90 if bottom_left_pixels == max(pixels) else \
            270 if top_right_pixels == max(pixels) else \
            180 if bottom_right_pixels == max(pixels) else 0
                  
    print(f'Orientation angle {angle}')
    k = angle // 90
    image =  np.rot90(image, k=k)

    return image

def orient_image(img):
    try:
        rotate = image_to_osd(img, output_type=Output.DICT)["rotate"]
        # This tells it to use the
        # highest quality interpolation algorithm that it has available,
        # and to expand the image to encompass the full rotated size
        # instead of cropping.
        # The documentation does not say what color
        # the background will be filled with.
        # https://stackoverflow.com/a/17822099

        angle = - float(rotate)
        # if angle > 0:
        #   angle = 360 - angle
        logger.info(f'Orientation angle {angle}')

        k = angle // 90
        if k != 0:
          img =  np.rot90(img, k=k)
        

        # img = img.rotate(-rotate, resample=Image.BICUBIC, expand=True)        
    # sometimes an error can occur with tesseract reading the image
    # maybe there is not enough text or dpi is not set
    # this need to be handled
    except Exception as e:
        raise e
    return img

def rotation(angle, image):
  if angle == 0.0:
    return image
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#skew correction
def deskew_angle(image):
  coords = np.column_stack(np.where(image > 0))
  angle = cv2.minAreaRect(coords)[-1]
  if angle < -45:
      angle = -(90 + angle)
  else:
      angle = -angle
  return angle

def deskew(image):
  angle = deskew_angle(image)
  rotated = rotation(angle, image)
  return rotated    