from logger import logger
from pytesseract import image_to_osd, Output
import numpy as np
import cv2
from logger import logger

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