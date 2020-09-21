import cv2
import re
import numpy as np
from collections import OrderedDict
from logger import logger
from string import printable
import pytesseract
from cv2_imshow import cv2_imshow

printable_ptrn = re.compile("[^{}]+".format(printable))
doublespace_ptrn = re.compile('\s+')
newline_ptrn = re.compile('\n')

def cv2_imshow(img):
    '''Mock function for collab'''
    pass

def draw_rect_text(x,y,w,h,image,text):
  image = cv2.rectangle(image, (x, y), (x + w, y + h), (255), 1)
  cv2.putText(image, text, (x+w//2-10, y+h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 2)

def draw_struct_table(table_struct):
  min_x = min([x['cell'][0] for x in table_struct])
  min_y = min([x['cell'][1] for x in table_struct])

  max_x = max([x['cell'][0] for x in table_struct])
  max_y = max([x['cell'][1] for x in table_struct])+50

  image = np.zeros((max_y,max_x))
  for struct in table_struct:
    x,y,w,h = struct['cell']
    # print(x['col'],x['row'])
    text = f"{struct['col']},{struct['row']}"
    draw_rect_text(x,y,w,h,image,text)
  cv2_imshow(image[min_y:,min_x:])

def draw_image_document(filename):
  img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
  if img is None:
      raise ValueError("File {0} does not exist".format(filename))
  logger.info((filename, img.shape))    
  cv2_imshow(img)  


def get_cell(col,row, table_struct):
      return [x['cell'] for x in table_struct if x['col']==col and x['row']==row]

def ocr(img, config, lang = 'rus+digits'):
  text = ''
  try:
    text = pytesseract.image_to_string(img, lang=lang, config=config)
  except Exception as E:
    logger.warning(E)

  return postprocess_ocr_text(text)

def alignment(coord, thresh = 7):
  coord_map = OrderedDict()
  sorted_coord = sorted(coord)
  for i,d in enumerate(np.diff(sorted_coord)):
    if d < thresh:
      coord_map[sorted_coord[i+1]] = sorted_coord[i]
      sorted_coord[i+1] = sorted_coord[i]
  return [c if c not in coord_map.keys() else coord_map[c]  for c in coord]


def postprocess_ocr_text(text):  
  # text = printable_ptrn.sub("", text)
  text = newline_ptrn.sub(" ", text)
  text = doublespace_ptrn.sub(" ", text)
  
  if (len(text) == 1) & (text == ' '):
    text = ''
  return text.strip()

def debug_box(text,img,text_boxes):
  # Visualize the result text_boxes
  vis = img.copy()
  logger.info(text)
  for (x, y, w, h) in text_boxes:
    cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)
  cv2_imshow(vis)  

def denoise(img):
      # Convert to gray
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply dilation and erosion to remove some noise
  kernel = np.ones((1, 1), np.uint8)
  img = cv2.dilate(img, kernel, iterations=1)
  img = cv2.erode(img, kernel, iterations=1)

  # #  Apply threshold to get image with only black and white
  # img = apply_threshold(img, method)
  # binarization
  _, img = cv2.threshold(img,110,255,cv2.THRESH_TOZERO)
  # cv2_imshow(img)
  return img

def apply_threshold(img, method, kernel_size = 1):
  # =============================================================================== #
  #    Threshold Methods                                                            #
  # =============================================================================== #
  # 1. Binary-Otsu                                                                  #
  # 2. Binary-Otsu w/ Gaussian Blur (kernel size, kernel size)                      #
  # 3. Binary-Otsu w/ Median Blur (kernel size, kernel size)                        #
  # 4. Adaptive Gaussian Threshold (31,2) w/ Gaussian Blur (kernel size)            #
  # 5. Adaptive Gaussian Threshold (31,2) w/ Median Blur (kernel size)              #
  # =============================================================================== #  
    switcher = {
        1: cv2.threshold(img, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.medianBlur(img, kernel_size), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        5: cv2.adaptiveThreshold(cv2.medianBlur(img, kernel_size), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(method, "Invalid method")

def to_gray(src):
  return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  

def to_joints(horizontal, vertical):
  return cv2.bitwise_and(horizontal, vertical)

def to_binary(gray):
  return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

def to_lines(bw, structure):
  # Apply morphology operations
  mat = cv2.erode(bw, structure, (-1, -1))
  mat = cv2.dilate(mat, structure, (-1, -1))
  return mat  
