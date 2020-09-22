import string
import re
import cv2
from typing import List, NamedTuple
from utils import denoise, get_cell, ocr
from logger import logger, DEBUG
import numpy as np
from cv2_imshow import cv2_imshow
import pytesseract
import os
import string
import math
from table_extractor import TableExtractor
from deskew import orient_image, orient_doc
from glob import glob
import random
import simplejson 
from utils import debug_box
import platform


DEBUG_LEVEL = 0

INN_ERROR_PATTERN = re.compile('ИННИКПП|ИННЖПП|ИННКПИ|ИННИЖПП|ИННИПП|ИННЖКПП|ИННКПП')
MAP_TEXT_ERRORS = {'поковкалитамповка':'поковка/штамповка',
  'ТЕВСТОЛИТА': 'ТЕКСТОЛИТ А',
  'С ОТКАНЬ ПРОПИТАННАЯ': 'СТЕКЛОТКАНЬ ПРОПИТАННАЯ',
  'ПАКОТКАНЬ': 'ЛАКОТКАНЬ',
  'СТЕКЛО ТЕКСТОЛИТ': 'СТЕКЛОТЕКСТОЛИТ',
  'СТЕКЛОТЕКС ТОЛИТ': 'СТЕКЛОТЕКСТОЛИТ',
  'ЛАКОПТКАНЬ': 'ЛАКОТКАНЬ',
  # 'ИННИКПП': 'ИНН/КПП',
  # 'ИННЖПП': 'ИНН/КПП',
  # 'ИННКПИ': 'ИНН/КПП'
  }


def error_correct(text):
  old = text
  for reg, rep in MAP_TEXT_ERRORS.items():
    text = re.sub(reg, rep, text)

  if old!=text:
    logger.info(f'Коррекция текста {old} -> {text}')     
  return text

def text2num(text):
  '''Разделитель на 2 или 3 знака'''
  # text = re.sub('\s', '', text)
  sep_idx = 0
  if len(text)>3 and text[-3:-2] not in string.digits:
    sep_idx = -3
  elif len(text)>4 and text[-4:-3] not in string.digits:
    sep_idx = -4

  if sep_idx == 0:
    text = ''.join([t for t in text if t in string.digits])
  else:
    t1 = [t for t in text[:sep_idx] if t in string.digits]
    t1.append('.')
    t2 = [t for t in text[sep_idx+1:] if t in string.digits]
    t1.extend(t2)
    text = ''.join(t1)
  return text

def repr_dict(obj):
  lines = ['{']
  for key, value in obj._asdict().items():
      lines.append('  {}:{}'.format(key, value))
  lines.append('}')
  return '\n'.join(lines)

class FactRawUnit(NamedTuple):
    name:               str # 0 наименование
    code_unit:          str # 1 код ед изм
    volume:             str # 3 кол, объем
    unit_price:         str # 4 цена за единицу
    amount_without_tax: str # 5 сумма без налогоа
    amount_tax:         str # 8 сумма налога
    amount_with_tax:    str # 9 сумма с налогом
    def __repr__(self):
      return repr_dict(self)

class FactUnit(NamedTuple):
    name:               str   # 0 наименование
    code_unit:          str   # 1 код ед изм
    volume:             int   # 3 кол, объем
    unit_price:         float # 4 цена за единицу
    amount_without_tax: float # 5 сумма без налогоа
    amount_tax:         float # 8 сумма налога
    amount_with_tax:    float # 9 сумма с налогом
    def __repr__(self):
      return repr_dict(self)

def cell_table_image(x, y, table_struct, img):
  cell = get_cell(x,y,table_struct)[0]
  x,y,w,h = cell[:4]
  cell = img[y+1:y+h-1, x+1:x+w-1]
  return cell

def get_cell_text(x,y, table_struct, img):
  cell = cell_table_image(x, y, table_struct, img)
  # cv2_imshow(cell)
  # return ocr(cell, config='--oem 1 --psm 6 -c textord_heavy_nr=true', lang='rus2+digits1')
  # return ocr(cell, config='--oem 1 --psm 6 -c textord_heavy_nr=true', lang='rus+digits1')
  text = ocr(cell, config='--oem 1 --psm 6 -c textord_heavy_nr=true', lang='rus')
  text = error_correct(text)
  return text

def get_cell_digits(x,y, table_struct, img):
  cell = cell_table_image(x, y, table_struct, img)
  # cv2_imshow(cell)
  # return ocr(cell, config='--oem 1 --psm 6 -c textord_heavy_nr=true', lang='digits1')
  text = ocr(cell, config='--oem 1 --psm 7', lang='digits')
  # text = text2num(text)
  return text
  
def text2digits(text):  
  """Оставляем только цифры"""
  return ''.join([t for t in text if t in string.digits])

def text2strfloat2(text):
  """ddd.dd"""
  text = text2digits(text)
  return text[:-2] + '.' + text[-2:]

def text2strfloat3(text):
  """ddd.ddd"""
  text = text2digits(text)
  return text[:-3] + '.' + text[-3:]

def parse_invoice_unit(n, struct_table, img, n_cols):
  # с 3 строки  
  # 0 наименование
  # 1 код ед изм
  # 3 кол, объем
  # 4 цена за единицу
  # 5 сумма без налогоа
  # 8 сумма налога
  # 9 сумма с налогом
  # посл строка итог
  # 5 сумма без налога
  # 8 сумма налога
  # 9 сумма с налогом

  # если есть доп колонка то сдвигаем на 1 вперед
  n_start = 0 if n_cols == 12 else 1  
  
  name = get_cell_text(0,n,struct_table, img)
  code_unit = get_cell_digits(n_start+1,n,struct_table, img)
  volume = get_cell_digits(n_start+3,n,struct_table, img)
  unit_price = get_cell_digits(n_start+4,n,struct_table, img)
  amount_without_tax = get_cell_digits(n_start+5,n,struct_table, img)
  amount_tax = get_cell_digits(n_start+8,n,struct_table, img)
  amount_with_tax = get_cell_digits(n_start+9,n,struct_table, img)
  return FactRawUnit(name=name,
                  code_unit=text2digits(code_unit),
                  volume=text2strfloat3(volume), 
                  unit_price=text2strfloat2(unit_price),
                  amount_without_tax=text2strfloat2(amount_without_tax),
                  amount_tax=text2strfloat2(amount_tax), 
                  amount_with_tax=text2strfloat2(amount_with_tax))
  
def is_header_table(struct_table):
  # если нет ячейки 1,1
  return len([x for x in struct_table if x['col']==0 and x['row'] == 1]) == 0

def is_lastrow_summary(struct_table):
  # если в последней строке меньше колонок чем в предыдущей, то это итог
  n_cols = max([x['col'] for x in struct_table])
  n_rows = max([x['row'] for x in struct_table])
   
  return len([x for x in struct_table if x['row'] == n_rows-1]) != len([x for x in struct_table if x['row'] == n_rows])

def correct_unit(unit):
  code_unit = 0 if unit.code_unit == '' else int(unit.code_unit)
  volume = float(unit.volume)
  unit_price = float(unit.unit_price)
  amount_without_tax = float(unit.amount_without_tax)
  amount_tax = float(unit.amount_tax)
  amount_with_tax = float(unit.amount_with_tax)

  amount_without_tax1 = round(volume*unit_price,2)
  amount_without_tax2 = amount_without_tax
  amount_without_tax3 = round(amount_with_tax-amount_tax,2)

  if not (amount_without_tax1 == amount_without_tax2 and amount_without_tax1 == amount_without_tax3):
    if amount_without_tax1 == amount_without_tax2:      
      old = amount_with_tax
      amount_with_tax = round(amount_without_tax1+amount_tax,2)
      logger.info(f'Коррекция amount_with_tax старое {old}, amount_with_tax = volume*unit_price+amount_tax, {amount_with_tax} ={volume}*{unit_price}+{amount_tax}')
    elif  amount_without_tax1 == amount_without_tax3:      
      old = amount_without_tax
      amount_without_tax = amount_without_tax1
      logger.info(f'Коррекция amount_without_tax старое {old}, amount_without_tax = volume*unit_price, {amount_without_tax} ={volume}*{unit_price}')
    # elif  amount_without_tax2 == amount_without_tax3:
    #   if unit.unit_price[-3:-2] == '.':
    #     volume = round(amount_without_tax3 / unit_price,3)
    #   elif unit.volume[-4:-3] == '.':
    #     unit_price = round(amount_without_tax3 / volume,2)

  return FactUnit(name=unit.name,
                  code_unit=code_unit,
                  volume=volume, 
                  unit_price=unit_price,
                  amount_without_tax=amount_without_tax,
                  amount_tax=amount_tax, 
                  amount_with_tax=amount_with_tax)

def parse_invoice_units(struct_table, img):
  invoice_units = []
  n_cols = max([x['col'] for x in struct_table])
  n_rows = max([x['row'] for x in struct_table])

  logger.info(f'строк {n_rows}, колонок {n_cols}')  
  
  n_start = 3
  if not is_header_table(struct_table): # если нет заголовка таблицы с 1 строки
    logger.info('Таблица без заголовка')
    n_start = 0

  if not is_lastrow_summary(struct_table):
    n_rows += 1

  for n in np.arange(n_start, n_rows): # последнняя строка - итог 
    unit = parse_invoice_unit(n, struct_table, img, n_cols)
    unit = correct_unit(unit)
    invoice_units.append(unit)
  return invoice_units

class InvoiceData(NamedTuple):
  invoice:        str # фактура 0
  seller:         str # продавец 2
  seller_address: str # 3
  inn_seller:     str # 4
  buyer:          str # покупатель 8
  buyer_address:  str # 9
  inn_buyer:      str # 10
  
  def __repr__(self):
    return repr_dict(self)
  
def get_invoice_data(image):
  text = pytesseract.image_to_string(image, lang='rus', config='--oem 1 --psm 6')
  text = INN_ERROR_PATTERN.sub('ИНН/КПП', text)
  
  text = re.sub('”|“','"',text)
  text = re.sub('‘Адрес','Адрес',text)  
  text = re.sub('\n+','\n',text)

  texts = text.split('\n')
  assert len(texts)>=12
  invoice = texts[0]
  seller = texts[2]
  seller_address = texts[3]
  inn_seller = texts[4]
  buyer = texts[8]
  buyer_address = texts[9]
  inn_buyer = texts[10]
  return InvoiceData(invoice=invoice,
                    seller = seller,
                    seller_address = seller_address,
                    inn_seller = inn_seller,
                    buyer = buyer,
                    buyer_address = buyer_address,
                    inn_buyer = inn_buyer)  


def invoice_image_rect(gray):
  """ Координаты параграфа счета фактуры"""
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (7,7), 0)
  thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Create rectangular structuring element and dilate
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
  dilate = cv2.dilate(thresh, kernel, iterations=6)

  # Find max contours 
  cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  max_contour = cnts[0]
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      if cv2.contourArea(max_contour) < cv2.contourArea(c):
        max_contour = c
      if DEBUG:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (36,255,12), 2)


  x,y,w,h = cv2.boundingRect(max_contour)
  # if DEBUG:
  #   cv2_imshow(thresh)
  #   cv2_imshow(dilate)
  #   cv2_imshow(gray)

  return x,y,w,h

def parse_document(struct_table, img):
  denoise_img = denoise(img)

  invoice_units = parse_invoice_units(struct_table, denoise_img)

  # total_n = len(y_coords)-1
  # total_amount_without_tax = get_cell_digits(5,total_n, table_coords, denoise_img)
  # total_amount_tax = get_cell_digits(8,total_n, table_coords, denoise_img)
  # total_amount_with_tax = get_cell_digits(9,total_n, table_coords, denoise_img)

  # print(total_amount_without_tax,total_amount_tax,total_amount_with_tax)

  x_table = min([x['cell'][0] for x in struct_table])
  y_table = min([x['cell'][1] for x in struct_table])  

  invoice_data = None
  if y_table > 100:
    invoice_header_cell = denoise_img[0:y_table,x_table:denoise_img.shape[1]//2]  
    x,y,w,h = invoice_image_rect(invoice_header_cell)
    invoice_header_cell = denoise_img[y:y+h,x:x+w]
    if DEBUG:
      cv2_imshow(invoice_header_cell)

    try:
      invoice_data = get_invoice_data(invoice_header_cell)
    except Exception as E:
      logger.info('Не определен заголовок фактуры')        
      logger.exception(E)      
  else:
    logger.info('Не определен заголовок фактуры')  

  return invoice_data, invoice_units

def read_image(filename):
  img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
  if img is None:
      raise ValueError("File {0} does not exist".format(filename))

  logger.info((filename, img.shape))    
  return img      

def extract(img):
  try:
    # img = orient_image(img)
    img = orient_doc(img)    
  except Exception as E:
    logger.warning('Упал tesseract')
    logger.exception(E)
    cv2_imshow(img)
    raise E
  # cv2_imshow(img)

  min_size = 80
  min_area = min_size*min_size
  delta = -5
  epsilon = 10

  te = TableExtractor(img, hor_min_size=70, ver_min_size=100, debug = DEBUG, debug_level = DEBUG_LEVEL)
  te.prepare()  

  if DEBUG:
    cv2_imshow(te.src)

  tables = te.locate_tables(min_area, epsilon, delta)

  if len(tables)==0 :
    logger.warning('Таблица не определилась')
    raise Exception('Таблица не определилась')
  if len(tables)>2 :
    logger.warning('Некорректная структура таблиц')
    raise Exception('Некорректная структура таблиц')

  logger.info(f'Find tables {len(tables)}')

  cells = te.locate_cells(table = tables[0], epsilon = epsilon)
  if DEBUG:
    debug_box('', img, [cell[:4] for cell in cells])

  struct_table = te.to_struct(cells)

  try:
    invoice_data, invoice_units = parse_document(struct_table, te.src)
    logger.info(invoice_data)
    logger.info(invoice_units)
  except Exception as E:
    logger.warning('Все сломалось')
    logger.exception(E)
    raise E

  return simplejson.dumps({'invoice':invoice_data, 'units':invoice_units})

def extract_all(path):
    files = glob(os.path.join(path,'*.jpg'))
    # random.shuffle(files)
    data = []
    for filename in files[:40]:    
        json = extract(filename)

    data.append([filename, json])
    return data


if platform.system() == 'Windows':
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

