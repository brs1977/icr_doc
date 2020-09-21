from dataclasses import dataclass, field
from typing import List, NamedTuple
from utils import *
from deskew import deskew_angle, rotation
from logger import logger
from cv2_imshow import cv2_imshow

@dataclass
class TableRoi:
    col_points: List[float] 
    row_points: List[float] 
    x: float
    y: float
    w: float
    h: float  


class TableExtractor():
  def __init__(self, src, hor_min_size = 200, ver_min_size = 200, 
               apply_method=4, kernel_size=3, 
               area_out_of_size = 300,
               debug = False, debug_level = 0):
    self.src = src
    self.apply_method=apply_method
    self.kernel_size=kernel_size
    self.hor_min_size = hor_min_size
    self.ver_min_size = ver_min_size
    self.horizontal = None
    self.vertical = None
    self.debug_level = debug_level
    # self.angle = 0.0
    # self.min_x = 0
    # self.min_y = 0
    self.debug = debug
    self.area_out_of_size = area_out_of_size

  def preprocess_image(self, img, method=1, kernel_size=5):
      gray = to_gray(img)
      gray = apply_threshold(gray, method=method)

      # dilate the text to make it solid spot
      struct = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
      gray = cv2.dilate(~gray, struct, anchor=(-1, -1), iterations=1)
      if self.debug and self.debug_level > 0:
        cv2_imshow(gray)
      return gray        

  def prepare(self):
    gray = self.preprocess_image(self.src, self.apply_method, self.kernel_size)
    self.bw = to_binary(gray)

    self.horizontal = to_lines(self.bw, cv2.getStructuringElement(cv2.MORPH_RECT, (self.hor_min_size, 1)) )
    self.vertical   = to_lines(self.bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.ver_min_size)) )

    mask = self.horizontal + self.vertical
    min_x,min_y,max_x,max_y = self.table_contour(mask)

    mask = mask[min_y:max_y, min_x:max_x]
    self.angle = deskew_angle(mask)

    logger.info(f'Rotate angle {self.angle}')
    self.src = rotation(self.angle, self.src)

    gray = self.preprocess_image(self.src, self.apply_method, self.kernel_size)
    self.bw = to_binary(gray)

    self.horizontal = to_lines(self.bw, cv2.getStructuringElement(cv2.MORPH_RECT, (self.hor_min_size, 1)) )
    self.vertical   = to_lines(self.bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.ver_min_size)) )

    if self.debug and self.debug_level > 0:
      logger.info('gray')
      cv2_imshow(gray)
      logger.info('bw')
      cv2_imshow(self.bw)
      logger.info('horizontal')
      cv2_imshow(self.horizontal)
      logger.info('vertical')
      cv2_imshow(self.vertical)

  def table_contour(self, img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = map(cv2.contourArea, contours)

    largest_contour_idx = np.argmax(list(contour_areas))
    largest_contour = contours[largest_contour_idx]

    x = [x[0][0] for x in largest_contour]
    y = [x[0][1] for x in largest_contour]

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    # return mask[min_y:max_y, min_x:max_x]
    # return (min_x-1,min_y-1,max_x+1,max_y+1)
    return min_x,min_y,max_x,max_y

  def locate_tables(self, min_area, epsilon, delta):
    # assert self.horizontal != None
    # assert self.vertical != None

    mask = self.horizontal + self.vertical
    #Find external contours from the mask, which most probably will belong to tables or to images    
    joints = to_joints(self.horizontal, self.vertical)

    contours, hierarchies = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if self.debug and self.debug_level > 0:
      logger.info('contours')
      mark_src = self.src.copy()    
      cv2.drawContours(mark_src, contours, -1, (255, 0, 0), 2)
      cv2_imshow(mark_src)

    cols, rows = self.bw.shape[:2]

    rois = []
    for contour in contours:
      # find the area of each contour
      area = cv2.contourArea(contour)

      # filter individual lines of blobs that might exist and they do not represent a table
      if area < min_area: #value is randomly chosen, you will need to find that by yourself with trial and error procedure
        continue

      contours_poly = cv2.approxPolyDP(contour, epsilon, True)
      x, y, w, h = cv2.boundingRect(np.array(contours_poly))
      
      # find the number of joints that each table has
      box = joints[y:y+h, x:x+w]
      joints_contours, _ = cv2.findContours(box, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

      # if the number is not more than 6 then most likely it not a table
      if len(joints_contours) < 6:
        continue
      if x - delta >= 0 & y - delta >= 0 & x + w + delta <= cols & y + h + delta <= rows: 
        x -= delta
        y -= delta
        w += delta * 2
        h += delta * 2

      # relative points
      col_points, row_points = self.parameter_table(joints_contours)
      # absolute points
      col_points = [col + x for col in col_points]
      row_points = [row + y for row in row_points]

      roi = TableRoi(col_points, row_points, x,y,w,h)
      rois.append(roi)
    
    return rois       

  def parameter_table(self, joints_contours):
    x_coor = []
    y_coor = []
    for i in joints_contours:
        x_coor.append(cv2.minEnclosingCircle(i)[0][0])
        y_coor.append(cv2.minEnclosingCircle(i)[0][1])

    x_coor = sorted(x_coor)
    y_coor = sorted(y_coor)

    xs = set()
    for index in range(len(x_coor) - 1):
        if abs(x_coor[index] - x_coor[index + 1]) < 14:
            x_coor[index + 1] = x_coor[index]
            xs.add(x_coor[index])

    ys = set()
    for index in range(len(y_coor) - 1):
        if abs(y_coor[index] - y_coor[index + 1]) < 14:
            y_coor[index + 1] = y_coor[index]
            ys.add(y_coor[index])

    return list(xs), list(ys)

  def locate_cells(self, table, epsilon = 10, round_alpha = 15):
    # assert self.horizontal != None
    # assert self.vertical != None
    
    # x,y,w,h = (self.min_x+table.x,self.min_y+table.y,self.min_x+table.w,self.min_y+table.h)
    x,y,w,h = (table.x,table.y,table.w,table.h)

    src = self.src[y:y+h,x:x+w]

    mask = self.horizontal + self.vertical
    # joints = self.to_joints(horizontal, vertical)
    # cols, rows, col_points, row_points = self.table_params(rect,joints)

    if self.debug:    
      mask_src = src.copy()

    contours, hierarchies = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for i, (contour,hierarchy) in enumerate(zip(contours, hierarchies[0])):
        
        area = cv2.contourArea(contour)    
        contours_poly = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(np.array(contours_poly))
        # x, y, w, h = x-3, y-3, w+3, h+3

        #out of size, contains contour
        if area < self.area_out_of_size or any([h < 14, w < 14]) or hierarchy[3]<0:    
          continue

        if self.debug:
          cv2.rectangle(mask_src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # roi = src[y:y+h, x:x+w]

        # round lines horiz, vert
        new_x = [col for col in table.col_points if (x - round_alpha < col) & (x + round_alpha > col)][0]
        new_y = [row for row in table.row_points if (y - round_alpha < row) & (y + round_alpha > row)][0]
        # + aligned coords
        rois.append([x, y, w, h, new_x, new_y])
    
    if self.debug:
      cv2_imshow(mask_src)   
    return rois     

  def struct_table(self, cells):
    #выравнивание координат
    x_coord = alignment([cell[0] for cell in cells], thresh=10)
    y_coord = alignment([cell[1] for cell in cells], thresh=10)

    #кривая координата : выровненная
    x_x_map = dict({ (x[0][0],x[1]) for x in zip(cells, x_coord) })
    y_y_map = dict({ (x[0][1],x[1]) for x in zip(cells, y_coord) })

    #перевод координат в номера столбцов колонок
    x_set = list(set(x_coord))
    col_idx = dict({ (x[1],x[0])  for x in enumerate(sorted(x_set)) })
    y_set = list(set(y_coord))
    row_idx = dict({ (x[1],x[0])  for x in enumerate(sorted(y_set)) })

    n_cols = len(x_set)
    n_rows = len(y_set)

    return n_cols,n_rows,x_x_map,y_y_map,col_idx,row_idx

  def to_struct(self, cells):
    table_struct = []
    n_cols,n_rows,x_x_map,y_y_map,col_idx,row_idx = self.struct_table(cells)

    #вывод в таблицу
    for cell in cells:
      x,y,w,h = cell[:4]
      col = col_idx[x_x_map[x]]
      row = row_idx[y_y_map[y]]
      table_struct.append({'col':col,'row':row,'cell':[x,y,w,h]})
    if self.debug:
      draw_struct_table(table_struct)  
    return table_struct  
