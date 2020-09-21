from doc_extractor import extract, read_image

if __name__ == '__main__':
  filename = 'data/ScanImage388.jpg'
  img = read_image(filename)
  json = extract(img)
  print(json)

  # # extract_all('data')
