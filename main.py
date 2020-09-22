from doc_extractor import extract, read_image
import timeit

def test():
  filename = 'data/ScanImage388.jpg'
  img = read_image(filename)
  json = extract(img)
  print(json)

  # # extract_all('data')

if __name__ == '__main__':
  print(timeit.timeit('test()', globals=globals(), number = 1))