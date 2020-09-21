
#tesseract
apt-get install tesseract-ocr-rus

# digits 0-9, dash sign and decimal point, 10 Non-italic fonts
wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits.traineddata
# Arial font
wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits1.traineddata
# digits 0-9, dash sign and decimal point
wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits_comma.traineddata
# rus model
wget -O /usr/share/tesseract-ocr/4.00/tessdata/rus1.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/master/rus.traineddata
wget -O /usr/share/tesseract-ocr/4.00/tessdata/rus2.traineddata https://github.com/tesseract-ocr/tessdata/raw/master/rus.traineddata
