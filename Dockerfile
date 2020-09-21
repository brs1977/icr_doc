
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
#Обновить информацию о репозиториях внутри контейнера
RUN apt-get update -y   
#Установить внутрь контейнера пакеты: python-pip, python-dev, build-essential
RUN apt-get install -y python-pip python-dev build-essential libgl1-mesa-dev tesseract-ocr-rus

# digits 0-9, dash sign and decimal point, 10 Non-italic fonts
RUN wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits.traineddata
# Arial font
RUN wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits1.traineddata
# digits 0-9, dash sign and decimal point
RUN wget -P /usr/share/tesseract-ocr/4.00/tessdata https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits_comma.traineddata
# rus model
RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/rus1.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/master/rus.traineddata
RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/rus2.traineddata https://github.com/tesseract-ocr/tessdata/raw/master/rus.traineddata


COPY requirements.txt /tmp
WORKDIR /tmp
#Установить зависимости, сохраненные вами в requirements.txt. Данная команда установить Flask и все, что необходимо для его запуска внутри контейнера.
RUN pip install -r requirements.txt 

#Скопировать содержимое текущей директории «.» в директорию /app внутри образа. Внимание: текущей директорией в процессе сборки будет считаться директория, содержащая Dockerfile
COPY . /app             
#Сменить рабочую директорию внутри контейнера. Все команды далее будут запускаться внутри директории /app внутри контейнера
WORKDIR /app    

# ENTRYPOINT ["unicorn"]   
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9095"]
