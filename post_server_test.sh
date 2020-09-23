# curl -iX POST -H "Content-Type: application/json" -d '{"text":"Автомобиль  "}' "localhost:9090/okpdsearch"

# curl -F "file=@data/ScanImage388.jpg" "localhost:9095/predict"
curl -F "file=@data/ScanImage388.jpg" "172.16.0.247:9095/predict"

