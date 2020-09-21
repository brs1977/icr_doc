#!/bin/bash
app="icr"

docker build -t ${app} .

docker run -d -p 9095:9095 ${app} 

