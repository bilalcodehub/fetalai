# Fetal AI


docker build --no-cache -t fetalai:latest .


docker run --gpus all -it --rm -p 8889:8888 -v ~/mlworks:/data fetalai:latest


http://127.0.0.1:8889/lab
