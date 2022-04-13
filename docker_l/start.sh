sudo service docker start

docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
        -v "$(pwd)/notebooks:/rapids/notebooks/host" \
        rapidsai/rapidsai:cuda10.1-runtime-ubuntu18.04