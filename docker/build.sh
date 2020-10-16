#!/bin/sh

NAME=$1

docker container stop -t 0 $NAME
docker image build -t $NAME docker/

