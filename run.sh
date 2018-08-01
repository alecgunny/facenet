#!/bin/bash
NV_GPU=0 nvidia-docker run --rm -it -v $(pwd)/src:/workspace/facenet/src/ $USER/facenet:test $1 $2
