FROM ubuntu:latest
LABEL authors="lbachmaier"

ENTRYPOINT ["top", "-b"]