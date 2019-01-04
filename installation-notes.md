# Installation Notes

All things that need to be done for this project to work should be documented here.


## cppyy

Tested with python3.7 - let pip point to the correct version.

Install C++17 version of cppyy:

```
pip install --find-links=https://cern.ch/wlav/wheels/cppyy-cling cppyy-cling --no-cache-dir --no-index
pip install cppyy --no-cache-dir --no-binary :all:
```
