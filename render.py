#@title Блоп {vertical-output: true}
from netty import Netty
from netty import imutil as im
import os
import numpy as np

def render(IMAGES, BACKGROUND, MIN_SIZE, MAX_SIZE, STEP, ITERS):

  if len(BACKGROUND) == 0:
    x0 = None
  else:
    x0 = im.load("IN"+"/"+BACKGROUND)
  if type(MIN_SIZE) in (tuple, list):
    size = np.int32(MIN_SIZE)
  else:
    size = np.int32([MIN_SIZE, MIN_SIZE])
  if type(IMAGES) not in (tuple, list):
    IMAGES = [IMAGES]
  imgs = []
  name = "bg_"+BACKGROUND+"_size_"+str(size)
  for img in IMAGES:
    name += "_"+img
    imgs.append(im.load("IN/"+img))

  it = 0
  while True:
    fs = [os.path.splitext(f)[0] for f in os.listdir("OUT")]
    if name+str(it) not in fs:
      name = name+str(it)
      break
    else:
      it += 1

  net = Netty()

  iters = ITERS
  while max(size) <= MAX_SIZE:
    net.clear()
    net.size(*size)
    net.x0(x0)
    for img in imgs:
      net.style(img, [1,4,7,9], [1,1,1,0.01])
    x0 = net.render(iters)
    size = np.int32(size*STEP/2)*2
    iters = iters // STEP
    im.save(x0, "OUT/"+name+".jpg")

#   files.download("OUT/"+name+".jpg")