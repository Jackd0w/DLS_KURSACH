
import os
import sys

cycle_gan_official_repository_gitcloned_root_dir = os.path.join(
    os.path.dirname(__file__),
    "pytorch-CycleGAN-and-pix2pix"
)

sys_path = sys.path
sys.path = [cycle_gan_official_repository_gitcloned_root_dir] + sys_path

import data
import models
import options
import util

sys.path = sys_path

del sys, os, sys_path