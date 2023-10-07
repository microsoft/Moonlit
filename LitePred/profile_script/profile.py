# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from ADBConnect import *
from bench_utils import*
from run_on_device  import*


device_seris = "1c19316"
adb = ADBConnect(device_seris)
model_path = '/path/to/your/model'
_,avg,_ = run_on_android(model_path,adb)