# LitePred: Transferable and Scalable Latency Prediction for Hardware-Aware Neural Architecture Search

LitePred, a lightweight approach for accurately predicting DNN inference latency on new platforms with minimal adaptation data by transferring existing predictors.At the core of LitePred lies the principle that knowledge from a pre-existing latency predictor for one platform can be transferred to new platforms that share similarities.


## Results and Predictors
our prebuild predictors can be accessed [here](https://huggingface.co/fcq/pred_lite/tree/main)

### (a)Selecting most similar kernel predictors from the whole knowledge pool

| Platform | Adaptation  cost <br> #Data #Time |  Prediction Accuracy <br>  5% 10%|
| ----------- | ----------- | ---------| 
| Xiaomi11CPU, ORT|  1400 0.48h |   90.5% 98.9%|
| Pixel5GPU,NCNN| 17400 0.96h | 84.3% 99.1% |
| Xiaomi11CPU, Mindspore| 4800 0.35h| 90.4% 99.9%|
| Xiaomi11GPU,TFLite2.7| 11000 0.17h| 83.7% 98.6%|
| Xiaomi11CPU,NCNN| 11400 0.88h| 80.3% 98.9%|
| Pixle6CPU,TFLite2.1| 3500 0.16h| 79.4% 100%|
| Pixel5CPU, TFLite2.7| 3400 0.13h| 79.6% 99.2%|
| Xiaomi12CPU,TFLite2.7,INT8| 3100 0.05h | 95.7% 100%|

### (b)Similarity detection of kernel predictors Excluding same inference frameworks
| Platform | Adaptation  cost <br> #Data #Time |  Prediction Accuracy <br>  5% 10%|
| ----------- | ----------- | ---------| 
|Xiaomi11CPU,ORT| 2400 0.72h|84.2% 99.2%|
|Xiaomi12GPU,TFLite2.7|16100 0.22h|79.4%,98.7%|
|Xiaomi11CPU,Mindspore|9700 0.80h|98.1%,99.2%|
|Pixel5GPU,NCNN|18500 1.73h|86.5% 99.3%|
|Xiaomi12CPU,TFLite2.1,low Freq| 1800 0.18h|94.7% 100%|
|Xiaomi12CPU,TFLite2.1|1800 0.10h|97.6% 99.9%|



##  Using  LitePred
[build your own predictor](https://github.com/microsoft/Moonlit/tree/main/LitePred/predictor_builder)  
[Use our predictors to predict your model latency](https://github.com/microsoft/Moonlit/tree/main/LitePred/predition_example)  
[Use vae to collect effective multi-dimensional data](https://github.com/microsoft/Moonlit/tree/main/LitePred/vae)  



