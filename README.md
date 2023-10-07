# Moonlit: Research for enhancing AI models' efficiency and performance.

**Moonlit** is a collection of our model compression work for efficient AI.

> [**ToP**](./ToP) (```@KDD'23```): [**Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference**](https://arxiv.org/abs/2306.14393)
>>**ToP** is a constraint-aware and ranking-distilled token pruning method, which selectively removes unnecessary tokens as input sequence pass through layers, allowing the model to improve online inference speed while preserving accuracy.
> 
> [**SpaceEvo**](./SpaceEvo) (```@ICCV'23```): [**SpaceEvo: Hardware-Friendly Search Space Design for Efficient INT8 Inference**](https://arxiv.org/abs/2303.08308)
>>**SpaceEvo** is an automatic method for designing a dedicated, quantization-friendly search space for target hardware. This work is featured on Microsoft Research blog: [Efficient and hardware-friendly neural architecture search with SpaceEvo](https://www.microsoft.com/en-us/research/blog/efficient-and-hardware-friendly-neural-architecture-search-with-spaceevo/)
> 
> [**ElasticViT**](./ElasticViT) (```@ICCV'23```): [**ElasticViT: Conflict-aware Supernet Training for Deploying Fast Vision Transformer on Diverse Mobile Devices**](https://arxiv.org/abs/2303.09730)
>>**ElasticViT** is a two-stage NAS approach that trains a high-quality ViT supernet over a very large search space for covering a wide range of mobile devices, and then searches an optimal sub-network (subnet) for direct deployment. 
>
> [**LitePred**](./LitePred/) (```@NSDI'24```): [**LitePred: Transferable and Scalable Latency Prediction for Hardware-Aware Neural Architecture Search**]()
>>**LitePred** is a lightweight transferrable approach for accurately predicting DNN inference latency. Instead of training a latency predictor from scratch, LitePred is the first to transfer pre-existing latency predictors and achieve accurate prediction on new edge platforms with a profiling cost of less than 1 hour. 

