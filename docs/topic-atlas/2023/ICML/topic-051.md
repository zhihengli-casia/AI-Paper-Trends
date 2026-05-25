# ICML 2023: LLM 推理系统与 GPU Kernel 优化

- Topic ID: `51`
- Papers: **16** (0.88%)
- Macro topic: 系统/效率/压缩
- English keywords: `quantization / language / gpu / memory`
- Keyword pool: quantization; language; gpu; memory; scaling; bit; llms; inference; decoding; quantized; pipeline; activations

[Back to ICML 2023](README.md) | [Atlas home](../../README.md)

## Representative Papers

- Cramming: Training a Language Model on a single GPU in one day.
- FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- Unit Scaling: Out-of-the-Box Low-Precision Training

## Papers

<a id="paper-https-proceedings-mlr-press-v202-vilnis23a-html"></a>
1. [Arithmetic Sampling: Parallel Diverse Decoding for Large Language Models](https://openreview.net/forum?id=EfhmBBrXY2)
   - Luke Vilnis, Yury Zemlyanskiy, Patrick Murray, Alexandre Tachard Passos, Sumit Sanghai; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-kim23l-html"></a>
2. [BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models](https://openreview.net/forum?id=HVKmLi1iR4)
   - Taebum Kim, Hyoungjoo Kim, Gyeong-In Yu, Byung-Gon Chun; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-geiping23a-html"></a>
3. [Cramming: Training a Language Model on a single GPU in one day.](https://openreview.net/forum?id=2snzoozOWH)
   - Jonas Geiping, Tom Goldstein; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-wang23aa-html"></a>
4. [Data Efficient Neural Scaling Law via Model Reusing](https://openreview.net/forum?id=iXYnIz4RRx)
   - Peihao Wang, Rameswar Panda, Zhangyang Wang; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-leviathan23a-html"></a>
5. [Fast Inference from Transformers via Speculative Decoding](https://openreview.net/forum?id=C9NEblP8vS)
   - Yaniv Leviathan, Matan Kalman, Yossi Matias; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-sheng23a-html"></a>
6. [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://openreview.net/forum?id=RRntzKrBTp)
   - Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, et al.; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-lee23h-html"></a>
7. [FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization](https://openreview.net/forum?id=EPnzNJTYsb)
   - Jung Hyun Lee, Jeonghoon Kim, Se Jung Kwon, Dongsoo Lee; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-nova23a-html"></a>
8. [Gradient-Free Structured Pruning with Unlabeled Data](https://openreview.net/forum?id=Ga6nQOAb7A)
   - Azade Nova, Hanjun Dai, Dale Schuurmans; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-zeng23a-html"></a>
9. [LookupFFN: Making Transformers Compute-lite for CPU inference](https://openreview.net/forum?id=MmYoDC7dH9)
   - Zhanpeng Zeng, Michael Davies, Pranav Pulijala, Karthikeyan Sankaralingam, Vikas Singh; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-de-jong23a-html"></a>
10. [Pre-computed memory or on-the-fly encoding? A hybrid approach to retrieval augmentation makes the most of your compute](https://openreview.net/forum?id=nlUAvrMbUZ)
   - Michiel De Jong, Yury Zemlyanskiy, Nicholas Fitzgerald, Joshua Ainslie, Sumit Sanghai, Fei Sha, et al.; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-markov23a-html"></a>
11. [Quantized Distributed Training of Large Models with Convergence Guarantees](https://openreview.net/forum?id=Nqp8A5IDzq)
   - Ilia Markov, Adrian Vladu, Qi Guo, Dan Alistarh; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-xiao23c-html"></a>
12. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://openreview.net/forum?id=sHfSV8eYEp)
   - Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-longpre23a-html"></a>
13. [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://openreview.net/forum?id=ZX4uS605XV)
   - Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, et al.; assignment: `nearest_centroid`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-dettmers23a-html"></a>
14. [The case for 4-bit precision: k-bit Inference Scaling Laws](https://openreview.net/forum?id=i8tGb1ab1j)
   - Tim Dettmers, Luke Zettlemoyer; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-wu23k-html"></a>
15. [Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases](https://openreview.net/forum?id=q1WGm3hItW)
   - Xiaoxia Wu, Cheng Li, Reza Yazdani Aminabadi, Zhewei Yao, Yuxiong He; assignment: `hdbscan`; source: `PMLR`
<a id="paper-https-proceedings-mlr-press-v202-blake23a-html"></a>
16. [Unit Scaling: Out-of-the-Box Low-Precision Training](https://openreview.net/forum?id=A8HOsNfish)
   - Charlie Blake, Douglas Orr, Carlo Luschi; assignment: `hdbscan`; source: `PMLR`
