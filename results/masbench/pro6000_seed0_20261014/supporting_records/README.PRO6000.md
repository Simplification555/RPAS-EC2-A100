# 本机 PRO 6000 运行配置

本目录从原 RTX5880 交接包独立复制，原包保留。
硬件：RTX PRO 6000 Blackwell Workstation Edition，97887 MiB，驱动596.49。
Qwen3.5-9B BF16，TP=1，上下文16384，输出上限6144，thinking关闭。
固定全任务16并发，显存比例0.85，batch token预算16384，chunked prefill与prefix caching开启。
MaAS控制器与MiniLM仍在CPU运行，GPU运行vLLM。
修复TransportOwnedGuard审计summary字段，并拒绝缺失usage的响应。
方法环境使用支持Qwen3.5的transformers 5与sentence-transformers 5；安装成功后保留freeze。
新结果目录：outputs/aflow_maas_9b_ctx16k_seed0_pro6000_20261013。
先运行预检，通过后自动进入AFlow/MaAS五轴seed0正式实验。

推理服务使用vLLM 0.18.1 / CUDA版Torch 2.10，方法环境使用CPU版Torch 2.10，均在独立Python3.12虚拟环境中安装。请求传输层共享16并发上限，避免工作流内部并行导致超额请求。

本机实测Torch 2.10.0+cu128、CUDA12.8、vLLM0.18.1在Blackwell sm120上BF16矩阵计算成功。MaAS原生导入检查成功；补齐tree-sitter0.26与tree-sitter-python0.25。
