export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${MACA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/mxdriver/lib:${LD_LIBRARY_PATH} 

export TOKENIZERS_PARALLELISM=true
export MACA_SMALL_PAGESIZE_ENABLE=1
# 用于调试内存问题，打开会降低执行效率
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

export MCCL_MAX_NCHANNELS=16
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_USE_FLASHATTN=1


python -m vllm.entrypoints.openai.api_server \
	--model="path to Qwen2.5-Coder-14B-Instruct folder" \
	--trust-remote-code \
	--device auto \
	--gpu-memory-utilization 0.7 \
	--served-model-name "Qwen2.5-Coder-14B-Instruct" \
	--host 0.0.0.0 \
	--dtype "float16" \
	--port 8300


# shell中测试方法
# curl -v -X POST 'http://127.0.0.1:8300/v1/chat/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{ "model": "Qwen2.5-Coder-14B-Instruct", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100 }'