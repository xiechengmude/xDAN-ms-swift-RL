#!/bin/bash

# 搜索专家智能体训练脚本
# 基于GRPO框架的多轮迭代搜索训练

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用8卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model_type qwen2_5 \
    --model /data/vayu/train/models/xDAN-R2-Thinking-Base \
    --dataset search_agent_dataset \
    --output_dir output/search_agent_grpo \
    --sft_type lora \
    --train_dataset_sample -1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_length 4096 \
    --max_prompt_length 3072 \
    --max_completion_length 1024 \
    --reward_funcs relevance completeness reliability depth \
    --reward_weights 0.4 0.3 0.2 0.1 \
    --num_generations 16 \
    --system examples/train/grpo/search_agent_prompt.txt \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --deepspeed zero3 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 50 \
    --log_completions true \
    --num_infer_workers 8 \
    --tensor_parallel_size 4 \
    --async_generate false \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --sleep_level 1 \
    --multi_turn_func search_quality_evaluator
