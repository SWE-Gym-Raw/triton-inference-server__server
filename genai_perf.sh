#!/bin/bash
FRONTEND=${FRONTEND:-"kserve"}
PROFILE_EXPORT_DIR="export"
CONCURRENCY=${CONCURRENCY:-200}
REQUESTS_PER_THREAD=${REQUESTS_PER_THREAD:-1}
REQUEST_COUNT=$((CONCURRENCY*REQUESTS_PER_THREAD))
ISL=${ISL:-200}
OSL=${OSL:-200}
EXTRA_FLAGS=${EXTRA_FLAGS:-""}
PASSTHROUGH_FLAGS=${PASSTHROUGH_FLAGS:-""}
TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL="vllm_dummy"

if [[ ${FRONTEND} == "kserve" ]]; then
    genai-perf profile -m ${MODEL} \
            -u localhost:8001 \
            --service-kind triton \
            --backend vllm \
            --output-tokens-mean-deterministic \
            --streaming \
            --num-prompts 100 \
            --synthetic-input-tokens-mean ${ISL} \
            --synthetic-input-tokens-stddev 0 \
            --concurrency ${CONCURRENCY} \
            --output-tokens-mean ${OSL} \
            --output-tokens-stddev 0 \
            --profile-export-file ${PROFILE_EXPORT_DIR} \
            --tokenizer=${TOKENIZER} \
            --measurement-interval=100000 \
            ${EXTRA_FLAGS} \
            -- --max-threads=${CONCURRENCY} --request-count=${REQUEST_COUNT} \
            ${PASSTHROUGH_FLAGS}
elif [[ ${FRONTEND} == "generate" ]]; then
    genai-perf profile -m ${MODEL} \
            -u localhost:8000 \
            --service-kind triton \
            --endpoint-type generate \
            --endpoint v2/models/${MODEL}/generate_stream \
            --streaming \
            --num-prompts 100 \
            --synthetic-input-tokens-mean ${ISL} \
            --synthetic-input-tokens-stddev 0 \
            --concurrency ${CONCURRENCY} \
            --output-tokens-mean ${OSL} \
            --output-tokens-stddev 0 \
            --profile-export-file ${PROFILE_EXPORT_DIR} \
            --tokenizer=${TOKENIZER} \
            --measurement-interval=100000 \
            ${EXTRA_FLAGS} \
            -- --max-threads=${CONCURRENCY} --request-count=${REQUEST_COUNT} \
            ${PASSTHROUGH_FLAGS}
            #--backend vllm \
            #--output-tokens-mean-deterministic \
elif [[ ${FRONTEND} == "openai" ]]; then
    genai-perf profile -m ${MODEL} \
            -u localhost:9000 \
            --service-kind openai \
            --endpoint-type chat \
            --streaming \
            --num-prompts 100 \
            --synthetic-input-tokens-mean ${ISL} \
            --synthetic-input-tokens-stddev 0 \
            --concurrency ${CONCURRENCY} \
            --output-tokens-mean ${OSL} \
            --output-tokens-stddev 0 \
            --profile-export-file ${PROFILE_EXPORT_DIR} \
            --tokenizer=${TOKENIZER} \
            --measurement-interval=100000 \
            --extra-input ignore_eos:true \
            --extra-inputs max_tokens:${OSL} \
            --extra-inputs min_tokens:${OSL} \
            ${EXTRA_FLAGS} \
            -- --max-threads=${CONCURRENCY} --request-count=${REQUEST_COUNT} \
            ${PASSTHROUGH_FLAGS}
else
    echo 'ERROR: Unknown frontend! Pick FRONTEND="kserve" or FRONTEND="openai"'
    exit 1
fi

#           --request-parameter max_tokens:200:int \
#           --request-parameter min_tokens:200:int \
#           --request-parameter ignore_eos:true:bool

#        --extra-input "sampling_parameters:{\"max_tokens\": 200, \"min_tokens\": 200, \"ignore_eos\": true}" \

