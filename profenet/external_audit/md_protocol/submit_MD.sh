#!/bin/bash

# --- 配置 ---
MAX_SUBMIT=10
SUBMIT_COUNT=0
OUTPUT_DIR="output"
SUBMIT_LOG="submitted_list.txt"
FINISH_LOG="finished_list.txt"
NOT_FINISHED_LOG="not_finished_list.txt"

# 初始化记录文件
touch "$SUBMIT_LOG" "$FINISH_LOG"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 找不到 $OUTPUT_DIR 文件夹，请先运行 Python 提取脚本。"
    exit 1
fi

########################################
# 1) 检测已提交但未成功完成的体系
########################################
> "$NOT_FINISHED_LOG"   # 清空/创建 not_finished_list.txt

echo "===== 检查已提交但未成功完成的体系 ====="

while read -r dir_name; do
    # 跳过空行
    [ -z "$dir_name" ] && continue

    # 如果已经在 finished_list 里，认为成功完成，跳过
    if grep -qxw "$dir_name" "$FINISH_LOG"; then
        continue
    fi

    # 如果没有 prod.xtc，认为没有成功跑完
    if [ ! -f "$OUTPUT_DIR/$dir_name/prod.xtc" ]; then
        echo "$dir_name" | tee -a "$NOT_FINISHED_LOG"
    fi
done < "$SUBMIT_LOG"

NOT_FINISHED_COUNT=$(wc -l < "$NOT_FINISHED_LOG")
echo "共找到 $NOT_FINISHED_COUNT 个已提交但未成功完成的体系，列表已写入 $NOT_FINISHED_LOG"
echo "========================================="
echo

########################################
# 2) 像之前一样提交新的任务
########################################

cd "$OUTPUT_DIR"

for dir in */; do
    # 去掉末尾的斜杠
    dir_name=${dir%/}
    
    # --- 过滤 seed-0 和 seed-1 ---
    if [[ ! "$dir_name" == *"seed-0"* && ! "$dir_name" == *"seed-1"* ]]; then
        continue
    fi
    # ------------------------------

    [ $SUBMIT_COUNT -ge $MAX_SUBMIT ] && break

    # 已经提交过 或 已经完成的，都跳过（这里的逻辑保持不变，不会自动重提）
    if grep -qxw "$dir_name" "../$SUBMIT_LOG" || grep -qxw "$dir_name" "../$FINISH_LOG"; then
        continue
    fi

    if [ -f "${dir_name}/complex.pdb" ]; then
        echo ">>> 提交任务: $dir_name"
        slurm_submit -n "MD_${dir_name}" -g 1 -c 8 '<ROOT>/result-20/slurm_task.sh '"$dir_name"''
        
        echo "$dir_name" >> "../$SUBMIT_LOG"
        ((SUBMIT_COUNT++))
    fi
done

echo "任务提交结束。当前提交数: $SUBMIT_COUNT"

