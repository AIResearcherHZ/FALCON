#!/usr/bin/env bash
# tmux 后台跑训练,断 SSH 不挂;自动激活 conda、tee 日志。
# 用法: ./run_train.sh <会话名> <完整训练命令...>   (命令见 注释.txt 步骤1)
#   指定卡(单卡/多卡通用): 在最前面设 CUDA_VISIBLE_DEVICES=... (会自动带进 tmux 会话)。
#     例(单卡用 2 号卡):     CUDA_VISIBLE_DEVICES=2 ./run_train.sh g1 python humanoidverse/train_agent.py +exp=... ...
#     例(多卡用 0,1,2,3 号卡): CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./run_train.sh g1m python humanoidverse/train_agent.py +exp=... ...
#   单卡: 默认 NPROC=1(不设 CUDA_VISIBLE_DEVICES 则用 0 号卡)。
#   多卡(单机多卡,同步数据并行 DDP): 设 NPROC=张数,训练命令须以 python 开头(会替换成 torchrun)。
#     例: NPROC=4 ./run_train.sh g1 python humanoidverse/train_agent.py +exp=... ...
#     说明: 每卡一个进程,梯度跨卡平均。strong 扩展(默认)总环境数 = num_envs(各卡均分 num_envs/NPROC),等价单卡配方;
#           num_envs 须能被 NPROC 整除;NPROC 不超过 CUDA_VISIBLE_DEVICES 卡数(不设则用全部卡);weak 扩展加 scaling_mode=weak(总数×卡数,需重新调参)。
#   CONDA_ENV=xxx 换环境,CONDA_ENV= 跳过激活。
#   看输出: tmux attach -t <会话名> (脱离 Ctrl-b 再 d) | 停止: tmux kill-session -t <会话名>
set -euo pipefail

SESSION="${1:?用法: ./run_train.sh <会话名> <训练命令...>}"
shift
[ "$#" -gt 0 ] || { echo "❌ 缺少训练命令。" >&2; exit 1; }

# 多卡: NPROC>1 时把 'python ...' 替换为 'torchrun --standalone --nnodes=1 --nproc_per_node=NPROC ...'
NPROC="${NPROC:-1}"
case "$NPROC" in (''|*[!0-9]*) echo "❌ NPROC 必须是正整数,当前: '$NPROC'。" >&2; exit 1;; esac
if [ "$NPROC" -gt 1 ]; then
  if [ "${1:-}" = "python" ] || [ "${1:-}" = "python3" ]; then
    shift
  else
    echo "❌ 多卡模式(NPROC=$NPROC)要求训练命令以 'python' 开头(会替换成 torchrun)。" >&2
    exit 1
  fi
  set -- torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" "$@"
fi

cd "$(dirname "$0")"   # 切到仓库根,保证 humanoidverse/train_agent.py 相对路径可用
REPO_DIR="$(pwd)"
mkdir -p logs/launch
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/launch/${SESSION}_${TS}.log"

# 激活 conda(tmux 干净 shell 不会自动激活);设 CONDA_ENV= 跳过。
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-fcgym}}"
CONDA_PREFIX_CMD=""
if [ -n "$CONDA_ENV" ]; then
  CONDA_BASE=""
  if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="${CONDA_EXE%/bin/conda}"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  fi
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    CONDA_PREFIX_CMD="source $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") && conda activate $(printf '%q' "$CONDA_ENV") && "
  else
    echo "⚠️  未找到 conda 安装根,不会自动激活 '$CONDA_ENV';请先手动 conda activate 或设 CONDA_ENV= 跳过。" >&2
  fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "❌ tmux 会话 '$SESSION' 已存在。先 'tmux attach -t $SESSION' 看看,或换个会话名。" >&2
  exit 1
fi

# 捕获外层 CUDA_VISIBLE_DEVICES 注入 tmux(否则在新会话里丢失;单卡/多卡均生效)。
CVD_CMD=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CVD_CMD="export CUDA_VISIBLE_DEVICES=$(printf '%q' "$CUDA_VISIBLE_DEVICES") && "
fi

# printf %q 安全拼接(保留 CUDA_VISIBLE_DEVICES=0 前缀);激活+cd 成功后才训练,退出后 read 保留窗口。
INNER="${CONDA_PREFIX_CMD}${CVD_CMD}cd $(printf '%q' "$REPO_DIR") && PYTHONUNBUFFERED=1 $(printf '%q ' "$@")2>&1 | tee $(printf '%q' "$LOG"); echo; echo '==== 训练进程已退出,按回车关闭本窗口 ===='; read"
tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$INNER")"

echo "✅ 已在 tmux 会话 '$SESSION' 启动训练(断 SSH 不中断)。"
[ "$NPROC" -gt 1 ] && echo "   多卡: torchrun $NPROC 进程(strong 扩展: 总环境数恒定 num_envs,各卡均分)" || echo "   单卡模式(NPROC=1)"
[ -n "$CONDA_PREFIX_CMD" ] && echo "   conda 环境: $CONDA_ENV"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "   指定卡: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   实时查看: tmux attach -t $SESSION   跟踪日志: tail -f $REPO_DIR/$LOG"
echo "   全部会话: tmux ls                 停止训练: tmux kill-session -t $SESSION"
