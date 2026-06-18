import math
import os
import cv2
import torch
import argparse
import config as cfg
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder
import time

# 解析命令行參數
parser = argparse.ArgumentParser(description='VFIMamba 視頻插幀')
parser.add_argument('--video', type=str, required=True, help='輸入視頻路徑')
parser.add_argument('--output', type=str, required=True, help='輸出視頻路徑')
parser.add_argument('--model', type=str, default='VFIMamba_S', choices=['VFIMamba_S', 'VFIMamba'], help='模型大小 (VFIMamba_S 較快, VFIMamba 較準)')
parser.add_argument('--multiplier', type=int, default=2, help='幀率倍數，須為 2 的冪 (2/4/8/...)')
args = parser.parse_args()

if args.multiplier < 2 or (args.multiplier & (args.multiplier - 1)) != 0:
    parser.error(f'--multiplier 必須是 2 的冪 (2, 4, 8, ...)，收到: {args.multiplier}')
num_recursions = int(math.log2(args.multiplier))

current_timestamp = time.time()

# 載入模型
TTA = False
if args.model == 'VFIMamba':
    TTA = True
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])

model = Model(-1)
model.load_model()
model.eval()
model.device()


def recursive_interpolate(frame1, frame2, depth):
    '''遞迴二分插幀，回傳 frame1~frame2 之間依時間排序的中間幀（不含端點）'''
    if depth == 0:
        return []
    mid = model.inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA)
    left = recursive_interpolate(frame1, mid, depth - 1)
    right = recursive_interpolate(mid, frame2, depth - 1)
    return left + [mid] + right


# 讀取影片
video = cv2.VideoCapture(args.video)
if not video.isOpened():
    raise RuntimeError(f"無法開啟輸入影片（檔案不存在或損毀）: {args.video}")
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"處理視頻: {args.video}")
print(f"模型: {args.model}, 倍數: {args.multiplier}x")
print(f"原始 FPS: {fps}, 輸出 FPS: {fps * args.multiplier}")
print(f"分辨率: {width}x{height}")

# 輸出影片（自動建立輸出目錄）
output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps * args.multiplier, (width, height))
if not out.isOpened():
    raise RuntimeError(f"無法建立輸出影片（路徑或編碼器不支援）: {args.output}")

ret, frame1 = video.read()
while True:
    ret, frame2 = video.read()
    if not ret:
        break

    # 轉換為 tensor
    I0 = (torch.tensor(frame1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2 = (torch.tensor(frame2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    # Padding
    padder = InputPadder(I0.shape, divisor=32)
    I0, I2 = padder.pad(I0, I2)

    # 生成中間幀（依倍數遞迴二分）
    mids = recursive_interpolate(I0, I2, num_recursions)
    mids = [padder.unpad(m)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0 for m in mids]
    mids = [m.astype('uint8') for m in mids]

    # 寫入影片
    out.write(frame1)
    for mid in mids:
        out.write(mid)

    frame1 = frame2

out.write(frame1)  # 最後一幀
video.release()
out.release()
end_timestamp = time.time()
print(f"✅ 處理完成！")
print(f"⏱️  總耗時: {end_timestamp - current_timestamp:.2f} 秒")
print(f"📁 輸出文件: {args.output}")
