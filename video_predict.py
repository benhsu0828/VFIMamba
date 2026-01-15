import cv2
import torch
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder
import time

current_timestamp = time.time()

# 載入模型
model = Model(-1)
model.load_model()
model.eval()
model.device()

# 讀取影片
video = cv2.VideoCapture('result_video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 輸出影片
out = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps*2, (width, height))

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
    
    # 生成中間幀
    mid = model.inference(I0, I2, True)
    mid = padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    mid = mid.astype('uint8')
    
    # 寫入影片
    out.write(frame1)
    out.write(mid)
    
    frame1 = frame2

out.write(frame1)  # 最後一幀
video.release()
out.release()
end_timestamp = time.time()
print(f"Processing time: {end_timestamp - current_timestamp} seconds")