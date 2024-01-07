import ffmpeg
import numpy as np

import random
import torch

import decord

def vread_ffmpeg(vpath, num_frames, mode="random"):
    
    assert mode in ["random", "mean"], f"video reading mode {mode} is not supported."
    
    probe = ffmpeg.probe(vpath)
    stream_dict = probe['streams'][0]
    format_dict = probe['format']
    width, height = stream_dict['width'], stream_dict['height'] #, stream_dict['pix_fmt']
    
    stream = ffmpeg.input(vpath, vsync='0')
    out, info = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24', v='trace')\
        .run(capture_stdout=True, capture_stderr=True)
    
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3]) # frames in np.array
    video_length = len(frames)
    
    num_frames = min(num_frames, video_length)
    seg_size = float(video_length - 1) / num_frames
    seq = []
    for i in range(num_frames):
        s = int(np.round(seg_size * i))
        e = int(np.round(seg_size * (i + 1)))
        if mode == "random":
            seq.append(random.randint(s, e))
        elif mode == "mean":
            seq.append((s + e) // 2)
    
    return torch.as_tensor(frames[seq]).permute(0, 3, 1, 2) # bchw

def vread_decord(vpath, num_frames, mode="random", max_batch_sz=-1):
    
    assert mode in ["random", "mean"], f"video reading mode {mode} is not supported."
    
    decord.bridge.set_bridge("torch")
    stream = decord.VideoReader(open(vpath, "rb"))
    fps = stream.get_avg_fps()
    video_length = len(stream)
    
    num_frames = min(num_frames, video_length)
    seg_size = float(video_length - 1) / num_frames
    seq = []
    for i in range(num_frames):
        s = int(np.round(seg_size * i))
        e = int(np.round(seg_size * (i + 1)))
        if mode == "random":
            seq.append(random.randint(s, e))
        elif mode == "mean":
            seq.append((s + e) // 2)
            
    if max_batch_sz == -1:
        return stream.get_batch(np.array(seq))

    out = []
    max_batch_sz = min(max_batch_sz, len(seq))
    for left in range(0, len(seq), max_batch_sz):
        out.append(stream.get_batch(np.array(seq)[np.arange(left, left + max_batch_sz)]))
    return torch.cat(out, dim=0).permute(0, 3, 1, 2) # bchw
