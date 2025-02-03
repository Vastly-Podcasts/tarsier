from typing import List
import os
from PIL import Image, ImageSequence
import decord

VALID_DATA_FORMAT_STRING = "Input data must be {'.jpg', '.jpeg', '.png', '.tif'} for image; or {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv', '.gif'}  for videos!"

def sample_frame_indices(start_frame, total_frames: int, n_frames: int):
    if n_frames == 1:
        return [0]  # sample first frame in default
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    sample_ids = [i + start_frame for i in sample_ids]
    return sample_ids

def sample_video(
    video_path: str, 
    n_frames: int = 8,
    start_time: int = 0,
    end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(video_path), f"File not found: {video_path}"
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    vr.seek(0)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    print(f"Video loaded: {total_frames} frames, {fps:.2f} fps")

    start_frame = 0
    end_frame = total_frames - 1
    if start_time > 0:
        start_frame = min((total_frames-1), int(fps*start_time))
    if end_time > 0:
        end_frame = max(start_frame, int(fps*end_time))
        end_frame = min(end_frame, (total_frames-1))
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )
    print(f"Sampling {n_frames} frames at indices: {frame_indices}")

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(f).convert('RGB') for f in frames]
    return frames

def sample_gif(
        gif_path: str,
        n_frames: int = 8,
        start_time: int = 0,
        end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(gif_path), f"File not found: {gif_path}"
    
    gif_frames = Image.open(gif_path)
    print(f"GIF loaded: {gif_frames.n_frames} frames")

    start_frame = 0
    end_frame = gif_frames.n_frames - 1
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )
    print(f"Sampling {n_frames} frames at indices: {frame_indices}")
        
    frames = []
    i = 0
    for frame in ImageSequence.Iterator(gif_frames):
        if i in frame_indices:
            frames.append(frame.convert('RGB'))
        i += 1
    return frames

def sample_image(
    image_path: str, 
    n_frames: int = 1,
    start_time: int = 0,
    end_time: int = -1
    ):
    assert os.path.exists(image_path), f"File not found: {image_path}"
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    return [image]

def get_visual_type(input_file):
    ext = os.path.splitext(input_file)[-1]
    if ext in {'.gif'}:
        return 'gif'
    elif ext in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv'}:
        return 'video'
    elif ext in {'.jpg', '.jpeg', '.png', '.tif'}:
        return 'image'
    else:
        print(f"{VALID_DATA_FORMAT_STRING} But found {ext}!")
        return 'unk'

def load_images(self, visual_data_path, n_frames=8, start_time=0, end_time=-1):
    """Load images from a video/image file.
    
    Args:
        visual_data_path (str): Path to video/image file
        n_frames (int): Number of frames to sample (default: 8)
        start_time (int): Start time in seconds
        end_time (int): End time in seconds
    """
    sampler = self.select_frames_sampler(visual_data_path)
    print(f"Using sampler for type: {get_visual_type(visual_data_path)}")
    return sampler(visual_data_path, n_frames=n_frames, start_time=start_time, end_time=end_time) 