import cv2
import numpy as np


def figure_to_image(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def frames_to_video(path, frames, fps=30):
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()
