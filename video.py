from constants import SAMPLING_RATE

class Frame:
    def __init__(self, image, keypoints_with_scores):
        self.image = image
        self.keypoints_with_scores = keypoints_with_scores
    
    def detect_defects(self):
        return None

class Video:
    def __init__(self):
        self.frames = []
    
    def add_frame(self, image, keypoints_with_scores):
        self.frames.append(Frame(image=image, keypoints_with_scores=keypoints_with_scores))

    def process_frames(self):
        pass