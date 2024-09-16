from abc import ABC, abstractmethod
from video import Video, Frame

class Defect(ABC):

    @abstractmethod
    def preprocess_frame(self, frame: Frame):
        pass

    @abstractmethod
    def detect_defect(self):
        pass
