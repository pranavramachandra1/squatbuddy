from defect import Defect
from video import Video, Frame
from tqdm import tqdm

# Local packages
from features import COM_SHIFT_FEATURES

class COM_SHIFT(Defect):

    """
    COM_SHIFT (or Center of Mass-Shift) defect detects if the user's hips are swaying to the right/left.
    This imbalance introduces uneven stressed placed on the spine and the hips, and compromises the integrity
    of the squat. 

    For more information on this defect, please checkout: <<INSERT LINK HERE>>
    """

    def __init__(self, name: str, video: Video):
        self.name = name
        self.video = video
        self.features = COM_SHIFT_FEATURES

    def peprocess_frame(self, frame: Frame):
        """
        This function takes in a frame and returns if it is a valid frame or not.

        Args:
            frame: (Frame) Frame object from a video object
        
        Returns:
            (bool): If the frame is valid or note
        """
        return frame

    def detect_defect(self):
        """
        This function process all the frames in a video and returns a dictionary of frames
        with their index in the video frame list and the frame object.
        
        Returns
            defects: (Dict[int] -> Frame) Dictionary mapping the index of the frame in the list
            to the frame in the video
        """
        defects = {}
        count = 0
        processed_frames = []
        for frame in tqdm(self.video.frames):
            
            # Preprocess the frame
            preprocessed_frame = self.processed_frame(frame = frame)
            
            # Disregard bad frame
            if not preprocessed_frame:
                continue

            # Add processed frame to list
            processed_frames.append(frame)

            # Determine if defect exists or not
            if self.com_shift(frame):
                defects[count] = frame
            count+=1
        
        # Assign new object attributes:
        self.processed_frames = processed_frames
        self.defects = defects

        return defects
    
    def com_shift(self, frame: Frame):
        """
        Takes in a frame and assesses if the frame has the com_shift defect.


        Args:
            frame: (Frame) frame object to assess
        
        Returns:
            (bool): if frame object has a com_shift defect or not.
        """
        if frame:
            return False
