from squat_buddy import SquatBuddy
from movement import Movement
from body_data import BodyData

MOVING_AVG_SAMP_RATE = 50
FRAME_SAMP_RATE = 10

class Squat(Movement):

    def __init__(self, name: str, body_data: BodyData) -> None:
        self.name = name
        self.body_data = body_data
    
    def preprocess_data(self):
        return


    def find_beginning(self):
        return
    
    def segment_squats(self):
        return

    def run_defect_analysis(self):
        return

    def get_defects(self):
        return
    
    def process_data(self):
        pass