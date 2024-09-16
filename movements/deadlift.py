from movement import Movement
from body_data import BodyData

class Deadlift(Movement):

    def __init__(self, name: str, body_data: BodyData):
        self.name = name
        self.body_data = body_data
    
    def get_defect(self):
        return

    def process_data(self, body_data):
        return