from abc import ABC, abstractmethod

class Movement:

    @abstractmethod
    def run_defect_analysis(self):
        pass

    @abstractmethod
    def get_defects(self):
        pass
    
    @abstractmethod
    def process_data(self):
        pass