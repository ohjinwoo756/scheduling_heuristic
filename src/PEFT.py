
class PEFT():

    # variable declaration
    # OCT (Optimal Cost Table) generated

    def __init__(self, network, profile):
        self.network = network
        self.profile = profile
        self.generate_OCT()
        pass

    def generate_OCT(self):
        print "Make OCT"

    def do_schedule(self):
        print "Do scheduling"

        # 2 phases
        self.prioritize_tasks() # 1st phase
        self.assign_task_to_processor() # 2nd phase

    def prioritize_tasks(self):
        print "Prioritize_task"

    def assign_task_to_processor(self):
        print "Assign task to specific processor"
