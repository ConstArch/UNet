class IterationLogger:
    
    def __init__(self, message_sender, duration):
        self.message_sender = message_sender
        self.duration = duration
        self.count = 0
    
    def tick(self):
        self.count += 1
        if self.count % self.duration == 0:
            self.message_sender(self.count)
    
    def reset(self):
        self.count = 0
