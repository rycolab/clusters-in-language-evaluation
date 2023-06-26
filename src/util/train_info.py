from tqdm import tqdm


class TrainInfo:
    batch_id = 0
    running_loss = []
    best_loss = float('inf')
    best_batch = 0

    def __init__(self, wait_iterations, eval_batches):
        self.wait_iterations = wait_iterations
        self.eval_batches = eval_batches

    @property
    def finish(self):
        return (self.batch_id - self.best_batch) >= self.wait_iterations

    @property
    def eval(self):
        return (self.batch_id % self.eval_batches) == 0

    @property
    def max_epochs(self):
        return self.best_batch + self.wait_iterations

    @property
    def avg_loss(self):
        return sum(self.running_loss) / len(self.running_loss)

    @property
    def description(self):
        batch_id = int(self.batch_id / 100)
        max_epochs = int(self.max_epochs / 100)
        return f'({batch_id:05d}/{max_epochs:05d})'

    def new_batch(self, loss):
        self.batch_id += 1
        self.running_loss += [loss]

    def is_best(self, dev_loss):
        if dev_loss < self.best_loss:
            self.best_loss = dev_loss
            self.best_batch = self.batch_id
            return True

        return False

    def reset_loss(self):
        self.running_loss = []

    def print_progress(self, dev_loss):
        tqdm.write(
            f'{self.description} Training loss: {self.avg_loss:.4f} Dev loss: {dev_loss:.4f}')

        self.reset_loss()
