from datetime import timedelta
import os
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
import heapq
import gc
import gzip
import sys

from script.test.model import Model

from models.FLA.architecture import LSTM
from models.FLA.guesser import Guesser
from models.FLA.fla_utils.dataloader import *
from models.FLA.heapitem import HeapItem
from models.FLA.mem_utils import kill_if_low_memory,get_memory_usage,get_memory_usage_byte

#TODO implement dictionary for maximum python compatibility  
def get_lower_probability_threshold(n_samples):
    n_samples = int(n_samples)
    if n_samples <= 10 ** 5: 
        return .000001
    elif n_samples <= 10 ** 6:
        return 0.00000001
    elif n_samples <= 10 ** 7:
        return 0.000000001
    elif n_samples <= 5 * (10 ** 8):
        return 0.0000000001
    else:
        return 0.00000000001

class FLA(Model):
    def __init__(self, settings):
        self.model = None
        self.optimizer = None

        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        return DataLoader(train_passwords, test_passwords, max_length, self.params)

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        self.params['eval']['evaluation_batch_size'] = self.n_samples + 1
        lstm_hidden_size = self.params["train"]['lstm_hidden_size']
        dense_hidden_size = self.params["train"]['dense_hidden_size']
        context_len = self.data.max_length
        vocab_size = self.data.tokenizer.vocab_size

        self.model = LSTM(lstm_hidden_size=lstm_hidden_size,
                     dense_hidden_size=dense_hidden_size,
                     vocab_size=vocab_size,
                     context_len=context_len
                     ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_step(self, x_train, y_train):
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(x_train)

        train_loss = F.cross_entropy(y_pred, y_train)
        train_loss.backward()

        self.optimizer.step()

    def train(self):
        print("[I] - Launching training")

        epochs = self.params["train"]["epochs"]
        batch_size = self.params["train"]["batch_size"]

        start = time.time()

        current_epoch = 0
        n_matches = 0
        n_passwords = self.data.get_train_size()

        checkpoint_frequency = self.params['eval']['checkpoint_frequency']

        self.init_model()

        while current_epoch < epochs:
            print(f"Epoch: {current_epoch + 1} / {epochs}")

            progress_bar = tqdm(range(n_passwords), desc="Epoch {}/{}".format(current_epoch, epochs))

            n_iter = 0
            for batch in self.data.get_batches(batch_size):
                x_train = np.array(batch[0])
                y_train = np.array(batch[1])

                x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

                self.train_step(x_train, y_train)
                progress_bar.update(batch_size)
                n_iter += 1

            if current_epoch % checkpoint_frequency == 0:
                matches, _, _ = self.evaluate(n_samples=10 ** 6, validation_mode=True)
                if matches >= n_matches:
                    n_matches = matches
                    obj = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    self.save(obj)

            current_epoch += 1

        end = time.time()
        time_delta = timedelta(seconds=end - start)
        print(f"[T] - Training completed after: {time_delta}")

    def eval_init(self, n_samples, evaluation_batch_size):
        self.model.eval()
        eval_dict = {
            'n_samples': n_samples,
            'output_file': os.path.join(self.path_to_guesses_dir, "total_guesses.gz"),
        }
        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict)->set[str]:
        lower_probability_threshold:float = get_lower_probability_threshold(eval_dict['n_samples'])
        guesser = Guesser(model=self.model, params=self.params, data=self.data,
                          lower_probability_threshold=lower_probability_threshold, output_file=eval_dict['output_file'],
                          device=self.device)
        n_gen:int = guesser.complete_guessing()

        print(f"[I] - Generated {n_gen} passwords")

        min_heap_n_most_prob:list[type] = []
        # Optimize the algorithm 
        
        memory_heap:int = 0

        with gzip.open(eval_dict['output_file'], "rt") as f_out:
            initial_memory:float = get_memory_usage()
            print(f"Memory Before creating  heap ==>{initial_memory:.2f} MB\n")
            for i,line in enumerate(f_out):
                if i%100_000 == 0:
                    kill_if_low_memory(threshold_percent=95) 

                line = line.split(" ")

                if len(line) != 2:
                    continue

                password, prob= line[0].replace("~", ""), float(line[1])
                # new method 
                if len(min_heap_n_most_prob) < eval_dict['n_samples']:
                    item = HeapItem(prob,password,len(password))
                    heapq.heappush(min_heap_n_most_prob, item)
                    memory_heap+=sys.getsizeof(item)
                elif prob > min_heap_n_most_prob[0].prob: 
                    item = HeapItem(prob,password,len(password))
                    heapq.heappushpop(min_heap_n_most_prob,item)
                    memory_heap+=sys.getsizeof(item)

            current_memory:float = get_memory_usage()

        size_of_heap:float = sys.getsizeof(min_heap_n_most_prob)
        print(f"Size of Heap ==>{size_of_heap}")
        size_of_heap+= memory_heap
        from pympler import asizeof

        total_size = asizeof.asizeof(min_heap_n_most_prob)  # measures Python overhead
        manual_buffer_total = sum(item.memory_size()['password_buffer'] for item in min_heap_n_most_prob)

        total_heap_size = total_size + manual_buffer_total
        print(f"Total heap memory including buffers ==> {total_heap_size / 1_000_000:.2f} MB")

         
        print(f"""Memory After the heap is created ==> {current_memory:.2f} MB\nMemory Of Heap ==> {size_of_heap/1000/1000:.2f}MB\nMemory Used During Creation ==> {current_memory-initial_memory:.2f}MB\n""")
        print(f"Size of HeapItem object:{min_heap_n_most_prob[0].memory_size()["object_size"]},Size of HeapItem pwd_buffer:{min_heap_n_most_prob[0].memory_size()["password_buffer"]},\nSize of HeapItem total_size:{min_heap_n_most_prob[0].memory_size()["total"]},\n")
        print(f"Number of items in heap ==> {len(min_heap_n_most_prob)}\n")
        n_most_prob_psw:set[str] = set()
        
        for hi in heapq.nlargest(eval_dict['n_samples'], min_heap_n_most_prob):
            n_most_prob_psw.add(hi.password_string)
            del hi      # engages __dealloc__

        return n_most_prob_psw
        # when generating return set()

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        gc.collect()
        #os.remove(eval_dict['output_file'])
        pass