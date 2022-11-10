import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Batch:
    def __init__(self,env_with_Dead):
        self.env_with_Dead=env_with_Dead

    def makeBatch(self,data):
        s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst = [], [], [], [], [], [], []
        for transition in data:
            state,act,reward,s_prime,logprob_a,done,dw=transition

            s_lst.append(state)
            a_lst.append(act)
            logprob_a_lst.append(logprob_a)
            r_lst.append([reward])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            dw_lst = (np.array(dw_lst)*False).tolist()

        data = [] 

        with torch.no_grad():
            state= torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
            act=torch.tensor(np.array(a_lst), dtype=torch.float).to(device)
            reward=torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
            s_prime=torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
            logprob_a=torch.tensor(np.array(logprob_a_lst), dtype=torch.float).to(device)
            done_mask=torch.tensor(np.array(done_lst), dtype=torch.float).to(device)
            dw_mask=torch.tensor(np.array(dw_lst), dtype=torch.float).to(device)

        return state, act, reward, s_prime, logprob_a, done_mask, dw_mask, data