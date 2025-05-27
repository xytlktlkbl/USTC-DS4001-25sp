import numpy as np

class BWtrainer:
    def __init__(self, n_states, n_obs, max_iter=500, tol=1e-6):
        self.n_states = n_states    # 隐状态数
        self.n_obs = n_obs          # 观测值种类数
        self.max_iter = max_iter    # 最大迭代次数
        self.tol = tol              # 收敛阈值
        
        # 随机初始化参数
        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()    
        
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True) 
        
        self.B = np.random.rand(n_states, n_obs)
        self.B /= self.B.sum(axis=1, keepdims=True) 

    def fit(self, obs_seqs):
        prev_log_prob = -np.inf
        for _ in range(self.max_iter):
            sum_gamma0 = np.zeros(self.n_states)
            sum_A = np.zeros((self.n_states, self.n_states))
            sum_B = np.zeros((self.n_states, self.n_obs))
            sum_gamma = np.zeros(self.n_states)
            current_log_prob = 0

            # E-step：遍历所有序列
            for seq in obs_seqs:
                alpha, scaling_factors = self._forward(seq)  
                beta = self._backward(seq, scaling_factors) 
                gamma, xi = self._compute_gamma_xi(seq, alpha, beta)
                
                # 计算对数似然
                current_log_prob += -np.sum(np.log(scaling_factors)) 

                sum_gamma0 += gamma[0]
                sum_A += xi.sum(axis=0)
                for t in range(len(seq)):
                    k = seq[t]
                    sum_B[:, k] += gamma[t]
                sum_gamma += gamma.sum(axis=0)

            # M-step：参数更新
            self.pi = sum_gamma0 / len(obs_seqs)
            self.A = sum_A / sum_gamma[:, None]  
            self.B = sum_B / sum_gamma[:, None]
            
            # 确保参数归一化
            self.pi /= self.pi.sum() 
            self.A /= self.A.sum(axis=1, keepdims=True)  
            self.B /= self.B.sum(axis=1, keepdims=True) 

            # 检查收敛
            if np.abs(current_log_prob - prev_log_prob) < self.tol:
                break
            prev_log_prob = current_log_prob

    # 前向算法
    def _forward(self, seq):
        T = len(seq)
        alpha = np.zeros((T, self.n_states))
        scaling_factors = np.zeros(T) 
        # 初始化第一个时间步
        alpha[0] = self.pi * self.B[:, seq[0]]  ## fill in this blank with an expression ##
        scaling_factors[0] = np.sum(alpha[0]) 
        alpha[0] /= scaling_factors[0]  
        for t in range(1, T):
            alpha[t] = [np.sum(alpha[t-1] * self.A[:, j]) * self.B[j][seq[t]] for j in range(self.n_states)]  ## fill in this blank with an expression ##
            scaling_factors[t] = np.sum(alpha[t]) 
            alpha[t] /= scaling_factors[t]  
            
        return alpha, scaling_factors 

    # 后向算法
    def _backward(self, seq, scaling_factors):  
        T = len(seq)
        beta = np.zeros((T, self.n_states))
        
        # 初始化最后一个时间步
        beta[T-1] = 1.0
        beta[T-1] /= scaling_factors[T-1] 
        
        for t in range(T-2, -1, -1):
            beta[t] = [np.sum(self.A[i] * self.B[:, seq[t+1]] * beta[t+1]) for i in range(self.n_states)]  ## fill in this blank with an expression ##
            beta[t] /= scaling_factors[t]  

        return beta

    def _compute_gamma_xi(self, seq, alpha, beta):
        T = len(seq)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) 
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            xi[t] = alpha[t][:, None] * self.A * self.B[:, seq[t+1]] * beta[t+1]
            xi[t] /= xi[t].sum()  

        return gamma, xi 

    @property
    def parameters(self):
        return {
            "初始状态": self.pi,
            "转移矩阵": self.A,
            "发射矩阵": self.B
        }

if __name__ == "__main__":
    # 指定的观察序列
    obs_seq_list = [
        np.array([0,0,1,2,1,0,1,2]),
        np.array([1,2,2,2,1,0,0,0]),
        np.array([2,1,1,0,0,2,1,2]),
        np.array([0,1,2,1,0,0,1,2])
    ]
    
    trainer = BWtrainer(n_states=3, n_obs=3, max_iter=500)
    trainer.fit(obs_seq_list)
    params = trainer.parameters
    print("π =")
    print(np.round(params["初始状态"], 3))
    print("A =")
    print(np.round(params["转移矩阵"], 3))
    print("B =")
    print(np.round(params["发射矩阵"], 3))
