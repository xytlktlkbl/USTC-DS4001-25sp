import numpy as np

def normalize(x):
    return x / np.sum(x)

def sample_categorical(p):
    return np.random.choice(len(p), p=p)

def gibbs_hmm_multi(sequences, N, M, num_iters=100, alpha=1.0, beta=1.0, gamma=1.0):
    Qs = [np.random.choice(N, size=len(obs)) for obs in sequences]
    samples = []

    for it in range(num_iters):
        pi_counts = np.zeros(N)
        A_counts = np.zeros((N, N))
        B_counts = np.zeros((N, M))

        for obs, Q in zip(sequences, Qs):
            T = len(obs)
            # Gibbs 
            for t in range(T):
                probs = np.ones(N)
                for k in range(N):
                    if t == 0:
                        probs[k] *= 1.0 / N
                    else:
                        probs[k] *= A_counts[Q[t - 1], k] + beta
                    if t < T - 1:
                        probs[k] *= A_counts[k, Q[t + 1]] + beta
                    probs[k] *= B_counts[k, obs[t]] + gamma
                Q[t] = sample_categorical(normalize(probs))

            pi_counts[Q[0]] += 1
            for t in range(T - 1):
                A_counts[Q[t], Q[t + 1]] += 1
            for t in range(T):
                B_counts[Q[t], obs[t]] += 1

        pi = np.random.dirichlet(pi_counts + alpha)
        A = np.array([np.random.dirichlet(A_counts[i] + beta) for i in range(N)])
        B = np.array([np.random.dirichlet(B_counts[i] + gamma) for i in range(N)])
        samples.append((pi, A, B))

    pi_avg = np.mean([s[0] for s in samples], axis=0)
    A_avg = np.mean([s[1] for s in samples], axis=0)
    B_avg = np.mean([s[2] for s in samples], axis=0)
    return pi_avg, A_avg, B_avg

if __name__ == "__main__":
    sequences = [
        np.array([0,0,1,2,1,0,1,2]),
        np.array([1,2,2,2,1,0,0,0]),
        np.array([2,1,1,0,0,2,1,2]),
        np.array([0,1,2,1,0,0,1,2])
    ]
    N = 3  # 隐状态个数
    M = 3  # 观测符号数

    pi, A, B = gibbs_hmm_multi(sequences, N, M, num_iters=300)
    np.set_printoptions(precision=3, suppress=True)
    print("π =", np.round(pi, 3))
    print("A =", np.round(A, 3))
    print("B =", np.round(B, 3))
