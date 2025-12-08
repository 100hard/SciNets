import numpy as np
import json
import time

# Simple numpy DQN (small MLP) to avoid external dependencies
class SimpleDQN:
    def __init__(self, input_dim, output_dim, hidden_dim=64, lr=1e-3, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / max(1, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / max(1, hidden_dim))
        self.b2 = np.zeros(output_dim)
        self.lr = lr

    def predict(self, x):
        # x: (batch, input_dim)
        h = np.maximum(0, x.dot(self.W1) + self.b1)
        q = h.dot(self.W2) + self.b2
        return q

    def train_on_batch(self, x, y_target):
        # MSE loss, simple SGD
        # Forward
        h_pre = x.dot(self.W1) + self.b1
        h = np.maximum(0, h_pre)
        q = h.dot(self.W2) + self.b2
        diff = (q - y_target) / x.shape[0]
        # Gradients
        dW2 = h.T.dot(diff)
        db2 = diff.sum(axis=0)
        dh = diff.dot(self.W2.T)
        dh_pre = dh * (h_pre > 0)
        dW1 = x.T.dot(dh_pre)
        db1 = dh_pre.sum(axis=0)
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # Return loss
        loss = ((q - y_target) ** 2).mean()
        return loss


class ReplayBuffer:
    def __init__(self, max_size=10000, seed=42):
        self.max_size = max_size
        self.buf = []
        self.pos = 0
        self.rng = np.random.RandomState(seed)

    def add(self, s, a, r, s2, done):
        item = (s.copy(), a, float(r), s2.copy() if s2 is not None else None, bool(done))
        if len(self.buf) < self.max_size:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        idx = self.rng.choice(len(self.buf), size=min(batch_size, len(self.buf)), replace=False)
        batch = [self.buf[i] for i in idx]
        s = np.vstack([b[0][None, :] for b in batch])
        a = np.array([b[1] for b in batch], dtype=int)
        r = np.array([b[2] for b in batch], dtype=float)
        s2 = np.vstack([b[3][None, :] if b[3] is not None else np.zeros_like(b[0])[None, :] for b in batch])
        done = np.array([b[4] for b in batch], dtype=bool)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def simulate_environment_step(gains, alpha=0.9):
    # gains: current gains matrix (N_sub, N_users)
    # AR(1)-like update per element
    noise = np.random.randn(*gains.shape) * 0.5
    gains = alpha * gains + np.sqrt(1 - alpha ** 2) * noise
    gains = np.abs(gains)  # positive channel gains
    return gains


def throughput_from_snr(snr_linear, bandwidth=1.0):
    # Shannon capacity per subband (bandwidth normalized to 1)
    return bandwidth * np.log2(1 + snr_linear)


def run_experiment():
    np.random.seed(123)
    try:
        # Environment parameters
        N_sub = 6           # number of sub-bands
        N_users = 12        # number of users
        Tx_power = 1.0      # normalized transmit power per subband
        noise_power = 0.1
        episodes_train = 200
        episodes_test = 100
        T = 50              # timesteps per episode

        # DQN parameters
        input_dim = N_users  # for per-subband network: vector of gains to all users
        output_dim = N_users # choose which user
        hidden = 64
        lr = 1e-3
        gamma = 0.95
        eps_start = 1.0
        eps_end = 0.05
        eps_decay = 0.995
        batch_size = 64

        # Initialize network and replay buffers (shared across subbands)
        agent = SimpleDQN(input_dim, output_dim, hidden_dim=hidden, lr=lr, seed=42)
        replay = ReplayBuffer(max_size=20000)

        # Initialize baseline static allocation selection strategy: we'll estimate long-term average gains
        # by simulating some channel realizations
        avg_gains = np.zeros((N_sub, N_users))
        # Warm-up to estimate long-term average gains
        gains = np.abs(np.random.randn(N_sub, N_users))
        warm_steps = 200
        for _ in range(warm_steps):
            gains = simulate_environment_step(gains)
            avg_gains += gains
        avg_gains /= warm_steps
        static_mapping = np.argmax(avg_gains, axis=1)  # fixed user per subband

        # Training loop (per-subband agent): treat each subband interaction as independent sample
        eps = eps_start
        start_time = time.time()
        for ep in range(episodes_train):
            gains = np.abs(np.random.randn(N_sub, N_users))
            ep_reward = 0.0
            for t in range(T):
                # update channel
                gains = simulate_environment_step(gains)
                # for each subband, choose action
                for s in range(N_sub):
                    state = gains[s]  # shape (N_users,)
                    # epsilon-greedy
                    if np.random.rand() < eps:
                        action = np.random.randint(N_users)
                    else:
                        q = agent.predict(state[None, :])[0]
                        action = int(np.argmax(q))
                    # compute reward
                    gain = gains[s, action]
                    snr = Tx_power * (gain ** 2) / noise_power
                    r = throughput_from_snr(snr)
                    ep_reward += r
                    # next state is next timestep gains for that subband; we'll simulate a small step for s alone
                    # but for simplicity, treat next_state as current gains (because interplay across timesteps small at per-step level)
                    next_state = state.copy()
                    done = (t == T - 1)
                    replay.add(state, action, r, next_state, done)
                # train from replay
                if len(replay) >= 32:
                    s_b, a_b, r_b, s2_b, done_b = replay.sample(batch_size)
                    q_next = agent.predict(s2_b)
                    q_curr = agent.predict(s_b)
                    q_target = q_curr.copy()
                    # Update targets for taken actions
                    for i in range(s_b.shape[0]):
                        if done_b[i]:
                            q_target[i, a_b[i]] = r_b[i]
                        else:
                            q_target[i, a_b[i]] = r_b[i] + gamma * np.max(q_next[i])
                    loss = agent.train_on_batch(s_b, q_target)
            eps = max(eps * eps_decay, eps_end)
            if (ep + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Trained ep {ep+1}/{episodes_train}, eps={eps:.3f}, elapsed={elapsed:.1f}s")

        # Evaluation: compute spectral efficiency per episode for both static and DRL
        def evaluate(policy='drl', episodes=episodes_test):
            se_list = []
            for ep in range(episodes):
                gains = np.abs(np.random.randn(N_sub, N_users))
                total_se = 0.0
                for t in range(T):
                    gains = simulate_environment_step(gains)
                    if policy == 'static':
                        for s in range(N_sub):
                            u = int(static_mapping[s])
                            gain = gains[s, u]
                            snr = Tx_power * (gain ** 2) / noise_power
                            total_se += throughput_from_snr(snr)
                    elif policy == 'drl':
                        for s in range(N_sub):
                            state = gains[s]
                            q = agent.predict(state[None, :])[0]
                            u = int(np.argmax(q))
                            gain = gains[s, u]
                            snr = Tx_power * (gain ** 2) / noise_power
                            total_se += throughput_from_snr(snr)
                    else:
                        raise ValueError('Unknown policy')
                # spectral efficiency normalized per subband per timestep: divide by (N_sub * T)
                se_per_timestep_per_sub = total_se / (N_sub * T)
                se_list.append(se_per_timestep_per_sub)
            return np.array(se_list)

        se_static = evaluate(policy='static', episodes=episodes_test)
        se_drl = evaluate(policy='drl', episodes=episodes_test)

        mean_static = float(np.mean(se_static))
        mean_drl = float(np.mean(se_drl))
        improvement = (mean_drl - mean_static) / mean_static if mean_static != 0 else float('inf')

        # Bootstrap p-value for improvement > 0 (one-sided): fraction of bootstrap resamples where mean_diff <= 0
        diffs = se_drl - se_static
        rng = np.random.RandomState(2025)
        n_boot = 2000
        boot_means = np.empty(n_boot)
        n = len(diffs)
        for i in range(n_boot):
            sample_idx = rng.randint(0, n, size=n)
            boot_means[i] = np.mean(diffs[sample_idx])
        p_value = float(np.mean(boot_means <= 0.0))

        result = {
            "mean_static_se": mean_static,
            "mean_drl_se": mean_drl,
            "improvement_ratio": improvement,
            "improvement_percent": improvement * 100.0,
            "meets_30_percent": improvement >= 0.30,
            "episodes_test": episodes_test,
            "episodes_train": episodes_train,
            "bootstrap_p_value_one_sided": p_value,
            "notes": "Environment is synthetic and simplified (per-subband shared DQN). Throughput uses Shannon formula with normalized bandwidth."
        }

        print(json.dumps(result))

    except Exception as e:
        # If an error occurs, return an error JSON explaining the failure
        err = {"error": str(e)}
        print(json.dumps(err))


if __name__ == '__main__':
    run_experiment()
