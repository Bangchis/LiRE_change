import os
import numpy as np
import wandb
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


class RewardModel:
    def __init__(self, config, obs_act_1, obs_act_2, labels, dimension, weights=None,
                 blocks=None, best_pos=None, worst_pos=None):
        self.env = config.env
        self.config = config
        self.dimension = dimension
        self.device = config.device
        self.obs_act_1 = obs_act_1
        self.obs_act_2 = obs_act_2
        self.labels = labels
        # Weights for weighted BT loss (for BW feedback with lambda)
        # Handle both training mode (labels+weights provided) and inference mode (labels=None)
        if weights is not None:
            self.weights = weights
        elif labels is not None:
            self.weights = np.ones(len(labels), dtype=np.float32)
        else:
            # Inference mode (IQL loading model): no labels/weights needed
            self.weights = None

        # NEW: PL model data (blocks instead of pairs)
        self.blocks = blocks          # [M, K, T, D] for PL training
        self.best_pos = best_pos      # [M] - best segment index per block
        self.worst_pos = worst_pos    # [M] - worst segment index per block

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.activation = config.activation
        # self.data_aug: For TDA data augmentation
        # (SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based Reinforcement Learning)
        self.data_aug = config.data_aug
        self.segment_size = config.segment_size
        self.lr = config.lr
        self.hidden_sizes = config.hidden_sizes
        self.loss = None
        self.model_type = config.model_type
        if self.model_type == "BT":
            self.loss = self.BT_loss
        elif self.model_type == "linear_BT":
            self.loss = self.linear_BT_loss
        elif self.model_type in ["PL", "linear_PL"]:
            # PL models use dedicated loss function in training loop
            self.loss = None
        self.ensemble_num = config.ensemble_num
        self.ensemble_method = config.ensemble_method
        self.paramlist = []
        self.optimizer = []
        self.lr_scheduler = []
        self.net = None
        self.ensemble_model = None
        self.feedback_type = config.feedback_type

    def save_test_dataset(
        self,
        test_obs_act_1,
        test_obs_act_2,
        test_labels,
        test_binary_labels,
    ):
        self.test_obs_act_1 = torch.from_numpy(test_obs_act_1).float().to(self.device)
        self.test_obs_act_2 = torch.from_numpy(test_obs_act_2).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels).float().to(self.device)
        self.test_binary_labels = (
            torch.from_numpy(test_binary_labels).float().to(self.device)
        )

    def model_net(self, in_dim=39, out_dim=1, H=128, n_layers=2):
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(in_dim, H))
            net.append(nn.LeakyReLU())
            in_dim = H
        net.append(nn.Linear(H, out_dim))
        if self.activation == "tanh":
            net.append(nn.Tanh())
        elif self.activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif self.activation == "relu":
            net.append(nn.ReLU())
        elif self.activation == "leaky_relu":
            net.append(nn.LeakyReLU())
        elif self.activation == "none":
            pass
        elif self.activation == "gelu":
            net.append(nn.GELU())

        return nn.Sequential(*net)

    def construct_ensemble(self):
        ensemble_model = []
        for i in range(self.ensemble_num):
            ensemble_model.append(
                self.model_net(
                    in_dim=self.dimension, out_dim=1, H=self.hidden_sizes
                ).to(self.device)
            )
        return ensemble_model

    def single_model_forward(self, obs_act):
        return self.net(obs_act)

    def ensemble_model_forward(self, obs_act):
        pred = []
        for i in range(self.ensemble_num):
            pred.append(self.ensemble_model[i](obs_act))
        pred = torch.stack(pred, dim=1)
        if self.ensemble_method == "mean":
            return torch.mean(pred, dim=1)
        elif self.ensemble_method == "min":
            return torch.min(pred, dim=1).values
        elif self.ensemble_method == "uwo":
            return torch.mean(pred, dim=1) - 5 * torch.std(pred, dim=1)

    def BT_loss_vec(self, pred_hat, label):
        # Returns per-sample loss for weighted BT loss
        # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
        logprobs = F.log_softmax(pred_hat, dim=1)    # [B, 2]
        return -(label * logprobs).sum(dim=1)        # [B] - per-sample loss

    def BT_loss(self, pred_hat, label):
        # Wrapper for backward compatibility
        return self.BT_loss_vec(pred_hat, label).sum()

    def linear_BT_loss_vec(self, pred_hat, label):
        # Returns per-sample loss for weighted linear_BT loss
        pred_hat = pred_hat + self.segment_size + 1e-5
        pred_prob = pred_hat / torch.sum(pred_hat, dim=1, keepdim=True)
        # label and pred_hat cross entropy loss
        return -(label * torch.log(pred_prob)).sum(dim=1)  # [B] - per-sample loss

    def linear_BT_loss(self, pred_hat, label):
        # Wrapper for backward compatibility
        return self.linear_BT_loss_vec(pred_hat, label).sum()

    def bw_pl_loss_vec(self, seg_scores, best_pos, worst_pos, lam, mode):
        """
        Plackett-Luce loss for Best-Worst blocks (Algorithm 1).

        Args:
            seg_scores: [B, K] - predicted segment returns (sum over timesteps)
            best_pos: [B] - index of best segment in each block
            worst_pos: [B] - index of worst segment in each block
            lam: float - weight for worst term (lambda_bw)
            mode: str - "PL" (exponential) or "linear_PL"

        Returns:
            [B] - per-block loss
        """
        if mode == "PL":
            # Exponential PL: f(σ) = exp(s) → utility is exponential
            # P(best) = exp(s_best) / sum(exp(s)) = softmax(s)[best]
            # -log P(best) = cross_entropy(s, best)
            loss_best = F.cross_entropy(seg_scores, best_pos, reduction="none")  # [B]

            # P(worst) with inverse utility: 1/exp(s) = exp(-s)
            # P(worst) = exp(-s_worst) / sum(exp(-s)) = softmax(-s)[worst]
            # -log P(worst) = cross_entropy(-s, worst)
            loss_worst = F.cross_entropy(-seg_scores, worst_pos, reduction="none")  # [B]

        else:  # mode == "linear_PL"
            # Linear PL: f(σ) = s + const (ensure positive, NO log of utility!)
            u = seg_scores + self.segment_size + 1e-5  # [B, K] - utilities (all positive)

            # P(best) = u_best / sum(u)
            # Gather best utilities: [B, K] → [B, 1] → [B]
            u_best = u.gather(1, best_pos.unsqueeze(1)).squeeze(1)  # [B]
            u_sum = u.sum(dim=1)  # [B]
            p_best = u_best / u_sum  # [B]
            loss_best = -torch.log(p_best + 1e-8)  # [B]

            # P(worst) with inverse utility: (1/u_worst) / sum(1/u)
            u_inv = 1.0 / u  # [B, K] - inverse utilities for "less preferred"
            u_inv_worst = u_inv.gather(1, worst_pos.unsqueeze(1)).squeeze(1)  # [B]
            u_inv_sum = u_inv.sum(dim=1)  # [B]
            p_worst = u_inv_worst / u_inv_sum  # [B]
            loss_worst = -torch.log(p_worst + 1e-8)  # [B]

        # Combined loss: L = -log P(best) - λ * log P(worst)
        return loss_best + lam * loss_worst  # [B]

    def save_model(self, path):
        for member in range(self.ensemble_num):
            # join path + member number
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            torch.save(self.ensemble_model[member].state_dict(), member_path)

    def load_model(self, path):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            member_path = os.path.join(path, "reward_" + str(member) + ".pt")
            self.ensemble_model[member].load_state_dict(torch.load(member_path))

    def get_reward(self, dataset):
        obs = dataset["observations"]
        act = dataset["actions"]
        obs_act = np.concatenate((obs, act), axis=-1)
        obs_act = torch.from_numpy(obs_act).float().to(self.device)
        with torch.no_grad():
            for i in range((obs_act.shape[0] - 1) // 10000 + 1):
                obs_act_batch = obs_act[i * 10000 : (i + 1) * 10000]
                pred_batch = self.ensemble_model_forward(obs_act_batch).reshape(-1)
                dataset["rewards"][
                    i * 10000 : (i + 1) * 10000
                ] = pred_batch.cpu().numpy()
        return dataset["rewards"]

    def eval(self, obs_act_1, obs_act_2, labels, binary_labels, name, epoch):
        eval_acc = 0
        eval_loss = 0
        for member in range(self.ensemble_num):
            self.ensemble_model[member].eval()
        with torch.no_grad():
            for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                obs_act_1_batch = obs_act_1[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                obs_act_2_batch = obs_act_2[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                labels_batch = labels[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                binary_labels_batch = binary_labels[
                    batch * self.batch_size : (batch + 1) * self.batch_size
                ]
                pred_1 = self.ensemble_model_forward(obs_act_1_batch)
                pred_2 = self.ensemble_model_forward(obs_act_2_batch)
                pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)
                pred_labels = torch.argmax(pred_hat, dim=-1)
                eval_acc += torch.sum(
                    pred_labels == torch.argmax(binary_labels_batch, dim=-1)
                ).item()
                # For PL models, self.loss is None (only compute accuracy)
                if self.loss is not None:
                    eval_loss += self.loss(pred_hat, labels_batch).item()

        eval_acc /= float(obs_act_1.shape[0])

        # Log metrics (only log loss if computed)
        if self.loss is not None:
            eval_loss /= obs_act_1.shape[0]
            wandb.log({name + "/loss": eval_loss, name + "/acc": eval_acc}, step=epoch)
        else:
            # PL models: only log accuracy
            wandb.log({name + "/acc": eval_acc}, step=epoch)

    def train_model(self):
        self.ensemble_model = self.construct_ensemble()
        for member in range(self.ensemble_num):
            self.ensemble_model[member].train()
            self.optimizer.append(
                optim.Adam(self.ensemble_model[member].parameters(), lr=self.lr)
            )
            self.lr_scheduler.append(
                optim.lr_scheduler.StepLR(
                    self.optimizer[member],
                    step_size=10 if self.epochs <= 500 else 1000,
                    gamma=0.9,
                )
            )

        # ---- NEW: BW + PL training (direct on blocks, no pairwise) ----
        if self.feedback_type == "BW" and self.model_type in ["PL", "linear_PL"]:
            blocks = torch.from_numpy(self.blocks).float().to(self.device)      # [M, K, T, D]
            best_pos = torch.from_numpy(self.best_pos).long().to(self.device)   # [M]
            worst_pos = torch.from_numpy(self.worst_pos).long().to(self.device) # [M]
            M, K, T, D = blocks.shape
            lam = float(getattr(self.config, "lambda_bw", 1.0))

            for epoch in tqdm.tqdm(range(self.epochs)):
                # Shuffle blocks at the start of each epoch
                idx = torch.randperm(M, device=self.device)
                blocks_shuf = blocks[idx]
                best_shuf = best_pos[idx]
                worst_shuf = worst_pos[idx]

                train_loss = 0.0
                for member in range(self.ensemble_num):
                    self.net = self.ensemble_model[member]
                    self.net.train()

                    for b in range((M - 1) // self.batch_size + 1):
                        # Get batch of blocks
                        bb = blocks_shuf[b * self.batch_size : (b + 1) * self.batch_size]  # [B, K, T, D]
                        bp = best_shuf[b * self.batch_size : (b + 1) * self.batch_size]    # [B]
                        wp = worst_shuf[b * self.batch_size : (b + 1) * self.batch_size]   # [B]

                        if bb.shape[0] == 0:  # Skip empty batches
                            continue

                        self.optimizer[member].zero_grad()

                        # Forward pass: compute rewards for all K segments in each block
                        # Input: [B, K, T, D] → Output: [B, K, T, 1]
                        pred = self.net(bb)                          # [B, K, T, 1]
                        seg_scores = pred.sum(dim=2).squeeze(-1)     # [B, K] - sum over time

                        # Compute PL loss
                        loss_vec = self.bw_pl_loss_vec(
                            seg_scores, bp, wp, lam, self.model_type
                        )  # [B]
                        loss = loss_vec.mean()

                        loss.backward()
                        self.optimizer[member].step()

                        train_loss += loss.item() * bb.shape[0]

                    self.lr_scheduler[member].step()

                train_loss /= (M * self.ensemble_num)

                if epoch % 20 == 0:
                    wandb.log({"train_eval/loss": train_loss}, step=epoch)

                # Evaluate on test set (pairwise) every 100 epochs
                if epoch % 100 == 0:
                    self.eval(
                        self.test_obs_act_1,
                        self.test_obs_act_2,
                        self.test_labels,
                        self.test_binary_labels,
                        "test_eval",
                        epoch,
                    )
            return
        # ---- END NEW: BW + PL training ----

        self.obs_act_1 = torch.from_numpy(self.obs_act_1).float().to(self.device)
        self.obs_act_2 = torch.from_numpy(self.obs_act_2).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).float().to(self.device)
        for epoch in tqdm.tqdm(range(self.epochs)):
            train_loss = 0
            for member in range(self.ensemble_num):
                self.optimizer[member].zero_grad()
                self.net = self.ensemble_model[member]
                # shuffle data
                idx = np.random.permutation(self.obs_act_1.shape[0])
                obs_act_1 = self.obs_act_1[idx]
                obs_act_2 = self.obs_act_2[idx]
                labels = self.labels[idx]
                weights_shuffled = self.weights[idx]  # Shuffle weights along with data

                for batch in range((obs_act_1.shape[0] - 1) // self.batch_size + 1):
                    loss = 0
                    obs_act_1_batch = obs_act_1[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    obs_act_2_batch = obs_act_2[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    labels_batch = labels[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    weights_batch = weights_shuffled[
                        batch * self.batch_size : (batch + 1) * self.batch_size
                    ]
                    if self.data_aug == "temporal":
                        # cut random segment from self.segment_size (20 ~ 25)
                        short_segment_size = np.random.randint(
                            self.segment_size - 5, self.segment_size + 1
                        )
                        start_idx_1 = np.random.randint(
                            0, self.segment_size - short_segment_size + 1
                        )
                        start_idx_2 = np.random.randint(
                            0, self.segment_size - short_segment_size + 1
                        )
                        obs_act_1_batch = obs_act_1_batch[
                            :, start_idx_1 : start_idx_1 + short_segment_size, :
                        ]
                        obs_act_2_batch = obs_act_2_batch[
                            :, start_idx_2 : start_idx_2 + short_segment_size, :
                        ]
                    pred_1 = self.single_model_forward(obs_act_1_batch)
                    pred_2 = self.single_model_forward(obs_act_2_batch)
                    pred_seg_sum_1 = torch.sum(pred_1, dim=1)
                    pred_seg_sum_2 = torch.sum(pred_2, dim=1)
                    pred_hat = torch.cat([pred_seg_sum_1, pred_seg_sum_2], dim=-1)

                    # Use weighted loss (supports both BT and linear_BT)
                    if self.model_type == "BT":
                        loss_vec = self.BT_loss_vec(pred_hat, labels_batch)  # [B]
                    elif self.model_type == "linear_BT":
                        loss_vec = self.linear_BT_loss_vec(pred_hat, labels_batch)  # [B]

                    weights_tensor = torch.from_numpy(weights_batch).to(self.device)
                    # Normalize by sum of weights (not batch size) to handle λ properly
                    loss = (weights_tensor * loss_vec).sum() / weights_tensor.sum()
                    train_loss += loss.item() * weights_tensor.sum().item()
                    loss.backward()
                    self.optimizer[member].step()
                self.lr_scheduler[member].step()

            # Normalize by total weights (not sample count) for proper weighted loss
            train_loss /= self.weights.sum() * self.ensemble_num

            if epoch % 20 == 0:
                wandb.log({"train_eval/loss": train_loss}, step=epoch)

            if epoch % 100 == 0:
                self.eval(
                    self.obs_act_1,
                    self.obs_act_2,
                    self.labels,
                    self.labels,
                    "train_eval",
                    epoch,
                )
                self.eval(
                    self.test_obs_act_1,
                    self.test_obs_act_2,
                    self.test_labels,
                    self.test_binary_labels,
                    "test_eval",
                    epoch,
                )
