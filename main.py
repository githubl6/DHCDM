import json
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from scipy.special import expit
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,mean_absolute_error

from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from DOA import DOA
from homogeneity import cosine_similarity, euclidean_similarity
from MMD import compute_mmd

class Hypergraph:
    # Calculate the value of the normalized incidence matrix
    def __init__(self, H: np.ndarray):
        self.H = H
        # avoid zero
        self.Dv = np.count_nonzero(H, axis=1) + 1
        self.De = np.count_nonzero(H, axis=0) + 1

        self.Omega = sp.eye(self.De.shape[0])

    def to_tensor_nadj(self):
        coo = sp.coo_matrix(self.H @ self.Omega @ np.diag(1 / self.De) @ self.H.T @ np.diag(1 / self.Dv))
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class ResponseLogs:
    # consturct hypergraph
    def __init__(self, path, train_size=0.8):
        print("Load dataset config file")
        with open(path+"/config.json") as file:
            config = json.load(file)
        print("Dataset name: {}".format(config["dataset"]))

        self.config = config
        self.q_matrix = read_csv(path+"/"+config["q_file"], header=None).to_numpy()
        self.h_matrix = read_csv(path+"/"+config["h_file"], header=None).to_numpy()
        self.response_logs = read_csv(path+"/"+config["data"], header=None).to_numpy(dtype=int)
        self.train_set, self.test_set = train_test_split(self.response_logs,
                                                         train_size=int(train_size*self.response_logs.shape[0]))

        print("Load successfully! Logs entry: {}".format(self.response_logs.shape[0]))

    def hyper_construct(self, choice="student"):
        if choice == "student":
            print("Construct student-exercise interactHypergraph")
            H = self.h_matrix.copy()
        elif choice == "exercise":
            print("Construct exercise-knowledge correlateHypergraph")
            H = self.q_matrix.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "knowledge":
            print("Construct knowledge-exercise correlateHypergraph")
            H = self.q_matrix.T.copy()
            H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        elif choice == "H":
            print("Construct exercise-student interactHypergraph")
            H = self.h_matrix.T.copy()
        else:
            raise ValueError("Only \"student\", \"exercise\" and \"knowledge\" are capable for parameter choice")

        return Hypergraph(H)

    def transform(self, choice="train", batch_size=32):
        if choice == "train":
            dataset = TensorDataset(
                torch.tensor(self.train_set[:, 0], dtype=torch.int64),
                torch.tensor(self.train_set[:, 1], dtype=torch.int64),
                torch.tensor(self.q_matrix[self.train_set[:, 1], :]),
                torch.tensor(self.train_set[:, 2], dtype=torch.float64)
            )
        elif choice == "test":
            dataset = TensorDataset(
                torch.tensor(self.test_set[:, 0], dtype=torch.int64),
                torch.tensor(self.test_set[:, 1], dtype=torch.int64),
                torch.tensor(self.q_matrix[self.test_set[:, 1], :]),
                torch.tensor(self.test_set[:, 2], dtype=torch.float64)
            )
        else:
            raise ValueError("Only \"train\" and \"test\" are capable for parameter choice")
        return DataLoader(dataset, batch_size, shuffle=True)

    def get_r_matrix(self, choice="train"):
        r_matrix = -1 * np.ones(shape=(int(self.config["student_num"]), int(self.config["exercise_num"])))
        if choice == "train":
            for line in self.train_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        elif choice == "test":
            for line in self.test_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        elif choice == "total":
            for line in self.train_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
            for line in self.test_set:
                student_id = line[0]
                exercise_id = line[1]
                score = line[2]
                r_matrix[student_id, exercise_id] = int(score)
        else:
            raise ValueError("Only \"train\" and \"test\" are capable for parameter choice")
        return r_matrix


class HSCD_Net(nn.Module):
    # Student performance prediction
    def __init__(self, student_num, exercise_num, knowledge_num, feature_dim, emb_dim,
                 student_adj, exercise_adj1, exercise_adj2, knowledge_adj, device, layers=3, leaky=0.8):#
        super(HSCD_Net, self).__init__()

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim

        self.device = device
        self.layers = layers
        self.leaky = leaky

        self.student_emb = nn.Embedding(student_num, emb_dim, dtype=torch.float64)
        self.exercise_emb = nn.Embedding(exercise_num, emb_dim, dtype=torch.float64)
        self.knowledge_emb = nn.Embedding(knowledge_num, emb_dim, dtype=torch.float64)

        self.student_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.knowledge_emb2feature = nn.Linear(emb_dim, feature_dim, dtype=torch.float64)
        self.exercise_emb2discrimination = nn.Linear(emb_dim, 1, dtype=torch.float64)

        self.w1 = nn.Linear(emb_dim, emb_dim, dtype=torch.float64)
        self.w2 = nn.Linear(emb_dim, emb_dim, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
       # Positive fully connected layer
        self.clipper = NoneNegClipper()

        self.state2response = nn.Sequential(
            nn.Linear(knowledge_num, 512, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=torch.float64),
            nn.Sigmoid()
        )

        self.student_adj = student_adj
        self.exercise_adj1 = exercise_adj1 #
        self.exercise_adj2 = exercise_adj2
        self.knowledge_adj = knowledge_adj

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def convolution(self, embedding, adj):
        all_emb = embedding.weight.to(self.device)
        final = [all_emb]
        for i in range(self.layers):
            # implement hypergraph momentum convolution
            all_emb = torch.sparse.mm(adj, all_emb) + 0.8 * all_emb  # momentum parameter
            final.append(all_emb)
        final_emb = torch.mean(torch.stack(final, dim=1), dim=1, dtype=torch.float64)
        return final_emb

    def forward(self, student_id, exercise_id, knowledge):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_exercise_emb1 = self.convolution(self.exercise_emb, self.exercise_adj1)
        convolved_exercise_emb2 = self.convolution(self.exercise_emb, self.exercise_adj2)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        batch_student = f.embedding(student_id, convolved_student_emb)
        batch_exercise1 = f.embedding(exercise_id, convolved_exercise_emb1)
        batch_exercise2 = f.embedding(exercise_id, convolved_exercise_emb2)
        gate_signal = self.sigmoid(self.w1(batch_exercise1) + self.w2(batch_exercise2))
        fused_exercise_feature = gate_signal * batch_exercise1 + (1 - gate_signal) * batch_exercise2 # Gating mechanism fusion

        student_feature = f.leaky_relu(self.student_emb2feature(batch_student), negative_slope=self.leaky) #embedding coversion
        exercise_feature = f.leaky_relu(self.exercise_emb2feature(fused_exercise_feature), negative_slope=self.leaky)

        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)
        discrimination = torch.sigmoid(self.exercise_emb2discrimination(fused_exercise_feature))

        state = discrimination * (student_feature @ knowledge_feature.T
                                  - exercise_feature @ knowledge_feature.T) * knowledge
        state = self.state2response(state)

        c_s_h1_loss = self.contrastive_loss(batch_exercise1, batch_exercise2)
        c_s_h2_loss = self.contrastive_loss(batch_exercise2, batch_exercise1)
        contrastive_loss = c_s_h1_loss + c_s_h2_loss

        mmd_loss = compute_mmd(batch_exercise1, batch_exercise2)
        return state.view(-1), contrastive_loss, mmd_loss

    def apply_clipper(self):
        for layer in self.state2response:
            if isinstance(layer, nn.Linear):
                layer.apply(self.clipper)

    def contrastive_loss(self, h1, h2):
        # enhanced fusion part
        t = 0.5
        batch_size = h1.shape[0]
        negatives_mask = (~torch.eye(batch_size, batch_size,
                          dtype=bool)).to(self.device).float()
        z1 = F.normalize(h1, dim=1)
        z2 = F.normalize(h2, dim=1)
        similarity_matrix1 = F.cosine_similarity(
            z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
        positives = torch.exp(torch.diag(similarity_matrix1) / t)
        negatives = negatives_mask * torch.exp(similarity_matrix1 / t)
        loss_partial = -torch.log(positives / (positives + torch.sum(negatives, dim=1)))
        loss = torch.sum(loss_partial) / batch_size
        return loss

    def get_proficiency_level(self):
        convolved_student_emb = self.convolution(self.student_emb, self.student_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)

        student_feature = f.leaky_relu(self.student_emb2feature(convolved_student_emb), negative_slope=self.leaky)
        knowledge_feature = f.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=self.leaky)

        return torch.sigmoid(student_feature @ knowledge_feature.T).detach().cpu().numpy()

class HyperCDM:
    def __init__(self, student_num, exercise_num, knowledge_num, feature_dim, emb_dim=64, layers=4,
                 device="cpu"):
        self.net: HSCD_Net

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.layers = layers

        self.student_hyper = None
        self.exercise_hyper = None
        self.knowledge_hyper = None
        self.h_hyper = None

        self.device = device

    def train(self, train_set, valid_set=None, q_matrix=None, r_matrix=None, epoch=4, lr=0.0001):
        # train is transformed
        if self.student_hyper is None or self.exercise_hyper is None or self.knowledge_hyper is None:
            raise RuntimeError("Use hyperbuild() method first")
        self.net = HSCD_Net(self.student_num, self.exercise_num, self.knowledge_num,
                            self.feature_dim, self.emb_dim,
                            student_adj=self.student_hyper.to_tensor_nadj().to(self.device),
                            exercise_adj1=self.h_hyper.to_tensor_nadj().to(self.device),
                            exercise_adj2=self.exercise_hyper.to_tensor_nadj().to(self.device),
                            knowledge_adj=self.knowledge_hyper.to_tensor_nadj().to(self.device),
                            layers=self.layers,
                            device=self.device)
        self.net.to(self.device)
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0005)
        total_params = sum(p.numel() for p in self.net.parameters())
        print("Total number of parameters of HGCDM: {}".format(total_params))

        for epoch_i in range(epoch):
            self.net.train()

            all_losses = []
            epoch_losses = []
            contrastive_losses = []
            mmd_losses = []
            for batch_data in tqdm(train_set, "Epoch {}".format(epoch_i + 1)):
                student_id, exercise_id, knowledge, y = batch_data
                student_id: torch.Tensor = student_id.to(self.device)
                exercise_id: torch.Tensor = exercise_id.to(self.device)
                knowledge = knowledge.to(self.device)
                y: torch.Tensor = y.to(self.device)

                pred_y, contrastive_loss, mmd_loss = self.net.forward(student_id, exercise_id, knowledge)
                bce_loss = bce_loss_function(pred_y, y)
                optimizer.zero_grad()
                all_loss = bce_loss + 0.001 * contrastive_loss + 0.001 * mmd_loss
                all_loss.backward()
                optimizer.step()
                self.net.apply_clipper()

                all_losses.append(all_loss.mean().item())
                epoch_losses.append(bce_loss.mean().item())
                contrastive_losses.append(contrastive_loss.mean().item())
                mmd_losses.append(mmd_loss.mean().item())
            print("[Epoch %d] average allloss: %.6f, bce_loss: %.6f, contrastive_loss:%.6f, mmd_loss: %.6f" % (epoch_i + 1, float(np.mean(all_losses)),float(np.mean(epoch_losses)),float(np.mean(contrastive_losses)),float(np.mean(mmd_losses))))

            if valid_set is not None:
                pprint(self.eval(epoch_i, valid_set, q_matrix, r_matrix))

    def eval(self, epoch_i, test_set, q_matrix=None, r_matrix=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        proficiency = self.net.get_proficiency_level()
        for batch_data in tqdm(test_set, "Evaluating"):
            student_id, exercise_id, knowledge, y = batch_data
            student_id: torch.Tensor = student_id.to(self.device)
            exercise_id: torch.Tensor = exercise_id.to(self.device)
            knowledge = knowledge.to(self.device)
            y: torch.Tensor = y.to(self.device)
            pred_y, contrastive_loss, mmd_loss = self.net.forward(student_id, exercise_id, knowledge)
            y_pred.extend(pred_y.detach().cpu().tolist())
            y_true.extend(y.tolist())
        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        auc = roc_auc_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        mae = mean_absolute_error(y_true, y_pred)
        f1 = f1_score(y_true, np.array(y_pred) >= 0.5)
        if q_matrix is None or r_matrix is None:
            print("Evaluation ACC: %.6f, AUC: %.6f, RMSE:%.6f, MAE:%.6f, F1: %.6f" % (acc, auc, rmse, mae, f1))
            return {"acc": acc, "auc": auc, "rmse": rmse, "f1": f1}
        else:
            doa = DOA(proficiency, q_matrix, r_matrix)
            print("Evaluation ACC: %.6f, AUC: %.6f, RMSE:%.6f, MAE:%.6f,F1: %.6f, DOA: %.6f" % (acc, auc, rmse, mae, f1, doa))
        with open('result/model_val.txt', 'a', encoding='utf8') as f:
            f.write("epoch= %d, ACC: %.6f, AUC: %.6f, RMSE:%.6f, MAE:%.6f,F1: %.6f, DOA: %.6f\n" % (epoch_i + 1, acc, auc, rmse, mae, f1, doa))
        return {"acc": acc, "auc": auc, "rmse": rmse, "mae": mae, "f1": f1, "doa": doa}

    def hyper_build(self, response_logs):
        self.student_hyper = response_logs.hyper_construct("student")
        self.exercise_hyper = response_logs.hyper_construct("exercise")
        self.knowledge_hyper = response_logs.hyper_construct("knowledge")
        self.h_hyper = response_logs.hyper_construct("H")


response_logs = ResponseLogs("data/Math2")
config = response_logs.config
cdm = HyperCDM(int(config["student_num"]), int(config["exercise_num"]), int(config["knowledge_num"]), 512, 16, 5, device="cuda")
cdm.hyper_build(response_logs)
cdm.train(response_logs.transform(choice="train", batch_size=64),
          response_logs.transform(choice="test", batch_size=64),
          q_matrix=response_logs.q_matrix,
          r_matrix=response_logs.get_r_matrix(choice="test"),
          epoch=10)

# measure HI and CI
proficiency_level = cdm.net.get_proficiency_level()

mask = np.zeros((response_logs.config["student_num"], response_logs.config["student_num"]))
scores = [0] * response_logs.config["student_num"]
for line in response_logs.response_logs:
    student_id = line[0]
    score = line[2]
    scores[student_id] += int(score)
for i in tqdm(range(len(scores))):
    for j in range(i + 1, len(scores)):
        if scores[i] == scores[j]:
            mask[i, j] = mask[j, i] = 1
normalized = np.count_nonzero(mask)

sim_r = cosine_similarity(expit(response_logs.get_r_matrix("total")))
sim_hypercdm = euclidean_similarity(proficiency_level)

HI = np.sum(sim_hypercdm * mask)/normalized
CI = np.mean(sim_hypercdm * sim_r)

with open('result/model_val.txt', 'a', encoding='utf8') as f:
    f.write(f"HI: {HI}\n")
    f.write(f"CI: {CI}\n")

print(f"HI: {HI}")
print(f"CI: {CI}")
