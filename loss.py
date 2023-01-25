import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def non_saturating_loss(x, dim=-1):
    max_idx = torch.argmax(x, dim=dim, keepdim=True)
    one_hots = torch.zeros_like(x).scatter(dim, max_idx, 1).to(x.device)
    return - torch.mean(one_hots * x) + torch.log(((1 - one_hots) * torch.exp(x)).sum(dim=dim)).mean()


def mcc_loss(x, reweight=False, dim=-1, class_num=32):
    mcc_loss = 0
    for x_split in x: # (B, L, D) -> (L, D)
        x_split = x_split.unsqueeze(0)
        p = x_split.softmax(dim) # (1, L, D)
        p = p.squeeze(0) # (L, D)

        if reweight: # (1, L, D) * (L, 1)
            target_entropy_weight = softmax_entropy(x_split, dim=-1).detach().squeeze(0) # instance-wise entropy (1, L, D)
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
            target_entropy_weight = x_split.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
        else:
            cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss += (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    mcc_loss /= len(x)
    return mcc_loss


def pseudo_labeling_loss(outputs, vocab, processor):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    target = []
    for s in transcription:
        if s == ' ':
            s = '|'
        target.append(vocab[s])

    logp = outputs.log_softmax(1).transpose(1, 0) # L,N,D
    input_len = logp.shape[0]
    tgt_len = len(target)
    loss = ctc_loss(logp, torch.tensor(target).int(), torch.tensor([input_len]), torch.tensor([tgt_len]))
    return loss


def js_divergence(p1, p2):
    total_m = 0.5 * (p1 + p2)
    loss = 0.5 * F.kl_div(torch.log(p1), total_m, reduction="batchmean") + 0.5 * F.kl_div(torch.log(p2), total_m, reduction="batchmean")
    return loss


def get_pl_loss(outputs, transcription, vocab):
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)
    target = []
    for s in transcription:
        if s == ' ':
            s = '|'
        target.append(vocab[s])
    logp = outputs.log_softmax(1).transpose(1, 0) # L,N,D
    input_len = logp.shape[0]
    tgt_len = len(target)
    loss = ctc_loss(logp, torch.tensor(target).int(), torch.tensor([input_len]), torch.tensor([tgt_len]))            
    return loss