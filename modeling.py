import random

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Language_Identification(nn.Module):
    r"""

    """

    def __init__(self, xlm, num_labels, id2label, label2id):
        super().__init__()
        self.xlm = xlm
        self.cls_fct = CrossEntropyLoss()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.classifier = ClassificationHead(xlm.config, num_labels)

    def forward(
            self,
            input_ids,
            input_mask,
            label,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

        outputs = self.xlm(
            input_ids,
            attention_mask=input_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        label = label.cuda()
        cls_loss = self.cls_fct(logits.squeeze(), label.squeeze())

        cls_hidden_states = sequence_output[:, 0, :].clone()
        cls_hidden_states = F.normalize(cls_hidden_states, p=2, dim=-1)
        similarities = torch.matmul(cls_hidden_states, cls_hidden_states.clone().permute(1, 0))
        similarities = similarities / 0.05
        group_num = similarities.size(0) // self.num_labels
        # random choose a pos_hidden
        random_pos_num = random.randint(1, group_num - 1)
        contras_loss = []
        CL_label = torch.arange(self.num_labels).cuda()
        for i in range(group_num):
            index = i + random_pos_num
            index = index % group_num
            group_sim = similarities[self.num_labels * i:self.num_labels * (i + 1),
                        self.num_labels * index:self.num_labels * (index + 1)]
            con_loss = self.cls_fct(group_sim, CL_label)
            contras_loss.append(con_loss)
        contras_loss = torch.stack(contras_loss).mean()
        return cls_loss, contras_loss

    def evaluate(self, input_ids, input_mask):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        outputs = self.xlm(input_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        pred_label = torch.argmax(logits, dim=-1)
        return pred_label
