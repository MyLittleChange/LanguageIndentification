# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import os
import os.path as op

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from utils.logger import setup_logger
from utils.misc import (mkdir)
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from data_process import LangIden_Dataset_train, LangIden_Dataset_eval
from modeling import Language_Identification
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from progressbar import ProgressBar
import xlwt

amazon_languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
xnli_languages = ['ar', 'el', 'hi', 'ru', 'th', 'tr', 'vi', 'bg', 'sw', 'ur']
stsb_languages = ['it', 'nl', 'pl', 'pt']
all_langs = sorted(list(set(amazon_languages + xnli_languages + stsb_languages)))
id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
label2id = {v: k for k, v in id2label.items()}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True,
                                batch_size=(opts.per_gpu_train_batch_size // len(all_langs)) * opts.num_gpus,
                                num_workers=0,
                                shuffle=True, collate_fn=dataset.data_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=0, shuffle=False, collate_fn=dataset.data_collate)
    return dataloader


def save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-latest')
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)
            # model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def save_checkpoint(model, tokenizer, args, epoch, iteration, optimizer, scheduler, num_trial=10, acc=0.0,
                    res_str=None):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}-acc-{:.4f}'.format(epoch, iteration, acc))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)

            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    if res_str is not None:
        file_name = 'res-acc-{:.4f}.json'.format(acc)
        with open(os.path.join(checkpoint_dir, file_name), 'w') as json_file:
            json_file.write(res_str)
        print('写入完成')
    return checkpoint_dir


def train(args, train_dataloader, val_dataset, model, tokenizer):
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    global_step = args.global_step

    if global_step > 0:
        model_file = os.path.join(args.eval_model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        optimizer.load_state_dict(torch.load(op.join(args.eval_model_dir, 'optimizer.pth')))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # an optimizer.cuda() method for this operation would be nice
        scheduler.load_state_dict(torch.load(op.join(args.eval_model_dir, 'scheduler.pth')))
        logger.info("  Resume from %s", args.eval_model_dir)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.per_gpu_train_batch_size * args.num_gpus * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Total examples = %d", len(train_dataloader) * args.per_gpu_train_batch_size * args.num_gpus)

    global_loss = 0.0
    global_cls_loss = 0.0
    global_contras_loss = 0.0

    model.zero_grad()
    model.train()
    n_examples = 0
    new_step = 0
    pbar_len = len(train_dataloader) // args.gradient_accumulation_steps

    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader) // args.gradient_accumulation_steps, desc='training')
        for step, batch in enumerate(train_dataloader):

            inputs = {'input_ids': batch['input_ids'],
                      'input_mask': batch['input_mask'], 'label': batch['label']}
            cls_loss, contras_loss = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            if args.num_gpus > 1:
                cls_loss = cls_loss.mean()
                contras_loss = contras_loss.mean()

            loss = cls_loss + contras_loss
            n_examples += args.per_gpu_train_batch_size * args.num_gpus
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                cls_loss = cls_loss / args.gradient_accumulation_steps
                contras_loss = contras_loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_loss += loss.item()
            global_cls_loss += cls_loss.item()
            global_contras_loss += contras_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                new_step += 1

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                pbar(step=new_step % pbar_len,
                     info={'Epoch': epoch, 'loss': global_loss / new_step,
                           'cls_loss': global_cls_loss / new_step, 'contras_loss': global_contras_loss / new_step})
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                                               optimizer.param_groups[
                                                                                                   0]["lr"], loss,
                                                                                               global_loss / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler)
                    # evaluation
                if global_step % args.valid_steps == 0 and epoch >= args.epoch_begin:
                    logger.info("Perform evaluation at step: %d" % (global_step))
                    acc, result_str = eval(args, val_dataset, model, tokenizer)
                    model.train()
                    logger.info("Evaluation acc: %f" % (acc))
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, optimizer, scheduler,
                                                     acc=acc, res_str=result_str)
    return checkpoint_dir


def eval(args, test_dataloader, model, tokenizer):
    n_examples = 0
    n_correct = 0

    model = model.module if hasattr(model, 'module') else model
    model.eval()
    pbar = ProgressBar(n_total=len(test_dataloader), desc='testing')
    label_res = {k: {'golden': 0, 'correct': 0, 'pred': 0} for k in label2id.keys()}

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'input_ids': batch['input_ids'],
                      'input_mask': batch['input_mask']}
            pred_label = model.evaluate(**inputs)
            for pred, golden in zip(pred_label, batch['label']):
                pred = id2label[pred.item()]
                n_examples += 1
                label_res[golden]['golden'] += 1
                label_res[pred]['pred'] += 1
                if pred == golden:
                    n_correct += 1
                    label_res[golden]['correct'] += 1
            pbar(step=step,
                 info={'acc': n_correct / n_examples})
    label_score = {k: {'precision': 0, 'recall': 0, 'F1': 0} for k in label2id.keys()}
    for k in label2id.keys():
        precision = label_res[k]['correct'] / label_res[k]['pred']
        recall = label_res[k]['correct'] / label_res[k]['golden']
        f1 = 2 * precision * recall / (precision + recall)
        label_score[k]['precision'] = precision
        label_score[k]['recall'] = recall
        label_score[k]['F1'] = f1

    label_score = json.dumps(label_score, indent=4)
    return n_correct / n_examples, label_score


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_file_train",
                        default="/mnt/inspurfs/user-fs/yangqian/MultiIden/datasets/languageIdentification/train.csv",
                        type=str)
    parser.add_argument("--example_file_dev",
                        default="/mnt/inspurfs/user-fs/yangqian/MultiIden/datasets/languageIdentification/valid.csv",
                        type=str)
    parser.add_argument("--example_file_test",
                        default="/mnt/inspurfs/user-fs/yangqian/MultiIden/datasets/languageIdentification/test.csv",
                        type=str)
    parser.add_argument("--num_gpus", default=2, type=int, help="Workers in dataloader.")

    parser.add_argument("--model_name_or_path",
                        default='/mnt/inspurfs/user-fs/yangqian/MultiIden/pretrained_model/xlm',
                        type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--eval_model_dir", default='./output/XLM_LangIden/checkpoint-latest', type=str, required=False,
                        help="The  directory of the model to perform test/evaluation.")
    parser.add_argument("--output_dir", default='./output/XLM_LangIden', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_local_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--load_DecP", action='store_true', help="Whether to load pretrained decoder.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out in BERT.")

    parser.add_argument("--per_gpu_train_batch_size", default=120, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=0, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization.")
    # for Constrained Beam Search
    parser.add_argument("--epoch_begin", default=1, type=int)
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="Run validation begin")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")

    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.distributed = False
    args.device = torch.device('cuda')

    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("Language_Identification", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Load pretrained model and tokenizer
    XLM = XLMRobertaModel.from_pretrained(args.model_name_or_path)

    model = Language_Identification(XLM, num_labels=len(all_langs), id2label=id2label, label2id=label2id)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = LangIden_Dataset_train(tokenizer, args.example_file_train, label2id,
                                               max_seq_length=args.max_seq_length)
        train_dataloader = build_dataloader(train_dataset, True, args)
        val_dataset = LangIden_Dataset_eval(tokenizer, args.example_file_dev, label2id,
                                            max_seq_length=args.max_seq_length)
        val_dataloader = build_dataloader(val_dataset,
                                          False, args)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)
    # inference and evaluation
    elif args.do_test:
        model_file = os.path.join(args.eval_model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        test_dataset = LangIden_Dataset_eval(tokenizer, args.example_file_test, label2id,
                                             max_seq_length=args.max_seq_length)
        test_dataloader = build_dataloader(test_dataset,
                                           False, args)
        acc, result_str = eval(args, test_dataloader, model, tokenizer)
        file_name = 'res-acc-{:.4f}.json'.format(acc)
        with open(os.path.join(args.eval_model_dir, file_name), 'w') as json_file:
            json_file.write(result_str)
        print(result_str)


if __name__ == "__main__":
    main()
