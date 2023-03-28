# Copyright (C) 2023  NASK PIB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import configparser
import gc
import locale
import os
import time
import torch
import torch.cuda.amp as amp
import torch.nn as nn

from collections import deque
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import Checkpoint, DiskSaver, Timer, EarlyStopping
from ignite.metrics import RunningAverage

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from libs.checkpoint import load_checkpoint
from libs.dataloader.spect_stn_rec_dl_producer import SpectStnRecLevelDataLoaderProducer
from libs.utils import get_file_logger, getConfigSection
from libs.model.classifier_ResNet_Attention import Classifier_ResNet_Time_Attention_Drop
from libs.model.classifier_ResNet_NoAttention import Classifier_ResNet_NoAttention

from libs.classes.roc_auc_safe import ROC_AUC_SAFE
from libs.classes.running_average_nan_safe import RunningAverageNanSafe


epoch_abort = 0

"""
Main learning script.
"""


def main():
    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    t0 = time.time()
    gpu_id = 0

    if epoch_abort > 0:
        for _ in range(10):
            print(f'************ epoch abort set to {epoch_abort} ************')

    options = {'dtype': 's1_0256', 'cut_off': False, 'normalize': True, 'scheduler': 'CyclicLR',
               'batch_size': 128, 'n_epochs': 500, 'patience': 25, 'rbsize': [6, 6, 6, 6], 'lr': 1e-7, 'wd': 1e-2}
    gpu_main('RAT6', gpu_id, classifier_class=Classifier_ResNet_Time_Attention_Drop, **options)
    gc.collect()
    torch.cuda.empty_cache()

    options = {'dtype': 's1_0256', 'cut_off': False, 'normalize': True, 'scheduler': 'CyclicLR',
               'batch_size': 128, 'n_epochs': 500, 'patience': 25, 'rbsize': [6, 6, 6, 6], 'lr': 1e-7, 'wd': 1e-2}
    gpu_main('RNA6', gpu_id, classifier_class=Classifier_ResNet_NoAttention, **options)
    gc.collect()
    torch.cuda.empty_cache()

    t1 = time.time()
    print(f'Finished in {t1 - t0:.2f} s.')

    # poweroff()


def gpu_main(prfx, gpu, **args):
    cfg_parser = configparser.ConfigParser()
    pwd_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.abspath(os.path.join(pwd_dir, '../../config'))
    cfg_path = os.path.join(cfg_dir, 'config_coi.ini')
    cfg_parser.read(cfg_path)
    config = cfg_parser[getConfigSection()]

    path_checkpoint = config.get('path_checkpoint')
    path_tb = config.get('path_tb')
    path_log = config.get('path_log')
    os.makedirs(path_checkpoint, exist_ok=True)
    os.makedirs(path_tb, exist_ok=True)
    os.makedirs(path_log, exist_ok=True)

    dtype = args['dtype']
    config = cfg_parser['COMMON']
    batch_size = args["batch_size"]
    cut_off = args["cut_off"]
    normalize = args["normalize"]
    run_id = f'{prfx}_[{dtype}]'

    logger = get_file_logger(path_log, f'_{run_id}', run_id, overwrite=True)

    # Device
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info('Running on: {} out of {} GPU(s)'.format(device, torch.cuda.device_count()))

    # setup data loaders
    logger.info("Creating data loaders")
    dlp = SpectStnRecLevelDataLoaderProducer(cfg_path, dtype, cut_off, normalize, batch_size=batch_size, cap=0)
    dl_tr, dl_vl, dl_ts = dlp.produce()

    # Define models, criterions and optimizers
    logger.info('Creating network')
    classifier_class = args['classifier_class']
    rbsize = args['rbsize']
    lr = args['lr']
    classifier = classifier_class(dlp.dataset.dim_0f, dlp.dataset.dim_1t, rbsize).to(device)

    # Loss
    classification_criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizer_classifier = Adam(classifier.parameters(), lr=lr, weight_decay=args['wd'])

    if args['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer_classifier, 'min', patience=10)
    elif args['scheduler'] == 'CyclicLR':
        scheduler = CyclicLR(optimizer_classifier, base_lr=1e-8, max_lr=1e-7, step_size_up=len(dl_tr) // 2, cycle_momentum=False)
    else:
        raise ValueError(f'Unknown scheduler: {args["scheduler"]}')

    # queues for best checkpoints
    dq_checkpoint_best_loss = deque()
    dq_checkpoint_best_acc = deque()
    dq_checkpoint_best_auc = deque()

    # state dict
    checkpoint_state_dict = {
        'classifier': classifier,
        'optimizer_classifier': optimizer_classifier,
        'scheduler': scheduler,
    }

    # Tensorboard writers
    path_tbr = os.path.join(path_tb, run_id)
    os.makedirs(path_tbr, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=path_tbr)

    # Init
    checkpoint_count = config.getint("checkpoint_count")
    checkpoint_every = config.getint("checkpoint_every")
    checkpoint_resume = config.get("checkpoint_resume")

    nvlneg = lambda x: x or -1.0
    best_loss = float(nvlneg(config.get("checkpoint_resume_best_loss")))
    best_acc = float(nvlneg(config.get("checkpoint_resume_best_acc")))
    best_auc = float(nvlneg(config.get("checkpoint_resume_best_auc")))

    patience = args['patience']
    start_epoch = 0

    # load pre-trained models
    if checkpoint_resume is not None:
        checkpoint_file = os.path.join(path_checkpoint, checkpoint_resume)
        if os.path.isfile(checkpoint_file):
            start_epoch = load_checkpoint(checkpoint_file, checkpoint_state_dict)
            logger.info(f'Loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    # Ignite it
    trainer: Engine = Engine(training_step)
    trainer.state.device = device
    trainer.state.classifier = classifier
    trainer.state.classification_criterion = classification_criterion
    trainer.state.optimizer_classifier = optimizer_classifier
    trainer.state.scaler = amp.GradScaler()
    trainer.state.scheduler = scheduler
    #
    trainer.state.v0 = torch.empty((0,), dtype=torch.float16, device='cpu')
    trainer.state.v1 = torch.empty((0,), dtype=torch.float16, device='cpu')

    evaluator = Engine(validation_step)
    evaluator.state.device = device
    evaluator.state.classifier = classifier
    evaluator.state.classification_criterion = classification_criterion
    evaluator.state.best_loss = best_loss
    evaluator.state.best_acc = best_acc
    evaluator.state.best_auc = best_auc
    evaluator.state.scheduler = scheduler
    #
    evaluator.state.v0 = torch.empty((0,), dtype=torch.float16, device='cpu')
    evaluator.state.v1 = torch.empty((0,), dtype=torch.float16, device='cpu')

    # Metrics
    attach_metrics(trainer, evaluator, config.getfloat("alpha"))

    # Progress bars
    attach_progressbars(trainer, evaluator)

    # Timers
    attach_timers(trainer, evaluator)

    def score_function(engine):
        metrics = engine.state.metrics
        return -metrics['m_vl_loss']

    if patience > 0:
        esh = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, esh)

    checkpointer = add_chekpointer(trainer, checkpoint_state_dict, path_checkpoint, checkpoint_count, run_id)
    checkpointer_snapper = add_interval_chekpointer(trainer, checkpoint_state_dict, path_checkpoint, checkpoint_every, run_id)

    # Handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        time_elapsed = time.strftime("%M:%S", time.gmtime(engine.state.timer.value()))

        metrics_train = engine.state.metrics

        vals = {
            'mp0': metrics_train['m_mp0'],
            'mp0_sd': metrics_train['m_mp0'] + metrics_train['m_sdp0'],
            'mp1_sd': metrics_train['m_mp1'] - metrics_train['m_sdp1'],
            'mp1': metrics_train['m_mp1'],
        }
        tb_writer.add_scalars(f"2.predicts/mp_train", vals, engine.state.epoch)
        msg = f"[{prfx}] T Results   - Epoch: {engine.state.epoch} loss: {metrics_train['m_tr_loss']:.5f}"
        msg += f" acc: {metrics_train['m_tr_acc']:.5f} auc: {metrics_train['m_tr_auc']:.5f}"
        msg += '\t\t\t\t\t\t\t\t'
        msg += f"{time_elapsed}"
        logger.info(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.state.timer.reset()
        evaluator.run(dl_vl)
        evaluator.state.timer.pause()
        time_elapsed = time.strftime("%M:%S", time.gmtime(evaluator.state.timer.value()))

        metrics_train = engine.state.metrics
        metrics_eval = evaluator.state.metrics

        tb_writer.add_scalars(f"1.metrics/loss", {'trn': metrics_train['m_tr_loss'], 'val': metrics_eval['m_vl_loss']}, engine.state.epoch)
        tb_writer.add_scalars(f"1.metrics/auc", {'trn': metrics_train['m_tr_auc'], 'val': metrics_eval['m_vl_auc']}, engine.state.epoch)
        tb_writer.add_scalars(f"3.a_s_s/acc", {'trn': metrics_train['m_tr_acc'], 'val': metrics_eval['m_vl_acc']}, engine.state.epoch)
        tb_writer.add_scalars(f"3.a_s_s/thr", {'val': metrics_eval['m_thr']}, engine.state.epoch)
        tb_writer.add_scalars(f"3.a_s_s/sens", {'val': metrics_eval['m_vl_sens']}, engine.state.epoch)
        tb_writer.add_scalars(f"3.a_s_s/spec", {'val': metrics_eval['m_vl_spec']}, engine.state.epoch)
        vals = {
            'mp0': metrics_eval['m_mp0'],
            'mp0_sd': metrics_eval['m_mp0'] + metrics_eval['m_sdp0'],
            'mp1_sd': metrics_eval['m_mp1'] - metrics_eval['m_sdp1'],
            'mp1': metrics_eval['m_mp1'],
        }
        tb_writer.add_scalars(f"2.predicts/mp_validate", vals, engine.state.epoch)

        msg = f"[{prfx}] V Results   - Epoch: {engine.state.epoch} loss: {metrics_eval['m_vl_loss']:.5f}"
        msg += f" acc: {metrics_eval['m_vl_acc']:.5f} auc: {metrics_eval['m_vl_auc']:.5f}"
        msg += f" sens: {metrics_eval['m_vl_sens']:.5f} spec: {metrics_eval['m_vl_spec']:.5f}"
        msg += f"\t{time_elapsed}"
        logger.info(msg)

        epoch = trainer.state.epoch
        tmp_state_dict = {
            'classifier': classifier.state_dict(),
            'optimizer_classifier': optimizer_classifier.state_dict(),
            'scheduler': scheduler,
        }

        loss = metrics_eval['m_vl_loss']     # the lower the better
        acc = metrics_eval['m_vl_acc']       # the higher the better
        auc = metrics_eval['m_vl_auc']       # the higher the better
        sens = metrics_eval['m_vl_sens']
        spec = metrics_eval['m_vl_spec']

        if epoch_abort > 0:
            return

        if evaluator.state.best_loss < 0 or loss < evaluator.state.best_loss:
            evaluator.state.best_loss = loss
            checkpoint_file = f'loss_{run_id}_{epoch:04d}__L_{loss:.5f}__ACC_{acc:.5f}__AUC_{auc:.5f}__SN_{sens:.5f}__SP_{spec:.5f}.pt'
            checkpoint_fpath = os.path.join(path_checkpoint, checkpoint_file)
            dq_checkpoint_best_loss.append(checkpoint_fpath)
            while len(dq_checkpoint_best_loss) > checkpoint_count:
                os.unlink(dq_checkpoint_best_loss.popleft())
            torch.save(tmp_state_dict, checkpoint_fpath)

        if evaluator.state.best_acc < 0 or acc > evaluator.state.best_acc:
            evaluator.state.best_acc = acc
            checkpoint_file = f'acc_{run_id}_{epoch:04d}__L_{loss:.5f}__ACC_{acc:.5f}__AUC_{auc:.5f}__SN_{sens:.5f}__SP_{spec:.5f}.pt'
            checkpoint_fpath = os.path.join(path_checkpoint, checkpoint_file)
            dq_checkpoint_best_acc.append(checkpoint_fpath)
            while len(dq_checkpoint_best_acc) > checkpoint_count:
                os.unlink(dq_checkpoint_best_acc.popleft())
            torch.save(tmp_state_dict, checkpoint_fpath)

        if evaluator.state.best_auc < 0 or auc > evaluator.state.best_auc:
            evaluator.state.best_auc = auc
            checkpoint_file = f'auc_{run_id}_{epoch:04d}__L_{loss:.5f}__ACC_{acc:.5f}__AUC_{auc:.5f}__SN_{sens:.5f}__SP_{spec:.5f}.pt'
            checkpoint_fpath = os.path.join(path_checkpoint, checkpoint_file)
            dq_checkpoint_best_auc.append(checkpoint_fpath)
            while len(dq_checkpoint_best_auc) > checkpoint_count:
                os.unlink(dq_checkpoint_best_auc.popleft())
            torch.save(tmp_state_dict, checkpoint_fpath)

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_histogram_data(engine_dummy):
        tb_writer.add_histogram('train/miss', trainer.state.v0, trainer.state.epoch)
        tb_writer.add_histogram('train/stn', trainer.state.v1, trainer.state.epoch)
        trainer.state.v0 = torch.empty((0,), dtype=torch.float16, device='cpu')
        trainer.state.v1 = torch.empty((0,), dtype=torch.float16, device='cpu')

    @evaluator.on(Events.COMPLETED)
    def evaluator_histogram_data(engine_dummy):
        tb_writer.add_histogram('validation/miss', evaluator.state.v0, trainer.state.epoch)
        tb_writer.add_histogram('validation/stn', evaluator.state.v1, trainer.state.epoch)
        evaluator.state.v0 = torch.empty((0,), dtype=torch.float16, device='cpu')
        evaluator.state.v1 = torch.empty((0,), dtype=torch.float16, device='cpu')

    @trainer.on(Events.EPOCH_STARTED)
    def reset_train_timer(engine):
        engine.state.timer.reset()

    # Set the initial epoch
    @trainer.on(Events.STARTED)
    def setup_state(engine):
        engine.state.epoch = start_epoch
        checkpointer._iteration = start_epoch
        checkpointer_snapper._iteration = start_epoch

    # Start training
    trainer.run(dl_tr, max_epochs=args['n_epochs'])


def training_step(engine, batch):
    with amp.autocast():
        # Init variables
        device = engine.state.device
        classifier = engine.state.classifier
        classification_criterion = engine.state.classification_criterion
        optimizer_classifier = engine.state.optimizer_classifier
        scaler = engine.state.scaler

        # Prepare input batch
        inputs, classes = batch
        inputs = inputs.to(device, non_blocking=True)
        classes = classes.to(device, non_blocking=True)
        classes = torch.unsqueeze(classes, 1).float()

        classifier.train()
        classifier.zero_grad()
        y_pred = classifier(inputs)

        acc, sens, spec, mp0, sdp0, mp1, sdp1, thr = sens_spec(y_pred, classes)
        classification_loss = classification_criterion(y_pred, classes)

    scaler.scale(classification_loss).backward()
    scaler.step(optimizer_classifier)
    scaler.update()

    scheduler = engine.state.scheduler
    if isinstance(scheduler, CyclicLR):
        scheduler.step()

    yp0 = torch.ravel(torch.sigmoid(y_pred[classes == 0]))
    yp1 = torch.ravel(torch.sigmoid(y_pred[classes == 1]))
    engine.state.v0 = torch.cat((engine.state.v0, yp0.detach().cpu()), dim=0)
    engine.state.v1 = torch.cat((engine.state.v1, yp1.detach().cpu()), dim=0)

    if 0 < epoch_abort <= engine.state.iteration:
        engine.state.iteration = 0
        engine.terminate_epoch()

    return {
        "y_pred": y_pred,
        "y": classes,
        "tr_acc": acc,
        "tr_loss": classification_loss,
        "mp0": mp0,
        "sdp0": sdp0,
        "mp1": mp1,
        "sdp1": sdp1,
        "thr": thr
    }


def validation_step(engine, batch):
    # Init variables
    device = engine.state.device
    classifier = engine.state.classifier
    scheduler = engine.state.scheduler
    classification_criterion = engine.state.classification_criterion

    classifier.eval()

    input, classes = batch
    classes = classes.to(device, non_blocking=True)
    classes = torch.unsqueeze(classes, 1).float()

    with amp.autocast():
        with torch.no_grad():
            input = input.to(device, non_blocking=True)
            y_pred = classifier(input)
            acc, sens, spec, mp0, sdp0, mp1, sdp1, thr = sens_spec(y_pred, classes)
            classification_loss = classification_criterion(y_pred, classes)

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(classification_loss)

    yp0 = torch.ravel(torch.sigmoid(y_pred[classes == 0]))
    yp1 = torch.ravel(torch.sigmoid(y_pred[classes == 1]))
    engine.state.v0 = torch.cat((engine.state.v0, yp0.detach().cpu()), dim=0)
    engine.state.v1 = torch.cat((engine.state.v1, yp1.detach().cpu()), dim=0)

    return {
        "y_pred": y_pred,
        "y": classes,
        "vl_loss": classification_loss,
        "vl_acc": acc,
        "vl_sens": sens,
        "vl_spec": spec,
        "mp0": mp0,
        "sdp0": sdp0,
        "mp1": mp1,
        "sdp1": sdp1,
        "thr": thr
    }


def sens_spec(results: torch.Tensor, classes: torch.Tensor):
    assert results.shape[0] == classes.shape[0]
    assert results.shape[0] > 0

    results = torch.sigmoid(results)

    pred_for_0 = results[classes == 0]
    pred_for_1 = results[classes == 1]
    sdp0, mp0 = (0, 0) if pred_for_0.shape[0] == 0 else torch.std_mean(pred_for_0, unbiased=False)
    sdp1, mp1 = (0, 0) if pred_for_1.shape[0] == 0 else torch.std_mean(pred_for_1, unbiased=False)

    threshold = 0.5
    if (pred_for_0.shape[0] > 0) and (pred_for_1.shape[0] > 0):
        threshold = (((mp0 + sdp0) + (mp1 - sdp1)) / 2.0).item()

    results_01 = torch.ge(results, threshold).float()
    eqv = torch.eq(results_01, classes)
    nqv = torch.logical_not(eqv)

    assert torch.sum(eqv).item() + torch.sum(nqv).item() == results.shape[0]

    tpt: torch.Tensor = torch.logical_and(eqv, torch.eq(results_01, 1))    # results is a true 1
    fnt: torch.Tensor = torch.logical_and(nqv, torch.eq(results_01, 0))    # results is a false 0
    tnt: torch.Tensor = torch.logical_and(eqv, torch.eq(results_01, 0))    # results is a true 0
    fpt: torch.Tensor = torch.logical_and(nqv, torch.eq(results_01, 1))    # results is a false 1

    tp = torch.sum(tpt)
    tn = torch.sum(tnt)
    fp = torch.sum(fpt)
    fn = torch.sum(fnt)

    acc = torch.sum(eqv) / eqv.shape[0]
    sens = torch.ones((1,), dtype=tp.dtype, device=tp.device) if (tp + fn) == 0 else tp / (tp + fn)
    spec = torch.ones((1,), dtype=tp.dtype, device=tp.device) if (tn + fp) == 0 else tn / (tn + fp)
    return acc.item(), sens.item(), spec.item(), mp0.item(), sdp0.item(), mp1.item(), sdp1.item(), threshold


def attach_metrics(trainer, evaluator, alpha):
    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["tr_loss"]
    ).attach(trainer, "m_tr_loss")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["tr_acc"]
    ).attach(trainer, "m_tr_acc")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["mp0"]
    ).attach(trainer, "m_mp0")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["sdp0"]
    ).attach(trainer, "m_sdp0")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["mp1"]
    ).attach(trainer, "m_mp1")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["sdp1"]
    ).attach(trainer, "m_sdp1")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["thr"]
    ).attach(trainer, "m_thr")

    # RunningAverage(ROC_AUC(), alpha=alpha).attach(trainer, "m_tr_auc")
    RunningAverageNanSafe(ROC_AUC_SAFE(), alpha=alpha).attach(trainer, "m_tr_auc")

    # --------------------------------------------------- #

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["vl_loss"]
    ).attach(evaluator, "m_vl_loss")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["vl_sens"]
    ).attach(evaluator, "m_vl_sens")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["vl_spec"]
    ).attach(evaluator, "m_vl_spec")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["vl_acc"]
    ).attach(evaluator, "m_vl_acc")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["mp0"]
    ).attach(evaluator, "m_mp0")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["sdp0"]
    ).attach(evaluator, "m_sdp0")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["mp1"]
    ).attach(evaluator, "m_mp1")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["sdp1"]
    ).attach(evaluator, "m_sdp1")

    RunningAverage(
        alpha=alpha,
        output_transform=lambda x: x["thr"]
    ).attach(evaluator, "m_thr")

    # RunningAverage(ROC_AUC(), alpha=alpha).attach(evaluator, "m_vl_auc")
    RunningAverageNanSafe(ROC_AUC_SAFE(), alpha=alpha).attach(evaluator, "m_vl_auc")


def attach_progressbars(trainer, evaluator):
    ProgressBar(persist=True,
                bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar:30}{postfix} [{elapsed}<{remaining}]').attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        closing_event_name=Events.EPOCH_COMPLETED,
    )

    ProgressBar(persist=True,
                bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar:30}{postfix} [{elapsed}<{remaining}]').attach(
        evaluator,
        event_name=Events.ITERATION_COMPLETED,
        closing_event_name=Events.EPOCH_COMPLETED,
    )


def attach_timers(trainer, evaluator):
    t = Timer()
    t.attach(trainer)
    trainer.state.timer = t

    t = Timer()
    t.attach(evaluator)
    evaluator.state.timer = t


def add_chekpointer(trainer, checkpoint_state_dict, path_checkpoint, checkpoint_count, run_id):
    checkpointer = Checkpoint(
        checkpoint_state_dict,
        DiskSaver(path_checkpoint, create_dir=True, require_empty=False),
        n_saved=checkpoint_count,
        global_step_transform=lambda engine, event_name: engine.state.epoch,
        filename_prefix=f'ch_{run_id}_',
    )

    if epoch_abort == 0:
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpointer
        )

    return checkpointer


def add_interval_chekpointer(trainer, checkpoint_state_dict, path_checkpoint, checkpoint_every, run_id):
    checkpointer = Checkpoint(
        checkpoint_state_dict,
        DiskSaver(path_checkpoint, create_dir=True, require_empty=False),
        n_saved=None,
        global_step_transform=lambda engine, event_name: engine.state.epoch,
        filename_prefix=f'snp_{run_id}_',
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=checkpoint_every),
        checkpointer
    )

    return checkpointer


if __name__ == '__main__':
    main()
