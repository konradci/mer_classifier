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


"""
Checkpoint utils to enable saving model, optimizer etc. as one file
To use with PyTorch Ignite ModelCheckpoint handler

How to use:

(1) Resume from checkpoint
    epoch = 0
    ckp_path = "models/Compare_2ch_checkpoint_1.pth"
    to_load = {'model': model, 'optimizer': optimizer}
    if ckp_path is not None and os.path.isfile(ckp_path):
        epoch = load_checkpoint(ckp_path, to_load)

(2) Save checkpoint with ignite handler
    checkpointer = ModelCheckpoint(dirname=model_dir
                                   , filename_prefix= model.__class__.__name__
                                   , save_interval=1
                                   , n_saved=2
                                   , create_dir=True
                                   , require_empty=False
                                   , save_as_state_dict=False) # IMPORTANT
    to_save = {'model': model, 'optimizer': optimizer}
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              checkpointer,
                              {'checkpoint': setup_checkpoint(to_save)})

(3) Set the initial epoch (to start from that one in checkpoint)
    @trainer.on(Events.STARTED)
    def setup_state(engine):
        engine.state.epoch = epoch
        checkpointer._iteration = epoch

"""
import torch


def setup_checkpoint(to_save):
    checkpoint = {}
    for k, obj in to_save.items():
        checkpoint[k] = obj.state_dict()
    return checkpoint


def load_checkpoint(checkpoint_fpath, to_load):
    epoch = int(checkpoint_fpath.split(".")[-2].split("_")[-1])
    checkpoint = torch.load(checkpoint_fpath)
    load_objects(to_load, checkpoint)
    return epoch


def load_objects(to_load, checkpoint):
    """
    Method to apply `load_state_dict` on the objects from `to_load`
    using states from `checkpoint`.

    Args:
        to_load (dict):         ex. {'model': model, 'optimizer': optimizer}
        checkpoint (dict):
    """
    out = []
    for k, obj in to_load.items():
        if not (hasattr(obj, "state_dict")
           and hasattr(obj, "load_state_dict")):
            raise TypeError("Object {} should have `state_dict` \
                and `load_state_dict` methods".format(type(obj)))
    if not isinstance(checkpoint, dict):
        raise TypeError("Argument checkpoint should be a dictionary, \
            but given {}".format(type(checkpoint)))
    for k, obj in to_load.items():
        if k not in checkpoint:
            raise ValueError("Object labeled by '{}' from `to_load` \
                is not found in the checkpoint".format(k))
        obj.load_state_dict(checkpoint[k])
