import threading


class Callbacks:
    """Handles all registered callbacks for YOLOv5 Hooks."""

    def __init__(self):
        """Initialize callback hooks and the training stop flag."""
        self._callbacks = {
            "on_pretrain_routine_start": [],
            "on_pretrain_routine_end": [],
            "on_train_start": [],
            "on_train_epoch_start": [],
            "on_train_batch_start": [],
            "optimizer_step": [],
            "on_before_zero_grad": [],
            "on_train_batch_end": [],
            "on_train_epoch_end": [],
            "on_val_start": [],
            "on_val_batch_start": [],
            "on_val_image_end": [],
            "on_val_batch_end": [],
            "on_val_end": [],
            "on_fit_epoch_end": [],
            "on_model_save": [],
            "on_train_end": [],
            "on_params_update": [],
            "teardown": [],
        }
        self.stop_training = False

    def register_action(self, hook, name="", callback=None):
        """Add a callable action to the specified callback hook."""
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({"name": name, "callback": callback})

    def run(self, hook, *args, thread=False, **kwargs):
        """Run all actions registered for a hook."""
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for action in self._callbacks[hook]:
            if thread:
                threading.Thread(
                    target=action["callback"],
                    args=args,
                    kwargs=kwargs,
                    daemon=True,
                ).start()
            else:
                action["callback"](*args, **kwargs)