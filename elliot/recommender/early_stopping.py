from types import SimpleNamespace
import typing as t

from elliot.utils import logging
import logging as pylog

import math


class EarlyStopping:
    def __init__(self, early_stopping_ns: SimpleNamespace, validation_metric: str, validation_k: int, cutoffs: t.List,
                 simple_metrics: t.List, ):
        self.logger = logging.get_logger(self.__class__.__name__, pylog.DEBUG)
        self.validation_metric = validation_metric
        self.validation_k = validation_k
        self.cutoffs = cutoffs
        self.simple_metrics = simple_metrics
        self.monitor = getattr(early_stopping_ns, "monitor", self.validation_metric)

        if not len(early_stopping_ns.__dict__):
            self.active = False
        else:
            if not hasattr(early_stopping_ns, "patience"):
                self.patience = 0
            else:
                self.patience = early_stopping_ns.patience

            if self.monitor == "loss":
                if not hasattr(early_stopping_ns, "mode"):
                    self.mode = "min"
                elif early_stopping_ns.mode == "auto":
                    self.mode = "min"
                # observed_quantity = self._losses
                self.metric = False

            else:
                if not hasattr(early_stopping_ns, "mode"):
                    self.mode = "max"
                elif early_stopping_ns.mode == "auto":
                    self.mode = "max"

                metric = self.monitor.split("@")
                if metric[0].lower() not in [m.lower() for m in self.simple_metrics]:
                    raise Exception("Early stopping metric must be in the list of simple metrics")

                self.metric_k = int(metric[1]) if len(metric) > 1 else self.validation_k
                if self.metric_k not in self.cutoffs:
                    raise Exception("Validation cutoff must be in general cutoff values")
                self.metric = metric[0]
                # observed_quantity = [r[early_stopping_nsmetric_k]["val_results"][early_stopping_ns.metric] for r in self._results]

            if hasattr(early_stopping_ns, "min_delta"):
                self.min_delta = early_stopping_ns.min_delta

            if hasattr(early_stopping_ns, "rel_delta"):
                self.rel_delta = early_stopping_ns.rel_delta

            if hasattr(early_stopping_ns, "baseline"):
                self.baseline = early_stopping_ns.baseline

            self.verbose = getattr(early_stopping_ns, "verbose", False)
            self.active = True

    def stop(self, losses, results):
        if not self.active:
            return False
        else:
            # check on loss if the last value is nan
            if len(losses) > 0 and (math.isnan(losses[-1]) or math.isinf(losses[-1]) or (not losses[-1])):
                return True
            if not self.metric:
                observed_quantity = losses[-(1 + self.patience):]
            else:
                observed_quantity = [r[self.metric_k]["val_results"][self.metric]
                                     for
                                     r in results[-(1 + self.patience):]]

            if len(observed_quantity) > self.patience:
                # observed_quantity = observed_quantity[-(1+self.patience):]
                # if self.mode == "min":
                #     observed_quantity = observed_quantity[::-1]
                check = []
                candidate_best = observed_quantity[0]
                for p in observed_quantity[1:]:
                    if self.check_conditions(candidate_best, p):
                        check.append(True)
                    else:
                        check.append(False)
                    if self.verbose:
                        self.logger.info(f"Analyzed pair: ({round(candidate_best, 5)}, {round(p, 5)}): {check[-1]}")
                if self.verbose:
                    self.logger.info(f"Check List: {check}")
                if check and all(check):
                    return True
                else:
                    return False

    def check_conditions(self, obs_0: float, obs_1: float):
        if hasattr(self, "min_delta") and hasattr(self, "rel_delta") and hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1)
        elif hasattr(self, "min_delta") and hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "min_delta") and hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_min_delta(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1)
        elif hasattr(self, "baseline") and hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) \
                   or self.condition_baseline(obs_0, obs_1) \
                   or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "min_delta"):
            return self.condition_base(obs_0, obs_1) or self.condition_min_delta(obs_0, obs_1)
        elif hasattr(self, "rel_delta"):
            return self.condition_base(obs_0, obs_1) or self.condition_rel_delta(obs_0, obs_1)
        elif hasattr(self, "baseline"):
            return self.condition_base(obs_0, obs_1) or self.condition_baseline(obs_0, obs_1)
        else:
            return self.condition_base(obs_0, obs_1)

    def condition_base(self, obs_0: float, obs_1: float):
        if self.mode == 'min':
            return obs_0 < obs_1
        elif self.mode == 'max':
            return obs_0 > obs_1
        else:
            raise ValueError("mode option must be in the list [min, max, auto]")

    def condition_min_delta(self, obs_0: float, obs_1: float):
        return (obs_0 - obs_1) <= self.min_delta

    def condition_rel_delta(self, obs_0: float, obs_1: float):
        return (obs_0 - obs_1) <= obs_0 * self.rel_delta

    def condition_baseline(self, obs_0: float, obs_1: float):
        if self.mode == "min":
            return obs_0 >= self.baseline
        elif self.mode == "max":
            return obs_0 <= self.baseline
        else:
            raise ValueError("mode option must be in the list [min, max, auto]")

    def __str__(self):
        return ", ".join([f"{str(k)}: {str(v)}" for k, v in self.__dict__.items()])