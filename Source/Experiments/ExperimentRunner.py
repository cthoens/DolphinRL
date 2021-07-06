"""
Experiment Runner
=================
Runs a suite of experiments. See SamplesSource/TableVsDeepModel.py for an example.

Usage: ExperimentRunner <experiments module>
    experiments module      Name of a module on the PYTHONPATH that defines a experiment_suite() function

TODO: Document properly
"""

import numpy as np
from Utilities.Eval import MetricsLogger
from console_progressbar import ProgressBar
import sys
import importlib
import asyncio
from Experiments.Experiment import Experiment, Suite


async def run_experiment_module(suite_module_name):
    module = importlib.import_module(suite_module_name)
    suite: Suite = module.experiment_suite()
    await run_experiment_suite(suite)


async def run_experiment_suite(suite):
    class ValidationMetrics:
        def __init__(self):
            self.training_avg_reward = 0.0
            self.validation_avg_reward = 0.0

    np.random.seed(643674)
    try:
        random_state = np.random.get_state()
        for experiment_constructor in suite.experiments:
            experiment: Experiment = experiment_constructor()
            # Restore the initial state of the random number generator such that every experiment start with the same
            # sequence or random numbers
            np.random.set_state(random_state)
            training_metrics_log = MetricsLogger(experiment.method.metrics, max_length=100000)
            validation_metrics = ValidationMetrics()
            validation_metrics_log = MetricsLogger(validation_metrics, max_length=10000)

            pb = ProgressBar(total=suite.episode_count, prefix=f"{experiment.name}", length=50, fill='X')
            pb.print_progress_bar(0)
            for i in range(suite.episode_count):
                reward = await experiment.method.run_episode()
                pb.print_progress_bar(i)
                # print(f" - {reward:.2f}", end="")
                training_metrics_log.append(experiment.method.metrics)

                #
                if i % suite.validation_frequency == suite.validation_frequency - 1:
                    validation_metrics.training_avg_reward = np.average(
                        training_metrics_log.data["episode_reward"][-suite.validation_frequency:])
                    validation_metrics.validation_avg_reward = await suite.validate(experiment)
                    validation_metrics_log.append(validation_metrics)

            print(f"\r{experiment.name}: {validation_metrics.validation_avg_reward:>10.3f}")
            experiment.model.save(f"{experiment.name}-model")
            np.save(f"{experiment.name}-validation_avg_reward.npy",
                    validation_metrics_log.data["validation_avg_reward"], )

    except KeyboardInterrupt:
        print("Keyboard interrupt")


if __name__ == '__main__':
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if len(args) == 0:
        raise SystemExit(f"Usage: {sys.argv[0]} <experiments module>")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_experiment_module(args[0]))
