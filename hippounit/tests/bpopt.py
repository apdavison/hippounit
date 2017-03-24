"""
Tests of the neuron responses to current steps of different amplitudes match experimental data.

The responses are quantified by extracting features from the voltage traces using eFEL.

Reference data (features extracted from experimental recordings) and experimental protocol configurations
 are extracted from .zip files produced by BluePyOpt.

Andrew Davison, UNIC, CNRS.
March 2017
"""

import os.path
from datetime import datetime
import json
from zipfile import ZipFile
import numpy as np
import sciunit
from sciunit.scores import ZScore
from neuronunit.capabilities import ProducesMembranePotential, ReceivesSquareCurrent
import neo
import efel
import matplotlib.pyplot as plt


class RMSZScore(ZScore):
    """
    Calculates the z-score for one or more variables and returns the root mean square
    """

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a z-score from an observation and a prediction.
        """
        scores = []
        table = []  # store intermediate results
        for obs, pred in zip(observation, prediction):
            assert obs.keys() == pred.keys()
            assert len(obs) == 1
            #print("O" + str(obs))
            #print("P" + str(pred))
            scores.append(ZScore.compute(list(obs.values())[0],
                                         list(pred.values())[0]).score)
            #print("Z" + str(scores[-1]) + "\n")
            key = obs.keys()[0]
            table.append((key, obs[key]["mean"], obs[key]["std"], pred[key]["value"], scores[-1]))
        sc = np.sqrt(np.mean(np.square(scores)))
        print("RMS(Z) " + str(sc))
        score = cls(sc, related_data={'score_table': table})
        return score

    _description = ('The root-mean-square of the z-scores for multiple variables')


class MultipleCurrentStepTest(sciunit.Test):
    """
    Tests of the neuron responses to current steps of different amplitudes match
    experimental data.

    The responses are quantified by extracting features from the voltage traces
    using eFEL.
    """
    required_capabilities = (ProducesMembranePotential, ReceivesSquareCurrent)
    score_type = RMSZScore

    def __init__(self, observation=None, name=None, protocol=None, plot_figure=False):
        sciunit.Test.__init__(self, observation, name)
        self.plot_figure = plot_figure
        if protocol is None:
            raise ValueError("Must provide a stimulation protocol")
        self.protocol = protocol
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    def validate_observation(self, observation):
        """
        Checks that the observation has the correct format, i.e.

        - a top-level dict with one entry per current step.
            - the key should be a label for the step
            - the value should be a dict containing one entry per feature of the voltage trace
                - the key of the feature dict should be a label for the feature
                - the value should be a dict with keys 'mean' and 'value'
        """
        pass   # todo

    def generate_prediction(self, model, verbose=False):
        use_cache = True
        cache_filename = "results.pkl"
        if use_cache and os.path.exists(cache_filename):
            io = neo.io.get_io(cache_filename)
            self.recordings = io.read_block()
        else:
            self.recordings = self._run_simulations(model)
            io = neo.io.PickleIO(cache_filename)
            io.write_block(self.recordings)
        if self.plot_figure:
            for i, seg in enumerate(self.recordings.segments):
                plt.plot(seg.analogsignals[0].times.rescale('ms'),
                         seg.analogsignals[0].rescale('mV').magnitude + i * 110.0,
                         label=seg.name)
            plt.legend()
            self.figure_path = "{}_{}_{}.png".format(self.name, model.name, self.timestamp)
            plt.savefig(self.figure_path)
        return self._calculate_features(self.recordings)

    def _run_simulations(self, model):
        """For each step in the protocol, run simulation and store recordings"""
        recordings = neo.Block()
        for step_name, step in self.protocol.items():
            segment = neo.Segment(name=step_name)
            recordings.segments.append(segment)
            segment.block = recordings

            model.inject_current(step["stimuli"])
            model.run(tstop=step["total_duration"])
            signal = model.get_membrane_potential()
            stimulus_on =  neo.Epoch(time=step["stimuli"]["delay"],
                                     duration=step["stimuli"]["duration"],
                                     label="stimulus")
            segment.analogsignals.append(signal)
            segment.epochs.append(stimulus_on)
        return recordings

    def _calculate_features(self, recordings):
        """For each recorded step, calculate the features."""
        features_from_simulation = {}
        for segment in recordings.segments:
            step_name = segment.name
            feature_names = self.observation[step_name].keys()
            trace = {
                'T': segment.analogsignals[0].times.rescale('ms').magnitude,
                'V': segment.analogsignals[0].rescale('mV').magnitude,
                'stim_start': [segment.epochs[0].time],
                'stim_end': [segment.epochs[0].time + segment.epochs[0].duration]
            }

            features = efel.getFeatureValues([trace], feature_names)[0]
            features_from_simulation[step_name] = dict([(k, {'value': v[0]})
                                                        for k, v in features.items()])
        return features_from_simulation

    def compute_score(self, observation, prediction, verbose=False):
        """
        Generates a score given the observations provided in the constructor
        and the prediction generated by generate_prediction.
        """
        # reformat the observations and predictions into the form needed by RMSZScore
        # i.e. from dict of dicts into a flat list of dicts
        observation_list = []
        prediction_list = []
        for step_name in observation:
            for feature_name in observation[step_name]:
                key = "{}_{}".format(step_name, feature_name)
                observation_list.append({key: observation[step_name][feature_name]})
                prediction_list.append({key: prediction[step_name][feature_name]})
        return self.score_type.compute(observation_list, prediction_list)

    def bind_score(self, score, model, observation,
                   prediction):
        """
        For the user to bind additional features to the score.
        """
        if hasattr(self, "figure_path"):
            score.related_data["figure"] = self.figure_path
        return score



class BluePyOpt_MultipleCurrentStepTest(MultipleCurrentStepTest):
    """
    Tests if the neuron responses to current steps of different amplitudes match
    experimental data.

    The responses are quantified by extracting features from the voltage traces
    using eFEL.

    Experimental protocol definitions and experimental features obtained from
    zip files produced by BluePyOpt
    """

    def __init__(self, observation=None, name=None, plot_figure=False):
        with ZipFile(observation) as zf:
            top_level_directory = os.path.splitext(os.path.basename(observation))[0]
            # load the protocol definition and the reference data
            with zf.open(top_level_directory + "/config/protocols.json") as fp:
                protocols = json.load(fp)
                assert len(protocols) == 1
                protocol_name = protocols.keys()[0]

            with zf.open(top_level_directory + "/config/features.json") as fp:
                reference_features = json.load(fp)
            assert reference_features.keys()[0] == protocol_name

        # reformat the reference_features dict into the necessary form
        observations = {}
        for step, value in reference_features[protocol_name].items():
            observations[step] = {}
            for feature_name, (mean, std) in value["soma"].items():
                observations[step][feature_name] = {"mean": mean, "std": std}

        # reformat the protocol definition into the form requested by NeuronUnit
        protocol = {}
        for step_name, content in protocols[protocol_name].items():
            stim = content["stimuli"][0]
            stim["amplitude"] = stim["amp"]
            protocol[step_name] = {
                "stimuli": stim,
                "total_duration": stim["totduration"]
            }
            del stim["amp"]
            del stim["totduration"]

        MultipleCurrentStepTest.__init__(self,
                                         observation=observations,
                                         name=name,
                                         protocol=protocol,
                                         plot_figure=plot_figure)
