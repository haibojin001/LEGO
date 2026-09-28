# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg626::lightautoml.addons.uplift.metalearners.TLearner+lightautoml.addons.uplift.metalearners.XLearner
# name: lightautoml_primitive
# summary: Uses lightautoml.addons.uplift.metalearners.TLearner, lightautoml.addons.uplift.metalearners.XLearner across 2 repos
# anchor_symbols: ['lightautoml.addons.uplift.metalearners.TLearner', 'lightautoml.addons.uplift.metalearners.XLearner']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/addons/uplift/base.py::AutoUpliftTX._init_metalearner ---
def _init_metalearner(self, metalearner_name: str, bls: Dict[MLStageFullName, str]) -> MetaLearner:
        """Initialize best metalearner from trained baselearners.

        Args:
            metalearner_name: Metalearner name.
            bls: Mapping metalearner stage to baselearner name.

        Returns:
            Metalearner.

        """
        ml: Optional[MetaLearner] = None
        if metalearner_name == "TLearner":
            ocl = self._get_trained_bl(("outcome_control",), bls[("outcome_control",)]).trained_model
            otl = self._get_trained_bl(("outcome_treatment",), bls[("outcome_treatment",)]).trained_model

            ml = TLearner(control_learner=ocl, treatment_learner=otl)
        elif metalearner_name == "XLearner":
            ocl = self._get_trained_bl(("outcome_control",), bls[("outcome_control",)]).trained_model
            otl = self._get_trained_bl(("outcome_treatment",), bls[("outcome_treatment",)]).trained_model
            pl = self._get_trained_bl(("propensity",), bls[("propensity",)]).trained_model
            ecl = self._get_trained_bl(
                ("outcome_treatment", "effect_control"), bls[("outcome_treatment", "effect_control")]
            ).trained_model
            etl = self._get_trained_bl(
                ("outcome_control", "effect_treatment"),
                bls[("outcome_control", "effect_treatment")],
            ).trained_model

            ml = XLearner(outcome_learners=[ocl, otl], effect_learners=[ecl, etl], propensity_learner=pl)
        else:
            raise Exception()

        return ml

# --- from sberbank-ai-lab__LightAutoML::lightautoml/addons/uplift/base.py::AutoUpliftTX._init_metalearner ---
def _init_metalearner(self, metalearner_name: str, bls: Dict[MLStageFullName, str]) -> MetaLearner:
        """Initialize best metalearner from trained baselearners.

        Args:
            metalearner_name: Metalearner name.
            bls: Mapping metalearner stage to baselearner name.

        Returns:
            Metalearner.

        """
        ml: Optional[MetaLearner] = None
        if metalearner_name == "TLearner":
            ocl = self._get_trained_bl(("outcome_control",), bls[("outcome_control",)]).trained_model
            otl = self._get_trained_bl(("outcome_treatment",), bls[("outcome_treatment",)]).trained_model

            ml = TLearner(control_learner=ocl, treatment_learner=otl)
        elif metalearner_name == "XLearner":
            ocl = self._get_trained_bl(("outcome_control",), bls[("outcome_control",)]).trained_model
            otl = self._get_trained_bl(("outcome_treatment",), bls[("outcome_treatment",)]).trained_model
            pl = self._get_trained_bl(("propensity",), bls[("propensity",)]).trained_model
            ecl = self._get_trained_bl(
                ("outcome_treatment", "effect_control"), bls[("outcome_treatment", "effect_control")]
            ).trained_model
            etl = self._get_trained_bl(
                ("outcome_control", "effect_treatment"),
                bls[("outcome_control", "effect_treatment")],
            ).trained_model

            ml = XLearner(outcome_learners=[ocl, otl], effect_learners=[ecl, etl], propensity_learner=pl)
        else:
            raise Exception()

        return ml
