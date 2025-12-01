from enum import Enum, auto
import numpy as np
import nutpie
from nutpie.compiled_pyfunc import from_pyfunc
from .stat_collector import Statistics


class MassMatrixAdaptation(Enum):
    STANDARD = auto()        # standard full matrix
    LOW_RANK = auto()        # low-rank adaptation
    GRADIENT_BASED = auto()  # gradient adaptation


class NutsSampler:
    def __init__(
        self,
        log_func,
        dim,
        cores=None,
        initial_guess=None,
        bounds=None,
    ):
        # fuunction returns (-E, -grad(E)) in x
        self.log_func = log_func
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.dim = dim
        self.sample = None
        self.statistics: Statistics | None = None

    @staticmethod
    def make_expand_func(*unused):
        def expand(x, **unused):
            return {"y": x}

        return expand

    def make_logp_func(self):
        def logp(x: np.ndarray, **unused):
            return self.log_func(x)

        return logp

    def create_sample(
        self,
        *,
        # main params
        draws: int = 100,
        tune: int = 100,
        chains: int = 1,
        target_accept: float = 0.8,
        maxdepth: int = 8,
        max_energy_error: float = 1000.0,
        # mass matrix adaptation mode
        mass_matrix_mode: MassMatrixAdaptation = MassMatrixAdaptation.STANDARD,

        save_warmup: bool = True,
        store_gradient: bool = True,
        store_divergences: bool = True,
        store_unconstrained: bool = True,
        **kwargs,
    ):
        # base arguments for nutpie.sample
        final_args = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "maxdepth": maxdepth,
            "max_energy_error": max_energy_error,
            "save_warmup": save_warmup,
            "store_gradient": store_gradient,
            "store_divergences": store_divergences,
            "store_unconstrained": store_unconstrained,
            **kwargs,
        }


        if mass_matrix_mode == MassMatrixAdaptation.LOW_RANK:
            final_args.update(
                {
                    "low_rank_modified_mass_matrix": True,
                    "store_mass_matrix": False,
                }
            )
        elif mass_matrix_mode == MassMatrixAdaptation.GRADIENT_BASED:
            final_args.update(
                {
                    "use_grad_based_mass_matrix": True,
                    "store_mass_matrix": True,
                }
            )
        else:  # STANDARD
            final_args.update(
                {
                    "use_grad_based_mass_matrix": False,
                    "store_mass_matrix": True,
                }
            )

        # nutpie model
        model = from_pyfunc(
            self.dim,
            self.make_logp_func,
            NutsSampler.make_expand_func,
            [np.dtype("float64")],  # y type
            [(self.dim,)],          # y dim
            ["y"],                  # arg name
        )

        # sampling
        fit = nutpie.sample(model, **final_args)

        # logging
        self.statistics = Statistics(fit)
        self.sample = fit.posterior.y

        return self.sample
